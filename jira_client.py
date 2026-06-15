"""Async Jira REST API v2 client with retries and circuit breaker."""
import base64
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import aiohttp
from config import CONFIG


class JiraClient:
    def __init__(self):
        self.base_url = f"{CONFIG.JIRA_URL}/rest/api/2"
        auth_str = f"{CONFIG.JIRA_EMAIL}:{CONFIG.JIRA_API_TOKEN}"
        self.auth_header = "Basic " + base64.b64encode(auth_str.encode()).decode()
        self.headers = {
            "Authorization": self.auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._failure_count = 0
        self._circuit_open_until: Optional[datetime] = None

    def _is_circuit_open(self) -> bool:
        if self._circuit_open_until is None:
            return False
        return datetime.now(timezone.utc) < self._circuit_open_until

    async def _request(self, method: str, endpoint: str, json_data: Optional[dict] = None, retries: int = 3) -> tuple:
        if self._is_circuit_open():
            raise RuntimeError("Jira circuit breaker is open. Try again later.")
        url = f"{self.base_url}{endpoint}"
        last_err = None
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG.JIRA_TIMEOUT)) as session:
                    async with session.request(method, url, headers=self.headers, json=json_data) as resp:
                        text = await resp.text()
                        if 500 <= resp.status < 600:
                            self._failure_count += 1
                            if self._failure_count >= 5:
                                self._circuit_open_until = datetime.now(timezone.utc) + timedelta(minutes=5)
                            raise RuntimeError(f"Jira server error {resp.status}: {text[:500]}")
                        if resp.status == 401:
                            raise RuntimeError(f"Jira auth failed (401): check email and API token.")
                        if resp.status == 400:
                            raise RuntimeError(f"Jira bad request (400): {text[:500]}")
                        self._failure_count = 0
                        return resp.status, text
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                wait = 2 ** attempt
                await asyncio.sleep(wait)
        raise RuntimeError(f"Jira request failed after {retries} retries: {last_err}")

    async def search_recent_issues(self, days: int = CONFIG.DEDUP_DAYS, max_results: int = 50) -> List[Dict[str, Any]]:
        jql = f'project={CONFIG.JIRA_PROJECT_KEY} AND created >= -{days}d AND status != Done ORDER BY created DESC'
        endpoint = f'/search?jql={aiohttp.helpers.quote(jql, safe="")}&fields=summary,description,key&maxResults={max_results}'
        status, text = await self._request("GET", endpoint)
        data = json.loads(text)
        issues = data.get("issues", [])
        return [{"key": i["key"], "summary": i["fields"].get("summary", ""), "description": self._extract_desc(i["fields"].get("description"))} for i in issues]

    def _extract_desc(self, desc) -> str:
        if not desc:
            return ""
        if isinstance(desc, str):
            return desc
        if isinstance(desc, dict):
            return json.dumps(desc, ensure_ascii=False)
        return str(desc)

    async def create_issue(self, summary: str, description: str, priority: str, labels: list) -> Dict[str, Any]:
        payload = {
            "fields": {
                "project": {"key": CONFIG.JIRA_PROJECT_KEY},
                "summary": summary[:255],
                "description": description,
                "issuetype": {"name": "Story"},
                "priority": {"name": priority},
                "labels": labels,
            }
        }
        status, text = await self._request("POST", "/issue", payload)
        data = json.loads(text)
        return {"key": data.get("key"), "url": f"{CONFIG.JIRA_URL}/browse/{data.get('key')}"}

    async def add_comment(self, issue_key: str, comment: str):
        payload = {"body": comment}
        await self._request("POST", f"/issue/{issue_key}/comment", payload)


JIRA_CLIENT = JiraClient()
