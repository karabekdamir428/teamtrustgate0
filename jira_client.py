"""Async Jira REST API v2 client with retries and circuit breaker."""
import base64
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from urllib.parse import quote

import aiohttp
from config import CONFIG

logger = logging.getLogger("teamtrustgate")

_NON_RETRYABLE_STATUSES = {400, 401, 403, 404}

_PRIORITY_MAP = {
    "Highest": "Highest",
    "High":    "High",
    "Medium":  "Medium",
    "Low":     "Low",
}


class CircuitBreaker:
    FAILURE_THRESHOLD = 5
    OPEN_DURATION_MINUTES = 5

    def __init__(self):
        self._failure_count = 0
        self._open_until: Optional[datetime] = None

    def is_open(self) -> bool:
        if self._open_until is None:
            return False
        if datetime.now(timezone.utc) < self._open_until:
            return True
        self._reset()
        return False

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self.FAILURE_THRESHOLD:
            self._open_until = datetime.now(timezone.utc) + timedelta(
                minutes=self.OPEN_DURATION_MINUTES
            )
            logger.error(f"jira: circuit breaker ОТКРЫТ до {self._open_until.isoformat()}")

    def record_success(self):
        if self._failure_count > 0:
            logger.info("jira: circuit breaker сброшен после успешного запроса")
        self._reset()

    def _reset(self):
        self._failure_count = 0
        self._open_until = None


class JiraClient:
    def __init__(self):
        self.base_url = f"{CONFIG.JIRA_URL}/rest/api/2"
        auth_str = f"{CONFIG.JIRA_EMAIL}:{CONFIG.JIRA_API_TOKEN}"
        self._headers = {
            "Authorization": "Basic " + base64.b64encode(auth_str.encode()).decode(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._circuit = CircuitBreaker()

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict] = None,
        retries: int = 3,
    ) -> tuple[int, str]:
        if self._circuit.is_open():
            raise RuntimeError(
                "Jira circuit breaker активен — сервис временно недоступен. "
                "Попробуйте через несколько минут."
            )

        url = f"{self.base_url}{endpoint}"
        last_err: Optional[Exception] = None

        for attempt in range(retries):
            try:
                timeout = aiohttp.ClientTimeout(total=CONFIG.JIRA_TIMEOUT)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.request(
                        method, url, headers=self._headers, json=json_data
                    ) as resp:
                        text = await resp.text()

                        if resp.status in _NON_RETRYABLE_STATUSES:
                            raise RuntimeError(
                                f"Jira {resp.status}: {_status_message(resp.status, text)}"
                            )

                        if 500 <= resp.status < 600:
                            self._circuit.record_failure()
                            raise RuntimeError(
                                f"Jira server error {resp.status}: {text[:300]}"
                            )

                        self._circuit.record_success()
                        logger.debug(f"jira: {method} {endpoint} → {resp.status}")
                        return resp.status, text

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    f"jira: попытка {attempt + 1}/{retries} провалилась "
                    f"({type(e).__name__}), ждём {wait}с"
                )
                await asyncio.sleep(wait)

            except RuntimeError:
                raise

        raise RuntimeError(f"Jira недоступна после {retries} попыток: {last_err}")

    async def search_recent_issues(
        self,
        days: int = CONFIG.DEDUP_DAYS,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        jql = (
            f"project={CONFIG.JIRA_PROJECT_KEY} "
            f"AND created >= -{days}d "
            f"AND status != Done "
            f"ORDER BY created DESC"
        )
        endpoint = (
            f"/search"
            f"?jql={quote(jql, safe='')}"
            f"&fields=summary,description,key"
            f"&maxResults={max_results}"
        )
        status, text = await self._request("GET", endpoint)
        data = json.loads(text)
        issues = data.get("issues", [])
        result = [
            {
                "key": i["key"],
                "summary": i["fields"].get("summary", ""),
                "description": self._extract_desc(i["fields"].get("description")),
            }
            for i in issues
        ]
        logger.info(f"jira: найдено {len(result)} тикетов за последние {days} дней")
        return result

    async def search_user_issues(
        self,
        username: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        jql = (
            f"project={CONFIG.JIRA_PROJECT_KEY} "
            f'AND labels = "teamtrustgate" '
            f'AND description ~ "{username}" '
            f"ORDER BY created DESC"
        )
        endpoint = (
            f"/search"
            f"?jql={quote(jql, safe='')}"
            f"&fields=summary,status,priority,key"
            f"&maxResults={max_results}"
        )
        status, text = await self._request("GET", endpoint)
        data = json.loads(text)
        issues = data.get("issues", [])
        result = [
            {
                "key": i["key"],
                "summary": i["fields"].get("summary", ""),
                "status":  i["fields"].get("status", {}).get("name", "—"),
                "priority": i["fields"].get("priority", {}).get("name", "Low"),
            }
            for i in issues
        ]
        logger.info(f"jira: найдено {len(result)} тикетов пользователя {username}")
        return result

    async def get_project_stats(self, days: int = 30) -> Dict[str, Any]:
        """Возвращает статистику по тикетам teamtrustgate из Jira."""
        jql_all = (
            f"project={CONFIG.JIRA_PROJECT_KEY} "
            f'AND labels = "teamtrustgate" '
            f"AND created >= -{days}d "
            f"ORDER BY created DESC"
        )
        endpoint = (
            f"/search"
            f"?jql={quote(jql_all, safe='')}"
            f"&fields=priority,status,created"
            f"&maxResults=200"
        )
        _, text = await self._request("GET", endpoint)
        data   = json.loads(text)
        issues = data.get("issues", [])
        total  = data.get("total", len(issues))

        by_priority: Dict[str, int] = {}
        by_status:   Dict[str, int] = {}
        done_count = 0

        for i in issues:
            p = i["fields"].get("priority", {}).get("name", "Low")
            s = i["fields"].get("status",   {}).get("name", "—")
            by_priority[p] = by_priority.get(p, 0) + 1
            by_status[s]   = by_status.get(s, 0) + 1
            if s.lower() in ("done", "готово", "closed", "resolved"):
                done_count += 1

        return {
            "total":       total,
            "by_priority": by_priority,
            "by_status":   by_status,
            "done":        done_count,
            "in_progress": by_status.get("In Progress", 0) + by_status.get("В работе", 0),
        }

    async def export_issues(self, days: int = 90, max_results: int = 200) -> List[Dict[str, Any]]:
        """
        Выгружает все тикеты teamtrustgate за период со всеми полями для CSV экспорта.
        """
        jql = (
            f"project={CONFIG.JIRA_PROJECT_KEY} "
            f'AND labels = "teamtrustgate" '
            f"AND created >= -{days}d "
            f"ORDER BY created DESC"
        )
        endpoint = (
            f"/search"
            f"?jql={quote(jql, safe='')}"
            f"&fields=summary,status,priority,created,updated,labels"
            f"&maxResults={max_results}"
        )
        _, text = await self._request("GET", endpoint)
        data   = json.loads(text)
        issues = data.get("issues", [])

        result = []
        for i in issues:
            f = i["fields"]
            result.append({
                "key":      i["key"],
                "summary":  f.get("summary", ""),
                "status":   f.get("status", {}).get("name", ""),
                "priority": f.get("priority", {}).get("name", ""),
                "created":  (f.get("created") or "")[:10],
                "updated":  (f.get("updated") or "")[:10],
                "labels":   ", ".join(f.get("labels", [])),
            })
        logger.info(f"jira: экспортировано {len(result)} тикетов за {days} дней")
        return result

    async def create_issue(
        self,
        summary: str,
        description: str,
        priority: str,
        labels: list,
    ) -> Dict[str, Any]:
        jira_priority = _PRIORITY_MAP.get(priority, "Low")
        payload = {
            "fields": {
                "project":     {"key": CONFIG.JIRA_PROJECT_KEY},
                "summary":     summary[:255],
                "description": description,
                "issuetype":   {"name": "Story"},
                "priority":    {"name": jira_priority},
                "labels":      labels,
            }
        }
        status, text = await self._request("POST", "/issue", payload)
        data = json.loads(text)
        key  = data.get("key")
        url  = f"{CONFIG.JIRA_URL}/browse/{key}"
        logger.info(f"jira: тикет создан {key}")
        return {"key": key, "url": url}

    async def delete_issue(self, issue_key: str) -> None:
        try:
            await self._request("DELETE", f"/issue/{issue_key}")
        except RuntimeError as e:
            if "204" in str(e):
                pass
            else:
                raise
        logger.info(f"jira: тикет удалён {issue_key}")

    async def add_comment(self, issue_key: str, comment: str) -> None:
        await self._request("POST", f"/issue/{issue_key}/comment", {"body": comment})
        logger.info(f"jira: комментарий добавлен к {issue_key}")

    def _extract_desc(self, desc) -> str:
        if not desc:
            return ""
        if isinstance(desc, str):
            return desc
        if isinstance(desc, dict):
            return json.dumps(desc, ensure_ascii=False)
        return str(desc)


def _status_message(status: int, text: str) -> str:
    messages = {
        400: f"Некорректный запрос: {text[:300]}",
        401: "Ошибка авторизации — проверьте JIRA_EMAIL и JIRA_API_TOKEN",
        403: "Нет прав доступа — проверьте права токена в Jira",
        404: "Ресурс не найден — проверьте JIRA_URL и JIRA_PROJECT_KEY",
    }
    return messages.get(status, text[:300])


JIRA_CLIENT = JiraClient()
