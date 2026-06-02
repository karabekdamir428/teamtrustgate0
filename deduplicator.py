"""Deduplication logic: LLM semantic compare + keyword fallback."""
import json
from typing import Optional, Dict, Any
from jira_client import JIRA_CLIENT
from llm_adapter import LLMProvider
from config import CONFIG

class Deduplicator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        with open("prompts/dedup.txt", "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    async def check_duplicate(self, problem_statement: str) -> Optional[Dict[str, Any]]:
        candidates = await JIRA_CLIENT.search_recent_issues(days=CONFIG.DEDUP_DAYS)
        if not candidates:
            return None
        for ticket in candidates:
            try:
                result = await self.llm.dedup_compare(problem_statement, ticket["summary"], self.prompt_template)
                if result == "DUPLICATE":
                    return ticket
            except Exception:
                continue
        return self._keyword_fallback(problem_statement, candidates)

    def _keyword_fallback(self, problem: str, candidates: list) -> Optional[Dict[str, Any]]:
        words = set(problem.lower().split())
        best = None
        best_score = 0.0
        for ticket in candidates:
            ticket_words = set(ticket["summary"].lower().split())
            if not ticket_words:
                continue
            intersection = words & ticket_words
            union = words | ticket_words
            score = len(intersection) / len(union) if union else 0
            if score > best_score and score > 0.6:
                best_score = score
                best = ticket
        return best
