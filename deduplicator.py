"""Deduplication logic: LLM semantic batch compare + keyword fallback."""
import logging
from typing import Optional, Dict, Any, List

from jira_client import JIRA_CLIENT
from llm_adapter import LLMProvider
from config import CONFIG

logger = logging.getLogger("teamtrustgate")

BATCH_SIZE = 15


def _load_prompt(name: str) -> str:
    """Загружает промпт из папки prompts/. Поддерживает .md и .txt форматы."""
    for ext in ("md", "txt"):
        path = f"prompts/{name}.{ext}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"✅ Промпт загружен: {path}")
                return content
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Промпт '{name}' не найден. Ожидается prompts/{name}.md или prompts/{name}.txt"
    )


class Deduplicator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.batch_prompt_template = _load_prompt("dedup_batch")
        self.single_prompt_template = _load_prompt("dedup")

    async def check_duplicate(
        self, problem_statement: str
    ) -> Optional[Dict[str, Any]]:
        """
        Проверяет problem_statement на дубликат среди недавних тикетов Jira.

        Стратегия:
        1. Батч-сравнение через LLM (по BATCH_SIZE тикетов за раз)
        2. Если LLM недоступен — keyword fallback (Jaccard similarity)
        """
        candidates = await JIRA_CLIENT.search_recent_issues(days=CONFIG.DEDUP_DAYS)
        if not candidates:
            logger.info("dedup: нет кандидатов в бэклоге")
            return None

        logger.info(f"dedup: проверяем {len(candidates)} тикетов батчами по {BATCH_SIZE}")

        # Собираем все ключи найденных дубликатов по всем батчам
        found_keys: List[str] = []
        llm_failed = True

        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i : i + BATCH_SIZE]
            try:
                duplicate_keys = await self.llm.dedup_compare_batch(
                    problem_statement, batch, self.batch_prompt_template
                )
                llm_failed = False

                if duplicate_keys:
                    found_keys.extend(duplicate_keys)
                    logger.info(f"dedup: батч {i//BATCH_SIZE + 1} — найдены дубликаты: {duplicate_keys}")

            except Exception as e:
                logger.warning(f"dedup: ошибка LLM на батче {i//BATCH_SIZE + 1}: {e}")
                continue

        if llm_failed:
            logger.warning("dedup: LLM недоступен, переключаемся на keyword fallback")
            return self._keyword_fallback(problem_statement, candidates)

        if not found_keys:
            logger.info("dedup: дубликатов не найдено")
            return None

        # Возвращаем первый найденный дубликат (с наибольшим приоритетом в списке)
        key_set = set(found_keys)
        for ticket in candidates:
            if ticket["key"] in key_set:
                logger.info(f"dedup: возвращаем дубликат {ticket['key']}")
                return ticket

        return None

    def _keyword_fallback(
        self, problem: str, candidates: list
    ) -> Optional[Dict[str, Any]]:
        """
        Резервный метод дедупликации через Jaccard similarity по словам.
        Используется только если LLM полностью недоступен.
        Порог: > 0.6 совпадения слов.
        """
        words = set(problem.lower().split())
        best = None
        best_score = 0.0

        for ticket in candidates:
            ticket_words = set(ticket["summary"].lower().split())
            if not ticket_words:
                continue
            union = words | ticket_words
            score = len(words & ticket_words) / len(union) if union else 0.0
            if score > best_score and score > 0.6:
                best_score = score
                best = ticket

        if best:
            logger.info(
                f"dedup fallback: найден кандидат {best['key']} "
                f"(Jaccard={best_score:.2f})"
            )
        else:
            logger.info("dedup fallback: дубликатов не найдено")

        return best
