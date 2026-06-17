"""RICE-based scoring and priority mapping."""
import logging
from typing import Dict, Any

from llm_adapter import LLMProvider
from parse_utils import parse_llm_number

logger = logging.getLogger("teamtrustgate")

# Соответствует промпту scoring.md
PRIORITY_THRESHOLDS = [
    (250, "Highest"),
    (120, "High"),
    (50,  "Medium"),
    (0,   "Low"),
]

REACH_MAP = {"one_client": 1, "segment": 5, "all_clients": 10}
MAX_SCORE = 400.0


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


def _map_priority(total: float) -> str:
    """Маппинг скора в приоритет по единой таблице из промпта."""
    for threshold, priority in PRIORITY_THRESHOLDS:
        if total >= threshold:
            return priority
    return "Low"


def _is_enterprise_risk(analysis: dict) -> bool:
    """
    Определяет Enterprise-риск через поле is_enterprise_risk из extraction.
    Это поле выставляется LLM в extraction.md — не нужно угадывать по словам.
    """
    return bool(analysis.get("is_enterprise_risk", False))


class Scorer:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompt_template = _load_prompt("scoring")

    async def score(self, analysis: dict) -> Dict[str, Any]:
        try:
            llm_result = await self.llm.score(analysis, self.prompt_template)
            return self._normalize(llm_result, analysis)
        except Exception as e:
            logger.warning(f"scorer: LLM недоступен, переключаемся на fallback. Причина: {e}")
            return self._fallback_score(analysis)

    def _normalize(self, result: dict, analysis: dict) -> Dict[str, Any]:
        """
        Нормализует ответ LLM:
        - Безопасно парсит все числовые поля (защита от '0.8 (medium)')
        - Применяет Enterprise-множитель если LLM его пропустил
        - Обрезает total_score до MAX_SCORE (400)
        """
        reach_score     = parse_llm_number(result.get("reach_score", 1))
        impact_score    = parse_llm_number(result.get("impact_score", 1))
        confidence_score = parse_llm_number(result.get("confidence_score", 0))
        strategy_fit    = parse_llm_number(result.get("strategy_fit_score", 1))
        total           = parse_llm_number(result.get("total_score", 0))

        revenue_at_risk = parse_llm_number(analysis.get("revenue_at_risk", 0))

        # Применяем Enterprise-множитель если LLM его не учёл в total_score
        # Это дублирует правило из промпта как страховку
        if _is_enterprise_risk(analysis) and revenue_at_risk >= 9:
            if reach_score < 5:
                logger.info("scorer: применяем Enterprise-множитель (reach 1→5, strategy→10)")
                reach_score = 5
            if strategy_fit < 10:
                strategy_fit = 10
            # Пересчитываем total если LLM вернул некорректное значение
            recalculated = reach_score * impact_score * confidence_score * strategy_fit
            if abs(total - recalculated) > 1:
                logger.warning(
                    f"scorer: total_score от LLM ({total}) не совпадает с расчётом "
                    f"({recalculated}), используем расчётное значение"
                )
                total = recalculated

        # Жёсткий потолок
        total = min(total, MAX_SCORE)

        priority = _map_priority(total)

        logger.info(
            f"scorer: R={reach_score} I={impact_score} C={confidence_score} "
            f"S={strategy_fit} → total={total} priority={priority}"
        )

        return {
            "reach_score": reach_score,
            "impact_score": impact_score,
            "confidence_score": confidence_score,
            "strategy_fit_score": strategy_fit,
            "total_score": round(total, 1),
            "priority": priority,
            "justification": result.get("justification", ""),
        }

    def _fallback_score(self, analysis: dict) -> Dict[str, Any]:
        """
        Резервный RICE-расчёт без LLM.
        Используется только если LLM полностью недоступен.
        Все правила из промпта scoring.md воспроизведены здесь явно.
        """
        reach       = REACH_MAP.get(analysis.get("reach", "one_client"), 1)
        impact      = parse_llm_number(analysis.get("revenue_at_risk", 5), default=5.0)
        confidence  = parse_llm_number(analysis.get("confidence", 0.5), default=0.5)
        strategy_fit = 5  # нейтральное значение при отсутствии LLM

        revenue_at_risk = parse_llm_number(analysis.get("revenue_at_risk", 0))

        # Enterprise-множитель (из промпта scoring.md)
        if _is_enterprise_risk(analysis) and revenue_at_risk >= 9:
            reach = max(reach, 5)
            strategy_fit = 10
            logger.info("scorer fallback: применён Enterprise-множитель")

        total = reach * impact * confidence * strategy_fit
        total = min(total, MAX_SCORE)  # жёсткий потолок 400

        priority = _map_priority(total)

        logger.info(
            f"scorer fallback: R={reach} I={impact} C={confidence} "
            f"S={strategy_fit} → total={total} priority={priority}"
        )

        return {
            "reach_score": reach,
            "impact_score": impact,
            "confidence_score": confidence,
            "strategy_fit_score": strategy_fit,
            "total_score": round(total, 1),
            "priority": priority,
            "justification": (
                "Fallback scoring (LLM недоступен). "
                "Автоматический пересчёт на основе данных из extraction."
            ),
        }
