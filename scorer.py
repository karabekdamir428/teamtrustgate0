"""RICE-based scoring and priority mapping."""
import json
from typing import Dict, Any
from llm_adapter import LLMProvider
from config import CONFIG

class Scorer:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        with open("prompts/scoring.txt", "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    async def score(self, analysis: dict) -> Dict[str, Any]:
        try:
            llm_result = await self.llm.score(analysis, self.prompt_template)
            return self._normalize(llm_result, analysis)
        except Exception:
            return self._fallback_score(analysis)

    def _normalize(self, result: dict, analysis: dict) -> Dict[str, Any]:
        total = result.get("total_score", 0)
        if isinstance(total, str):
            total = float(total)

        # Жесткое бизнес-правило (Enterprise-множитель) напрямую в коде
        revenue_at_risk = float(analysis.get("revenue_at_risk", 0))
        is_enterprise = "альфа" in str(analysis.get("client_context", "")).lower() or "банк" in str(analysis.get("client_context", "")).lower()
        
        # Если критический риск для крупного клиента — вытягиваем приоритет вверх
        if revenue_at_risk >= 9 and is_enterprise:
            if total < 200:
                total = max(total, 250.0) # Искусственно поднимаем скор для Jira

        if total >= 200:
            priority = "Highest"
        elif total >= 100:
            priority = "High"
        elif total >= 50:
            priority = "Medium"
        else:
            priority = "Low"

        return {
            "reach_score": result.get("reach_score", 0),
            "impact_score": result.get("impact_score", 0),
            "confidence_score": result.get("confidence_score", 0),
            "strategy_fit_score": result.get("strategy_fit_score", 0),
            "total_score": total,
            "priority": priority,
            "justification": result.get("justification", ""),
        }

    def _fallback_score(self, analysis: dict) -> Dict[str, Any]:
        reach_map = {"one_client": 1, "segment": 5, "all_clients": 10}
        reach = reach_map.get(analysis.get("reach", "one_client"), 1)
        impact = float(analysis.get("revenue_at_risk", 5))
        confidence = float(analysis.get("confidence", 0.5))
        strategy_fit = 5
        
        # Защита от занижения в fallback режиме
        if impact >= 9:
            reach = max(reach, 5) # Считаем как важный сегмент
            strategy_fit = max(strategy_fit, 9)

        total = reach * impact * confidence * strategy_fit
        
        # СТРОГИЙ СРЕЗ ДО 400 ПРЯМО В ФОЛБЕКЕ, ЧТОБЫ НЕ БЫЛО 450/400
        total = min(total, 400.0)

        if total >= 250: # Соответствует нашей новой шкале из промпта
            priority = "Highest"
        elif total >= 120:
            priority = "High"
        elif total >= 50:
            priority = "Medium"
        else:
            priority = "Low"

        return {
            "reach_score": reach,
            "impact_score": impact,
            "confidence_score": confidence,
            "strategy_fit_score": strategy_fit,
            "total_score": round(total, 1),
            "priority": priority,
            "justification": "Fallback scoring (LLM failed). Автоматический пересчет веса Enterprise-клиента.",
        }
