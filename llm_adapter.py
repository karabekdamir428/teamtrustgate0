"""LLM Provider Adapter with Gemini, DeepSeek, OpenAI, Anthropic support and Auto-Retry/Failover resilience."""
import json
import re
import asyncio
import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import aiohttp
from config import CONFIG

logger = logging.getLogger("teamtrustgate")

# ── Константы ─────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_BASE_DELAY  = 2.0
_RETRYABLE_MARKERS = ("503", "429", "unavailable", "timeout", "exhausted", "overloaded")


# ── Базовые утилиты ───────────────────────────────────────────────────────

def _build_candidates_list(candidates: list) -> str:
    """Форматирует список кандидатов для промпта дедупликации."""
    return "\n".join(f"- {t['key']}: {t['summary']}" for t in candidates)


def _fill_template(template: str, **kwargs) -> str:
    """Заменяет плейсхолдеры {KEY} в шаблоне промпта."""
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value) if value is not None else "")
    return result


# ── Абстрактный базовый класс ─────────────────────────────────────────────

class LLMProvider(ABC):

    @abstractmethod
    async def _call(self, messages: list, system_prompt: str = "") -> str: ...

    @abstractmethod
    async def analyze(
        self, user_message: str, collected_answers: list, prompt_template: str
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def dedup_compare(
        self, problem_a: str, problem_b: str, prompt_template: str
    ) -> str: ...

    @abstractmethod
    async def dedup_compare_batch(
        self, problem_a: str, candidates: list, prompt_template: str
    ) -> List[str]: ...

    @abstractmethod
    async def score(
        self, analysis: dict, prompt_template: str
    ) -> Dict[str, Any]: ...

    # ── Общие методы (используются всеми провайдерами) ────────────────────

    def _parse_batch_dedup_response(
        self, text: str, candidates: list
    ) -> List[str]:
        """
        Парсит ответ batch-дедупликации.
        Возвращает список всех найденных ключей-дубликатов (не только первый).
        """
        try:
            result = self._extract_json(text)
            duplicates = result.get("duplicates", [])
            if isinstance(duplicates, list):
                # Фикс пункта 5: не логируем ключи — только факт нахождения
                valid = [k for k in duplicates if isinstance(k, str)]
                return valid
        except ValueError:
            # Fallback: ищем ключи тикетов прямо в тексте
            found = []
            for ticket in candidates:
                if ticket["key"] in text:
                    found.append(ticket["key"])
            return found
        return []

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Извлекает JSON из ответа LLM.
        Обрабатывает markdown-обёртки, обрезанные ответы и мусор вокруг JSON.
        """
        text = text.strip()
        # Убираем markdown-блоки
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
        text = text.strip()

        # Прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Ищем JSON-объект внутри текста
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"JSON не найден в ответе LLM: {text[:200]}")

        candidate = match.group()

        # Прямой парсинг найденного фрагмента
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Попытка починить обрезанный JSON (модель упёрлась в лимит токенов)
        for fix in _attempt_json_repairs(candidate):
            try:
                return json.loads(fix)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Не удалось распарсить JSON из ответа LLM: {text[:200]}")

    async def _call_with_retry(self, call_func, *args, **kwargs) -> str:
        """
        Выполняет вызов API с экспоненциальным backoff.
        Ретраит только временные ошибки (429, 503, timeout).
        Пробрасывает сразу постоянные ошибки (401, 400 и т.д.).
        """
        last_err: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await call_func(*args, **kwargs)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                await self._maybe_retry(attempt, str(e))
            except RuntimeError as e:
                error_msg = str(e)
                # Постоянные ошибки (авторизация, плохой запрос) — не ретраим
                if any(code in error_msg for code in ("401", "403", "400")):
                    raise
                if any(marker in error_msg.lower() for marker in _RETRYABLE_MARKERS):
                    last_err = e
                    await self._maybe_retry(attempt, error_msg)
                else:
                    raise

        raise RuntimeError(
            f"LLM API недоступен после {_MAX_RETRIES} попыток: {last_err}"
        )

    async def _maybe_retry(self, attempt: int, error_msg: str):
        if attempt == _MAX_RETRIES:
            raise RuntimeError(
                f"LLM API: попытки исчерпаны ({_MAX_RETRIES}/{_MAX_RETRIES}). "
                f"Последняя ошибка: {error_msg[:200]}"
            )
        delay = (_BASE_DELAY ** attempt) + random.uniform(0.5, 1.5)
        logger.warning(
            f"llm: попытка {attempt}/{_MAX_RETRIES} провалилась. "
            f"Повтор через {delay:.1f}с. Ошибка: {error_msg[:100]}"
        )
        await asyncio.sleep(delay)


def _attempt_json_repairs(text: str) -> List[str]:
    """Генерирует варианты починки обрезанного JSON."""
    candidates = []
    # Убрать хвостовую запятую
    stripped = text.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1]
    # Вариант 1: закрыть строку и объект
    candidates.append(stripped + '"}"' if not stripped.endswith("}") else stripped)
    # Вариант 2: просто закрыть объект
    if not stripped.endswith("}"):
        candidates.append(stripped + "}")
    # Вариант 3: отрезать последнее незакрытое поле
    last_comma = stripped.rfind(",")
    if last_comma > 0:
        candidates.append(stripped[:last_comma] + "}")
    return candidates


# ── Миксин для стандартных методов (OpenAI-совместимый формат) ───────────

class OpenAICompatibleMixin:
    """
    Общая реализация методов analyze/dedup/score для провайдеров
    с OpenAI-совместимым форматом messages (DeepSeek, OpenAI, Anthropic).
    """

    def _fmt_msg(self, content: str) -> dict:
        return {"role": "user", "content": content}

    async def analyze(
        self, user_message: str, collected_answers: list, prompt_template: str
    ) -> Dict[str, Any]:
        context = " | ".join(collected_answers) if collected_answers else ""
        prompt = _fill_template(
            prompt_template,
            USER_MESSAGE=user_message,
            COLLECTED_ANSWERS=context,
            PRODUCT_STRATEGY=CONFIG.PRODUCT_STRATEGY,
        )
        text = await self._call(
            [self._fmt_msg(prompt)],
            system_prompt="You are a product analyst. Respond ONLY with a valid JSON object. No markdown.",
        )
        return self._extract_json(text)

    async def dedup_compare(
        self, problem_a: str, problem_b: str, prompt_template: str
    ) -> str:
        prompt = _fill_template(prompt_template, PROBLEM_A=problem_a, PROBLEM_B=problem_b)
        text = await self._call(
            [self._fmt_msg(prompt)],
            system_prompt="Respond only with DUPLICATE or UNIQUE.",
        )
        return "DUPLICATE" if "DUPLICATE" in text.upper() else "UNIQUE"

    async def dedup_compare_batch(
        self, problem_a: str, candidates: list, prompt_template: str
    ) -> List[str]:
        prompt = _fill_template(
            prompt_template,
            PROBLEM_A=problem_a,
            CANDIDATES_LIST=_build_candidates_list(candidates),
        )
        text = await self._call(
            [self._fmt_msg(prompt)],
            system_prompt="You are a deduplication system. Respond ONLY with a valid JSON object. No markdown.",
        )
        return self._parse_batch_dedup_response(text, candidates)

    async def score(
        self, analysis: dict, prompt_template: str
    ) -> Dict[str, Any]:
        prompt = _fill_template(
            prompt_template,
            ANALYSIS_JSON=json.dumps(analysis, ensure_ascii=False),
            PRODUCT_STRATEGY=CONFIG.PRODUCT_STRATEGY,
        )
        text = await self._call(
            [self._fmt_msg(prompt)],
            system_prompt="You are a product prioritization expert. Respond ONLY with a valid JSON object. No markdown.",
        )
        return self._extract_json(text)


# ── Провайдеры ────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Google Gemini 2.5 Flash."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = CONFIG.GEMINI_MODEL
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{self.model}:generateContent"
        )

    async def _call(self, contents: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload: dict = {
                "contents": contents,
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
            }
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            url = f"{self.base_url}?key={self.api_key}"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)
            ) as session:
                async with session.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Gemini API {resp.status}: {text[:300]}")
                    data = await resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]

        return await self._call_with_retry(_raw_post)

    # Gemini использует свой формат contents (parts), поэтому методы здесь

    async def analyze(
        self, user_message: str, collected_answers: list, prompt_template: str
    ) -> Dict[str, Any]:
        context = " | ".join(collected_answers) if collected_answers else ""
        prompt = _fill_template(
            prompt_template,
            USER_MESSAGE=user_message,
            COLLECTED_ANSWERS=context,
            PRODUCT_STRATEGY=CONFIG.PRODUCT_STRATEGY,
        )
        text = await self._call(
            [{"role": "user", "parts": [{"text": prompt}]}],
            system_prompt="You are a product analyst. Respond ONLY with a valid JSON object. No markdown.",
        )
        return self._extract_json(text)

    async def dedup_compare(
        self, problem_a: str, problem_b: str, prompt_template: str
    ) -> str:
        prompt = _fill_template(prompt_template, PROBLEM_A=problem_a, PROBLEM_B=problem_b)
        text = await self._call(
            [{"role": "user", "parts": [{"text": prompt}]}],
            system_prompt="Respond only with DUPLICATE or UNIQUE.",
        )
        return "DUPLICATE" if "DUPLICATE" in text.upper() else "UNIQUE"

    async def dedup_compare_batch(
        self, problem_a: str, candidates: list, prompt_template: str
    ) -> List[str]:
        prompt = _fill_template(
            prompt_template,
            PROBLEM_A=problem_a,
            CANDIDATES_LIST=_build_candidates_list(candidates),
        )
        text = await self._call(
            [{"role": "user", "parts": [{"text": prompt}]}],
            system_prompt="You are a deduplication system. Respond ONLY with a valid JSON object. No markdown.",
        )
        return self._parse_batch_dedup_response(text, candidates)

    async def score(
        self, analysis: dict, prompt_template: str
    ) -> Dict[str, Any]:
        prompt = _fill_template(
            prompt_template,
            ANALYSIS_JSON=json.dumps(analysis, ensure_ascii=False),
            PRODUCT_STRATEGY=CONFIG.PRODUCT_STRATEGY,
        )
        text = await self._call(
            [{"role": "user", "parts": [{"text": prompt}]}],
            system_prompt="You are a product prioritization expert. Respond ONLY with a valid JSON object. No markdown.",
        )
        return self._extract_json(text)


class DeepSeekProvider(OpenAICompatibleMixin, LLMProvider):
    """DeepSeek Chat — резервный провайдер."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload = {
                "model": CONFIG.DEEPSEEK_MODEL,
                "messages": (
                    [{"role": "system", "content": system_prompt}] if system_prompt else []
                ) + messages,
                "temperature": 0.2,
                "max_tokens": 4096,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)
            ) as session:
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"DeepSeek API {resp.status}: {text[:300]}")
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]

        return await self._call_with_retry(_raw_post)


class OpenAIProvider(OpenAICompatibleMixin, LLMProvider):
    """OpenAI GPT-4o-mini."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload = {
                "model": "gpt-4o-mini",
                "messages": (
                    [{"role": "system", "content": system_prompt}] if system_prompt else []
                ) + messages,
                "temperature": 0.2,
                "max_tokens": 4096,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)
            ) as session:
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"OpenAI API {resp.status}: {text[:300]}")
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]

        return await self._call_with_retry(_raw_post)


class AnthropicProvider(OpenAICompatibleMixin, LLMProvider):
    """Anthropic Claude Haiku."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload = {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 4096,
                "temperature": 0.2,
                "system": system_prompt,
                "messages": messages,
            }
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)
            ) as session:
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Anthropic API {resp.status}: {text[:300]}")
                    data = await resp.json()
                    return data["content"][0]["text"]

        return await self._call_with_retry(_raw_post)


# ── Failover orchestrator ─────────────────────────────────────────────────

class ResilientFailoverProvider(LLMProvider):
    """
    Гибридный провайдер с прозрачным переключением на резервные модели.
    При ошибке основного провайдера последовательно пробует fallback-ы.
    """

    def __init__(self, primary: LLMProvider, fallbacks: List[LLMProvider]):
        self.primary = primary
        self.fallbacks = fallbacks
        self._all = [primary] + fallbacks

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        return await self.primary._call(messages, system_prompt)

    async def _with_failover(self, method: str, *args, **kwargs):
        last_err: Optional[Exception] = None
        for provider in self._all:
            try:
                return await getattr(provider, method)(*args, **kwargs)
            except Exception as e:
                last_err = e
                name = provider.__class__.__name__
                logger.warning(f"llm failover: {name} провалился на [{method}]: {e}")
                if provider is not self._all[-1]:
                    logger.info(f"llm failover: переключаемся на следующий провайдер")
        raise RuntimeError(
            f"Все LLM-провайдеры недоступны для [{method}]: {last_err}"
        )

    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]:
        return await self._with_failover("analyze", user_message, collected_answers, prompt_template)

    async def dedup_compare(self, problem_a: str, problem_b: str, prompt_template: str) -> str:
        return await self._with_failover("dedup_compare", problem_a, problem_b, prompt_template)

    async def dedup_compare_batch(self, problem_a: str, candidates: list, prompt_template: str) -> List[str]:
        return await self._with_failover("dedup_compare_batch", problem_a, candidates, prompt_template)

    async def score(self, analysis: dict, prompt_template: str) -> Dict[str, Any]:
        return await self._with_failover("score", analysis, prompt_template)


# ── Фабрика ───────────────────────────────────────────────────────────────

def get_llm_provider() -> LLMProvider:
    """Инициализирует отказоустойчивую цепочку провайдеров из конфига."""

    def _create(name: str) -> LLMProvider:
        n = name.lower()
        if n == "gemini":
            return GeminiProvider(CONFIG.LLM_API_KEY)
        if n == "deepseek":
            key = CONFIG.DEEPSEEK_API_KEY or CONFIG.LLM_API_KEY
            return DeepSeekProvider(key)
        if n == "openai":
            return OpenAIProvider(CONFIG.LLM_API_KEY)
        if n == "anthropic":
            return AnthropicProvider(CONFIG.LLM_API_KEY)
        raise ValueError(f"Неизвестный LLM_PROVIDER: '{name}'")

    primary_name = CONFIG.LLM_PROVIDER
    primary = _create(primary_name)

    logger.info(f"LLM_PROVIDER = {primary_name}")
    logger.info(f"LLM_API_KEY loaded = {bool(CONFIG.LLM_API_KEY)}")
    logger.info(f"DEEPSEEK_API_KEY loaded = {bool(CONFIG.DEEPSEEK_API_KEY)}")

    fallbacks: List[LLMProvider] = []

    # Авто-добавляем DeepSeek как резерв если основной — Gemini
    if primary_name.lower() == "gemini" and CONFIG.DEEPSEEK_API_KEY:
        try:
            fallbacks.append(_create("deepseek"))
            logger.info("🛡️ Резервный контур DeepSeek успешно интегрирован в отказоустойчивую фабрику.")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось инициализировать резервный DeepSeek: {e}")

    if fallbacks:
        return ResilientFailoverProvider(primary, fallbacks)

    return primary
