"""LLM Provider Adapter with Gemini, DeepSeek, OpenAI, Anthropic support and Auto-Retry/Failover resilience."""
import json
import re
import asyncio
import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import aiohttp
from config import CONFIG

logger = logging.getLogger("teamtrustgate")

class LLMProvider(ABC):
    @abstractmethod
    async def _call(self, messages: list, system_prompt: str = "") -> str: ...

    @abstractmethod
    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def dedup_compare(self, problem_a: str, problem_b: str, prompt_template: str) -> str: ...

    @abstractmethod
    async def score(self, analysis: dict, prompt_template: str) -> Dict[str, Any]: ...

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        # Полноценное удаление markdown-тегов формата json
        text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^```\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Поиск валидного JSON объекта внутри текста, если ИИ прислал лишний текст
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"No JSON found in response: {text[:200]}")

    async def _call_with_retry(self, call_func, *args, **kwargs) -> str:
        """Обертка с экспоненциальным ретраем для обработки ошибок 503, 429 и таймаутов."""
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(1, max_retries + 1):
            try:
                return await call_func(*args, **kwargs)
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                error_msg = str(e)
                # Ловим только сетевые сбои, перегрузки (503) и рейт-лимиты (429)
                if any(marker in error_msg for marker in ["503", "429", "unavailable", "timeout", "exhausted"]):
                    if attempt == max_retries:
                        logger.error(f"❌ [Attempt {attempt}/{max_retries}] Проблемы с API. Попытки исчерпаны.")
                        raise e
                    
                    # Экспоненциальная задержка + случайный джиттер
                    delay = (base_delay ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning(f"⚠️ [Attempt {attempt}/{max_retries}] Сбой API ({error_msg}). Повтор через {delay:.2f} сек...")
                    await asyncio.sleep(delay)
                else:
                    # Если ошибка содержательная (401 Bad Auth, 400 Bad Request) — ретраить бесполезно
                    raise e


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "[https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent](https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent)"

    async def _call(self, contents: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload = {
                "contents": contents,
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
            }
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            url = f"{self.base_url}?key={self.api_key}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)) as session:
                async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Gemini API error {resp.status}: {text[:500]}")
                    data = await resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                    
        return await self._call_with_retry(_raw_post)

    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]:
        context = " | ".join(collected_answers) if collected_answers else ""
        prompt = prompt_template.replace("{USER_MESSAGE}", user_message).replace("{COLLECTED_ANSWERS}", context).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "parts": [{"text": prompt}]}], system_prompt="You are a product analyst. Respond ONLY with a valid JSON object. Do not include markdown block formatting.")
        return self._extract_json(text)

    async def dedup_compare(self, problem_a: str, problem_b: str, prompt_template: str) -> str:
        prompt = prompt_template.replace("{PROBLEM_A}", problem_a).replace("{PROBLEM_B}", problem_b)
        text = await self._call([{"role": "user", "parts": [{"text": prompt}]}])
        return "DUPLICATE" if "DUPLICATE" in text.upper() else "UNIQUE"

    async def score(self, analysis: dict, prompt_template: str) -> Dict[str, Any]:
        analysis_json = json.dumps(analysis, ensure_ascii=False)
        prompt = prompt_template.replace("{ANALYSIS_JSON}", analysis_json).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "parts": [{"text": prompt}]}], system_prompt="You are a product prioritization expert. Respond ONLY with a valid JSON object.")
        return self._extract_json(text)


class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "[https://api.deepseek.com/v1/chat/completions](https://api.deepseek.com/v1/chat/completions)"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload = {
                "model": "deepseek-chat",
                "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
                "temperature": 0.2,
                "max_tokens": 2048,
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)) as session:
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"DeepSeek API error {resp.status}: {text[:500]}")
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                    
        return await self._call_with_retry(_raw_post)

    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]:
        context = " | ".join(collected_answers) if collected_answers else ""
        prompt = prompt_template.replace("{USER_MESSAGE}", user_message).replace("{COLLECTED_ANSWERS}", context).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "content": prompt}], "You are a product analyst. Respond only in JSON.")
        return self._extract_json(text)

    async def dedup_compare(self, problem_a: str, problem_b: str, prompt_template: str) -> str:
        prompt = prompt_template.replace("{PROBLEM_A}", problem_a).replace("{PROBLEM_B}", problem_b)
        text = await self._call([{"role": "user", "content": prompt}], "Respond only DUPLICATE or UNIQUE.")
        return "DUPLICATE" if "DUPLICATE" in text.upper() else "UNIQUE"

    async def score(self, analysis: dict, prompt_template: str) -> Dict[str, Any]:
        analysis_json = json.dumps(analysis, ensure_ascii=False)
        prompt = prompt_template.replace("{ANALYSIS_JSON}", analysis_json).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "content": prompt}], "You are a product prioritization expert. Respond only in JSON.")
        return self._extract_json(text)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "[https://api.openai.com/v1/chat/completions](https://api.openai.com/v1/chat/completions)"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload = {
                "model": "gpt-4o-mini",
                "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
                "temperature": 0.2,
                "max_tokens": 2048,
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)) as session:
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"OpenAI API error {resp.status}: {text[:500]}")
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                    
        return await self._call_with_retry(_raw_post)

    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]:
        context = " | ".join(collected_answers) if collected_answers else ""
        prompt = prompt_template.replace("{USER_MESSAGE}", user_message).replace("{COLLECTED_ANSWERS}", context).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "content": prompt}], "You are a product analyst. Respond only in JSON.")
        return self._extract_json(text)

    async def dedup_compare(self, problem_a: str, problem_b: str, prompt_template: str) -> str:
        prompt = prompt_template.replace("{PROBLEM_A}", problem_a).replace("{PROBLEM_B}", problem_b)
        text = await self._call([{"role": "user", "content": prompt}], "Respond only DUPLICATE or UNIQUE.")
        return "DUPLICATE" if "DUPLICATE" in text.upper() else "UNIQUE"

    async def score(self, analysis: dict, prompt_template: str) -> Dict[str, Any]:
        analysis_json = json.dumps(analysis, ensure_ascii=False)
        prompt = prompt_template.replace("{ANALYSIS_JSON}", analysis_json).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "content": prompt}], "You are a product prioritization expert. Respond only in JSON.")
        return self._extract_json(text)


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "[https://api.anthropic.com/v1/messages](https://api.anthropic.com/v1/messages)"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        async def _raw_post():
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 2048,
                "temperature": 0.2,
                "system": system_prompt,
                "messages": messages,
            }
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG.LLM_TIMEOUT)) as session:
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Anthropic API error {resp.status}: {text[:500]}")
                    data = await resp.json()
                    return data["content"][0]["text"]
                    
        return await self._call_with_retry(_raw_post)

    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]:
        context = " | ".join(collected_answers) if collected_answers else ""
        prompt = prompt_template.replace("{USER_MESSAGE}", user_message).replace("{COLLECTED_ANSWERS}", context).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "content": prompt}], "You are a product analyst. Respond only in JSON.")
        return self._extract_json(text)

    async def dedup_compare(self, problem_a: str, problem_b: str, prompt_template: str) -> str:
        prompt = prompt_template.replace("{PROBLEM_A}", problem_a).replace("{PROBLEM_B}", problem_b)
        text = await self._call([{"role": "user", "content": prompt}], "Respond only DUPLICATE or UNIQUE.")
        return "DUPLICATE" if "DUPLICATE" in text.upper() else "UNIQUE"

    async def score(self, analysis: dict, prompt_template: str) -> Dict[str, Any]:
        analysis_json = json.dumps(analysis, ensure_ascii=False)
        prompt = prompt_template.replace("{ANALYSIS_JSON}", analysis_json).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        text = await self._call([{"role": "user", "content": prompt}], "You are a product prioritization expert. Respond only in JSON.")
        return self._extract_json(text)


class ResilientFailoverProvider(LLMProvider):
    """Гибридный провайдер, который прозрачно переключается на резервные модели при отказе основной."""
    def __init__(self, primary: LLMProvider, fallbacks: List[LLMProvider]):
        self.primary = primary
        self.fallbacks = fallbacks

    async def _call(self, messages: list, system_prompt: str = "") -> str:
        # Этот метод не должен вызываться напрямую извне для FailoverProvider,
        # так как логика переключения реализована на уровне высокоуровневых методов.
        return await self.primary._call(messages, system_prompt)

    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]:
        providers = [self.primary] + self.fallbacks
        for i, provider in enumerate(providers):
            try:
                return await provider.analyze(user_message, collected_answers, prompt_template)
            except Exception as e:
                logger.error(f"🚨 Ошибка в провайдере {provider.__class__.__name__} при анализе: {e}")
                if i == len(providers) - 1:
                    raise e
                logger.warning(f"🔄 Переключаемся на резервный LLM-провайдер для этапа [Analyze]...")

    async def dedup_compare(self, problem_a: str, problem_b: str, prompt_template: str) -> str:
        providers = [self.primary] + self.fallbacks
        for i, provider in enumerate(providers):
            try:
                return await provider.dedup_compare(problem_a, problem_b, prompt_template)
            except Exception as e:
                logger.error(f"🚨 Ошибка в провайдере {provider.__class__.__name__} при дедупликации: {e}")
                if i == len(providers) - 1:
                    raise e
                logger.warning(f"🔄 Переключаемся на резервный LLM-провайдер для этапа [Dedup]...")

    async def score(self, analysis: dict, prompt_template: str) -> Dict[str, Any]:
        providers = [self.primary] + self.fallbacks
        for i, provider in enumerate(providers):
            try:
                return await provider.score(analysis, prompt_template)
            except Exception as e:
                logger.error(f"🚨 Ошибка в провайдере {provider.__class__.__name__} при скоринге: {e}")
                if i == len(providers) - 1:
                    raise e
                logger.warning(f"🔄 Переключаемся на резервный LLM-провайдер для этапа [Scoring]...")


def get_llm_provider() -> LLMProvider:
    """Инициализирует отказоустойчивую цепочку провайдеров."""
    def _create_provider(name: str) -> LLMProvider:
        name_lower = name.lower()
        if name_lower == "gemini":
            return GeminiProvider(CONFIG.LLM_API_KEY)
        elif name_lower == "deepseek":
            return DeepSeekProvider(CONFIG.LLM_API_KEY)
        elif name_lower == "openai":
            return OpenAIProvider(CONFIG.LLM_API_KEY)
        elif name_lower == "anthropic":
            return AnthropicProvider(CONFIG.LLM_API_KEY)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {name}")

    primary_name = CONFIG.LLM_PROVIDER
    primary_provider = _create_provider(primary_name)
    
    # Резервный пул. Если в CONFIG заведены резервные ключи/модели, можно их распределить.
    # Для базового Failover добавим OpenAI или DeepSeek в качестве альтернативы, если они настроены.
    fallback_providers = []
    
    # Пример логики: если основной провайдер Gemini, то резервным ставим OpenAI (или наоборот)
    # Выбирай резерв в зависимости от того, какие токены у тебя лежат в окружении.
    try:
        if primary_name.lower() == "gemini":
            # Если упадет джемини, бот попытается подняться через OpenAI (если ключ совместим/указан)
            # Или просто укажи вторую модель, чьи лимиты у тебя стабильны.
            fallback_providers.append(_create_provider("openai"))
    except Exception:
        pass # Если резервный не сконфигурирован — игнорируем

    if fallback_providers:
        return ResilientFailoverProvider(primary_provider, fallback_providers)
    
    return primary_provider
