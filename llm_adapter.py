"""LLM Provider Adapter with Gemini, DeepSeek, OpenAI, Anthropic support."""
import json
import re
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any
import aiohttp
from config import CONFIG

class LLMProvider(ABC):
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

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    async def _call(self, contents: list, system_prompt: str = "") -> str:
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

    async def analyze(self, user_message: str, collected_answers: list, prompt_template: str) -> Dict[str, Any]:
        context = " | ".join(collected_answers) if collected_answers else ""
        prompt = prompt_template.replace("{USER_MESSAGE}", user_message).replace("{COLLECTED_ANSWERS}", context).replace("{PRODUCT_STRATEGY}", CONFIG.PRODUCT_STRATEGY)
        # Явное указание формата для предотвращения ошибок парсинга
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
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
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
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
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
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def _call(self, messages: list, system_prompt: str = "") -> str:
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

def get_llm_provider() -> LLMProvider:
    provider = CONFIG.LLM_PROVIDER.lower()
    if provider == "gemini":
        return GeminiProvider(CONFIG.LLM_API_KEY)
    elif provider == "deepseek":
        return DeepSeekProvider(CONFIG.LLM_API_KEY)
    elif provider == "openai":
        return OpenAIProvider(CONFIG.LLM_API_KEY)
    elif provider == "anthropic":
        return AnthropicProvider(CONFIG.LLM_API_KEY)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {CONFIG.LLM_PROVIDER}")
