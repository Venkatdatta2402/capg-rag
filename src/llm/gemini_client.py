"""Google Gemini LLM client implementation."""

import google.generativeai as genai

from config.settings import settings
from src.llm.base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models (Flash, Pro)."""

    def __init__(self, model: str):
        super().__init__(model)
        genai.configure(api_key=settings.google_api_key)
        self._model = genai.GenerativeModel(model)

    async def generate(self, system_prompt: str, user_message: str) -> str:
        prompt = f"{system_prompt}\n\n{user_message}"
        response = await self._model.generate_content_async(prompt)
        return response.text or ""

    async def generate_with_messages(self, messages: list[dict]) -> str:
        combined = "\n\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in messages
        )
        response = await self._model.generate_content_async(combined)
        return response.text or ""
