"""OpenAI LLM client implementation."""

from openai import AsyncOpenAI

from config.settings import settings
from src.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models (GPT-4.1, GPT-4o-mini, etc.)."""

    def __init__(self, model: str):
        super().__init__(model)
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(self, system_prompt: str, user_message: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    async def generate_with_messages(self, messages: list[dict]) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
