"""Google Gemini LLM client implementation."""

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from config.settings import settings
from src.llm.base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models (Flash, Pro)."""

    def __init__(self, model: str):
        super().__init__(model)
        genai.configure(api_key=settings.google_api_key)
        self._model_name = model

    def _get_model(self, tools=None):
        if tools:
            gemini_tools = [
                Tool(function_declarations=[
                    FunctionDeclaration(
                        name=t["function"]["name"],
                        description=t["function"].get("description", ""),
                        parameters=t["function"].get("parameters", {}),
                    )
                    for t in tools
                ])
            ]
            return genai.GenerativeModel(self._model_name, tools=gemini_tools)
        return genai.GenerativeModel(self._model_name)

    async def generate(self, system_prompt: str, user_message: str) -> str:
        prompt = f"{system_prompt}\n\n{user_message}"
        response = await self._get_model().generate_content_async(prompt)
        return response.text or ""

    async def generate_with_messages(self, messages: list[dict]) -> str:
        combined = "\n\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in messages
        )
        response = await self._get_model().generate_content_async(combined)
        return response.text or ""

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
    ) -> tuple[str, list[dict]]:
        prompt = f"{system_prompt}\n\n{user_message}"
        response = await self._get_model(tools).generate_content_async(prompt)

        content = ""
        tool_calls = []

        for part in response.parts:
            if hasattr(part, "text") and part.text:
                content += part.text
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append({
                    "name": fc.name,
                    "arguments": dict(fc.args),
                })

        return content, tool_calls
