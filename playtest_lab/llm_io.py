from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class LLMTransportError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    """Dependency-free OpenAI-compatible chat client.

    It reuses the environment variable names already documented by upstream
    MiroFish. CI never calls a live model; tests inject a fake client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("LLM_MODEL_NAME")
        self.timeout = timeout
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("LLM_API_KEY, LLM_BASE_URL and LLM_MODEL_NAME are required")

    def _request(self, messages: List[Dict[str, str]], response_format: Optional[dict] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMTransportError(str(exc)) from exc

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMTransportError("invalid chat completion response") from exc

    def complete_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        text = self._request(messages, response_format={"type": "json_object"})
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMTransportError("model did not return valid JSON") from exc
        if not isinstance(value, dict):
            raise LLMTransportError("model JSON root must be an object")
        return value

    def complete_text(self, messages: List[Dict[str, str]]) -> str:
        return self._request(messages)
