from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class LLMTransportError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    """Dependency-free OpenAI Responses API client.

    Environment variable names intentionally match upstream MiroFish:
    LLM_API_KEY, LLM_BASE_URL and LLM_MODEL_NAME.

    Live CI uses OpenAI's Responses API. Unit tests inject fake clients and do
    not require network access.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 45.0,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("LLM_MODEL_NAME")
        self.timeout = timeout
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("LLM_API_KEY, LLM_BASE_URL and LLM_MODEL_NAME are required")

    @staticmethod
    def _extract_output_text(data: Dict[str, Any]) -> str:
        for item in data.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str):
                    return part["text"]

        if isinstance(data.get("output_text"), str):
            return data["output_text"]

        raise LLMTransportError("Responses API returned no output text")

    def _request(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": messages,
        }
        # Intent classification is a low-complexity task. GPT-5.6 supports
        # reasoning.effort=none, which reduces latency/cost for this benchmark.
        if self.model and self.model.startswith("gpt-5.6"):
            payload["reasoning"] = {"effort": "none"}
        if json_mode:
            payload["text"] = {"format": {"type": "json_object"}}

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            detail = error_body[:2000] if error_body else str(exc)
            raise LLMTransportError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMTransportError(str(exc)) from exc

        return self._extract_output_text(data)

    def complete_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        text = self._request(messages, json_mode=True)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMTransportError("model did not return valid JSON") from exc
        if not isinstance(value, dict):
            raise LLMTransportError("model JSON root must be an object")
        return value

    def complete_text(self, messages: List[Dict[str, str]]) -> str:
        return self._request(messages, json_mode=False)
