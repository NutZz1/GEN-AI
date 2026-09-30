"""Provider-agnostic LLM client for the avatar-design agent.

Two backends:
  - Anthropic Claude API (default) - needs ANTHROPIC_API_KEY. Supports tool use.
  - Ollama (local, optional)       - needs a running `ollama serve`. Best-effort tool use: only
    recent tool-calling-capable local models (e.g. llama3.1+) will actually emit tool calls: others
    will just answer in text, which the agent loop treats as "no tool call" and handles like any
    other model that declined to call a tool.

If neither is configured/reachable, `get_client()` returns a client whose `.available` is False and
every part of the agent (agent/agent.py) falls back to the pre-existing rule-based text2avatar.py
parser and a heuristic planner, so the app keeps working with zero LLM setup.
"""
import json
from dataclasses import dataclass, field
from typing import Any

from . import config


@dataclass
class LLMResponse:
    text: str = ""                      # concatenated text blocks
    tool_calls: list = field(default_factory=list)   # [{"id", "name", "input"}]
    raw_content: list = field(default_factory=list)  # provider-native content blocks, for continuing a tool loop
    stop_reason: str = "end_turn"
    error: str = ""


class LLMClient:
    name = "none"
    available = False
    reason = "no LLM backend configured"   # why this client is unavailable, shown in the app sidebar

    def chat(self, system, messages, tools=None, json_mode=False) -> LLMResponse:
        return LLMResponse(error="no LLM backend configured")


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, api_key, model):
        import anthropic  # deferred: only required if this backend is actually used
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.available = True

    def chat(self, system, messages, tools=None, json_mode=False) -> LLMResponse:
        try:
            kwargs = {"model": self.model, "max_tokens": config.MAX_TOKENS, "system": system, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            resp = self._client.messages.create(**kwargs)
        except Exception as e:  # network errors, auth errors, rate limits, ... -> caller falls back
            return LLMResponse(error=f"anthropic error: {e}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [{"id": b.id, "name": b.name, "input": b.input} for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=tool_calls, raw_content=resp.content, stop_reason=resp.stop_reason)


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, model, host):
        import requests  # deferred
        self._requests = requests
        self.model, self.host = model, host.rstrip("/")
        r = requests.get(f"{self.host}/api/tags", timeout=5)
        r.raise_for_status()
        installed = [m.get("name", "") for m in r.json().get("models", [])]
        # "llama3.1" matches "llama3.1:latest" / "llama3.1:8b"; an exact tag ("llama3.1:8b") must match exactly.
        if not any(n == model or (":" not in model and n.split(":")[0] == model) for n in installed):
            raise RuntimeError(f"model {model!r} is not installed on {self.host} (installed: "
                               f"{', '.join(installed) or 'none'}) - run `ollama pull {model}` there")
        self.available = True

    def chat(self, system, messages, tools=None, json_mode=False) -> LLMResponse:
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}] +
                   [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])}
                    for m in messages], "stream": False,
                   # the parser prompt + tool schemas are ~2-3k tokens: Ollama's default context window can
                   # silently truncate the start of the prompt (the instructions and ontology), so set it.
                   "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0},
                   "keep_alive": config.OLLAMA_KEEP_ALIVE}  # keep the model loaded between chat turns
        if json_mode and not tools:  # constrain the parser to valid JSON (tool calls use their own format)
            payload["format"] = "json"
        if tools:  # OpenAI-style tool schema; only understood by newer local models
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        try:
            r = self._requests.post(f"{self.host}/api/chat", json=payload, timeout=config.REQUEST_TIMEOUT_S)
            r.raise_for_status()
            msg = r.json()["message"]
        except Exception as e:
            return LLMResponse(error=f"ollama error: {e}")
        calls = [{"id": f"ollama_{i}", "name": c["function"]["name"], "input": c["function"].get("arguments", {})}
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        return LLMResponse(text=msg.get("content", ""), tool_calls=calls, stop_reason="tool_use" if calls else "end_turn")


def get_client() -> LLMClient:
    """Build the configured backend; never raises - returns an unavailable LLMClient on any failure.
    Reads AVATARBLEND_LLM_PROVIDER from the environment directly (not the cached config.PROVIDER),
    so changing it at runtime (env var, monkeypatch in tests, .env edit + reload) takes effect."""
    import os
    provider = os.environ.get("AVATARBLEND_LLM_PROVIDER", "anthropic")
    try:
        if provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                return _unavailable("no ANTHROPIC_API_KEY set")
            model = os.environ.get("AVATARBLEND_AGENT_MODEL", config.ANTHROPIC_MODEL)
            return AnthropicClient(key, model)
        if provider == "ollama":
            model = os.environ.get("AVATARBLEND_OLLAMA_MODEL", config.OLLAMA_MODEL)
            host = os.environ.get("OLLAMA_HOST", config.OLLAMA_HOST)
            return OllamaClient(model, host)
    except Exception as e:
        return _unavailable(f"{provider} backend failed: {type(e).__name__}: {str(e)[:200]}")
    return _unavailable(f"LLM provider {provider!r} is not 'anthropic' or 'ollama'")


def _unavailable(reason) -> LLMClient:
    client = LLMClient()
    client.reason = reason
    return client
