"""The Ollama backend, exercised over real HTTP against a fake Ollama server (same endpoints and payload shapes as
the real one), so the connection path is tested without needing a model installed."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent import agent as agent_core
from agent import llm as agent_llm


class FakeOllama:
    def __init__(self, models=("llama3.1:latest",), fence_json=True):
        self.models, self.fence_json, self.requests = list(models), fence_json, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output clean
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"models": [{"name": m} for m in fake.models]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(payload)
                system = payload["messages"][0]["content"]
                if payload.get("tools"):
                    msg = {"role": "assistant", "content": "", "tool_calls": [{"function": {
                        "name": "generate_from_constraints",
                        "arguments": {"constraints": {"hair": [5], "hair_color": [3]}}}}]}
                elif system.startswith("You are the trait parser"):
                    js = json.dumps({"hair": [5], "hair_color": [3], "understood": ["afro, red"], "unknown": []})
                    msg = {"role": "assistant", "content": f"```json\n{js}\n```" if fake.fence_json else js}
                else:
                    msg = {"role": "assistant", "content": "Here is your afro avatar with red hair."}
                self._send({"message": msg})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_ollama():
    servers = []

    def make(**kw):
        servers.append(FakeOllama(**kw))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


def _use(monkeypatch, server, model="llama3.1"):
    monkeypatch.setenv("AVATARBLEND_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", server.url)
    monkeypatch.setenv("AVATARBLEND_OLLAMA_MODEL", model)


def test_connects_when_the_model_is_installed(fake_ollama, monkeypatch):
    _use(monkeypatch, fake_ollama())
    client = agent_llm.get_client()
    assert client.available and client.name == "ollama"


def test_unavailable_with_a_clear_reason_when_the_model_is_not_installed(fake_ollama, monkeypatch):
    _use(monkeypatch, fake_ollama(models=("smollm2:360m",)))
    client = agent_llm.get_client()
    assert not client.available
    assert "llama3.1" in client.reason and "not installed" in client.reason and "smollm2" in client.reason


def test_unavailable_with_a_reason_when_the_host_is_unreachable(monkeypatch):
    monkeypatch.setenv("AVATARBLEND_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    client = agent_llm.get_client()
    assert not client.available and client.reason


def test_request_sets_context_window_keep_alive_and_json_mode(fake_ollama, monkeypatch):
    server = fake_ollama()
    _use(monkeypatch, server)
    client = agent_llm.get_client()
    client.chat(system="s", messages=[{"role": "user", "content": "x"}], json_mode=True)
    sent = server.requests[-1]
    assert sent["options"]["num_ctx"] >= 4096 and sent["options"]["temperature"] == 0
    assert sent["keep_alive"] and sent["format"] == "json" and sent["stream"] is False


def test_full_agent_run_through_ollama(fake_ollama, monkeypatch, agent_ctx):
    """parse (fenced JSON) -> plan (tool call) -> verify -> explain, all over HTTP."""
    server = fake_ollama(fence_json=True)
    _use(monkeypatch, server)
    client = agent_llm.get_client()
    result = agent_core.run_once(agent_ctx, client, "afro, red", log=False)
    assert result.parse.source == "llm"                      # the ```json fences were tolerated
    assert result.parse.constraints == {"hair": [5], "hair_color": [3]}
    assert result.plan_source == "llm" and result.trace[0].tool == "generate_from_constraints"
    assert agent_ctx.avatars[result.final_avatar]["labels"]["hair"] == 5
    assert result.explanation == "Here is your afro avatar with red hair."
    assert result.llm_backend == "ollama"


def test_candidates_use_the_ollama_parser(fake_ollama, monkeypatch, agent_ctx):
    _use(monkeypatch, fake_ollama(fence_json=False))
    client = agent_llm.get_client()
    parse, cands = agent_core.generate_candidates(agent_ctx, client, "afro, red", n=2)
    assert parse.source == "llm" and len(cands) == 2
