"""No-API-key path: every stage of the agent must still produce a usable result."""
import numpy as np
import pytest

from agent import agent as agent_core
from agent import llm as agent_llm


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AVATARBLEND_LLM_PROVIDER", "anthropic")


def test_client_unavailable_without_key():
    client = agent_llm.get_client()
    assert client.available is False
    assert client.name == "none"


def test_null_client_chat_returns_error_not_raise():
    client = agent_llm.LLMClient()
    resp = client.chat(system="x", messages=[{"role": "user", "content": "y"}])
    assert resp.error


def test_parse_request_falls_back_to_rule_parser():
    client = agent_llm.LLMClient()
    result = agent_core.parse_request(client, "curly red hair, no glasses")
    assert result.source == "rule_fallback"
    assert result.constraints  # non-empty: the rule parser does understand this phrase


def test_run_once_generates_an_avatar(agent_ctx):
    client = agent_llm.LLMClient()
    result = agent_core.run_once(agent_ctx, client, "bald head, no glasses", log=False)
    assert result.plan_source == "heuristic"
    assert result.final_avatar in agent_ctx.avatars
    assert result.explanation


def test_heuristic_planner_routes_blend(agent_ctx):
    client = agent_llm.LLMClient()
    agent_core.run_once(agent_ctx, client, "blonde straight hair", log=False)
    agent_core.run_once(agent_ctx, client, "black curly hair", log=False)
    result = agent_core.run_once(agent_ctx, client, "blend the two avatars", log=False)
    assert result.trace[0].tool == "blend"


def test_generate_candidates_produces_n_verified_results(agent_ctx):
    client = agent_llm.LLMClient()
    parse, candidates = agent_core.generate_candidates(agent_ctx, client, "red curly hair", n=3)
    assert len(candidates) == 3
    assert len(set(c["name"] for c in candidates)) == 3  # distinct avatars, not the same one 3x
