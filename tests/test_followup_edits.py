"""Follow-up chat edits in the no-LLM path: they must modify the avatar the user is looking at, never silently
replace it with a new random one."""
import pytest

from agent import agent as agent_core
from agent import llm as agent_llm
from agent.tools import edit_traits, generate_from_constraints


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return agent_llm.LLMClient()


def _start(ctx, client, prompt="black straight hair, no glasses, clean-shaven"):
    return agent_core.run_once(ctx, client, prompt, log=False).final_avatar


@pytest.mark.parametrize("prompt, trait, check", [
    ("add sunglasses", "glasses", lambda v: v != 11),
    ("give it a beard", "facial_hair", lambda v: v not in (0, 1, 14)),
    ("make the eyes blue", "eye_color", lambda v: v == 1),
    ("make the hair red", "hair_color", lambda v: v in (2, 3)),
])
def test_edit_changes_the_same_avatar(agent_ctx, client, prompt, trait, check):
    name = _start(agent_ctx, client)
    result = agent_core.run_once(agent_ctx, client, prompt, log=False, active=name)
    assert result.trace[0].tool == "edit_traits"
    assert result.final_avatar == name                      # edited in place, not replaced
    assert check(agent_ctx.avatars[name]["labels"][trait])


def test_removals_take_the_feature_away(agent_ctx, client):
    name = _start(agent_ctx, client, "a beard, sunglasses")
    agent_core.run_once(agent_ctx, client, "remove the beard", log=False, active=name)
    assert agent_ctx.avatars[name]["labels"]["facial_hair"] in (0, 1, 14)
    agent_core.run_once(agent_ctx, client, "take off the glasses", log=False, active=name)
    assert agent_ctx.avatars[name]["labels"]["glasses"] == 11


def test_edit_only_touches_the_requested_trait_labels(agent_ctx, client):
    name = _start(agent_ctx, client)
    before = dict(agent_ctx.avatars[name]["labels"])
    agent_core.run_once(agent_ctx, client, "make the eyes green", log=False, active=name)
    after = agent_ctx.avatars[name]["labels"]
    assert {t for t in before if before[t] != after[t]} == {"eye_color"}


def test_follow_up_edits_the_picked_candidate_not_the_last_generated(agent_ctx, client):
    parse, cands = agent_core.generate_candidates(agent_ctx, client, "curly red hair", n=3)
    picked, last = cands[0]["name"], cands[-1]["name"]
    assert picked != last
    last_before = agent_ctx.avatars[last]["z"].copy()
    agent_core.run_once(agent_ctx, client, "make the eyes blue", log=False, active=picked)
    assert agent_ctx.avatars[picked]["labels"]["eye_color"] == 1
    assert (agent_ctx.avatars[last]["z"] == last_before).all()   # the other candidate is untouched


def test_unmappable_edit_keeps_the_avatar(agent_ctx, client):
    name = _start(agent_ctx, client)
    z_before = agent_ctx.avatars[name]["z"].copy()
    n_before = len(agent_ctx.avatars)
    result = agent_core.run_once(agent_ctx, client, "make it look happier", log=False, active=name)
    assert result.final_avatar is None and result.trace == []
    assert len(agent_ctx.avatars) == n_before and (agent_ctx.avatars[name]["z"] == z_before).all()
    assert "left your avatar as it is" in result.explanation


def test_describing_a_new_avatar_still_generates(agent_ctx, client):
    name = _start(agent_ctx, client)
    result = agent_core.run_once(agent_ctx, client, "a new avatar with curly red hair", log=False, active=name)
    assert result.trace[0].tool == "generate_from_constraints"
    assert result.final_avatar != name


def test_retry_restores_then_pushes_harder(agent_ctx):
    """A retried edit starts from the avatar's original state: re-applying on the already-edited avatar would be a
    no-op because the first attempt already relabelled it."""
    generate_from_constraints(agent_ctx, {"eye_color": [0]}, name="a")
    z0 = agent_ctx.avatars["a"]["z"].copy()
    snapshots = {}
    call = {"name": "edit_traits", "input": {"avatar": "a", "edits": {"eye_color": [1]}, "strength": 1.0}}
    agent_core._prepare_edit(agent_ctx, call, snapshots)
    edit_traits(agent_ctx, **call["input"])
    z1 = agent_ctx.avatars["a"]["z"].copy()
    assert not (z1 == z0).all()
    agent_core._prepare_edit(agent_ctx, call, snapshots)          # restore for the retry
    assert (agent_ctx.avatars["a"]["z"] == z0).all() and agent_ctx.avatars["a"]["labels"]["eye_color"] == 0
    edit_traits(agent_ctx, "a", {"eye_color": [1]}, strength=1.5)
    z2 = agent_ctx.avatars["a"]["z"]
    assert not (z2 == z1).all()                                   # a real second attempt, not a no-op
