"""Each LLM tool, exercised directly against a real (not mocked) AgentContext - the "dummy latent"
here is whatever generate_from_constraints samples, since the VAE has no meaningful all-zeros latent
to test against (mu is already the near-average face, not a degenerate case worth special-casing)."""
import numpy as np
import pytest

from agent.tools import TOOL_SCHEMAS, blend, breed, classify_avatar, edit_trait, generate_from_constraints, trait_mix


def test_tool_schemas_match_implementations():
    from agent.tools import TOOL_IMPLS
    for schema in TOOL_SCHEMAS:
        assert schema["name"] in TOOL_IMPLS


def test_generate_from_constraints(agent_ctx):
    result = generate_from_constraints(agent_ctx, {"hair_color": [7]})
    assert result["labels"]["hair_color"] == 7
    assert result["img"].shape == (agent_ctx.model.img_size, agent_ctx.model.img_size, 3)
    assert result["avatar"] in agent_ctx.avatars


def test_generate_unconstrained_traits_are_randomised(agent_ctx):
    r1 = generate_from_constraints(agent_ctx, {"hair_color": [0]}, name="a")
    r2 = generate_from_constraints(agent_ctx, {"hair_color": [0]}, name="b")
    assert r1["labels"]["hair_color"] == r2["labels"]["hair_color"] == 0
    assert not np.array_equal(r1["img"], r2["img"])  # different residual/unconstrained traits


def test_blend(agent_ctx):
    generate_from_constraints(agent_ctx, {}, name="a")
    generate_from_constraints(agent_ctx, {}, name="b")
    result = blend(agent_ctx, "a", "b", alpha=0.5)
    assert result["avatar"] in agent_ctx.avatars


def test_trait_mix(agent_ctx):
    generate_from_constraints(agent_ctx, {"hair": [5]}, name="a")
    generate_from_constraints(agent_ctx, {"hair": [40]}, name="b")
    result = trait_mix(agent_ctx, "a", "b", traits_from_a=["hair"])
    assert result["labels"]["hair"] == 5


def test_edit_trait(agent_ctx):
    generate_from_constraints(agent_ctx, {"hair_color": [0]}, name="a")
    result = edit_trait(agent_ctx, "a", "hair_color", 7, strength=1.0)
    assert result["labels"]["hair_color"] == 7


def test_breed(agent_ctx):
    generate_from_constraints(agent_ctx, {}, name="a")
    generate_from_constraints(agent_ctx, {}, name="b")
    result = breed(agent_ctx, "a", "b", n=4)
    assert len(result["children"]) == 4
    assert all(c["avatar"] in agent_ctx.avatars for c in result["children"])


def test_classify_avatar_or_skips_cleanly(agent_ctx):
    generate_from_constraints(agent_ctx, {}, name="a")
    predicted = classify_avatar(agent_ctx, "a")
    if predicted is None:
        pytest.skip("no checkpoints/attr_classifier.pt in this environment")
    assert "hair" in predicted


def test_unknown_avatar_raises(agent_ctx):
    from agent.tools import UnknownAvatarError
    with pytest.raises(UnknownAvatarError):
        agent_ctx.get("does_not_exist")
