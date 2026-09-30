import io

import numpy as np
import pytest
from PIL import Image

import photo2avatar


def _fake_photo_bytes(w=300, h=220):
    """A synthetic non-square RGB image, standing in for an arbitrary photo upload."""
    arr = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_prepare_image_crops_to_square_and_resizes(model):
    img = photo2avatar.prepare_image(_fake_photo_bytes(), model.img_size)
    assert img.shape == (model.img_size, model.img_size, 3)
    assert img.dtype == np.uint8


def test_prepare_image_handles_portrait_and_landscape(model):
    for w, h in [(300, 220), (220, 300), (128, 128)]:
        img = photo2avatar.prepare_image(_fake_photo_bytes(w, h), model.img_size)
        assert img.shape == (model.img_size, model.img_size, 3)


def test_guess_constraints_shape_or_skips_cleanly(model):
    img = photo2avatar.prepare_image(_fake_photo_bytes(), model.img_size)
    cons, conf = photo2avatar.guess_constraints(model, img)
    if cons is None:
        pytest.skip("no checkpoints/attr_classifier.pt in this environment")
    from judge import JUDGE_TRAITS
    assert set(cons) == set(JUDGE_TRAITS)
    for trait, ids in cons.items():
        assert len(ids) == 1
        assert 0 <= conf[trait] <= 1


def test_guess_constraints_feed_generate_from_constraints(agent_ctx):
    img = photo2avatar.prepare_image(_fake_photo_bytes(), agent_ctx.model.img_size)
    cons, conf = photo2avatar.guess_constraints(agent_ctx.model, img)
    if cons is None:
        pytest.skip("no checkpoints/attr_classifier.pt in this environment")
    from agent.tools import generate_from_constraints
    result = generate_from_constraints(agent_ctx, cons)
    assert result["avatar"] in agent_ctx.avatars
    for trait, ids in cons.items():
        assert result["labels"][trait] == ids[0]
