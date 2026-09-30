"""Photo -> avatar: crop/resize an uploaded photo and guess Cartoon Set trait variants from it, using
the same independent judge classifier the agent uses to verify its own generations
(checkpoints/attr_classifier.pt - trained in evaluate.py on real Cartoon Set illustrations, never on
real human photographs). Treat the guesses as a best-effort starting point for the trait-constrained
generator (agent/tools.py's generate_from_constraints), not a measured-accuracy feature - the judge
classifier's accuracy numbers in results/ are all on cartoon images, not photos.

No face detector: `prepare_image` just centre-crops to a square and resizes, so it works best on a
front-facing, roughly-centred photo. The uploaded photo is processed in memory only - never written to
disk, never sent to an LLM API (trait guessing is 100% local classifier inference).

    constraints, confidence = guess_constraints(model, prepare_image(photo_bytes, model.img_size))
"""
import io

import numpy as np
from PIL import Image

from judge import load_classifier, predict_traits


def prepare_image(file_bytes, size):
    """Uploaded file bytes -> uint8 (size,size,3) RGB, centre-cropped to square then resized."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return np.array(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)  # np.array copies; avoids a non-writable buffer


def guess_constraints(model, img_uint8):
    """-> ({trait: [variant id]}, {trait: confidence 0-1}), or (None, None) with no classifier checkpoint."""
    clf = load_classifier(model)
    if clf is None:
        return None, None
    labels, conf = predict_traits(clf, model, img_uint8)
    return {t: [v] for t, v in labels.items()}, conf
