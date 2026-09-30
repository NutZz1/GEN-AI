"""Shared loader for the independent attribute classifier (checkpoints/attr_classifier.pt, trained in
evaluate.py on real Cartoon Set images - never on real human photographs, never seen by the VAE). Used
both to verify agent-generated avatars (agent/tools.py's classify_avatar) and to guess trait
constraints from an uploaded photo (photo2avatar.py).
"""
import torch
import torch.nn.functional as F

from avatarblend import CKPT_DIR, load_dataset, model_device, to_tensor
from evaluate import TRAITS as JUDGE_TRAITS
from evaluate import AttrClassifier

_CACHE = {}


def load_classifier(model):
    """-> AttrClassifier on model's device, cached per img_size. None if the checkpoint doesn't exist."""
    size = model.img_size
    if size in _CACHE:
        return _CACHE[size]
    path = CKPT_DIR / "attr_classifier.pt"
    if not path.exists():
        return None
    data = load_dataset(size)
    names = list(data["attr_names"])
    n_classes = [int(data["attr_nvariants"][names.index(t)]) for t in JUDGE_TRAITS]
    clf = AttrClassifier(n_classes).to(model_device(model))
    clf.load_state_dict(torch.load(path, map_location=model_device(model)))
    clf.eval()
    _CACHE[size] = clf
    return clf


@torch.no_grad()
def predict_traits(clf, model, img_uint8):
    """One (H,W,3) uint8 image -> ({trait: predicted variant id}, {trait: confidence 0-1})."""
    x = to_tensor(img_uint8[None]).to(model_device(model))
    probs = [F.softmax(o, 1)[0] for o in clf(x)]
    labels = {t: int(p.argmax().item()) for t, p in zip(JUDGE_TRAITS, probs)}
    conf = {t: float(p.max().item()) for t, p in zip(JUDGE_TRAITS, probs)}
    return labels, conf
