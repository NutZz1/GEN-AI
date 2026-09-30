"""Text -> avatar: a small rule-based parser from a description to trait variants.

The word -> variant-id vocabulary lives in knowledge/trait_ontology.json (built by
knowledge/build_ontology.py from the original curation: inspecting Cartoon Set's example sheets of
every variant, plus measurements for skin tone/sunglasses). This module is now a pure consumer of
that file, so the LLM agent (agent/prompts.py) and this rule-based parser share one vocabulary instead
of drifting apart.

    parse("curly red hair, heart sunglasses and a beard")
    -> {'hair': [...curly ids], 'hair_color': [3, 2], 'glasses': [10], 'facial_hair': [...]}, understood, unknown
"""
import json
import re
from pathlib import Path

import numpy as np

ONTOLOGY_PATH = Path(__file__).resolve().parent / "knowledge" / "trait_ontology.json"
ONTOLOGY = json.loads(ONTOLOGY_PATH.read_text())

HAIR_STYLE = ONTOLOGY["traits"]["hair"]["index"]
HAIR_COLOR = ONTOLOGY["traits"]["hair_color"]["index"]
GLASSES_SHAPE = ONTOLOGY["traits"]["glasses"]["index"]
GLASSES_COLOR = ONTOLOGY["traits"]["glasses_color"]["index"]
FACIAL_HAIR = ONTOLOGY["traits"]["facial_hair"]["index"]
SKIN = ONTOLOGY["traits"]["face_color"]["index"]
NO_GLASSES = ONTOLOGY["traits"]["glasses"]["special"]["no_glasses"]
CLEAN_SHAVEN = ONTOLOGY["traits"]["facial_hair"]["special"]["clean_shaven"]
FILLER = {"a", "an", "the", "with", "and", "has", "have", "having", "wearing", "wears", "person", "guy", "girl",
          "man", "woman", "avatar", "some", "of", "in", "very", "lots", "big", "hair", "haired", "on", "her",
          "his", "their", "who", "is", "skin", "skinned", "complexion", "eyes", "face", "cut", "style", "tone"}


def _norm(text):
    t = text.lower()
    for a, b in [("cat-eye", "cateye"), ("cat eye", "cateye"), ("top knot", "topknot"), ("side part", "sidepart"),
                 ("side-swept", "swept"), ("buzz cut", "buzz"), ("clean-shaven", "cleanshaven"),
                 ("clean shaven", "cleanshaven"), ("no beard", "cleanshaven"), ("no glasses", "noglasses"),
                 ("without glasses", "noglasses"), ("space buns", "buns"), ("sun glasses", "sunglasses"),
                 ("light-skinned", "light skin"), ("dark-skinned", "dark skin")]:
        t = t.replace(a, b)
    return t


def _merge(cur, new):
    """Intersect constraints on the same trait ("long" + "curly" -> long AND curly); fall back to the newer."""
    if cur is None:
        return list(new)
    both = [v for v in cur if v in new]
    return both or list(new)


def parse(text):
    """-> (constraints {trait: [allowed variants]}, understood [str], unknown [str])."""
    cons, understood, unknown = {}, [], []

    def add(trait, ids, word):
        cons[trait] = _merge(cons.get(trait), ids)
        understood.append(f"{word} → {trait.replace('_', ' ')}")

    for phrase in re.split(r",|;|\band\b|\bwith\b|\bwearing\b|\bplus\b", _norm(text)):
        words = re.findall(r"[a-z]+", phrase)
        if not words:
            continue
        # the head noun decides what colour words refer to
        if any(w in ("glasses", "specs", "spectacles", "sunglasses", "shades") for w in words):
            target = "glasses"
        elif any(w in ("skin", "skinned", "complexion") for w in words):
            target = "skin"
        elif any(w in FACIAL_HAIR for w in words):
            target = "facial_hair"
        else:
            target = "hair"
        for w in words:
            if w == "noglasses":
                add("glasses", NO_GLASSES, "no glasses")
            elif w == "cleanshaven":
                add("facial_hair", CLEAN_SHAVEN, "clean-shaven")
            elif target == "glasses" and w in GLASSES_SHAPE:
                add("glasses", GLASSES_SHAPE[w], w)
            elif target == "glasses" and w in GLASSES_COLOR:
                add("glasses_color", GLASSES_COLOR[w], w)
            elif target == "skin" and w in SKIN:
                add("face_color", SKIN[w], w)
            elif w in FACIAL_HAIR:
                add("facial_hair", FACIAL_HAIR[w], w)
            elif target == "facial_hair" and w in HAIR_COLOR:
                add("hair_color", HAIR_COLOR[w], w)  # beard colour follows hair colour in Cartoon Set
            elif w in HAIR_STYLE:
                add("hair", HAIR_STYLE[w], w)
            elif w in HAIR_COLOR:
                add("hair_color", HAIR_COLOR[w], w)
            elif w not in FILLER:
                unknown.append(w)
    if cons.get("glasses") == NO_GLASSES:
        cons.pop("glasses_color", None)
    return cons, understood, unknown


def generate(traits, cons, n, rng, residuals, residual_scale=0.7):
    """Sample n latents satisfying the constraints; unconstrained traits are random.
    `residuals` = pool of real-avatar residuals giving each candidate its own leftover details."""
    zs, labels = [], []
    for _ in range(n):
        lab = {t: int(rng.choice(cons[t])) if t in cons else int(rng.integers(traits.n_variants(t)))
               for t in traits.traits}
        res = residual_scale * residuals[rng.integers(len(residuals))]
        zs.append(traits.mu + traits.trait_part(lab) + res)
        labels.append(lab)
    return np.stack(zs).astype(np.float32), labels
