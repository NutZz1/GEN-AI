"""Text -> avatar: a small rule-based parser from a description to trait variants.

Cartoon Set only numbers its variants, so the vocabulary below was built by inspecting example sheets
of every variant (hairstyles, hair colours, glasses shapes/colours, facial hair) and by measurements:
skin-tone variants are ordered dark -> light by cheek-pixel luminance, and the "sunglasses" shapes are
the glasses variants whose lens region is dark on 100% of avatars (1, 8, 9, 10).

    parse("curly red hair, heart sunglasses and a beard")
    -> {'hair': [...curly ids], 'hair_color': [3, 2], 'glasses': [10], 'facial_hair': [...]}, understood, unknown
"""
import re

import numpy as np

HAIR_STYLE = {
    "bald": [0, 1, 2, 19, 108, 109, 110],
    "buzz": [18, 20, 22], "shaved": [18, 20, 22],
    "bun": [55, 68, 84, 102], "topknot": [55, 68, 84], "updo": [55, 68, 84, 102],
    "pigtails": [3, 4, 28, 29, 41, 96], "buns": [3, 28],
    "ponytail": [87, 95, 102],
    "afro": [5, 15, 16],
    "dreadlocks": [6, 13], "dreads": [6, 13], "locs": [6, 13], "braids": [13],
    "spiky": [23, 43, 46, 50, 52, 54, 77], "mohawk": [43, 46, 50],
    "curly": [5, 7, 8, 9, 10, 11, 15, 16, 17, 24, 25, 27, 30, 31, 34, 35, 36, 37, 38, 39, 56, 78, 90, 97, 106],
    "wavy": [26, 32, 33, 65, 82, 85, 91, 93, 94, 100, 103, 104],
    "straight": [40, 42, 47, 57, 58, 59, 60, 63, 69, 71, 79, 80],
    "long": [6, 7, 13, 25, 33, 40, 47, 57, 58, 59, 60, 63, 65, 69, 71, 78, 79, 80, 82, 85, 93, 100, 103, 104],
    "medium": [26, 30, 32, 42, 48, 88, 98],
    "short": [12, 14, 17, 21, 22, 23, 44, 45, 49, 51, 53, 61, 62, 66, 72, 73, 74, 76, 83, 89, 92, 99, 105, 107],
    "bob": [67, 75, 81, 86, 94, 101],
    "swept": [48, 49, 53, 64, 70, 72, 88, 89], "sidepart": [48, 49, 53, 64, 70, 72, 88, 89],
}
HAIR_COLOR = {
    "platinum": [0], "blonde": [0, 1], "blond": [0, 1], "golden": [1], "yellow": [1],
    "ginger": [2], "orange": [2], "red": [3, 2], "redhead": [3, 2], "auburn": [3],
    "honey": [4], "sandy": [4], "copper": [5], "brown": [6, 5], "brunette": [6], "chestnut": [5],
    "black": [7], "dark": [7, 6], "grey": [8], "gray": [8], "silver": [9, 8], "white": [9],
}
GLASSES_SHAPE = {
    "glasses": list(range(11)), "specs": list(range(11)), "spectacles": list(range(11)),
    "sunglasses": [1, 8, 9, 10], "shades": [1, 8, 9, 10],
    "heart": [10], "cateye": [8], "round": [0, 9], "rectangular": [2, 3], "square": [2, 3],
    "clear": [0, 2, 3, 4, 5, 6, 7],
}
GLASSES_COLOR = {"black": [0], "brown": [1], "blue": [2], "red": [3], "grey": [4], "gray": [4],
                 "gold": [5], "golden": [5], "yellow": [5], "pink": [6]}
FACIAL_HAIR = {
    "beard": [2, 3, 8, 9, 10, 11], "bearded": [2, 3, 8, 9, 10, 11],
    "goatee": [4, 7], "mustache": [5, 6], "moustache": [5, 6],
}
SKIN = {  # face_color variants are ordered dark (0) -> light (10)
    "dark": [0, 1, 2], "deep": [0, 1], "brown": [3, 4, 5], "tan": [5, 6, 7], "tanned": [5, 6, 7],
    "olive": [5, 6], "medium": [5, 6, 7], "light": [8, 9, 10], "fair": [9, 10], "pale": [10],
}
NO_GLASSES = [11]
CLEAN_SHAVEN = [0, 1, 14]
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
