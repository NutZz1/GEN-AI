"""Builds knowledge/trait_ontology.json - the single source of truth for every editable Cartoon Set
trait: variant counts, the natural-language vocabulary, and a per-variant description.

The word -> variant-id dictionaries below are the original curation from `text2avatar.py` (built by
inspecting Cartoon Set's example sheets of every variant, plus measurements for skin tone/sunglasses -
see that file's old docstring). This script is now their only home; `text2avatar.py` loads the JSON
this produces instead of hard-coding vocabulary, so there is exactly one place trait words are defined.

Variant *counts* come from the dataset itself (`attr_nvariants` in data/cartoonset{size}.npz), not from
guessing - if no dataset file is present we fall back to the counts observed when this file was written
(logged below), and say so loudly rather than silently inventing numbers.

    python knowledge/build_ontology.py    # writes knowledge/trait_ontology.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "trait_ontology.json"

# ---- curated vocabulary (moved verbatim from text2avatar.py) ----------------------------------

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

# trait -> (human label, word index, special named variant-sets)
TRAIT_DEFS = {
    "hair": ("Hairstyle", HAIR_STYLE, {}),
    "hair_color": ("Hair colour", HAIR_COLOR, {}),
    "glasses": ("Glasses", GLASSES_SHAPE, {"no_glasses": NO_GLASSES}),
    "glasses_color": ("Glasses colour", GLASSES_COLOR, {}),
    "facial_hair": ("Facial hair", FACIAL_HAIR, {"clean_shaven": CLEAN_SHAVEN}),
    "face_color": ("Skin tone", SKIN, {}),
    # no curated word vocabulary exists yet for these two (never inspected/measured) - counts only.
    "face_shape": ("Face shape", {}, {}),
    "eye_color": ("Eye colour", {}, {}),
}

# counts observed directly from data/cartoonset128.npz on 2026-09-29 (see git log for this file);
# used only if no dataset file is present locally, so the ontology can still build in a fresh checkout.
FALLBACK_N_VARIANTS = {
    "hair": 111, "hair_color": 10, "glasses": 12, "glasses_color": 7,
    "facial_hair": 15, "face_color": 11, "face_shape": 7, "eye_color": 5,
}


def dataset_n_variants():
    """attr_nvariants from the largest cartoonset{size}.npz present, else the logged fallback."""
    for size in (128, 64):
        path = ROOT / "data" / f"cartoonset{size}.npz"
        if path.exists():
            d = np.load(path)
            names, nvar = list(d["attr_names"]), d["attr_nvariants"]
            return {t: int(nvar[names.index(t)]) for t in TRAIT_DEFS}, f"data/cartoonset{size}.npz"
    print("WARNING: no data/cartoonset{64,128}.npz found - using counts logged from a prior run "
          "instead of measuring them. Run prepare_data.py and re-run this script to verify.")
    return dict(FALLBACK_N_VARIANTS), "fallback (no dataset file found)"


def build():
    n_variants, source = dataset_n_variants()
    traits = {}
    for trait, (label, words, special) in TRAIT_DEFS.items():
        n = n_variants[trait]
        tags_per_variant = {i: [] for i in range(n)}
        for word, ids in words.items():
            for i in ids:
                if i < n:
                    tags_per_variant[i].append(word)
        variants = {str(i): {"tags": tags, "description": ", ".join(tags) if tags else ""}
                    for i, tags in tags_per_variant.items()}
        traits[trait] = {"label": label, "n_variants": n, "index": words, "special": special,
                         "variants": variants}
    return {"dataset_source": source, "built_by": "knowledge/build_ontology.py", "traits": traits}


if __name__ == "__main__":
    ontology = build()
    OUT.write_text(json.dumps(ontology, indent=1))
    covered = {t: sum(1 for v in d["variants"].values() if v["tags"]) for t, d in ontology["traits"].items()}
    for t, d in ontology["traits"].items():
        print(f"{t:15s} {covered[t]:3d}/{d['n_variants']:3d} variants have a natural-language tag")
    print(f"wrote {OUT}")
