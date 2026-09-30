"""Builds knowledge/prompt_testset.jsonl: ~100 natural-language avatar requests with gold trait
constraints, for evaluating the rule-based parser vs. the LLM parser (see evaluate_agent.py).

Categories:
  easy            - one trait, in-vocabulary word (gold generated from the ontology + cross-checked
                    against text2avatar.parse(), so it is correct by construction).
  compositional   - 2-4 traits combined in one sentence, same guarantee.
  negation        - "no glasses" / "clean-shaven" phrasing.
  vague           - a persona/style description with no ontology keywords (gold is hand-written and
                    intentionally sparse: only traits a person would confidently infer are scored).
  code_mixed      - Tanglish/Manglish requests (English trait nouns, Tamil/Malayalam function words -
                    the common code-mixing pattern; the rule parser has no vocabulary for the
                    non-English tokens so this category is expected to separate it from an LLM parser).
  blend_edit      - requests that reference two avatars or "change X to Y"; gold includes an
                    `intent` field (generate / blend / edit) in addition to any trait constraints, for
                    scoring agent tool-routing rather than single-avatar constraint parsing.

Gold constraints use the same schema as text2avatar.parse()'s first return value: {trait: [variant
ids...]}. `hand-written` gold for vague/code_mixed items is this repo author's best-effort reading of
the phrase, not a user study - see DATASET.md for that caveat.

    python knowledge/build_prompt_testset.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import text2avatar as t2a  # noqa: E402

OUT = Path(__file__).resolve().parent / "prompt_testset.jsonl"
SEED = 0


def gold_for(prompt):
    """Ground truth for in-vocabulary prompts = what a correct rule parser must produce."""
    cons, understood, unknown = t2a.parse(prompt)
    assert not unknown, f"prompt uses a word outside the ontology: {prompt!r} -> unknown={unknown}"
    return cons


def easy_items(rng):
    pools = [
        ("hair", list(t2a.HAIR_STYLE), "{w} hair"),
        ("hair_color", list(t2a.HAIR_COLOR), "{w} hair"),
        ("glasses", [w for w in t2a.GLASSES_SHAPE if w not in ("specs", "spectacles")], "{w} glasses"),
        ("glasses_color", list(t2a.GLASSES_COLOR), "{w} glasses"),
        ("facial_hair", [w for w in t2a.FACIAL_HAIR if w != "bearded"], "a {w}"),
        ("face_color", list(t2a.SKIN), "{w} skin"),
    ]
    items = []
    for trait, words, tmpl in pools:
        for w in rng.sample(words, k=min(5, len(words))):
            prompt = tmpl.format(w=w)
            items.append({"prompt": prompt, "category": "easy", "gold": gold_for(prompt)})
    return items


def compositional_items(rng):
    hair = list(t2a.HAIR_STYLE)
    hcolor = list(t2a.HAIR_COLOR)
    glasses = [w for w in t2a.GLASSES_SHAPE if w not in ("specs", "spectacles", "glasses")]
    gcolor = list(t2a.GLASSES_COLOR)
    facial = [w for w in t2a.FACIAL_HAIR if w != "bearded"]
    skin = list(t2a.SKIN)
    templates = [
        "{hc} {h} hair, {g} glasses",
        "{h} {hc} hair with {gc} {g} glasses and a {f}",
        "{skin} skin, {hc} {h} hair",
        "{gc} {g} glasses, {f}, {h} hair",
        "{h} {hc} hair and {skin} skin, no beard",
    ]
    items = []
    for _ in range(30):
        tmpl = rng.choice(templates)
        prompt = tmpl.format(h=rng.choice(hair), hc=rng.choice(hcolor), g=rng.choice(glasses),
                             gc=rng.choice(gcolor), f=rng.choice(facial), skin=rng.choice(skin))
        items.append({"prompt": prompt, "category": "compositional", "gold": gold_for(prompt)})
    return items


def negation_items(rng):
    hair = list(t2a.HAIR_STYLE)
    hcolor = list(t2a.HAIR_COLOR)
    templates = [
        "{hc} {h} hair, no glasses",
        "{h} hair, clean-shaven, no glasses",
        "{hc} hair and a beard, no glasses",
        "no glasses, {h} hair, clean shaven",
    ]
    items = []
    for _ in range(15):
        tmpl = rng.choice(templates)
        prompt = tmpl.format(h=rng.choice(hair), hc=rng.choice(hcolor))
        items.append({"prompt": prompt, "category": "negation", "gold": gold_for(prompt)})
    return items


# hand-written: no ontology keywords, gold is sparse and subjective (see module docstring)
VAGUE_ITEMS = [
    ("looks like a friendly librarian", {"glasses": t2a.GLASSES_SHAPE["glasses"]}),
    ("a punk rocker", {"hair": t2a.HAIR_STYLE["spiky"]}),
    ("someone's grandma", {"hair_color": t2a.HAIR_COLOR["white"] + t2a.HAIR_COLOR["silver"]}),
    ("a corporate executive, very clean cut", {"facial_hair": t2a.CLEAN_SHAVEN, "hair": t2a.HAIR_STYLE["short"]}),
    ("a surfer dude", {"hair": t2a.HAIR_STYLE["wavy"] + t2a.HAIR_STYLE["long"]}),
    ("a wizard", {"facial_hair": t2a.FACIAL_HAIR["beard"]}),
    ("an anime protagonist", {"hair": t2a.HAIR_STYLE["spiky"]}),
    ("a goth", {"hair_color": t2a.HAIR_COLOR["black"], "glasses": t2a.GLASSES_SHAPE["sunglasses"]}),
    ("a scientist", {"glasses": t2a.GLASSES_SHAPE["glasses"]}),
    ("a rockstar with attitude", {"glasses": t2a.GLASSES_SHAPE["sunglasses"]}),
    ("a monk", {"hair": t2a.HAIR_STYLE["bald"], "facial_hair": t2a.CLEAN_SHAVEN}),
    ("a hipster barista", {"facial_hair": t2a.FACIAL_HAIR["beard"] + t2a.FACIAL_HAIR["mustache"]}),
    ("an old sailor", {"facial_hair": t2a.FACIAL_HAIR["beard"], "hair_color": t2a.HAIR_COLOR["grey"] + t2a.HAIR_COLOR["white"]}),
    ("a k-pop idol", {"hair": t2a.HAIR_STYLE["straight"]}),
    ("a detective in a noir film", {"glasses": t2a.GLASSES_SHAPE["round"]}),
]

# hand-written Tanglish/Manglish: English trait nouns, Tamil/Malayalam function words - the rule
# parser (English-only vocabulary) is expected to miss the non-English tokens entirely.
CODE_MIXED_ITEMS = [
    ("romba curly hair venum", {"hair": t2a.HAIR_STYLE["curly"]}),
    ("black kalar hair mathiri onnu vendum", {"hair_color": t2a.HAIR_COLOR["black"]}),
    ("kannadi venam, sunglasses type", {"glasses": t2a.GLASSES_SHAPE["sunglasses"]}),
    ("thadi vendam, clean shaven aa irukanum", {"facial_hair": t2a.CLEAN_SHAVEN}),
    ("red colour hair venum, curly aa", {"hair_color": t2a.HAIR_COLOR["red"], "hair": t2a.HAIR_STYLE["curly"]}),
    ("glasses illama, bald head mathiri", {"glasses": t2a.NO_GLASSES, "hair": t2a.HAIR_STYLE["bald"]}),
    ("nalla tanned skin venam", {"face_color": t2a.SKIN["tanned"]}),
    ("beard onnu venum, black colour", {"facial_hair": t2a.FACIAL_HAIR["beard"]}),
    ("straight hair, blonde aa maathunga", {"hair": t2a.HAIR_STYLE["straight"], "hair_color": t2a.HAIR_COLOR["blonde"]}),
    ("chinna mustache mathiri onnu vendum", {"facial_hair": t2a.FACIAL_HAIR["mustache"]}),
]

# hand-written: routing intent beyond single-avatar generation
BLEND_EDIT_ITEMS = [
    ("blend avatar A and avatar B evenly", {"intent": "blend"}, {}),
    ("mix parent A with parent B, mostly A", {"intent": "blend"}, {}),
    ("take avatar A but give it avatar B's hair", {"intent": "trait_mix"}, {}),
    ("use avatar 3's hairstyle on avatar 7", {"intent": "trait_mix"}, {}),
    ("change this avatar's hair colour to red", {"intent": "edit"}, {"hair_color": None}),
    ("make the hair shorter", {"intent": "edit"}, {"hair": None}),
    ("give this avatar sunglasses instead", {"intent": "edit"}, {"glasses": None}),
    ("breed avatar A and avatar B", {"intent": "breed"}, {}),
    ("make a new avatar from scratch with curly red hair", {"intent": "generate"},
     {"hair": t2a.HAIR_STYLE["curly"], "hair_color": t2a.HAIR_COLOR["red"]}),
    ("blend A and B then make the result clean-shaven", {"intent": "blend"}, {"facial_hair": t2a.CLEAN_SHAVEN}),
]


def hand_items(rows, category):
    items = []
    for row in rows:
        if len(row) == 2:
            prompt, gold = row
            items.append({"prompt": prompt, "category": category, "gold": gold})
        else:
            prompt, meta, gold = row
            items.append({"prompt": prompt, "category": category, "gold": {**gold, **meta}})
    return items


def main():
    rng = random.Random(SEED)
    items = (easy_items(rng) + compositional_items(rng) + negation_items(rng)
             + hand_items(VAGUE_ITEMS, "vague") + hand_items(CODE_MIXED_ITEMS, "code_mixed")
             + hand_items(BLEND_EDIT_ITEMS, "blend_edit"))
    rng.shuffle(items)
    n_test = 30
    for i, item in enumerate(items):
        item["split"] = "test" if i < n_test else "dev"
        item["id"] = f"p{i:03d}"
    with OUT.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    by_cat = {}
    for item in items:
        by_cat[item["category"]] = by_cat.get(item["category"], 0) + 1
    print(f"wrote {len(items)} prompts to {OUT} ({n_test} test / {len(items) - n_test} dev)")
    print("by category:", by_cat)


if __name__ == "__main__":
    main()
