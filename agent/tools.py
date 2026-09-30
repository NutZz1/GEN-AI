"""LLM tools: thin wrappers around the existing VAE/trait/genetics functions, each with an Anthropic
`input_schema` and a Python implementation that runs against an `AgentContext` (model + traits +
rng + a registry of avatars that exist in the current session). No new generative logic lives here -
every tool delegates to avatarblend.py / traits.py / genetics.py / text2avatar.py.
"""
import numpy as np

import text2avatar as t2a
from genetics import Lineage
from traits import TRAITS as TRAIT_NAMES


class UnknownAvatarError(Exception):
    pass


def compute_residual_pool(traits, mu, data):
    """Per-avatar leftover latent detail (z - mu - trait_part(labels)) for every training avatar -
    what text2avatar.generate() samples from so generated avatars have plausible fine detail instead
    of a bland trait-mean face."""
    names = list(data["attr_names"])
    train = np.nonzero(data["split"] == 0)[0]
    out = np.empty((len(train), mu.shape[1]), dtype=np.float32)
    for k, i in enumerate(train):
        labels = traits.labels_from_attrs(data["attrs"][i], names)
        out[k] = traits.residual(mu[i], labels)
    return out


class AgentContext:
    """Everything a tool call needs: the model/traits (shared, read-only) and one session's mutable
    state (registered avatars, a Lineage for breeding, an rng)."""

    def __init__(self, model, traits, residual_pool, rng=None):
        self.model, self.traits, self.residual_pool = model, traits, residual_pool
        self.rng = rng or np.random.default_rng()
        self.avatars = {}          # name -> {"z", "labels", "img"}
        self.lineage = Lineage()
        self._lineage_ids = {}     # avatar name -> lineage node id (only for bred/registered avatars)

    def register(self, name, z, labels, img):
        self.avatars[name] = {"z": np.asarray(z, dtype=np.float32), "labels": dict(labels), "img": img}
        self._lineage_ids[name] = self.lineage.add(self.avatars[name]["z"], labels, img, 0, name=name)
        return name

    def get(self, name):
        if name not in self.avatars:
            raise UnknownAvatarError(f"no avatar named {name!r} in this session; known: {list(self.avatars)}")
        return self.avatars[name]

    def decode(self, z):
        from avatarblend import decode_np
        return decode_np(self.model, z)


def _new_name(ctx, prefix):
    i = 1
    while f"{prefix}_{i}" in ctx.avatars:
        i += 1
    return f"{prefix}_{i}"


# ---------------- tool implementations ----------------

def generate_from_constraints(ctx, constraints, residual_strength=0.7, name=None):
    """constraints: {trait: [allowed variant ids]} (LLM-parser or rule-parser shape)."""
    cons = {t: list(ids) for t, ids in constraints.items() if t in TRAIT_NAMES and ids}
    z, labels = t2a.generate(ctx.traits, cons, 1, ctx.rng, ctx.residual_pool, residual_strength)
    img = ctx.decode(z)[0]
    name = name or _new_name(ctx, "avatar")
    ctx.register(name, z[0], labels[0], img)
    return {"avatar": name, "labels": labels[0], "img": img}


def blend(ctx, avatar_a, avatar_b, alpha=0.5, name=None):
    a, b = ctx.get(avatar_a), ctx.get(avatar_b)
    z = alpha * a["z"] + (1 - alpha) * b["z"]
    img = ctx.decode(z)[0]
    labels = a["labels"] if alpha >= 0.5 else b["labels"]  # whole-avatar blend has no single "true" label
    name = name or _new_name(ctx, "blend")
    ctx.register(name, z, labels, img)
    return {"avatar": name, "labels": labels, "img": img}


def trait_mix(ctx, avatar_a, avatar_b, traits_from_a, rest_from="B", name=None):
    a, b = ctx.get(avatar_a), ctx.get(avatar_b)
    weights = {t: (1.0 if t in traits_from_a else 0.0) for t in ctx.traits.traits}
    rest_weight = 1.0 if str(rest_from).upper() == "A" else 0.0
    z, labels = ctx.traits.selective_blend(a["z"], a["labels"], b["z"], b["labels"], weights, rest_weight)
    img = ctx.decode(z)[0]
    name = name or _new_name(ctx, "mix")
    ctx.register(name, z, labels, img)
    return {"avatar": name, "labels": labels, "img": img}


def edit_trait(ctx, avatar, trait, variant, strength=1.0, name=None):
    a = ctx.get(avatar)
    z, labels = ctx.traits.edit(a["z"], a["labels"], trait, int(variant), float(strength))
    img = ctx.decode(z)[0]
    name = name or avatar  # edits default to overwriting the avatar in place, like the Avatar Editor tab
    ctx.register(name, z, labels, img)
    return {"avatar": name, "labels": labels, "img": img}


def breed(ctx, avatar_a, avatar_b, n=6, noise=0.3, blend_prob=0.2, point_mutation=0.05):
    pa, pb = ctx._lineage_ids[avatar_a], ctx._lineage_ids[avatar_b]
    child_ids = ctx.lineage.breed(pa, pb, ctx.traits, ctx.decode, n, ctx.rng, noise, blend_prob, point_mutation)
    children = []
    for nid in child_ids:
        node = ctx.lineage.nodes[nid]
        ctx.avatars[node["name"]] = {"z": node["z"], "labels": node["labels"], "img": node["img"]}
        ctx._lineage_ids[node["name"]] = nid
        children.append({"avatar": node["name"], "labels": node["labels"], "img": node["img"],
                         "genotype": node["genotype"]})
    return {"children": children}


def classify_avatar(ctx, avatar):
    """Independent-classifier prediction for each judge trait -> {trait: predicted_variant_id}.
    Returns None (not {}) if no classifier checkpoint exists, so callers can skip verification cleanly."""
    from judge import load_classifier, predict_traits
    clf = load_classifier(ctx.model)
    if clf is None:
        return None
    labels, _ = predict_traits(clf, ctx.model, ctx.get(avatar)["img"])
    return labels


TOOL_IMPLS = {"generate_from_constraints": generate_from_constraints, "blend": blend, "trait_mix": trait_mix,
              "edit_trait": edit_trait, "breed": breed, "classify_avatar": classify_avatar}

TOOL_SCHEMAS = [
    {"name": "generate_from_constraints", "description":
        "Generate a brand-new avatar from trait constraints (variant ids). Unconstrained traits are "
        "randomised.",
     "input_schema": {"type": "object", "properties": {
         "constraints": {"type": "object", "description":
             "trait name -> list of acceptable variant ids, e.g. {\"hair\": [5,7], \"hair_color\": [3]}"},
         "residual_strength": {"type": "number", "description": "0-1, how much random fine detail to add"}},
         "required": ["constraints"]}},
    {"name": "blend", "description": "Whole-avatar latent interpolation between two existing session avatars.",
     "input_schema": {"type": "object", "properties": {
         "avatar_a": {"type": "string"}, "avatar_b": {"type": "string"},
         "alpha": {"type": "number", "description": "share of avatar_a, 0-1"}},
         "required": ["avatar_a", "avatar_b"]}},
    {"name": "trait_mix", "description": "Take specific traits from avatar_a, the rest from avatar_b.",
     "input_schema": {"type": "object", "properties": {
         "avatar_a": {"type": "string"}, "avatar_b": {"type": "string"},
         "traits_from_a": {"type": "array", "items": {"type": "string", "enum": TRAIT_NAMES}}},
         "required": ["avatar_a", "avatar_b", "traits_from_a"]}},
    {"name": "edit_trait", "description": "Change one trait of one existing session avatar.",
     "input_schema": {"type": "object", "properties": {
         "avatar": {"type": "string"}, "trait": {"type": "string", "enum": TRAIT_NAMES},
         "variant": {"type": "integer"},
         "strength": {"type": "number", "description": "1.0 = fully switch, >1 exaggerates, <0 moves away"}},
         "required": ["avatar", "trait", "variant"]}},
    {"name": "breed", "description": "Genetic crossover: breed two existing session avatars into n children.",
     "input_schema": {"type": "object", "properties": {
         "avatar_a": {"type": "string"}, "avatar_b": {"type": "string"},
         "n": {"type": "integer", "description": "number of children, default 6"}},
         "required": ["avatar_a", "avatar_b"]}},
    {"name": "classify_avatar", "description":
        "Ask the independent attribute classifier what traits an existing session avatar actually has "
        "(useful to check an avatar before deciding what to change).",
     "input_schema": {"type": "object", "properties": {"avatar": {"type": "string"}}, "required": ["avatar"]}},
]
