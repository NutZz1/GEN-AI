"""Pydantic validation for the LLM parser's JSON output (the most error-prone LLM call in the
pipeline - free-form generation, not a provider-enforced tool schema). Tool-call arguments are
already constrained by Anthropic's `input_schema` (see agent/tools.py) so they don't need a second
validation layer here.
"""
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError, model_validator

ONTOLOGY = json.loads((Path(__file__).resolve().parent.parent / "knowledge" / "trait_ontology.json").read_text())
TRAIT_NAMES = list(ONTOLOGY["traits"])


class ParsedConstraints(BaseModel):
    """One field per editable trait: a list of acceptable variant ids, or None if unconstrained."""
    hair: list[int] | None = None
    hair_color: list[int] | None = None
    face_shape: list[int] | None = None
    face_color: list[int] | None = None
    glasses: list[int] | None = None
    glasses_color: list[int] | None = None
    facial_hair: list[int] | None = None
    eye_color: list[int] | None = None
    understood: list[str] = []
    unknown: list[str] = []

    @model_validator(mode="after")
    def ids_in_range(self):
        errors = []
        for trait in TRAIT_NAMES:
            ids = getattr(self, trait)
            if ids is None:
                continue
            n = ONTOLOGY["traits"][trait]["n_variants"]
            bad = [i for i in ids if not (0 <= i < n)]
            if bad:
                errors.append(f"{trait}: variant id(s) {bad} out of range [0, {n})")
            if not ids:
                errors.append(f"{trait}: must be a non-empty list of ids, or omitted/null entirely")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def constraints(self):
        """-> {trait: [ids]} in the exact shape text2avatar.parse()/generate() use."""
        return {t: getattr(self, t) for t in TRAIT_NAMES if getattr(self, t) is not None}


def _strip_wrapping(raw_text):
    """Small local models often wrap the JSON in ```json fences or a sentence; keep just the outermost {...}.
    Anything that still isn't valid JSON after this is rejected (and re-asked) as before."""
    text = (raw_text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def validate_parser_json(raw_text):
    """raw_text -> (ParsedConstraints, None) or (None, error_message)."""
    try:
        data = json.loads(_strip_wrapping(raw_text))
    except json.JSONDecodeError as e:
        return None, f"not valid JSON: {e}"
    try:
        return ParsedConstraints.model_validate(data), None
    except ValidationError as e:
        return None, str(e)
