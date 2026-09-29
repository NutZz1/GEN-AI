import json
from pathlib import Path

import numpy as np

from traits import TRAITS as TRAIT_NAMES

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY = json.loads((ROOT / "knowledge" / "trait_ontology.json").read_text())


def test_ontology_has_every_editable_trait():
    assert set(ONTOLOGY["traits"]) == set(TRAIT_NAMES)


def test_ontology_matches_dataset_variant_counts():
    data_path = ROOT / "data" / "cartoonset128.npz"
    if not data_path.exists():
        import pytest
        pytest.skip("data/cartoonset128.npz not present")
    d = np.load(data_path)
    names, nvar = list(d["attr_names"]), d["attr_nvariants"]
    for trait, info in ONTOLOGY["traits"].items():
        assert info["n_variants"] == int(nvar[names.index(trait)]), f"{trait} variant count mismatch"


def test_every_indexed_id_is_in_range():
    for trait, info in ONTOLOGY["traits"].items():
        n = info["n_variants"]
        for word, ids in info["index"].items():
            assert all(0 <= i < n for i in ids), f"{trait}/{word} has an out-of-range id: {ids}"
        for name, ids in info.get("special", {}).items():
            assert all(0 <= i < n for i in ids), f"{trait}/special/{name} has an out-of-range id: {ids}"


def test_variants_map_covers_every_id():
    for trait, info in ONTOLOGY["traits"].items():
        assert set(info["variants"]) == {str(i) for i in range(info["n_variants"])}


def test_text2avatar_reads_the_same_ontology():
    import text2avatar as t2a
    assert t2a.HAIR_STYLE == ONTOLOGY["traits"]["hair"]["index"]
    assert t2a.NO_GLASSES == ONTOLOGY["traits"]["glasses"]["special"]["no_glasses"]
