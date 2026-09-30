import json

from agent.schemas import validate_parser_json


def test_valid_json_parses():
    ok, err = validate_parser_json(json.dumps({"hair": [5, 7], "understood": ["curly -> hair"], "unknown": []}))
    assert err is None
    assert ok.hair == [5, 7]


def test_missing_keys_default_to_none():
    ok, err = validate_parser_json("{}")
    assert err is None
    assert ok.hair is None and ok.constraints() == {}


def test_out_of_range_id_rejected():
    ok, err = validate_parser_json(json.dumps({"hair": [99999]}))
    assert ok is None
    assert "out of range" in err


def test_empty_list_rejected():
    ok, err = validate_parser_json(json.dumps({"hair": []}))
    assert ok is None
    assert "non-empty" in err


def test_invalid_json_rejected():
    ok, err = validate_parser_json("this is not json")
    assert ok is None
    assert "not valid JSON" in err


def test_unknown_trait_key_ignored_not_crashed():
    # pydantic's default (non-strict) model just ignores extra keys - a made-up trait shouldn't crash the parser
    ok, err = validate_parser_json(json.dumps({"eyebrow_shape": [1]}))
    assert err is None
    assert ok.constraints() == {}
