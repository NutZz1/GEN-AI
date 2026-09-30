"""Rule-based parser (text2avatar.parse) phrasing that used to fail: removals, comparatives, eye colour, and
colour words attached to the right noun. No model needed."""
import text2avatar as t2a


def cons(text):
    return t2a.parse(text)[0]


def test_eye_colour_is_parsed():
    assert cons("black hair with blue eyes") == {"hair_color": [7], "eye_color": [1]}
    assert cons("make the eyes green") == {"eye_color": [2]}
    assert cons("brown eyes") == {"eye_color": [0]}


def test_eye_colour_does_not_leak_into_hair_or_skin():
    c = cons("red hair and grey eyes")
    assert c["eye_color"] == [3] and c["hair_color"] == [3, 2]


def test_removal_phrases_map_to_the_no_variants():
    for phrase in ("remove the beard", "without a beard", "no mustache", "shave off the beard"):
        assert cons(phrase) == {"facial_hair": t2a.CLEAN_SHAVEN}, phrase
    for phrase in ("take off the glasses", "take the glasses off", "remove his sunglasses", "no glasses",
                   "without glasses"):
        assert cons(phrase) == {"glasses": t2a.NO_GLASSES}, phrase


def test_adding_is_not_confused_with_removal():
    assert cons("add glasses")["glasses"] == t2a.GLASSES_SHAPE["glasses"]
    assert cons("now add a mustache") == {"facial_hair": t2a.FACIAL_HAIR["mustache"]}


def test_comparatives_map_to_the_absolute_trait():
    assert cons("make the hair shorter") == {"hair": t2a.HAIR_STYLE["short"]}
    assert cons("make the hair longer") == {"hair": t2a.HAIR_STYLE["long"]}


def test_colour_word_binds_to_the_noun_it_modifies():
    c = cons("blue glasses brown hair")
    assert c["glasses_color"] == t2a.GLASSES_COLOR["blue"]
    assert c["hair_color"] == t2a.HAIR_COLOR["brown"]


def test_edit_verbs_are_not_reported_as_unknown_words():
    assert t2a.parse("make it blonde")[2] == []
    assert t2a.parse("give it a beard")[2] == []


def test_previously_working_prompts_unchanged():
    c = cons("curly red hair, heart sunglasses and a beard")
    assert c["hair_color"] == [3, 2] and c["glasses"] == [10]
    assert cons("no glasses, clean-shaven, pale skin") == {"glasses": t2a.NO_GLASSES,
                                                           "facial_hair": t2a.CLEAN_SHAVEN, "face_color": [10]}
    assert cons("a friendly librarian") == {}
