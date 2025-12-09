import datacreek.utils.emotion as emo


def test_detect_emotion_default():
    assert emo.detect_emotion("I am very happy today!") == "happy"
    assert emo.detect_emotion("He is furious about this") == "angry"
    assert emo.detect_emotion("Nothing special here") == "neutral"


def test_detect_emotion_custom():
    lex = {"cool": {"wow"}}
    assert emo.detect_emotion.__wrapped__("wow it works", lexicon=lex) == "cool"
