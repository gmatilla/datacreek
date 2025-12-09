from datacreek.parsers import ImageParser, WhisperAudioParser, get_parser_for_extension


def test_image_parser_registry():
    parser = get_parser_for_extension(".png")
    assert isinstance(parser, ImageParser)


def test_audio_parser_registry():
    parser = get_parser_for_extension(".wav")
    assert isinstance(parser, WhisperAudioParser)
