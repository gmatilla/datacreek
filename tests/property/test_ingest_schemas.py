import pytest

pytest.importorskip("pydantic")
from pydantic import ValidationError

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from schemas import AudioIngest, ImageIngest, PdfIngest


@given(
    path=st.text(min_size=1),
    high_res=st.booleans(),
    width=st.integers(min_value=256, max_value=1024),
    height=st.integers(min_value=256, max_value=1024),
    blur=st.floats(min_value=0.0, max_value=0.2),
)
@settings(max_examples=1000)
def test_image_ingest_valid(
    path: str, high_res: bool, width: int, height: int, blur: float
) -> None:
    ImageIngest(
        path=path,
        high_res=high_res,
        width=width,
        height=height,
        blur_score=blur,
    )


@given(
    path=st.text(min_size=1),
    width=st.integers(max_value=255),
)
@settings(max_examples=1000)
def test_image_ingest_invalid(path: str, width: int) -> None:
    with pytest.raises(ValidationError):
        ImageIngest(path=path, width=width, height=256, blur_score=0.1)


@given(
    path=st.text(min_size=1),
    sr=st.one_of(st.none(), st.integers(min_value=8_000, max_value=48_000)),
    dur=st.floats(min_value=0, max_value=4 * 3600),
    snr=st.floats(min_value=10.0, max_value=60.0),
)
@settings(max_examples=1000)
def test_audio_ingest_valid(path: str, sr: int | None, dur: float, snr: float) -> None:
    AudioIngest(path=path, sample_rate=sr, duration=dur, snr=snr)


@given(
    path=st.text(min_size=1),
    dur=st.floats(min_value=4 * 3600 + 1, max_value=5 * 3600),
)
@settings(max_examples=1000)
def test_audio_ingest_invalid(path: str, dur: float) -> None:
    with pytest.raises(ValidationError):
        AudioIngest(path=path, duration=dur, snr=5.0)


@given(
    path=st.text(min_size=1),
    ocr=st.booleans(),
    entropy=st.floats(min_value=3.5, max_value=8.0),
)
@settings(max_examples=1000)
def test_pdf_ingest_valid(path: str, ocr: bool, entropy: float) -> None:
    PdfIngest(path=path, ocr=ocr, entropy=entropy)


@given(
    path=st.text(min_size=1),
    entropy=st.floats(max_value=3.4),
)
@settings(max_examples=1000)
def test_pdf_ingest_invalid(path: str, entropy: float) -> None:
    with pytest.raises(ValidationError):
        PdfIngest(path=path, entropy=entropy)
