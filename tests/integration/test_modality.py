from datacreek import DatasetBuilder, DatasetType, ingest_file, to_kg
from datacreek.utils.modality import detect_modality


def test_modality_detection(tmp_path):
    f = tmp_path / "speech.txt"
    text = "um this is a test"
    f.write_text(text)
    ds = DatasetBuilder(DatasetType.TEXT)
    to_kg(ingest_file(str(f)), ds, "d1")
    chunk_id = ds.get_chunks_for_document("d1")[0]
    assert ds.graph.graph.nodes[chunk_id].get("modality") == detect_modality(text)


def test_file_modality_detection(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"00")
    assert detect_modality(str(img)) == "IMAGE"
