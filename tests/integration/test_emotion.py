from datacreek import DatasetBuilder, DatasetType, ingest_file, to_kg
from datacreek.utils.emotion import detect_emotion


def test_emotion_detection(tmp_path):
    f = tmp_path / "emo.txt"
    text = "I am very happy today!"
    f.write_text(text)
    ds = DatasetBuilder(DatasetType.TEXT)
    to_kg(ingest_file(str(f)), ds, "d1")
    chunk_id = ds.get_chunks_for_document("d1")[0]
    assert ds.graph.graph.nodes[chunk_id].get("emotion") == detect_emotion(text)


def test_custom_emotion_fn(tmp_path):
    f = tmp_path / "emo.txt"
    text = "A neutral sentence."
    f.write_text(text)

    def custom_fn(t: str) -> str:
        return "custom"

    ds = DatasetBuilder(DatasetType.TEXT)
    to_kg(ingest_file(str(f)), ds, "d2", emotion_fn=custom_fn)
    chunk_id = ds.get_chunks_for_document("d2")[0]
    assert ds.graph.graph.nodes[chunk_id].get("emotion") == "custom"
