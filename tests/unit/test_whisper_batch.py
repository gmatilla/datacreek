from datacreek.utils import whisper_batch


class DummyModel:
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = []

    def transcribe(self, path, max_length=None):
        self.calls.append((path, max_length))
        return f"{self.prefix}:{path}"


class DummyCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1


def test_transcribe_cpu(monkeypatch):
    dummy = DummyModel("cpu")
    monkeypatch.setattr(whisper_batch, "_get_model", lambda *a, **k: dummy)
    metrics = {}
    monkeypatch.setitem(metrics, "calls", [])

    def update_metric(name, value, tags):
        metrics["calls"].append((name, value, tags))

    monkeypatch.setitem(whisper_batch.__dict__, "torch", None)
    monkeypatch.setitem(whisper_batch.__dict__, "bnb_fn", None)
    monkeypatch.setattr(whisper_batch, "Whisper", DummyModel, raising=False)
    from datacreek.analysis import monitoring

    monkeypatch.setattr(monitoring, "update_metric", update_metric)
    # ensure imported name resolved to patched function
    result = whisper_batch.transcribe_audio_batch(
        ["f1.wav", "f2.wav"], device="cpu", batch_size=4
    )
    assert result == ["cpu:f1.wav", "cpu:f2.wav"]
    assert metrics["calls"][0][2]["device"] == "cpu"
    assert dummy.calls == [("f1.wav", 30), ("f2.wav", 30)]


def test_gpu_fallback(monkeypatch):
    first = DummyModel("gpu")
    second = DummyModel("cpu")
    sequence = [first, second]

    def fake_get_model(*a, **k):
        return sequence[0]

    def clear():
        if sequence:
            sequence.pop(0)

    fake_get_model.cache_clear = clear
    monkeypatch.setattr(whisper_batch, "_get_model", fake_get_model)

    # simulate torch with cuda available
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

    monkeypatch.setitem(whisper_batch.__dict__, "torch", Torch)
    fallback_counter = DummyCounter()

    def update_metric(name, value, tags):
        pass

    from datacreek.analysis import monitoring

    monkeypatch.setattr(monitoring, "update_metric", update_metric)
    monkeypatch.setattr(monitoring, "whisper_fallback_total", fallback_counter)

    def failing_transcribe(path, max_length=None):
        raise RuntimeError("Out of memory")

    first.transcribe = failing_transcribe

    result = whisper_batch.transcribe_audio_batch(["x.wav"], device=None, batch_size=2)
    assert result == ["cpu:x.wav"]
    assert fallback_counter.count == 1
