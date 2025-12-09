import types

from datacreek.utils import image_captioning as ic


def test_caption_image(monkeypatch):
    def fake_pipeline(task, model):
        class M:
            def __call__(self, path):
                return [{"generated_text": path + " cap"}]

        return M()

    monkeypatch.setattr(ic, "pipeline", fake_pipeline)
    ic._get_model.cache_clear()
    assert ic.caption_image("img.png") == "img.png cap"
    # second call uses cache
    assert ic.caption_image("img.png") == "img.png cap"


def test_caption_images_parallel(monkeypatch):
    monkeypatch.setattr(ic, "caption_image", lambda p: f"{p}-c")
    res = ic.caption_images_parallel(["a", "b"], max_workers=2, chunk_size=1)
    assert res == ["a-c", "b-c"]
