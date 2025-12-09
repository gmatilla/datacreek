import builtins
import json
import sys
import types

import pytest
from PIL import Image

import datacreek.generators.vqa_generator as vqa


class DummyClient:
    def __init__(self):
        self.config = {
            "prompt": "Q",
            "generation": {"temperature": 0.1, "max_tokens": 5, "batch_size": 1},
        }

    def batch_completion(self, message_batches, temperature, max_tokens, batch_size):
        # return label for each batch
        return ["answer" for _ in message_batches]


class DummyDataset:
    def __init__(self, data):
        self.data = data

    def map(self, func, batch_size=None, batched=False):
        self.data = func(self.data)
        return self

    def to_dict(self):
        out = {
            "image": [],
            "query": list(self.data["query"]),
            "label": list(self.data["label"]),
        }
        for img in self.data["image"]:
            out["image"].append("img" if hasattr(img, "save") else img)
        return out

    def select(self, rng):
        return self

    def __len__(self):
        return len(self.data["image"])

    def __getitem__(self, idx):
        return self


class DummyDatasetsModule(types.ModuleType):
    def __init__(self):
        super().__init__("datasets")
        self.Dataset = types.SimpleNamespace(from_dict=lambda d: DummyDataset(d))


def test_transform_updates_labels(monkeypatch):
    monkeypatch.setattr(vqa, "_check_optional_deps", lambda: None)
    gen = vqa.VQAGenerator(DummyClient())
    img = Image.new("RGB", (1, 1))
    data = {"image": [img], "query": ["q"], "label": ["x"]}
    vqa.logger.setLevel("DEBUG")
    result = gen.transform(data)
    assert result["label"][0] == "answer"


def test_process_dataset_from_file(monkeypatch):
    monkeypatch.setattr(vqa, "_check_optional_deps", lambda: None)
    ds_mod = DummyDatasetsModule()

    def load_dataset(_src):
        img = Image.new("RGB", (1, 1))
        return DummyDataset({"image": [img], "query": ["q"], "label": ["a"]})

    ds_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", ds_mod)

    class DummyAPI:
        def repo_exists(self, **_):
            return True

    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.HfApi = lambda: DummyAPI()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)
    # simulate file not found to trigger load_dataset branch
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *_args, **_kw: (_ for _ in ()).throw(FileNotFoundError()),
    )
    gen = vqa.VQAGenerator(DummyClient())
    ds = gen.process_dataset("dummy_repo", num_examples=1, input_split="train")
    assert isinstance(ds, DummyDataset)
    assert ds.data["label"][0] == "answer"


class DummyBackend(vqa.StorageBackend):
    def __init__(self):
        self.saved = None

    def save(self, key: str, data: str) -> str:
        self.saved = (key, data)
        return key


def test_process_dataset_with_backend(monkeypatch):
    monkeypatch.setattr(vqa, "_check_optional_deps", lambda: None)
    ds_mod = DummyDatasetsModule()
    img = Image.new("RGB", (1, 1))
    ds_mod.load_dataset = lambda _src: DummyDataset(
        {"image": [img], "query": ["q"], "label": ["a"]}
    )
    monkeypatch.setitem(sys.modules, "datasets", ds_mod)

    class DummyAPI:
        repo_exists = lambda self, **_: True

    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.HfApi = lambda: DummyAPI()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)
    monkeypatch.setattr(
        builtins, "open", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError())
    )
    backend = DummyBackend()
    gen = vqa.VQAGenerator(DummyClient())
    key = gen.process_dataset(
        "dummy_repo",
        num_examples=1,
        input_split="train",
        backend=backend,
        redis_key="k",
    )
    assert key == "k"
    assert backend.saved[0] == "k"


def test_process_dataset_missing_lib(monkeypatch):
    monkeypatch.setattr(vqa, "_check_optional_deps", lambda: None)
    monkeypatch.setattr(
        builtins, "open", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError())
    )
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError):
        vqa.VQAGenerator(DummyClient()).process_dataset("src")


def test_process_dataset_from_json(monkeypatch, tmp_path):
    monkeypatch.setattr(vqa, "_check_optional_deps", lambda: None)
    ds_mod = DummyDatasetsModule()
    monkeypatch.setitem(sys.modules, "datasets", ds_mod)
    monkeypatch.setattr(vqa.VQAGenerator, "encode_image_base64", lambda self, img: "s")
    data = {"image": ["img"], "query": ["q"], "label": ["x"]}
    path = tmp_path / "d.json"
    path.write_text(json.dumps(data))
    gen = vqa.VQAGenerator(DummyClient())
    ds = gen.process_dataset(str(path), input_split="train")
    assert isinstance(ds, DummyDataset)
