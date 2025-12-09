import json
import sys
import types
from unittest.mock import ANY

import pytest

sys.modules.setdefault(
    "redisgraph", types.SimpleNamespace(Graph=object, Node=object, Edge=object)
)
from datacreek.core import dataset_full


class GraphMeta(dict):
    """Dictionary storing graph metrics with self reference for ``.graph``."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.graph = self


class DummyGraph:
    def __init__(self):
        self.calls = []
        self.graph = GraphMeta(
            fractal_sigma=0.1,
            gw_entropy=0.2,
            recall10=0.3,
            tpl_w1=0.4,
            j_cost=0.5,
        )
        self.use_hnsw = False

    def __post_init__(self):
        pass

    def add_document(self, *args, **kwargs):
        self.calls.append(("doc", args, kwargs))

    def add_section(self, *args, **kwargs):
        self.calls.append(("section", args, kwargs))

    def add_chunk(self, *args, **kwargs):
        self.calls.append(("chunk", args, kwargs))

    def add_image(self, *args, **kwargs):
        self.calls.append(("image", args, kwargs))

    def add_audio(self, *args, **kwargs):
        self.calls.append(("audio", args, kwargs))

    def fractal_coverage(self):
        return 0.5

    def sheaf_consistency_score(self):
        return 0.6

    def to_neo4j(self, *a, **k):
        self.calls.append(("neo4j", a, k))

    def to_dict(self):
        return {}

    def neighborhood_to_sentence(self, path):
        return " ".join(path)


class DummyRedis:
    def __init__(self):
        self.store = {}
        self.commands = []

    def pipeline(self):
        return self

    def sadd(self, *args):
        self.commands.append(("sadd", args))

    def execute(self):
        self.commands.append(("exec",))

    def set(self, key, value):
        self.store[key] = value

    def rpush(self, *args):
        self.store.setdefault(args[0], []).append(args[-1])

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)


class DummyDriver:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return method


def stub_get_redis_graph(name):
    return None


@pytest.mark.heavy
def test_dataset_builder_basic(monkeypatch):
    graph = DummyGraph()
    redis_client = DummyRedis()
    driver = DummyDriver()
    monkeypatch.setattr(dataset_full, "get_redis_graph", stub_get_redis_graph)
    from datacreek.models import llm_client

    monkeypatch.setattr(
        llm_client.LLMClient, "_check_vllm_server", lambda self: (True, {})
    )
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="ds",
        graph=graph,
        redis_client=redis_client,
        neo4j_driver=driver,
        auto_monitor=False,
    )
    builder.policy.loops = 1
    calls = []
    monkeypatch.setattr(builder, "_persist", lambda: calls.append("persist"))
    monkeypatch.setattr(builder, "_enforce_policy", lambda r: calls.append("policy"))
    builder.configure_llm_service(provider="openai")
    builder.add_document("d1", "src")
    builder.add_section("d1", "s1")
    builder.add_chunk("d1", "c1", "txt")
    builder.add_image("d1", "i1", "img.png")
    builder.add_audio("d1", "a1", "aud.wav", lang="en")
    card = builder.generate_model_card(0.1, "sha")
    html = dataset_full.DatasetBuilder.model_card_html(card)
    assert "bias" in html
    assert any(c[0] == "doc" for c in graph.calls)
    assert builder.events[0].operation == "add_document"
    builder._touch()
    assert builder.redis_key is None or "dataset" in redis_client.store
    wrapped = dataset_full.DatasetBuilder.persist_after(
        lambda self: calls.append("func")
    )
    wrapped(builder)
    assert "policy" in calls
    assert "persist" in calls and "func" in calls


@pytest.mark.heavy
def test_dataset_validation_and_persistence(monkeypatch):
    monkeypatch.setenv("DATACREEK_REQUIRE_PERSISTENCE", "1")
    graph = DummyGraph()
    redis_client = DummyRedis()
    driver = DummyDriver()
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="ds",
        graph=graph,
        redis_client=redis_client,
        neo4j_driver=driver,
        auto_monitor=False,
    )
    with pytest.raises(ValueError):
        dataset_full.DatasetBuilder.validate_name("bad name!")

    builder._record_event("op", "msg")
    assert builder.events[-1].operation == "op"
    builder._persist()

    metrics = []
    monkeypatch.setattr(
        dataset_full, "update_metric", lambda k, v: metrics.append((k, v))
    )
    builder.log_cycle_metrics()
    assert metrics


@pytest.mark.heavy
def test_dataset_requires_persistence(monkeypatch):
    monkeypatch.setenv("DATACREEK_REQUIRE_PERSISTENCE", "1")
    with pytest.raises(ValueError):
        dataset_full.DatasetBuilder(dataset_full.DatasetType.TEXT, name="bad")


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_generation_layer_async(monkeypatch):
    import sys
    import types

    async def fake_si(*a, **k):
        return "OK"

    sys.modules["datacreek.utils.self_instruct"] = types.SimpleNamespace(
        generate_with_self_instruct_async=fake_si
    )
    graph = DummyGraph()
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="ds",
        graph=graph,
        redis_client=DummyRedis(),
        neo4j_driver=DummyDriver(),
        auto_monitor=False,
    )
    monkeypatch.setattr(builder, "_enforce_policy", lambda r: None)
    monkeypatch.setattr(
        builder, "search_with_links_data", lambda query, k=1, hops=1: [{"path": ["a"]}]
    )

    async def fake_call(prompt):
        return "resp"

    records = await builder.run_generation_layer_async(fake_call, query="q", k=1)
    assert records == [{"prompt": ANY, "response": "OK", "confidence": 1.0}]


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_ingest_file_async(monkeypatch):
    import sys
    import types

    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="ds",
        graph=DummyGraph(),
        redis_client=DummyRedis(),
        neo4j_driver=DummyDriver(),
        auto_monitor=False,
    )

    async def fake_ingest(path, builder, **kw):
        return "doc1"

    sys.modules["datacreek.core.ingest"] = types.SimpleNamespace(
        ingest_into_dataset_async=fake_ingest
    )
    result = await builder.ingest_file_async("p.txt", "doc1")
    assert result == "doc1"
    assert builder.stage == dataset_full.DatasetStage.INGESTED


@pytest.mark.heavy
def test_persist_and_touch(monkeypatch):
    graph = DummyGraph()
    client = DummyRedis()
    driver = DummyDriver()
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="ds",
        graph=graph,
        redis_client=client,
        neo4j_driver=driver,
        auto_monitor=False,
    )
    builder.redis_key = "dataset:ds"
    monkeypatch.setattr(builder, "to_redis", lambda pipe, key: None)
    builder._persist()
    builder._touch()
    assert client.store.get(builder.redis_key) is not None


@pytest.mark.heavy
def test_reload_dataset_full(monkeypatch):
    import importlib
    import sys
    import types

    sys.modules["redisgraph"] = types.SimpleNamespace(
        Graph=object, Node=object, Edge=object
    )
    import datacreek.core.dataset_full as df

    importlib.reload(df)
    assert hasattr(df, "DatasetBuilder")


@pytest.mark.heavy
def test_monitor_and_owner(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dataset_full.DatasetBuilder,
        "start_policy_monitor_thread",
        lambda self, r: calls.append(r),
    )
    graph = DummyGraph()
    graph.use_hnsw = False
    redis_client = DummyRedis()
    driver = DummyDriver()
    monkeypatch.setenv("DATACREEK_REQUIRE_PERSISTENCE", "1")
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="name1",
        graph=graph,
        redis_client=redis_client,
        neo4j_driver=driver,
        auto_monitor=True,
        use_hnsw=True,
        owner_id=7,
    )
    assert graph.use_hnsw
    assert calls == [[1]]
    builder.redis_key = "dataset:name1"
    monkeypatch.setattr(builder, "to_redis", lambda pipe, key: None)
    builder._persist()
    assert ("sadd", ("user:7:datasets", "name1")) in redis_client.commands


@pytest.mark.heavy
def test_persist_redisgraph(monkeypatch):
    graph = DummyGraph()
    redis_client = DummyRedis()
    driver = DummyDriver()
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="ds2",
        graph=graph,
        redis_client=redis_client,
        neo4j_driver=driver,
        auto_monitor=False,
    )
    monkeypatch.setattr(dataset_full, "get_redis_graph", lambda name: graph)
    called = []

    def stub_save(self, g):
        called.append("rg")

    monkeypatch.setattr(dataset_full.DatasetBuilder, "save_redis_graph", stub_save)
    dataset_full.DatasetBuilder.save_redis_graph.__wrapped__ = stub_save
    builder._persist()
    assert "rg" in called


@pytest.mark.heavy
def test_model_card_template(monkeypatch):
    import sys
    import types

    tmpl = types.SimpleNamespace(render=lambda card: "<html>card</html>")
    sys.modules["jinja2"] = types.SimpleNamespace(Template=lambda s: tmpl)
    html = dataset_full.DatasetBuilder.model_card_html({"bias_wasserstein": 0.1})
    assert html.startswith("<html>")


@pytest.mark.heavy
def test_monitor_after(monkeypatch):
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT,
        name="ds3",
        graph=DummyGraph(),
        redis_client=DummyRedis(),
        neo4j_driver=DummyDriver(),
        auto_monitor=False,
    )
    called = []
    monkeypatch.setattr(builder, "_enforce_policy", lambda r: called.append(r))

    @dataset_full.DatasetBuilder.monitor_after([2])
    def func(self):
        return "ok"

    assert func(builder) == "ok"
    assert called == [[2]]


@pytest.mark.heavy
def test_model_card_html_with_template(monkeypatch):
    import sys
    import types

    tmpl = types.SimpleNamespace(render=lambda card: "HTML")
    sys.modules["jinja2"] = types.SimpleNamespace(Template=lambda s: tmpl)
    builder = dataset_full.DatasetBuilder(
        dataset_full.DatasetType.TEXT, name="x", graph=DummyGraph()
    )
    card = builder.generate_model_card(0.2, "sha1", code_commit="123")
    html = dataset_full.DatasetBuilder.model_card_html(card)
    assert html == "HTML"
