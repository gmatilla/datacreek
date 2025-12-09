import pytest

try:
    from datacreek.core.dataset import DatasetBuilder
except Exception:  # pragma: no cover - optional dependency missing
    DatasetBuilder = None  # type: ignore


def test_dataset_rollback_wrapper(tmp_path):
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()
    path = ds.rollback_gremlin_diff(output="tmp.diff")
    assert path.endswith("tmp.diff")


def test_dataset_sheaf_sla_wrapper():
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()
    mttr = ds.sheaf_checker_sla([0, 1800])
    assert mttr == 0.5


def test_dataset_prune_fractalnet_weights_wrapper():
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    db = DatasetBuilder()
    w = [0.1 * i for i in range(10)]
    out = db.prune_fractalnet_weights(w, ratio=0.3)
    assert len([x for x in out if x != 0]) == 3


def test_dataset_colour_box_dimension_wrapper():
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.add_node("a")
    ds.graph.add_node("b")
    ds.graph.add_edge("a", "b")
    dim, counts = ds.colour_box_dimension([1])
    assert isinstance(dim, float) and len(counts) == 1


def test_dataset_search_hybrid_wrapper(monkeypatch):
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()

    def fake(self, query, k=5, node_type="chunk"):
        assert query == "hello" and k == 2 and node_type == "chunk"
        return ["a", "b"]

    monkeypatch.setattr(ds.graph.__class__, "search_hybrid", fake)
    res = ds.search_hybrid("hello", k=2)
    assert res == ["a", "b"]


def test_dataset_cypher_ann_query_wrapper(monkeypatch):
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()
    driver = object()

    def fake(self, drv, query, cypher, k=5, node_type="chunk"):
        assert drv is driver and query == "hello" and cypher == "MATCH n" and k == 3
        return [{"id": "c1"}]

    monkeypatch.setattr(ds.graph.__class__, "cypher_ann_query", fake)
    res = ds.cypher_ann_query(driver, "hello", "MATCH n", k=3)
    assert res == [{"id": "c1"}]


def test_dataset_ann_hybrid_search_wrapper(monkeypatch):
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()

    def fake(
        self,
        q_n2v,
        q_gw,
        q_hyp,
        k=5,
        ann_k=2000,
        node_type="chunk",
        n2v_attr="embedding",
        gw_attr="graphwave_embedding",
        hyper_attr="poincare_embedding",
        gamma=0.5,
        eta=0.25,
    ):
        assert q_n2v == [1] and q_gw == [2] and q_hyp == [3]
        assert k == 4 and ann_k == 100
        assert node_type == "chunk" and n2v_attr == "embedding"
        return [("c1", 0.9)]

    monkeypatch.setattr(ds.graph.__class__, "ann_hybrid_search", fake)
    res = ds.ann_hybrid_search([1], [2], [3], k=4, ann_k=100)
    assert res == [("c1", 0.9)]
