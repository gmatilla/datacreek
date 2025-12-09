from datacreek.core.dataset import DatasetBuilder, DatasetType


def test_generate_model_card_records_event():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.graph.graph["fractal_sigma"] = 0.1
    ds.graph.graph.graph["gw_entropy"] = 0.2
    ds.graph.graph.graph["bias_W"] = 0.3
    card = ds.generate_model_card(
        prune_ratio=0.5, cca_sha="abc", code_commit="deadbeef"
    )
    assert card["sigma_db"] == 0.1
    assert card["H_wave"] == 0.2
    assert card["bias_wasserstein"] == 0.3
    assert card["code_commit"] == "deadbeef"
    assert any(e.operation == "model_card" for e in ds.events)


def test_model_card_html_generation():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.graph.graph["fractal_sigma"] = 0.1
    ds.graph.graph.graph["gw_entropy"] = 0.2
    ds.graph.graph.graph["bias_W"] = 0.3
    card = ds.generate_model_card(
        prune_ratio=0.5, cca_sha="abc", code_commit="deadbeef"
    )
    html = DatasetBuilder.model_card_html(card)
    assert "<html" in html
    assert "bias" in html
