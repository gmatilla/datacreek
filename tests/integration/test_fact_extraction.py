def test_extract_facts_regex():
    from datacreek.utils.fact_extraction import extract_facts

    text = "Berlin is the capital of Germany. Beethoven was born in Bonn."
    facts = extract_facts(text)
    objs = {f["object"] for f in facts}
    assert "Bonn" in objs
    assert any("Germany" in o for o in objs)


def test_dataset_extract_facts():
    from datacreek import DatasetBuilder, DatasetType

    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="src")
    ds.add_chunk("d", "c1", "Paris is the capital of France.")
    ds.extract_facts()
    fact_nodes = [
        n for n, d in ds.graph.graph.nodes(data=True) if d.get("type") == "fact"
    ]
    assert len(fact_nodes) == 1
    edges = ds.graph.graph.edges(fact_nodes[0])
    assert any(ds.graph.graph.edges[e]["relation"] == "subject" for e in edges)
