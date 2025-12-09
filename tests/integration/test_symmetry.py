import networkx as nx

from datacreek.analysis.symmetry import (
    automorphism_group_order,
    automorphism_orbits,
    automorphisms,
    quotient_graph,
)
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


def test_symmetry_functions():
    g = nx.cycle_graph(4)
    autos = automorphisms(g, max_count=2)
    assert autos and isinstance(autos[0], dict)
    assert automorphism_group_order(g, max_count=3) >= 1
    orbits = automorphism_orbits(g, max_count=2)
    assert any(len(o) > 1 for o in orbits)
    q, mapping = quotient_graph(g, orbits)
    assert q.number_of_nodes() <= g.number_of_nodes()
    assert mapping


def test_dataset_symmetry_wrappers():
    ds = DatasetBuilder(DatasetType.QA)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "a")
    ds.add_chunk("d", "c2", "b")
    ds.add_chunk("d", "c3", "c")
    ds.graph.graph.add_edge("c1", "c2")
    ds.graph.graph.add_edge("c2", "c3")
    autos = ds.detect_automorphisms(max_count=1)
    assert isinstance(autos, list)
    order = ds.automorphism_group_order(max_count=2)
    assert order >= 1
    q, mapping = ds.quotient_by_symmetry(max_count=1)
    assert q.number_of_nodes() <= ds.graph.graph.number_of_nodes()
    assert mapping
