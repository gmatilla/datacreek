import networkx as nx
import pytest

import datacreek.analysis.symmetry as sym


@pytest.mark.heavy
def test_automorphisms_and_group_order():
    g = nx.cycle_graph(4)
    autos = sym.automorphisms(g, max_count=5)
    assert autos
    order = sym.automorphism_group_order(g, max_count=5)
    assert order == 5


@pytest.mark.heavy
def test_orbits_and_quotient():
    g = nx.path_graph(3)
    orbits = sym.automorphism_orbits(g, max_count=5)
    q, mapping = sym.quotient_graph(g, orbits)
    # quotient graph should have fewer or equal nodes
    assert q.number_of_nodes() <= g.number_of_nodes()
    assert set(mapping) == set(g.nodes())
