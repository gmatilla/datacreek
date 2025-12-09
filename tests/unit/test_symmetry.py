import networkx as nx

from datacreek.analysis import symmetry


def test_automorphisms_and_orbits():
    g = nx.Graph()
    g.add_edge("a", "b")
    autos = symmetry.automorphisms(g)
    # Either direct swap or identity + swap may appear
    assert any(m.get("a") == "b" and m.get("b") == "a" for m in autos)
    orbits = symmetry.automorphism_orbits(g)
    assert {"a", "b"} in orbits


def test_group_order_and_quotient():
    g = nx.Graph()
    g.add_edges_from([(1, 2), (1, 3), (2, 3)])  # triangle
    order = symmetry.automorphism_group_order(g)
    assert order >= 6  # triangle has 6 automorphisms
    orbits = symmetry.automorphism_orbits(g)
    q, mapping = symmetry.quotient_graph(g, orbits)
    # all nodes collapse to the same class
    assert mapping == {1: 0, 2: 0, 3: 0}
    assert q.number_of_nodes() == 0
