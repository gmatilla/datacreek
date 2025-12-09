import json

import networkx as nx
import pytest

from datacreek.analysis import mapper


@pytest.mark.heavy
def test_mapper_full_and_json():
    g = nx.path_graph(5)
    nerve, cover = mapper.mapper_full(g, cover=(3, 0.5), clusterer="single")
    assert nerve.number_of_nodes() == len(cover)

    data = json.loads(
        mapper.mapper_to_json(g, cover=(3, 0.5), clusterer="single", autotune=False)
    )
    assert len(data["nodes"]) == len(cover)
    assert len(data["links"]) == nerve.number_of_edges()


@pytest.mark.heavy
def test_mapper_full_dbscan():
    g = nx.path_graph(6)
    nerve, cover = mapper.mapper_full(g, cover=(2, 0.5), clusterer="dbscan")
    assert nerve.number_of_nodes() == len(cover)
    assert nerve.number_of_nodes() > 0
