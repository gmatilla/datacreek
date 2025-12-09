import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx

from datacreek.analysis.fractal import colour_box_dimension


def test_colour_box_dimension_basic():
    g = nx.cycle_graph(6)
    dim, counts = colour_box_dimension(g, [1, 2])
    assert isinstance(dim, float)
    assert len(counts) == 2
