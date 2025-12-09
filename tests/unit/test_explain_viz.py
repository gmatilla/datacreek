import re

from datacreek.analysis.explain_viz import explain_to_svg


def test_explain_to_svg_empty():
    svg = explain_to_svg({"nodes": []})
    # Ensure minimal SVG is returned with width and height 10
    assert (
        svg == '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
    )


def test_explain_to_svg_basic_edge():
    data = {
        "nodes": ["A", "B"],
        "edges": [("A", "B")],
        "attention": {"A->B": 0.5},
    }
    svg = explain_to_svg(data)
    # Should contain svg root with dimension 160x160 (2*R)
    assert svg.startswith(
        '<svg xmlns="http://www.w3.org/2000/svg" width="160.0" height="160.0"'
    )
    # Line element with computed color and stroke-width
    assert (
        '<line x1="160.0" y1="80.0" x2="0.0" y2="80.0" stroke="rgb(127,0,127)" stroke-width="2.0" />'
        in svg
    )
    # Ensure nodes and labels are present
    assert svg.count("<circle") == 2
    # Expect labels near each node (A at 165, B at 5)
    assert '<text x="165.0" y="85.0" font-size="6">A</text>' in svg
    assert '<text x="5.0" y="85.0" font-size="6">B</text>' in svg
