import math
from typing import Dict, List, Tuple


def explain_to_svg(data: Dict) -> str:
    """Render explanation data as a simple SVG.

    Parameters
    ----------
    data : dict
        Output from :func:`KnowledgeGraph.explain_node` with ``nodes`` and
        ``edges`` lists and an ``attention`` mapping.

    Returns
    -------
    str
        SVG string visualising the subgraph with colored edges.
    """
    nodes: List[str] = data.get("nodes", [])
    edges: List[Tuple[str, str]] = data.get("edges", [])
    att: Dict[str, float] = data.get("attention", {})

    if not nodes:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'

    R = 80.0
    coords: Dict[str, Tuple[float, float]] = {}
    for i, n in enumerate(nodes):
        angle = 2.0 * math.pi * i / len(nodes)
        coords[n] = (R + R * math.cos(angle), R + R * math.sin(angle))

    elements: List[str] = []
    # edges first
    for u, v in edges:
        x1, y1 = coords.get(u, (R, R))
        x2, y2 = coords.get(v, (R, R))
        score = att.get(f"{u}->{v}", att.get(f"{v}->{u}", 0.0))
        width = 1.0 + 2.0 * score
        color = f"rgb({int(255*score)},0,{int(255*(1-score))})"
        elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width:.1f}" />'
        )

    # nodes
    for n, (x, y) in coords.items():
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="white" stroke="black" />'
        )
        elements.append(
            f'<text x="{x + 5:.1f}" y="{y + 5:.1f}" font-size="6">{n}</text>'
        )

    body = "\n".join(elements)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*R}" height="{2*R}">'
        f"{body}</svg>"
    )
    return svg
