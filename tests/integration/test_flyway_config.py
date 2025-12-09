from pathlib import Path


def test_haa_index_in_flyway_conf():
    conf = Path("ops/flyway.conf").read_text()
    assert "2025-07-haa_index.cypher" in conf
