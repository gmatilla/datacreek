import datacreek.analysis.fractal as fractal
from datacreek.analysis import fractal_dim_embedding


def test_fractal_dim_embedding(monkeypatch):
    monkeypatch.setattr(
        fractal, "latent_box_dimension", lambda c, r, sample=512: (2.0, [])
    )
    dim = fractal_dim_embedding({1: [0.0]}, [1.0])
    assert dim == 2.0
