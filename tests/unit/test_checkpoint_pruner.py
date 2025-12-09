from training.monitoring import CheckpointPruner


def test_pruner_keeps_top_k(tmp_path):
    pruner = CheckpointPruner(k=2, mode="max")
    scores = [0.1, 0.4, 0.3]
    for i, s in enumerate(scores):
        d = tmp_path / f"ckpt{i}"
        d.mkdir()
        # simulate saving some file inside checkpoint
        (d / "weights.bin").write_text("x")
        pruner.step(str(d), s)
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"ckpt1", "ckpt2"}


def test_pruner_min_mode(tmp_path):
    pruner = CheckpointPruner(k=2, mode="min")
    scores = [0.4, 0.1, 0.3]
    for i, s in enumerate(scores):
        d = tmp_path / f"ckpt{i}"
        d.mkdir()
        (d / "weights.bin").write_text("x")
        pruner.step(str(d), s)
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"ckpt1", "ckpt2"}
