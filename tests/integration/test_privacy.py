import random

from datacreek.analysis.privacy import k_out_randomized_response


def test_k_out_randomized_response_length():
    ids = ["a", "b", "c", "d"]
    res = k_out_randomized_response(ids, k=2)
    assert len(res) == len(ids)


def test_dataset_privacy_wrapper():
    from datacreek.core.dataset import DatasetBuilder, DatasetType

    ds = DatasetBuilder(DatasetType.TEXT)
    out = ds.apply_k_out_privacy(["x", "y", "z"], k=1)
    assert len(out) == 3
