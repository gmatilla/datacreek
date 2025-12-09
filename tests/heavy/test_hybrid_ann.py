import types

import numpy as np

import datacreek.analysis.hybrid_ann as ha


def fake_faiss():
    ns = types.SimpleNamespace()

    class IndexFlatIP:
        def __init__(self, d):
            self.d = d
            self.xb = None

        def add(self, xb):
            self.xb = np.asarray(xb, dtype=np.float32)

        def search(self, xq, k):
            scores = np.dot(np.asarray(xq, dtype=np.float32), self.xb.T)
            idx = np.argsort(-scores, axis=1)[:, :k]
            return scores, idx

    class IndexIVFPQ(IndexFlatIP):
        def __init__(self, quantizer, d, nlist, m, nbits):
            super().__init__(d)
            self.pq = types.SimpleNamespace(M=m)
            self.nprobe = 0
            self.nprobe_multi = []
            self.nsq = m

        def train(self, xb):
            pass

    class IndexHNSWFlat(IndexFlatIP):
        def __init__(self, d, M):
            super().__init__(d)
            self.hnsw = types.SimpleNamespace(efSearch=0)

    def index_cpu_to_gpu(res, dev, index):
        return index

    def read_index(path):
        idx = IndexFlatIP(2)
        idx.pq = types.SimpleNamespace(M=4)
        return idx

    def get_num_gpus():
        return 0

    ns.IndexFlatIP = IndexFlatIP
    ns.IndexIVFPQ = IndexIVFPQ
    ns.IndexHNSWFlat = IndexHNSWFlat
    ns.StandardGpuResources = types.SimpleNamespace
    ns.index_cpu_to_gpu = index_cpu_to_gpu
    ns.read_index = read_index
    ns.get_num_gpus = get_num_gpus
    return ns


def test_full_flow(monkeypatch):
    fake = fake_faiss()
    monkeypatch.setattr(ha, "faiss", fake)
    index = ha.load_ivfpq_cpu("dummy", 3)
    assert index.nprobe_multi == [3] * 4
    xb = np.eye(5, dtype=np.float32)
    xq = xb[:1]
    res = ha.rerank_pq(xb, xq, k=2, gpu=False, n_subprobe=2)
    assert res.shape == (1, 2)
    neighbors = ha.search_hnsw_pq(xb, xq, k=1, prefetch=5, n_subprobe=2)
    assert neighbors == [0]
    assert ha.expected_recall(2, 8, 4) > 0
    assert ha.choose_nprobe_multi(256, 16) == 4


def test_ivfpq_branch(monkeypatch):
    fake = fake_faiss()
    monkeypatch.setattr(ha, "faiss", fake)
    xb = np.random.rand(300, 4).astype(np.float32)
    xq = xb[:1]
    res = ha.rerank_pq(xb, xq, k=3, gpu=False)
    assert res.shape == (1, 3)
