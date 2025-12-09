import datacreek.utils.batch as batch
from datacreek.generators.qa_generator import QAGenerator
from datacreek.models.qa import QAPair


class DummyClient:
    def __init__(self):
        self.config = {
            "generation": {
                "chunk_size": 10,
                "overlap": 0,
                "temperature": 0.0,
                "batch_size": 2,
                "similarity_drop": 0.0,
                "retrieval_top_k": 2,
                "summary_temperature": 0.0,
                "summary_max_tokens": 50,
            },
            "prompts": {
                "summary": "Summ",
                "qa_generation": "{text}",
                "qa_rating": "rate",
            },
            "curate": {"threshold": 0.5, "batch_size": 1, "temperature": 0.0},
        }

    def chat_completion(self, messages, temperature=None, max_tokens=None):
        return "summary-text"

    def batch_completion(
        self, message_batches, temperature=None, batch_size=None, max_tokens=None
    ):
        return ["[{'question':'q','answer':'a'}]" for _ in message_batches]


def test_generate_summary_and_pairs(monkeypatch):
    client = DummyClient()
    gen = QAGenerator(client)

    # patch process_batches to bypass LLM
    def fake_process_batches(
        client_obj, batches, *, batch_size, temperature, parse_fn, raise_on_error=False
    ):
        return [[QAPair(question=f"q{i}", answer="a")] for i in range(len(batches))]

    monkeypatch.setattr(batch, "process_batches", fake_process_batches)

    summary = gen.generate_summary("doc")
    pairs = gen.generate_qa_pairs("hello world", summary, num_pairs=2)

    assert summary == "summary-text"
    assert len(pairs) == 2
    assert pairs[0].question.startswith("q")


def test_rate_qa_pairs(monkeypatch):
    client = DummyClient()
    gen = QAGenerator(client)

    qs = [QAPair(question="q", answer="a")]

    monkeypatch.setattr(batch, "process_batches", lambda *a, **k: ["[{'rating': 0.9}]"])
    monkeypatch.setattr(
        "datacreek.generators.qa_generator.parse_ratings",
        lambda text, orig=None: [QAPair(question="q", answer="a", rating=0.9)],
    )

    rated, metrics = gen.rate_qa_pairs(qs, "sum", threshold=0.5)
    assert metrics["filtered"] == 1
    assert rated[0]["rating"] == 0.9
