import asyncio

import fakeredis
import pytest

from datacreek.core import curate
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.content_type import ContentType
from datacreek.pipelines import (
    DatasetType,
    PipelineExecutionError,
    PipelineStep,
    TrainingGoal,
    get_dataset_types_for_training,
    get_pipelines_for_training,
    get_trainings_for_dataset,
    run_generation_pipeline,
    run_generation_pipeline_async,
)


def test_get_trainings_for_dataset():
    qa_trainings = get_trainings_for_dataset(DatasetType.QA)
    assert TrainingGoal.SFT in qa_trainings
    assert TrainingGoal.CPT not in qa_trainings

    text_trainings = get_trainings_for_dataset(DatasetType.TEXT)
    assert text_trainings == [TrainingGoal.CPT]

    pair_trainings = get_trainings_for_dataset(DatasetType.PREF_PAIR)
    assert TrainingGoal.DPO in pair_trainings
    assert TrainingGoal.GRPO not in pair_trainings

    kg_trainings = get_trainings_for_dataset(DatasetType.KG)
    assert TrainingGoal.SFT in kg_trainings
    assert TrainingGoal.CPT not in kg_trainings

    tool_trainings = get_trainings_for_dataset(DatasetType.TOOL)
    assert TrainingGoal.SFT in tool_trainings
    assert TrainingGoal.CPT not in tool_trainings


def test_reverse_mapping():
    qa_datasets = get_dataset_types_for_training(TrainingGoal.SFT)
    assert DatasetType.QA in qa_datasets
    assert DatasetType.TEXT not in qa_datasets

    cpt_pipes = get_pipelines_for_training(TrainingGoal.CPT)
    assert len(cpt_pipes) == 1
    assert cpt_pipes[0].dataset_type == DatasetType.TEXT

    rrhf_datasets = get_dataset_types_for_training(TrainingGoal.RRHF)
    assert DatasetType.PREF_LIST in rrhf_datasets

    sft_datasets = get_dataset_types_for_training(TrainingGoal.SFT)
    assert DatasetType.KG in sft_datasets
    assert DatasetType.TOOL in sft_datasets
    assert DatasetType.CONVERSATION in sft_datasets
    assert DatasetType.MULTI_TOOL in sft_datasets


def test_pipelines_include_kg_step():
    for pipeline in get_pipelines_for_training(TrainingGoal.SFT):
        if pipeline.dataset_type != DatasetType.TEXT:
            assert pipeline.steps[1] == PipelineStep.TO_KG


def test_run_generation_pipeline(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        # content_type is passed positionally
        assert args[5] == ContentType.QA
        assert kwargs.get("async_mode") is True
        assert "kg" in kwargs
        calls.append("gen")
        return {"qa_pairs": [{"question": "q", "answer": "a"}]}

    def fake_curate(data, *args, **kwargs):
        assert kwargs.get("async_mode") is True
        calls.append("curate")
        return {"qa_pairs": data["qa_pairs"], "summary": ""}

    def fake_save(data, fmt, cfg, storage_format="json"):
        calls.append("save")
        assert fmt == "jsonl"
        return "done"

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", fake_save)

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.QA, kg, verbose=True, async_mode=True)

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_invalid():
    with pytest.raises(ValueError):
        kg = KnowledgeGraph()
        kg.add_document("d", source="s", text="text")
        run_generation_pipeline("wrong", kg)


def test_run_generation_pipeline_cot(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.COT
        assert "kg" in kwargs
        calls.append("gen")
        return {"cot_examples": [{"question": "q", "reasoning": "r", "answer": "a"}]}

    def fake_curate(data, *args, **kwargs):
        calls.append("curate")
        return {"qa_pairs": [{"question": "q", "answer": "a"}], "summary": ""}

    def fake_save(data, fmt, cfg, storage_format="json"):
        calls.append("save")
        return "done"

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", fake_save)

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.COT, kg)

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_async_cot(monkeypatch):
    """Async pipeline should call async handlers for CoT."""

    called = {}

    async def fake_generate(*args, **kwargs):
        called["gen"] = True
        return {"cot_examples": []}

    async def fake_curate(d, *a, **k):
        called["curate"] = True
        return {"qa_pairs": [], "summary": ""}

    monkeypatch.setattr("datacreek.pipelines.process_file_async", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs_async", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    res = asyncio.run(run_generation_pipeline_async(DatasetType.COT, kg))

    assert res == "done"
    assert called == {"gen": True, "curate": True}


def test_run_generation_pipeline_vqa(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        # VQA pipeline uses the special content type
        assert args[5] == ContentType.VQA_ADD_REASONING
        assert "kg" in kwargs
        calls.append("gen")
        return {"qa_pairs": [{"question": "q", "answer": "a"}]}

    def fake_curate(data, *args, **kwargs):
        calls.append("curate")
        return {"qa_pairs": data["qa_pairs"], "summary": ""}

    def fake_save(data, fmt, cfg, storage_format="json"):
        calls.append("save")
        return "done"

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", fake_save)

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="img")

    result = run_generation_pipeline(DatasetType.VQA, kg)

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_tool(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.TOOL_CALL
        assert "kg" in kwargs
        calls.append("gen")
        return {"conversations": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.TOOL, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_conversation(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.CONVERSATION
        assert "kg" in kwargs
        calls.append("gen")
        return {"conversations": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.CONVERSATION, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_multi_tool(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.MULTI_TOOL
        assert "kg" in kwargs
        calls.append("gen")
        return {"conversations": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.MULTI_TOOL, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_pref_pair(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.PREF_PAIR
        assert "kg" in kwargs
        calls.append("gen")
        return {"pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.PREF_PAIR, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_pref_list(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.PREF_LIST
        assert "kg" in kwargs
        calls.append("gen")
        return {"responses": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.PREF_LIST, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_from_kg(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.FROM_KG
        assert "kg" in kwargs
        assert kwargs.get("multi_answer") is True
        calls.append("gen")
        return {"qa_pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.KG, kg, multi_answer=True)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_overrides(monkeypatch):
    received = {}

    def fake_generate(*args, **kwargs):
        received["overrides"] = kwargs.get("config_overrides")
        assert "kg" in kwargs
        return {"qa_pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    run_generation_pipeline(DatasetType.QA, kg, overrides={"foo": 1})

    assert received["overrides"] == {"foo": 1}


def test_run_generation_pipeline_validation(monkeypatch):
    """Ensure missing keys raise an error during validation."""

    monkeypatch.setattr("datacreek.pipelines.process_file", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.QA, kg)


def test_run_generation_pipeline_unsupported(monkeypatch):
    """Ensure unsupported dataset types raise errors."""

    with pytest.raises(ValueError):
        kg = KnowledgeGraph()
        kg.add_document("d", source="s", text="text")
        run_generation_pipeline(DatasetType.TEXT, kg)


def test_run_generation_pipeline_dataclass(monkeypatch):
    """Dataclass results should pass validation."""

    from datacreek.models.qa import QAPair
    from datacreek.models.results import QAGenerationResult

    def fake_generate(*args, **kwargs):
        return QAGenerationResult(
            summary="", qa_pairs=[QAPair(question="q", answer="a")]
        )

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.QA, kg)

    assert result == "done"


def test_run_generation_pipeline_dataclass_invalid(monkeypatch):
    """Invalid dataclass should raise error."""

    from datacreek.models.cot import COTExample
    from datacreek.models.results import COTGenerationResult

    def fake_generate(*args, **kwargs):
        return COTGenerationResult(
            summary="",
            cot_examples=[COTExample(question="q", reasoning="r", answer="a")],
            conversations=[],
        )

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.QA, kg)


def test_validate_qa_pair_items(monkeypatch):
    """QA pair items must contain question and answer."""

    def fake_generate(*args, **kwargs):
        return {"qa_pairs": [{"question": "q"}]}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.QA, kg)


def test_validate_cot_example_items(monkeypatch):
    """COT example items must contain question, reasoning and answer."""

    def fake_generate(*args, **kwargs):
        return {"cot_examples": [{"question": "q", "answer": "a"}]}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.COT, kg)


def test_validate_qa_pair_items_empty(monkeypatch):
    """QA pair items must not have empty question or answer."""

    def fake_generate(*args, **kwargs):
        return {"qa_pairs": [{"question": " ", "answer": ""}]}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.QA, kg)


def test_validate_cot_example_items_empty(monkeypatch):
    """COT items must not have empty fields."""

    def fake_generate(*args, **kwargs):
        return {"cot_examples": [{"question": "q", "reasoning": "", "answer": ""}]}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.COT, kg)


def test_validate_pref_pair_items(monkeypatch):
    """Pairwise preference items must contain question, chosen and rejected."""

    def fake_generate(*args, **kwargs):
        return {"pairs": [{"question": "q", "chosen": "a"}]}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.PREF_PAIR, kg)


def test_validate_pref_list_items(monkeypatch):
    """Listwise preference items must include non-empty answers."""

    def fake_generate(*args, **kwargs):
        return {"responses": [{"question": "q", "answers": [{"text": ""}]}]}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    with pytest.raises(ValueError):
        run_generation_pipeline(DatasetType.PREF_LIST, kg)


def test_run_generation_pipeline_deduplicate(monkeypatch):
    """Duplicate QA pairs should be removed after curation."""

    def fake_generate(*args, **kwargs):
        return {
            "qa_pairs": [
                {"question": "q", "answer": "a"},
                {"question": "q", "answer": "a"},
            ]
        }

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)

    def fake_curate(data, *a, **k):
        from datacreek.utils import deduplicate_pairs

        return {"qa_pairs": deduplicate_pairs(data["qa_pairs"])}

    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: a[0])

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    res = run_generation_pipeline(DatasetType.QA, kg)

    assert len(res["qa_pairs"]) == 1


def test_run_generation_pipeline_async(monkeypatch):
    called = {}

    async def fake_generate(*args, **kwargs):
        called["async"] = True
        return {"qa_pairs": []}

    async def fake_curate(d, *a, **k):
        return d

    monkeypatch.setattr("datacreek.pipelines.process_file_async", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs_async", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    res = asyncio.run(run_generation_pipeline_async(DatasetType.QA, kg))

    assert res == "done"
    assert called.get("async") is True


def test_generation_pipeline_error(monkeypatch):
    """Failing step should raise PipelineExecutionError."""

    def bad_generate(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("datacreek.pipelines.process_file", bad_generate)
    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")
    with pytest.raises(PipelineExecutionError) as exc:
        run_generation_pipeline(DatasetType.QA, kg)
    assert exc.value.step is PipelineStep.GENERATE_QA
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "boom" in exc.value.traceback


def test_generation_pipeline_async_error(monkeypatch):
    async def bad_generate(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("datacreek.pipelines.process_file_async", bad_generate)
    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")
    with pytest.raises(PipelineExecutionError):
        asyncio.run(run_generation_pipeline_async(DatasetType.QA, kg))


def test_pipeline_curation_error(monkeypatch):
    """Curation errors should be wrapped in PipelineExecutionError."""

    monkeypatch.setattr(
        "datacreek.pipelines.process_file",
        lambda *a, **k: {"qa_pairs": [{"question": "q", "answer": "a"}]},
    )

    def bad_curate(*a, **k):
        raise curate.CurationError("oops", [ValueError("bad")])

    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", bad_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: a[0])

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    with pytest.raises(PipelineExecutionError) as exc:
        run_generation_pipeline(DatasetType.QA, kg)

    assert exc.value.step is PipelineStep.CURATE


def test_pipeline_error_details(monkeypatch):
    def bad_generate(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("datacreek.pipelines.process_file", bad_generate)
    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")
    with pytest.raises(PipelineExecutionError) as exc:
        run_generation_pipeline(DatasetType.QA, kg)

    info = exc.value.info
    assert info.step is PipelineStep.GENERATE_QA
    assert info.exc_type == "RuntimeError"
    assert "boom" in info.traceback


def test_start_step_resume(monkeypatch):
    """Pipeline should resume from the specified step."""

    called = {}

    def fake_generate(*a, **k):
        called["generate"] = True

    def fake_curate(data, *a, **k):
        called["curate"] = True
        return {"qa_pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: a[0])

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="t")

    res = run_generation_pipeline(
        DatasetType.QA,
        kg,
        start_step=PipelineStep.CURATE,
    )

    assert "generate" not in called
    assert called.get("curate") is True
    assert isinstance(res, curate.CurationResult)


def test_redis_resume(monkeypatch):
    """Pipeline should load progress from Redis when resuming."""

    client = fakeredis.FakeStrictRedis()
    called = {"generate": 0, "curate": 0}

    def fake_generate(*a, **k):
        called["generate"] += 1
        return {"qa_pairs": [{"question": "q", "answer": "a"}]}

    def fake_curate(data, *a, **k):
        called["curate"] += 1
        return {"qa_pairs": data["qa_pairs"]}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: a[0])

    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d", source="s", text="t")

    run_generation_pipeline(
        DatasetType.QA, ds.graph, dataset_builder=ds, redis_client=client
    )
    assert client.hget("dataset:demo:progress", "generate_qa") is not None

    res = run_generation_pipeline(
        DatasetType.QA,
        ds.graph,
        dataset_builder=ds,
        start_step=PipelineStep.CURATE,
        redis_client=client,
    )

    assert called["generate"] == 1
    assert called["curate"] == 2
    assert isinstance(res, curate.CurationResult)


def test_kg_cleanup_step(monkeypatch):
    """KG_CLEANUP should deduplicate and clean chunk texts."""

    def fake_generate(*a, **k):
        return {"qa_pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: a[0])

    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "<b>Hello</b>")
    kg.add_chunk("d", "c2", "<b>Hello</b>")

    run_generation_pipeline(DatasetType.QA, kg, start_step=PipelineStep.KG_CLEANUP)

    assert "c2" not in kg.graph.nodes
    assert kg.graph.nodes["c1"]["text"] == "Hello"


def test_kg_cleanup_with_params(monkeypatch):
    called = {}

    def fake_generate(*a, **k):
        return {"qa_pairs": []}

    def fake_cleanup(
        self,
        *,
        resolve_threshold=0.8,
        resolve_aliases=None,
        dedup_similarity=1.0,
        normalize_dates=True,
        mark_conflicts=False,
        validate=False,
    ):
        called["threshold"] = resolve_threshold
        called["aliases"] = resolve_aliases
        called["sim"] = dedup_similarity
        called["norm"] = normalize_dates
        called["conflict"] = mark_conflicts
        called["validate"] = validate
        return 0, 0

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr(
        "datacreek.core.dataset.DatasetBuilder.cleanup_graph", fake_cleanup
    )
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: a[0])

    kg = KnowledgeGraph()
    kg.add_document("d", source="s")

    run_generation_pipeline(
        DatasetType.QA,
        kg,
        dataset_builder=DatasetBuilder(DatasetType.QA),
        resolve_threshold=0.9,
        resolve_aliases={"IBM": ["International Business Machines"]},
        start_step=PipelineStep.KG_CLEANUP,
    )

    assert called["threshold"] == 0.9
    assert called["aliases"] == {"IBM": ["International Business Machines"]}
    assert called["sim"] == 1.0
    assert called["norm"] is True
    assert called["conflict"] is False
    assert called["validate"] is False


def test_pipeline_records_step_events(monkeypatch):
    """Pipeline should log events for each executed step."""

    monkeypatch.setattr(
        "datacreek.pipelines.process_file", lambda *a, **k: {"qa_pairs": []}
    )
    monkeypatch.setattr(
        "datacreek.pipelines.curate_qa_pairs",
        lambda d, *a, **k: {"qa_pairs": d["qa_pairs"]},
    )
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d", source="s", text="t")
    ds.add_chunk("d", "c1", "t")

    run_generation_pipeline(DatasetType.QA, ds.graph, dataset_builder=ds)

    ops = [e.operation for e in ds.events]
    assert "kg_cleanup" in ops
    assert "generate_qa" in ops
    assert "curate" in ops
    assert "save" in ops


def test_curation_threshold_param(monkeypatch):
    """Pipeline should pass curation_threshold to curate_qa_pairs."""

    received = {}

    def fake_generate(*a, **k):
        return {"qa_pairs": []}

    def fake_curate(data, *a, **k):
        # threshold is passed positionally
        received["threshold"] = a[1] if len(a) > 1 else None
        return {}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    kg = KnowledgeGraph()
    kg.add_document("d", source="s")

    run_generation_pipeline(DatasetType.QA, kg, curation_threshold=7)

    assert received["threshold"] == 7


def test_kg_cleanup_failure(monkeypatch):
    """Errors during KG cleanup should raise PipelineExecutionError."""

    def bad_cleanup(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "datacreek.core.dataset.DatasetBuilder.cleanup_graph", bad_cleanup
    )
    monkeypatch.setattr("datacreek.pipelines.process_file", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: d)
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: a[0])

    kg = KnowledgeGraph()
    kg.add_document("d", source="s")

    with pytest.raises(PipelineExecutionError) as exc:
        run_generation_pipeline(
            DatasetType.QA,
            kg,
            dataset_builder=DatasetBuilder(DatasetType.QA),
            start_step=PipelineStep.KG_CLEANUP,
        )

    assert exc.value.step is PipelineStep.KG_CLEANUP


def test_pipeline_progress_timestamps(monkeypatch):
    """Each pipeline step should record start and finish times."""

    client = fakeredis.FakeStrictRedis()

    monkeypatch.setattr(
        "datacreek.pipelines.process_file", lambda *a, **k: {"qa_pairs": []}
    )
    monkeypatch.setattr(
        "datacreek.pipelines.curate_qa_pairs",
        lambda d, *a, **k: {"qa_pairs": d["qa_pairs"]},
    )
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d", source="s", text="t")
    ds.add_chunk("d", "c1", "t")

    run_generation_pipeline(
        DatasetType.QA,
        ds.graph,
        dataset_builder=ds,
        redis_client=client,
    )

    key = "dataset:demo:progress"
    assert client.hget(key, "kg_cleanup_start") is not None
    assert client.hget(key, "kg_cleanup_finish") is not None
    assert client.hget(key, "generate_qa_start") is not None
    assert client.hget(key, "generate_qa_finish") is not None
    assert client.hget(key, "curate_start") is not None
    assert client.hget(key, "curate_finish") is not None
    assert client.hget(key, "save_start") is not None
    assert client.hget(key, "save_finish") is not None
