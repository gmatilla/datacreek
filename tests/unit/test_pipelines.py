from pathlib import Path

import pytest

from datacreek import pipelines
from datacreek.models.qa import QAPair
from datacreek.models.results import QAGenerationResult


class DummyClient:
    def ping(self):
        return True


class BadClient:
    def ping(self):
        raise RuntimeError


def test_get_redis_client(monkeypatch):
    monkeypatch.setattr(pipelines, "backend_get_redis_client", lambda: DummyClient())
    assert isinstance(pipelines.get_redis_client(), DummyClient)


def test_get_redis_client_none(monkeypatch):
    monkeypatch.setattr(pipelines, "backend_get_redis_client", lambda: BadClient())
    assert pipelines.get_redis_client() is None


def test_load_pipelines_from_file(tmp_path):
    yaml_text = (
        "qa:\n  steps: [ingest, generate_qa]\n  trainings: [sft]\n  description: foo\n"
    )
    path = tmp_path / "p.yaml"
    path.write_text(yaml_text)
    data = pipelines.load_pipelines_from_file(path)
    pl = data[pipelines.DatasetType.QA]
    assert pl.steps == [
        pipelines.PipelineStep.INGEST,
        pipelines.PipelineStep.GENERATE_QA,
    ]
    assert pl.description == "foo"


def test_generation_options_model_to_options(tmp_path):
    conf = tmp_path / "c.yaml"
    model = pipelines.GenerationOptionsModel(start_step="ingest", config_path=str(conf))
    opts = model.to_options()
    assert opts.start_step is pipelines.PipelineStep.INGEST
    assert opts.config_path == conf


def test_validate_step_result_success():
    res = QAGenerationResult(summary="s", qa_pairs=[QAPair("q", "a")])
    assert (
        pipelines._validate_step_result(
            pipelines.DatasetType.QA,
            pipelines.PipelineStep.GENERATE_QA,
            res,
        )
        is res
    )


def test_validate_step_result_missing_field():
    with pytest.raises(ValueError):
        pipelines._validate_step_result(
            pipelines.DatasetType.QA,
            pipelines.PipelineStep.GENERATE_QA,
            {},
        )


def test_pipeline_execution_error():
    err = pipelines.PipelineExecutionError(
        pipelines.PipelineStep.INGEST, ValueError("bad")
    )
    info = err.info.to_dict()
    assert info["step"] == "ingest"
    assert info["exc_type"] == "ValueError"


def test_training_lookups():
    types_ = pipelines.get_dataset_types_for_training(pipelines.TrainingGoal.SFT)
    assert pipelines.DatasetType.QA in types_
    pipelines_list = pipelines.get_pipelines_for_training(pipelines.TrainingGoal.SFT)
    assert all(isinstance(p, pipelines.GenerationPipeline) for p in pipelines_list)


def test_serialize_nested_dataclass():
    qa = QAPair("q", "a")
    res = QAGenerationResult(summary="s", qa_pairs=[qa])
    assert pipelines._serialize(res) == {
        "summary": "s",
        "qa_pairs": [
            {
                "question": "q",
                "answer": "a",
                "rating": None,
                "confidence": None,
                "chunk": None,
                "source": None,
                "facts": None,
            }
        ],
    }


def test_generation_options_invalid_step():
    with pytest.raises(ValueError):
        pipelines.GenerationOptionsModel(start_step="unknown").to_options()


def test_serialize_collections():
    qa = QAPair("x", "y")
    data = {"items": [qa]}
    result = pipelines._serialize(data)
    assert result["items"][0]["question"] == "x"


def test_validate_step_result_pref_pair():
    payload = {"pairs": [{"question": "q", "chosen": "c", "rejected": "r"}]}
    assert (
        pipelines._validate_step_result(
            pipelines.DatasetType.PREF_PAIR,
            pipelines.PipelineStep.GENERATE_CANDIDATES,
            payload,
        )
        == payload
    )


def test_validate_step_result_pref_pair_invalid():
    with pytest.raises(ValueError):
        pipelines._validate_step_result(
            pipelines.DatasetType.PREF_PAIR,
            pipelines.PipelineStep.GENERATE_CANDIDATES,
            {"pairs": [{"question": "q", "chosen": "c"}]},
        )


def test_gen_options_start_variants(tmp_path):
    conf = tmp_path / "c.yaml"
    model_none = pipelines.GenerationOptionsModel(config_path=str(conf))
    opts_none = model_none.to_options()
    assert opts_none.start_step is None
    model_enum = pipelines.GenerationOptionsModel(
        start_step=pipelines.PipelineStep.SAVE
    )
    opts_enum = model_enum.to_options()
    assert opts_enum.start_step is pipelines.PipelineStep.SAVE


def test_gen_options_pipeline_path(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    model = pipelines.GenerationOptionsModel(pipeline_config_path=str(cfg))
    opts = model.to_options()
    assert opts.pipeline_config_path == cfg


def test_validate_step_result_qa_pair_too_long():
    long_q = "q" * 1001
    long_a = "a" * 5001
    with pytest.raises(ValueError):
        pipelines._validate_step_result(
            pipelines.DatasetType.QA,
            pipelines.PipelineStep.GENERATE_QA,
            {"qa_pairs": [{"question": long_q, "answer": long_a}]},
        )


def test_validate_step_result_pref_list(tmp_path):
    responses = [{"question": "q", "answers": [{"text": "a1"}, {"text": "a2"}]}]
    payload = {"responses": responses}
    assert (
        pipelines._validate_step_result(
            pipelines.DatasetType.PREF_LIST,
            pipelines.PipelineStep.GENERATE_CANDIDATES,
            payload,
        )
        == payload
    )


def test_validate_step_result_pref_list_invalid():
    with pytest.raises(ValueError):
        pipelines._validate_step_result(
            pipelines.DatasetType.PREF_LIST,
            pipelines.PipelineStep.GENERATE_CANDIDATES,
            {"responses": [{"question": "", "answers": [{"text": "a"}]}]},
        )


def test_load_pipelines_from_file_unknown_type(tmp_path, caplog):
    yaml_text = "foo:\n  steps: [ingest]\n  trainings: [sft]\n"
    path = tmp_path / "p.yaml"
    path.write_text(yaml_text)
    with caplog.at_level("WARNING"):
        data = pipelines.load_pipelines_from_file(path)
    assert pipelines.DatasetType.QA not in data
    assert any("Unknown dataset type" in rec.message for rec in caplog.records)


def test_load_pipelines_invalid_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(": invalid")
    data = pipelines.load_pipelines(str(bad))
    assert pipelines.DatasetType.QA in data


def test_get_pipeline_unknown(monkeypatch):
    monkeypatch.setattr(pipelines, "load_pipelines", lambda: {})
    with pytest.raises(KeyError):
        pipelines.get_pipeline(pipelines.DatasetType.QA)
