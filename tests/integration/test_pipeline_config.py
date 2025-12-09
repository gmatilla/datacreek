import importlib
from pathlib import Path

import datacreek.pipelines as pipelines


def test_env_pipeline_config(monkeypatch, tmp_path):
    custom = tmp_path / "pipes.yaml"
    custom.write_text(
        "qa:\n  steps: [ingest, to_kg]\n  trainings: [sft]\n  description: short\n"
    )
    monkeypatch.setenv("DATACREEK_PIPELINES_CONFIG", str(custom))
    importlib.reload(pipelines)
    pipe = pipelines.get_pipeline(pipelines.DatasetType.QA)
    assert pipe.steps == [
        pipelines.PipelineStep.INGEST,
        pipelines.PipelineStep.TO_KG,
    ]
