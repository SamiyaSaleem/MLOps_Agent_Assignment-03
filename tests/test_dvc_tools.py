import pytest
from agent.tools import dvc_tools

def test_list_datasets():
    result = dvc_tools.list_datasets()

    assert result["status"] == "success"

    paths = [
        item["path"]
        for item in result["data"]
    ]

    assert "data/train.csv" in paths


def test_current_dataset_version():
    result = dvc_tools.get_current_dataset_version()

    assert result["status"] == "success"
    assert result["dataset"] == "data/train.csv"
    assert len(result["git_commit"]) == 40
    assert result["dvc_hash"]
    assert result["tracked_size_bytes"] > 0
    assert isinstance(result["version_tags"], list)


def test_compare_dataset_versions():
    result = dvc_tools.compare_dataset_versions(
        "dataset-v1",
        "dataset-v2"
    )

    assert result["status"] == "success"

    modified_paths = [
        item["path"]
        for item in result["data"]["modified"]
    ]

    assert "data/train.csv" in modified_paths


def test_detect_dataset_changes():
    result = dvc_tools.detect_dataset_changes()

    assert result["status"] == "success"
    assert isinstance(result["data"], dict)


def test_reproduction_instructions():
    result = dvc_tools.get_reproduction_instructions(
        "dataset-v1"
    )

    assert result["status"] == "success"
    assert result["dataset_version"] == "dataset-v1"
    assert len(result["git_commit"]) == 40

    instructions = " ".join(result["instructions"])

    assert "dvc pull" in instructions
    assert "dvc checkout" in instructions


def test_invalid_dataset_version():
    result = dvc_tools.compare_dataset_versions(
        "invalid-version",
        "dataset-v2"
    )

    assert result["status"] == "error"
    assert "Invalid" in result["error"]


def test_command_timeout(monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=30
        )

    monkeypatch.setattr(
        dvc_tools.subprocess,
        "run",
        fake_run
    )

    result = dvc_tools.run_command([
        "dvc",
        "status"
    ])

    assert result["status"] == "error"
    assert "timed out" in result["error"]


def test_unavailable_command(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("dvc unavailable")

    monkeypatch.setattr(
        dvc_tools.subprocess,
        "run",
        fake_run
    )

    result = dvc_tools.run_command([
        "dvc",
        "status"
    ])

    assert result["status"] == "error"
    assert "unavailable" in result["error"]


def test_invalid_json_response():
    result = dvc_tools.parse_json_result({
        "status": "success",
        "output": "not valid json"
    })

    assert result["status"] == "error"
    assert "invalid JSON" in result["error"]
