import json
import re
import subprocess
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION_PATTERN = re.compile(r"^dataset-v[0-9]+$")

def run_command(command):
    """Execute an approved command without invoking a shell."""

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stderr.strip()
                or result.stdout.strip()
                or "Command failed",
            }

        return {
            "status": "success",
            "output": result.stdout.strip(),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Tool execution timed out",
        }

    except OSError:
        return {
            "status": "error",
            "error": "Required command is unavailable",
        }


def parse_json_result(result):
    """Convert successful command output into structured data."""

    if result["status"] != "success":
        return result

    try:
        return {
            "status": "success",
            "data": json.loads(result["output"]),
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "DVC returned invalid JSON",
        }


def list_datasets():
    """List datasets tracked by DVC."""

    result = run_command([
        "dvc",
        "list",
        ".",
        "--recursive",
        "--dvc-only",
        "--json",
    ])

    return parse_json_result(result)


def get_current_dataset_version():
    """Retrieve Git and DVC provenance for the current dataset."""

    commit = run_command([
        "git",
        "rev-parse",
        "HEAD",
    ])

    if commit["status"] != "success":
        return commit

    tags = run_command([
        "git",
        "tag",
        "--points-at",
        "HEAD",
    ])

    if tags["status"] != "success":
        return tags

    metadata_path = PROJECT_ROOT / "data/train.csv.dvc"

    try:
        metadata = yaml.safe_load(
            metadata_path.read_text(encoding="utf-8")
        )

        output = next(
            item
            for item in metadata["outs"]
            if item["path"] == "train.csv"
        )

    except (OSError, KeyError, StopIteration, TypeError, yaml.YAMLError):
        return {
            "status": "error",
            "error": "Dataset metadata is unavailable or invalid",
        }

    matching_tags = [
        tag
        for tag in tags["output"].splitlines()
        if VERSION_PATTERN.fullmatch(tag)
    ]

    return {
        "status": "success",
        "dataset": "data/train.csv",
        "version_tags": matching_tags,
        "git_commit": commit["output"],
        "dvc_hash": output.get("md5"),
        "tracked_size_bytes": output.get("size"),
        "note": (
            "The version tags identify HEAD. "
            "The DVC hash comes from the current metadata file. "
            "Check dataset status for uncommitted changes."
        ),
    }


def validate_version(version):
    """Accept only existing dataset version tags."""

    if not isinstance(version, str):
        return False

    if not VERSION_PATTERN.fullmatch(version):
        return False

    result = run_command([
        "git",
        "rev-parse",
        "--verify",
        f"refs/tags/{version}",
    ])

    return result["status"] == "success"


def compare_dataset_versions(version1, version2):
    """Compare two approved dataset versions."""

    if not validate_version(version1):
        return {
            "status": "error",
            "error": "Invalid or unknown first dataset version",
        }

    if not validate_version(version2):
        return {
            "status": "error",
            "error": "Invalid or unknown second dataset version",
        }

    result = run_command([
        "dvc",
        "diff",
        "--json",
        version1,
        version2,
    ])

    return parse_json_result(result)


def detect_dataset_changes():
    """Check for changes to DVC-tracked data."""

    result = run_command([
        "dvc",
        "status",
        "--json",
    ])

    return parse_json_result(result)


def get_reproduction_instructions(version):
    """Provide safe instructions for reproducing a dataset version.

    This function does not modify the working directory.
    """

    if not validate_version(version):
        return {
            "status": "error",
            "error": "Invalid or unknown dataset version",
        }

    commit = run_command([
        "git",
        "rev-parse",
        "--verify",
        f"refs/tags/{version}",
    ])

    if commit["status"] != "success":
        return commit

    return {
        "status": "success",
        "dataset_version": version,
        "git_commit": commit["output"],
        "instructions": [
            "Ensure the Git working tree is clean.",
            "Make sure the configured DVC remote is accessible.",
            f"git restore --source {version} -- data/train.csv.dvc",
            "dvc pull",
            "Verify the restored dataset contents.",
            "git restore --source HEAD -- data/train.csv.dvc",
            "dvc checkout",
        ],
        "warning": (
            "These commands modify local dataset files. "
            "Review changes before executing them. "
            "The agent only provides instructions."
        ),
    }
