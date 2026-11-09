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
    """Retrieve current dataset provenance and match historical tags."""

    commit = run_command([
        "git",
        "rev-parse",
        "HEAD",
    ])

    if commit["status"] != "success":
        return commit

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

        current_hash = output.get("md5")

        if not current_hash:
            raise ValueError("Missing DVC hash")

    except (OSError, KeyError, StopIteration, TypeError,
            ValueError, yaml.YAMLError):
        return {
            "status": "error",
            "error": "Dataset metadata is unavailable or invalid",
        }

    tags_result = run_command([
        "git",
        "tag",
        "--list",
        "dataset-v*",
    ])

    if tags_result["status"] != "success":
        return tags_result

    matching_versions = []

    for tag in tags_result["output"].splitlines():

        if not VERSION_PATTERN.fullmatch(tag):
            continue

        historical_metadata = run_command([
            "git",
            "show",
            f"refs/tags/{tag}:data/train.csv.dvc",
        ])

        if historical_metadata["status"] != "success":
            continue

        try:
            historical_data = yaml.safe_load(
                historical_metadata["output"]
            )

            historical_output = next(
                item
                for item in historical_data["outs"]
                if item["path"] == "train.csv"
            )

            if historical_output.get("md5") == current_hash:
                matching_versions.append(tag)

        except (KeyError, TypeError, StopIteration,
                yaml.YAMLError):
            continue

    status_result = detect_dataset_changes()

    if status_result["status"] != "success":
        return status_result

    return {
        "status": "success",
        "dataset": "data/train.csv",
        "version_tags": matching_versions,
        "git_commit": commit["output"],
        "dvc_hash": current_hash,
        "tracked_size_bytes": output.get("size"),
        "workspace_clean": not bool(status_result["data"]),
        "note": (
            "Dataset versions are identified by matching "
            "DVC content hashes against historical Git tags. "
            "git_commit represents current HEAD, not necessarily "
            "the commit that originally introduced the dataset."
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
