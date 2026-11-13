import hashlib
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from airflow.sdk import dag, task, get_current_context


PROJECT_ROOT = Path(
    "/home/user/Desktop/assignment-03-mlops-agent"
)

ML_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

log = logging.getLogger(__name__)

def execute_stage(stage_name):
    """
    Execute a machine-learning pipeline stage using
    the project's ML Python virtual environment.

    Stages:
        validate
        preprocess
        train
        evaluate
        quality_gate
    """

    ctxt = get_current_context()

    run_ID_AIRFLOW = ctxt["run_id"]

    run_key = hashlib.sha256(
        run_ID_AIRFLOW.encode("utf-8")
    ).hexdigest()[:20]

    log.info("Starting stage: %s", stage_name)
    log.info("Pipeline run key: %s", run_key)

    env = os.environ.copy()

    env["PATH"] = (
        str(ML_PYTHON.parent)
        + os.pathsep
        + env.get("PATH", "")
    )

    env["MLFLOW_TRACKING_URI"] = (
        "http://127.0.0.1:5000"
    )


    cmd = [
        str(ML_PYTHON),
        "-m",
        "training.airflow_stages",
        "--stage",
        stage_name,
        "--run-key",
        run_key,
    ]

    log.info("Executing approved stage: %s", stage_name)

    try:
        r = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    except subprocess.TimeoutExpired:
        log.error(
            "Pipeline stage timed out: %s",
            stage_name
        )

        raise RuntimeError(
            f"Pipeline stage timed out: {stage_name}"
        )

    except OSError as exc:
        log.error(
            "Unable to execute stage %s: %s",
            stage_name,
            type(exc).__name__,
        )

        raise RuntimeError(
            f"Unable to execute pipeline stage: {stage_name}"
        ) from exc

    if r.stdout:
        log.info(
            "Stage output:\n%s",
            r.stdout
        )

    if r.stderr:
        log.info(
            "Stage diagnostic output:\n%s",
            r.stderr
        )

    if r.returncode != 0:
        log.error(
            "Stage %s failed with exit code %s",
            stage_name,
            r.returncode,
        )

        raise RuntimeError(
            f"Pipeline stage failed: {stage_name}"
        )

    log.info(
        "Pipeline stage completed successfully: %s",
        stage_name,
    )

    return {
        "stage": stage_name,
        "status": "success",
        "run_key": run_key,
    }

@dag(
    dag_id="mlops_training_pipeline",

    description=(
        "Production-oriented MLOps training workflow "
        "with DVC, MLflow, validation, and quality gate"
    ),

    starting_date=datetime(2026, 11, 1),

    schedule=None,
    catchup=False,
    maximum_active_runs=1,

    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
        "execution_timeout": timedelta(minutes=10),
    },

    tags=[
        "mlops",
        "dvc",
        "mlflow",
        "training",
    ],
)
def mlops_training_pipeline():

    #validate the dataset
    @task
    def validate_dataset():
        """
        Validate the dataset schema, quality,
        and DVC dataset provenance.
        """
        return execute_stage("validate")


    #preprocess the data
    @task
    def preprocess_data():
        """
        Perform reproducible train/test splitting
        and save processed data.
        """
        return execute_stage("preprocess")


    #training of the model
    @task
    def train_model():
        """
        Train the classification model using
        the validated dataset.
        """
        return execute_stage("train")


    #evaluate and log to mlflow
    @task
    def evaluate_and_log():
        """
        Evaluate the trained model and record
        metrics, artifacts, and DVC provenance
        in MLflow.
        """
        return execute_stage("evaluate")
        

    #model quality gate
    @task
    def model_quality_gate():
        """
        Check whether model performance meets
        the minimum required quality threshold.

        A failure prevents downstream promotion.
        """
        return execute_stage("quality_gate")


    #task dependencies
    validation = validate_dataset()

    preprocessing = preprocess_data()

    training = train_model()

    evaluation = evaluate_and_log()

    quality_gate = model_quality_gate()

    (
        validation
        >> preprocessing
        >> training
        >> evaluation
        >> quality_gate
    )

mlops_training_pipeline()
