"""
End-to-end Prefect orchestration for the RecoMart pipeline:

  ingest_csv --> ingest_api --> validate --> prepare --> build_features
             --> feast_apply_materialize --> train_and_evaluate

Runs everything with a single command, including standing up (and tearing down)
the local mock product-signals API used by the near-real-time ingestion task.
Retries and failure logging are handled per-task via Prefect's built-in retry
policy; a run-level log is also written to logs/pipeline_run.log.
"""
import subprocess
import sys
import time
from pathlib import Path

import requests
from prefect import flow, task, get_run_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MOCK_API_BASE_URL, LOGS_DIR, ROOT  # noqa: E402
from ingestion.ingest_csv_sources import ingest_all  # noqa: E402
from ingestion.ingest_api_source import ingest_api  # noqa: E402
from validation.validate_data import run_validation  # noqa: E402
from preparation.clean_and_eda import run_preparation  # noqa: E402
from features.build_features import run_build_features  # noqa: E402
from models.train_collaborative import run_training as train_collaborative  # noqa: E402
from models.train_content_based import run_training as train_content_based  # noqa: E402

VENV_BIN = ROOT / ".venv-linux" / "bin"
FEATURE_REPO_DIR = ROOT / "feature_store" / "feature_repo"


@task(retries=1, retry_delay_seconds=2)
def start_mock_api_server():
    logger = get_run_logger()
    proc = subprocess.Popen(
        [str(VENV_BIN / "python"), str(ROOT / "ingestion" / "mock_api_server.py")],
        stdout=open(LOGS_DIR / "mock_api_server.log", "a"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        try:
            if requests.get(f"{MOCK_API_BASE_URL}/health", timeout=2).ok:
                logger.info("Mock product-signals API is healthy (pid=%d)", proc.pid)
                return proc.pid
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise RuntimeError("Mock API server failed to start")


@task
def stop_mock_api_server(pid: int):
    logger = get_run_logger()
    subprocess.run(["kill", str(pid)], check=False)
    logger.info("Stopped mock API server (pid=%d)", pid)


@task(retries=2, retry_delay_seconds=5, log_prints=True)
def ingest_csv_task():
    print("Running batch CSV ingestion...")
    return str(ingest_all())


@task(retries=2, retry_delay_seconds=5, log_prints=True)
def ingest_api_task(_dep=None, sample_size: int = 300):
    print("Running near-real-time API ingestion...")
    return str(ingest_api(sample_size=sample_size))


@task(retries=1, log_prints=True)
def validate_task(_dep=None):
    print("Running data validation...")
    return str(run_validation())


@task(retries=1, log_prints=True)
def prepare_task(_dep=None):
    print("Running data preparation and EDA...")
    return str(run_preparation())


@task(retries=1, log_prints=True)
def build_features_task(_dep=None):
    print("Running feature engineering...")
    return run_build_features()


@task(retries=1, log_prints=True)
def feast_apply_materialize_task(_dep=None):
    print("Applying and materializing the Feast feature store...")
    subprocess.run(
        [str(VENV_BIN / "feast"), "apply"], cwd=FEATURE_REPO_DIR, check=True,
    )
    subprocess.run(
        [str(VENV_BIN / "feast"), "materialize", "2016-01-01T00:00:00", "2030-01-01T00:00:00"],
        cwd=FEATURE_REPO_DIR, check=True,
    )
    return "feast_ready"


@task(retries=1, log_prints=True)
def train_and_evaluate_task(_dep=None):
    print("Training collaborative filtering model...")
    cf_metrics, cf_run_id = train_collaborative()
    print("Training content-based model...")
    cb_metrics, cb_run_id = train_content_based()
    return {
        "collaborative": {"metrics": cf_metrics, "mlflow_run_id": cf_run_id},
        "content_based": {"metrics": cb_metrics, "mlflow_run_id": cb_run_id},
    }


@flow(name="recomart_data_pipeline", log_prints=True)
def recomart_pipeline_flow(api_sample_size: int = 300):
    logger = get_run_logger()
    logger.info("Starting RecoMart end-to-end data management pipeline")

    api_pid = start_mock_api_server()
    try:
        csv_manifest = ingest_csv_task()
        api_manifest = ingest_api_task(csv_manifest, sample_size=api_sample_size)
        quality_report = validate_task(api_manifest)
        processed_path = prepare_task(quality_report)
        feature_counts = build_features_task(processed_path)
        feast_status = feast_apply_materialize_task(feature_counts)
        results = train_and_evaluate_task(feast_status)
    finally:
        stop_mock_api_server(api_pid)

    logger.info("Pipeline complete. Results: %s", results)
    return results


if __name__ == "__main__":
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    result = recomart_pipeline_flow()
    print("\n=== Final pipeline result ===")
    print(result)
