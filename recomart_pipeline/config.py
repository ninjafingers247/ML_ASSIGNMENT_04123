"""Central paths and constants shared across the RecoMart pipeline stages."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCE_DATA_DIR = ROOT / "source_data"          # upstream vendor drop (simulates external feeds)
RAW_LAKE_DIR = ROOT / "data" / "raw"             # partitioned raw data lake, populated by ingestion
PROCESSED_DIR = ROOT / "data" / "processed"      # cleaned/prepared datasets
FEATURES_DIR = ROOT / "data" / "features"        # engineered feature tables (parquet)

LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
EDA_PLOTS_DIR = ROOT / "preparation" / "eda_plots"

FEATURE_REPO_DIR = ROOT / "feature_store" / "feature_repo"
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"sqlite:///{MLRUNS_DIR}/mlflow.db"
FEATURES_DB_PATH = ROOT / "features" / "recomart_features.db"

MOCK_API_HOST = "127.0.0.1"
MOCK_API_PORT = 8879
MOCK_API_BASE_URL = f"http://{MOCK_API_HOST}:{MOCK_API_PORT}"

RANDOM_SEED = 42

for _d in (RAW_LAKE_DIR, PROCESSED_DIR, FEATURES_DIR, LOGS_DIR, REPORTS_DIR, EDA_PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
