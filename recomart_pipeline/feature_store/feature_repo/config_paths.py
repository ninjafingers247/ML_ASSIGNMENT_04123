"""Resolves absolute paths to the parquet feature tables for the Feast repo."""
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
_FEATURES_DIR = _PIPELINE_ROOT / "data" / "features"

USER_FEATURES_PATH = _FEATURES_DIR / "user_features.parquet"
ITEM_FEATURES_PATH = _FEATURES_DIR / "item_features.parquet"
INTERACTIONS_PATH = _FEATURES_DIR / "interactions.parquet"
