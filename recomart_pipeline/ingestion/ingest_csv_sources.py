"""
Batch ingestion of the Olist e-commerce CSV feeds (customers, orders, order items,
order reviews, products, sellers, payments, category translation) into the
partitioned raw data lake.

Simulates a vendor batch drop landing in `source_data/` and being periodically
picked up, validated as readable, and copied into
`data/raw/<source>/<type>/<ingestion_timestamp>/<file>.csv`.
"""
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SOURCE_DATA_DIR, RAW_LAKE_DIR  # noqa: E402
from ingestion.common import get_logger, retry  # noqa: E402

logger = get_logger("ingest_csv", "ingestion.log")

SOURCE_NAME = "olist_csv_batch"

FILE_TO_TYPE = {
    "olist_customers_dataset.csv": "customers",
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
}

# Injected once to demonstrate the retry/backoff path firing on a transient error.
_simulated_failure_done = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@retry(max_attempts=3, base_delay=1.0, logger=logger)
def _copy_one(src_file: Path, dest_dir: Path) -> dict:
    global _simulated_failure_done
    if not _simulated_failure_done and src_file.name == "olist_sellers_dataset.csv":
        _simulated_failure_done = True
        raise ConnectionError("simulated transient read error from vendor feed")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / src_file.name
    shutil.copyfile(src_file, dest_file)

    with open(src_file, "r", encoding="utf-8", errors="ignore") as f:
        row_count = sum(1 for _ in f) - 1  # minus header

    return {
        "file": src_file.name,
        "rows": row_count,
        "bytes": dest_file.stat().st_size,
        "sha256": _sha256(dest_file),
    }


def ingest_all(ingestion_ts: str = None) -> Path:
    ingestion_ts = ingestion_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {"source": SOURCE_NAME, "ingestion_ts": ingestion_ts, "files": []}

    logger.info("Starting batch CSV ingestion run %s", ingestion_ts)

    for filename, type_name in FILE_TO_TYPE.items():
        src_file = SOURCE_DATA_DIR / filename
        if not src_file.exists():
            logger.error("Missing expected source file: %s", src_file)
            continue

        dest_dir = RAW_LAKE_DIR / SOURCE_NAME / type_name / ingestion_ts
        try:
            info = _copy_one(src_file, dest_dir)
            info["type"] = type_name
            manifest["files"].append(info)
            logger.info(
                "Ingested %s -> %s (%d rows)", filename, dest_dir, info["rows"]
            )
        except Exception as exc:
            logger.error("Failed to ingest %s after retries: %s", filename, exc)

    run_dir = RAW_LAKE_DIR / SOURCE_NAME / "_manifests"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / f"manifest_{ingestion_ts}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Finished batch CSV ingestion run %s: %d/%d files ingested. Manifest: %s",
        ingestion_ts, len(manifest["files"]), len(FILE_TO_TYPE), manifest_path,
    )
    return manifest_path


if __name__ == "__main__":
    ingest_all()
