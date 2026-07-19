"""
Near-real-time ingestion source: polls the mock product-signals REST API
(`mock_api_server.py`) per product and lands the responses in the raw data lake.

Designed to be re-run on a short interval (via the Prefect schedule / a cron) so
the recommendation features always reflect fresh popularity/sentiment signals,
independent of the batch CSV ingestion above.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_LAKE_DIR, MOCK_API_BASE_URL  # noqa: E402
from ingestion.common import get_logger, retry  # noqa: E402

logger = get_logger("ingest_api", "ingestion.log")

SOURCE_NAME = "product_signals_api"


@retry(max_attempts=4, base_delay=0.5, logger=logger)
def _fetch_product_signals(product_id: str) -> dict:
    resp = requests.get(
        f"{MOCK_API_BASE_URL}/products/{product_id}/signals", timeout=5
    )
    if resp.status_code >= 500:
        raise ConnectionError(f"server error {resp.status_code} for {product_id}")
    resp.raise_for_status()
    return resp.json()


def ingest_api(sample_size: int = 300, ingestion_ts: str = None) -> Path:
    ingestion_ts = ingestion_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Starting API ingestion run %s", ingestion_ts)

    health = requests.get(f"{MOCK_API_BASE_URL}/health", timeout=5).json()
    logger.info("Mock API healthy, catalog size=%d", health["product_count"])

    product_ids = requests.get(f"{MOCK_API_BASE_URL}/products", timeout=10).json()[
        "product_ids"
    ][:sample_size]

    records, failures = [], 0
    start = time.time()
    for pid in product_ids:
        try:
            records.append(_fetch_product_signals(pid))
        except Exception as exc:
            failures += 1
            logger.error("Giving up on product %s after retries: %s", pid, exc)

    dest_dir = RAW_LAKE_DIR / SOURCE_NAME / "products" / ingestion_ts
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "product_signals.jsonl"
    with open(dest_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "source": SOURCE_NAME,
        "ingestion_ts": ingestion_ts,
        "requested": len(product_ids),
        "succeeded": len(records),
        "failed": failures,
        "duration_seconds": round(time.time() - start, 2),
        "output_file": str(dest_file),
    }
    manifest_dir = RAW_LAKE_DIR / SOURCE_NAME / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"manifest_{ingestion_ts}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Finished API ingestion run %s: %d/%d succeeded, %d failed. Manifest: %s",
        ingestion_ts, len(records), len(product_ids), failures, manifest_path,
    )
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300)
    args = parser.parse_args()
    ingest_api(sample_size=args.sample_size)
