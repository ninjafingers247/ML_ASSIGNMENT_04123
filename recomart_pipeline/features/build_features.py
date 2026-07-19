"""
Feature engineering and transformation: turns the prepared interactions dataset
(+ the near-real-time product-signals API ingestion) into user/item/interaction
feature tables, stored in both SQLite (structured warehouse, per schema.sql) and
parquet (for Feast offline-store ingestion in the feature_store stage).
"""
import itertools
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, FEATURES_DIR, RAW_LAKE_DIR  # noqa: E402
from ingestion.common import get_logger  # noqa: E402

logger = get_logger("build_features", "features.log")

FEATURES_DB_PATH = Path(__file__).resolve().parent / "recomart_features.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _latest_api_signals() -> pd.DataFrame:
    api_dir = RAW_LAKE_DIR / "product_signals_api" / "products"
    partitions = sorted(p for p in api_dir.iterdir() if p.is_dir())
    latest = partitions[-1]
    records = [json.loads(line) for line in open(latest / "product_signals.jsonl")]
    return pd.DataFrame(records)


def build_user_features(interactions: pd.DataFrame, now_ts: str) -> pd.DataFrame:
    agg = interactions.groupby("customer_unique_id").agg(
        user_interaction_count=("product_id", "count"),
        user_avg_rating=("review_score", "mean"),
        user_avg_spend=("price", "mean"),
    ).reset_index()
    agg["event_timestamp"] = now_ts
    return agg


def build_item_features(interactions: pd.DataFrame, api_signals: pd.DataFrame, now_ts: str) -> pd.DataFrame:
    agg = interactions.groupby("product_id").agg(
        item_interaction_count=("customer_unique_id", "count"),
        item_avg_rating=("review_score", "mean"),
        category_encoded=("category_encoded", "first"),
        product_category_name_english=("product_category_name_english", "first"),
    ).reset_index()

    api = api_signals.rename(columns={
        "popularity_score": "api_popularity_score",
        "sentiment_score": "api_sentiment_score",
        "avg_price": "api_avg_price",
    })[["product_id", "api_popularity_score", "api_sentiment_score", "api_avg_price"]]

    merged = agg.merge(api, on="product_id", how="left")
    merged["event_timestamp"] = now_ts
    return merged


def build_item_cooccurrence(interactions: pd.DataFrame, top_n: int = 5000) -> pd.DataFrame:
    pair_counts = Counter()
    for _, items in interactions.groupby("customer_unique_id")["product_id"]:
        unique_items = sorted(set(items))
        if len(unique_items) < 2:
            continue
        for a, b in itertools.combinations(unique_items, 2):
            pair_counts[(a, b)] += 1

    top_pairs = pair_counts.most_common(top_n)
    rows = [{"product_id_a": a, "product_id_b": b, "cooccurrence_count": c} for (a, b), c in top_pairs]
    logger.info("Computed %d co-occurring item pairs (kept top %d)", len(pair_counts), len(rows))
    return pd.DataFrame(rows, columns=["product_id_a", "product_id_b", "cooccurrence_count"])


def write_to_sqlite(tables: dict):
    FEATURES_DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(FEATURES_DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    for name, df in tables.items():
        df.to_sql(name, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    logger.info("Wrote %d tables to SQLite warehouse at %s", len(tables), FEATURES_DB_PATH)


def run_build_features() -> dict:
    interactions_prepared = pd.read_parquet(PROCESSED_DIR / "interactions_prepared.parquet")
    api_signals = _latest_api_signals()
    now_ts = pd.Timestamp.utcnow()

    interactions = interactions_prepared.rename(columns={"order_purchase_timestamp": "interaction_ts"})
    interactions["event_timestamp"] = pd.to_datetime(interactions["interaction_ts"], utc=True)
    interactions_out = interactions[[
        "customer_unique_id", "product_id", "review_score",
        "price_norm", "freight_value_norm", "timestamp_norm", "event_timestamp",
    ]].drop_duplicates(subset=["customer_unique_id", "product_id"])

    user_features = build_user_features(interactions_prepared, now_ts)
    item_features = build_item_features(interactions_prepared, api_signals, now_ts)
    cooccurrence = build_item_cooccurrence(interactions_prepared)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    interactions_out.to_parquet(FEATURES_DIR / "interactions.parquet", index=False)
    user_features.to_parquet(FEATURES_DIR / "user_features.parquet", index=False)
    item_features.to_parquet(FEATURES_DIR / "item_features.parquet", index=False)
    cooccurrence.to_parquet(FEATURES_DIR / "item_cooccurrence.parquet", index=False)

    write_to_sqlite({
        "interactions": interactions_out,
        "user_features": user_features,
        "item_features": item_features,
        "item_cooccurrence": cooccurrence,
    })

    logger.info(
        "Feature build complete: %d users, %d items, %d interactions, %d cooc pairs",
        len(user_features), len(item_features), len(interactions_out), len(cooccurrence),
    )
    return {
        "user_features": len(user_features),
        "item_features": len(item_features),
        "interactions": len(interactions_out),
        "item_cooccurrence": len(cooccurrence),
    }


if __name__ == "__main__":
    print(run_build_features())
