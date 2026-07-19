"""
Demonstrates Feast's two retrieval paths against the RecoMart feature repo:
- get_historical_features: point-in-time-correct features for model training
- get_online_features: latest materialized features for low-latency inference
"""
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore

FEATURE_REPO_DIR = Path(__file__).resolve().parent / "feature_repo"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES_DIR  # noqa: E402


def demo_historical_retrieval(store: FeatureStore, n: int = 5):
    # Training snapshot "as of now": query the feature values as they currently
    # stand for a sample of interactions (the user/item feature tables carry the
    # build-time event_timestamp, not the original interaction date, so a
    # point-in-time join needs an equally current "as of" timestamp here).
    interactions = pd.read_parquet(FEATURES_DIR / "interactions.parquet")
    entity_df = interactions[["customer_unique_id", "product_id"]].head(n).copy()
    entity_df["event_timestamp"] = pd.Timestamp.utcnow()

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_features:user_interaction_count",
            "user_features:user_avg_rating",
            "item_features:item_avg_rating",
            "item_features:api_popularity_score",
            "item_features:api_sentiment_score",
        ],
    ).to_df()

    print("=== Historical (training-time) feature retrieval ===")
    print(training_df.to_string(index=False))
    return training_df


def demo_online_retrieval(store: FeatureStore, n: int = 5):
    interactions = pd.read_parquet(FEATURES_DIR / "interactions.parquet")
    sample_users = interactions["customer_unique_id"].head(n).tolist()
    sample_items = interactions["product_id"].head(n).tolist()

    online_df = store.get_online_features(
        features=[
            "user_features:user_interaction_count",
            "user_features:user_avg_rating",
            "item_features:item_avg_rating",
            "item_features:api_popularity_score",
        ],
        entity_rows=[
            {"customer_unique_id": u, "product_id": i}
            for u, i in zip(sample_users, sample_items)
        ],
    ).to_df()

    print("\n=== Online (inference-time) feature retrieval ===")
    print(online_df.to_string(index=False))
    return online_df


if __name__ == "__main__":
    fs = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    demo_historical_retrieval(fs)
    demo_online_retrieval(fs)
