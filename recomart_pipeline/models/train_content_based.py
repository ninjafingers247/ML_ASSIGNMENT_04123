"""
Content-based filtering: represents each product by its category (one-hot) plus
normalized physical attributes, finds nearest-neighbor items by cosine
similarity, and recommends items similar to what a user has previously rated
highly. Tracked in MLflow alongside the collaborative model for comparison.
"""
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MLRUNS_DIR, MLFLOW_TRACKING_URI, SOURCE_DATA_DIR, RANDOM_SEED  # noqa: E402
from ingestion.common import get_logger  # noqa: E402
from models.data_split import load_interactions, leave_one_out_split  # noqa: E402
from models.evaluate import evaluate_recommender  # noqa: E402

logger = get_logger("train_content_based", "training.log")

MODEL_PATH = Path(__file__).resolve().parent / "content_based_model.joblib"
N_NEIGHBORS = 15
LIKE_THRESHOLD = 4  # review_score >= this counts as a "liked" seed item


def build_item_content_matrix():
    products = pd.read_csv(SOURCE_DATA_DIR / "olist_products_dataset.csv")
    numeric_cols = [
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    ]
    products[numeric_cols] = products[numeric_cols].fillna(products[numeric_cols].median())
    products["product_category_name"] = products["product_category_name"].fillna("unknown")

    ohe = OneHotEncoder(handle_unknown="ignore")
    category_matrix = ohe.fit_transform(products[["product_category_name"]])

    scaler = MinMaxScaler()
    numeric_matrix = scaler.fit_transform(products[numeric_cols])

    feature_matrix = hstack([category_matrix, numeric_matrix]).tocsr()
    item_ids = products["product_id"].tolist()
    return feature_matrix, item_ids


def make_recommend_fn(model: dict):
    nn_model = model["nn_model"]
    item_id_to_row = model["item_id_to_row"]
    row_to_item_id = model["row_to_item_id"]
    feature_matrix = model["feature_matrix"]
    liked_items_by_user = model["liked_items_by_user"]
    seen_by_user = model["seen_by_user"]

    def recommend(user_id: str, k: int = 10) -> list:
        liked = liked_items_by_user.get(user_id, [])
        liked_rows = [item_id_to_row[i] for i in liked if i in item_id_to_row]
        if not liked_rows:
            return []

        scores = {}
        for row in liked_rows:
            distances, neighbor_rows = nn_model.kneighbors(
                feature_matrix[row], n_neighbors=min(N_NEIGHBORS, feature_matrix.shape[0])
            )
            for dist, nrow in zip(distances[0], neighbor_rows[0]):
                item_id = row_to_item_id[nrow]
                similarity = 1 - dist
                scores[item_id] = max(scores.get(item_id, 0.0), similarity)

        for item_id in seen_by_user.get(user_id, ()):
            scores.pop(item_id, None)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [item_id for item_id, _ in ranked]

    return recommend


def run_training():
    logger.info("Loading prepared interactions for content-based filtering")
    df = load_interactions()
    train_df, test_holdout = leave_one_out_split(df)

    feature_matrix, item_ids = build_item_content_matrix()
    item_id_to_row = {item_id: row for row, item_id in enumerate(item_ids)}
    row_to_item_id = {row: item_id for item_id, row in item_id_to_row.items()}

    nn_model = NearestNeighbors(metric="cosine")
    nn_model.fit(feature_matrix)

    liked_items_by_user = (
        train_df[train_df["review_score"] >= LIKE_THRESHOLD]
        .groupby("customer_unique_id")["product_id"]
        .apply(list)
        .to_dict()
    )
    seen_by_user = train_df.groupby("customer_unique_id")["product_id"].apply(set).to_dict()

    model = {
        "nn_model": nn_model,
        "feature_matrix": feature_matrix,
        "item_id_to_row": item_id_to_row,
        "row_to_item_id": row_to_item_id,
        "liked_items_by_user": liked_items_by_user,
        "seen_by_user": seen_by_user,
    }
    recommend_fn = make_recommend_fn(model)

    metrics = evaluate_recommender(recommend_fn, test_holdout, k=10)
    logger.info("Content-based filtering metrics: %s", metrics)

    joblib.dump(model, MODEL_PATH)

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("recomart_recommender")
    with mlflow.start_run(run_name="content_based_knn"):
        mlflow.log_param("model_type", "content_based_cosine_knn")
        mlflow.log_param("n_neighbors", N_NEIGHBORS)
        mlflow.log_param("like_threshold", LIKE_THRESHOLD)
        mlflow.log_param("n_items", len(item_ids))
        mlflow.log_param("n_users_with_liked_items", len(liked_items_by_user))
        mlflow.log_param("random_seed", RANDOM_SEED)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_artifact(str(MODEL_PATH))
        run_id = mlflow.active_run().info.run_id

    logger.info("MLflow run %s logged for content_based_knn", run_id)
    return metrics, run_id


if __name__ == "__main__":
    print(run_training())
