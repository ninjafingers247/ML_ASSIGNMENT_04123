"""
Collaborative filtering via matrix factorization (truncated SVD) over the
user x item explicit-rating matrix, tracked in MLflow.
"""
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MLRUNS_DIR, MLFLOW_TRACKING_URI, RANDOM_SEED  # noqa: E402
from ingestion.common import get_logger  # noqa: E402
from models.data_split import load_interactions, leave_one_out_split  # noqa: E402
from models.evaluate import evaluate_recommender  # noqa: E402

logger = get_logger("train_collaborative", "training.log")

MODEL_PATH = Path(__file__).resolve().parent / "collaborative_svd_model.joblib"
N_FACTORS = 20


def build_train_matrix(train_df: pd.DataFrame):
    user_ids = sorted(train_df["customer_unique_id"].unique())
    item_ids = sorted(train_df["product_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {p: i for i, p in enumerate(item_ids)}

    rows = train_df["customer_unique_id"].map(user_to_idx).values
    cols = train_df["product_id"].map(item_to_idx).values
    global_mean = train_df["review_score"].mean()
    data = train_df["review_score"].values - global_mean

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return matrix, user_to_idx, item_to_idx, global_mean


def train_svd(matrix: csr_matrix, n_factors: int):
    k = min(n_factors, min(matrix.shape) - 1)
    U, sigma, Vt = svds(matrix.astype(float), k=k)
    order = np.argsort(-sigma)
    return U[:, order], sigma[order], Vt[order, :], k


def make_recommend_fn(model: dict):
    U, sigma, Vt = model["U"], model["sigma"], model["Vt"]
    user_to_idx, item_to_idx = model["user_to_idx"], model["item_to_idx"]
    idx_to_item = {i: p for p, i in item_to_idx.items()}
    global_mean = model["global_mean"]
    seen_by_user = model["seen_by_user"]

    factors = sigma[:, None] * Vt  # (k, n_items)

    def recommend(user_id: str, k: int = 10, with_scores: bool = False):
        if user_id not in user_to_idx:
            return []
        uidx = user_to_idx[user_id]
        scores = global_mean + U[uidx, :] @ factors
        for item_id in seen_by_user.get(user_id, ()):
            if item_id in item_to_idx:
                scores[item_to_idx[item_id]] = -np.inf
        top_idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        if with_scores:
            return [(idx_to_item[i], float(scores[i])) for i in top_idx]
        return [idx_to_item[i] for i in top_idx]

    return recommend


def run_training():
    logger.info("Loading prepared interactions for collaborative filtering")
    df = load_interactions()
    train_df, test_holdout = leave_one_out_split(df)
    logger.info(
        "Train rows=%d, evaluable users with held-out item=%d",
        len(train_df), len(test_holdout),
    )

    matrix, user_to_idx, item_to_idx, global_mean = build_train_matrix(train_df)
    U, sigma, Vt, k_used = train_svd(matrix, N_FACTORS)

    seen_by_user = train_df.groupby("customer_unique_id")["product_id"].apply(set).to_dict()

    model = {
        "U": U, "sigma": sigma, "Vt": Vt,
        "user_to_idx": user_to_idx, "item_to_idx": item_to_idx,
        "global_mean": global_mean, "seen_by_user": seen_by_user,
    }
    recommend_fn = make_recommend_fn(model)

    metrics = evaluate_recommender(recommend_fn, test_holdout, k=10)
    logger.info("Collaborative filtering metrics: %s", metrics)

    joblib.dump(model, MODEL_PATH)

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("recomart_recommender")
    with mlflow.start_run(run_name="collaborative_svd"):
        mlflow.log_param("model_type", "matrix_factorization_svd")
        mlflow.log_param("n_factors_requested", N_FACTORS)
        mlflow.log_param("n_factors_used", int(k_used))
        mlflow.log_param("n_users", matrix.shape[0])
        mlflow.log_param("n_items", matrix.shape[1])
        mlflow.log_param("matrix_density", matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        mlflow.log_param("random_seed", RANDOM_SEED)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_artifact(str(MODEL_PATH))
        run_id = mlflow.active_run().info.run_id

    logger.info("MLflow run %s logged for collaborative_svd", run_id)
    return metrics, run_id


if __name__ == "__main__":
    print(run_training())
