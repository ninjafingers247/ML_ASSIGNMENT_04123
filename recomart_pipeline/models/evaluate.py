"""Ranking evaluation metrics: Precision@K, Recall@K, NDCG@K."""
import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(idx + 2) for idx, item in enumerate(top_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(recommend_fn, test_holdout: dict, k: int = 10) -> dict:
    """
    recommend_fn(user_id, k) -> ranked list of recommended item_ids (already
    excluding items the user has already seen in train).
    test_holdout: {user_id: set(held_out_item_ids)}
    """
    precisions, recalls, ndcgs = [], [], []
    for user_id, relevant in test_holdout.items():
        recommended = recommend_fn(user_id, k)
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_evaluated_users": len(test_holdout),
    }
