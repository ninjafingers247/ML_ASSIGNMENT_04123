"""
Deployable inference interface for RecoMart: loads the trained collaborative
and content-based models and exposes a single `recommend(user_id, k, strategy)`
function, as required by the "deployable recommendation model and inference
interface" deliverable.
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.train_collaborative import make_recommend_fn as make_cf_recommend_fn  # noqa: E402
from models.train_content_based import make_recommend_fn as make_cb_recommend_fn  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent
CF_MODEL_PATH = MODELS_DIR / "collaborative_svd_model.joblib"
CB_MODEL_PATH = MODELS_DIR / "content_based_model.joblib"

_cf_recommend = None
_cb_recommend = None


def _load():
    global _cf_recommend, _cb_recommend
    if _cf_recommend is None:
        _cf_recommend = make_cf_recommend_fn(joblib.load(CF_MODEL_PATH))
    if _cb_recommend is None:
        _cb_recommend = make_cb_recommend_fn(joblib.load(CB_MODEL_PATH))


def recommend(user_id: str, k: int = 10, strategy: str = "hybrid") -> list:
    """
    strategy: "collaborative" | "content" | "hybrid" (interleaved union, de-duped)
    Returns a ranked list of up to k product_ids.
    """
    _load()

    if strategy == "collaborative":
        return _cf_recommend(user_id, k)
    if strategy == "content":
        return _cb_recommend(user_id, k)

    cf_items = _cf_recommend(user_id, k)
    cb_items = _cb_recommend(user_id, k)
    merged, seen = [], set()
    for a, b in zip(cf_items, cb_items):
        for item in (b, a):  # content-based first: it evaluated stronger on this sparse dataset
            if item not in seen:
                merged.append(item)
                seen.add(item)
    for item in cb_items + cf_items:
        if item not in seen and len(merged) < k:
            merged.append(item)
            seen.add(item)
    return merged[:k]


def recommend_collaborative_detailed(user_id: str, k: int = 10) -> list:
    """Returns [(product_id, predicted_rating_score), ...] for the Streamlit app."""
    _load()
    return _cf_recommend(user_id, k, with_scores=True)


def recommend_content_detailed(user_id: str, k: int = 10) -> list:
    """Returns [(product_id, similarity_score, because_of_product_id), ...]."""
    _load()
    return _cb_recommend(user_id, k, with_scores=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--strategy", default="hybrid", choices=["collaborative", "content", "hybrid"])
    args = parser.parse_args()

    print(f"Top-{args.k} ({args.strategy}) recommendations for {args.user_id}:")
    for product_id in recommend(args.user_id, args.k, args.strategy):
        print(" -", product_id)
