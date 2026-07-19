"""
Local mock REST API standing in for RecoMart's external product-signal provider
(the assignment brief's "External APIs e.g., sentiment or popularity scores").

Serves per-product popularity (order volume) and a lightweight lexicon-based
sentiment score computed from Portuguese review comments. Runs entirely offline
against the already-downloaded Olist CSVs; occasionally returns 500s/latency to
give the ingestion client's retry logic something real to do.
"""
import random
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SOURCE_DATA_DIR, MOCK_API_HOST, MOCK_API_PORT  # noqa: E402

app = Flask(__name__)

POSITIVE_WORDS = [
    "bom", "otimo", "ótimo", "excelente", "recomendo", "rapido", "rápido",
    "adorei", "perfeito", "gostei", "maravilhoso", "eficiente", "top",
    "chegou antes", "qualidade", "satisfeito",
]
NEGATIVE_WORDS = [
    "ruim", "pessimo", "péssimo", "atraso", "quebrado", "defeito",
    "nao gostei", "não gostei", "horrivel", "horrível", "demora", "errado",
    "cancelado", "veio errado", "arrependi",
]


def _sentiment_score(comments: pd.Series) -> float:
    text = " ".join(str(c).lower() for c in comments.dropna())
    if not text.strip():
        return 0.0
    pos = sum(text.count(w) for w in POSITIVE_WORDS)
    neg = sum(text.count(w) for w in NEGATIVE_WORDS)
    return round((pos - neg) / (pos + neg + 1), 4)


def _build_product_signals() -> pd.DataFrame:
    order_items = pd.read_csv(SOURCE_DATA_DIR / "olist_order_items_dataset.csv")
    reviews = pd.read_csv(SOURCE_DATA_DIR / "olist_order_reviews_dataset.csv")

    popularity = (
        order_items.groupby("product_id")
        .agg(order_count=("order_id", "nunique"), avg_price=("price", "mean"))
        .reset_index()
    )

    order_to_products = order_items[["order_id", "product_id"]].drop_duplicates()
    reviews_with_product = reviews.merge(order_to_products, on="order_id", how="inner")

    sentiment = (
        reviews_with_product.groupby("product_id")
        .apply(
            lambda g: pd.Series(
                {
                    "avg_review_score": g["review_score"].mean(),
                    "sentiment_score": _sentiment_score(g["review_comment_message"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    signals = popularity.merge(sentiment, on="product_id", how="left")
    max_orders = signals["order_count"].max()
    signals["popularity_score"] = (signals["order_count"] / max_orders).round(4)
    signals["avg_review_score"] = signals["avg_review_score"].fillna(0.0)
    signals["sentiment_score"] = signals["sentiment_score"].fillna(0.0)
    return signals.set_index("product_id")


print("Precomputing product popularity/sentiment signals for mock API...")
PRODUCT_SIGNALS = _build_product_signals()
print(f"Mock API ready with signals for {len(PRODUCT_SIGNALS)} products.")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "product_count": len(PRODUCT_SIGNALS)})


@app.route("/products")
def list_products():
    return jsonify({"product_ids": PRODUCT_SIGNALS.index.tolist()})


@app.route("/products/<product_id>/signals")
def product_signals(product_id):
    # Simulate occasional transient server errors and latency, like a real
    # third-party API under load, to exercise the client's retry/backoff path.
    if random.random() < 0.15:
        return jsonify({"error": "temporarily unavailable"}), 503

    if product_id not in PRODUCT_SIGNALS.index:
        return jsonify({"error": "product not found"}), 404

    row = PRODUCT_SIGNALS.loc[product_id]
    return jsonify(
        {
            "product_id": product_id,
            "order_count": int(row["order_count"]),
            "avg_price": float(row["avg_price"]),
            "avg_review_score": float(row["avg_review_score"]),
            "sentiment_score": float(row["sentiment_score"]),
            "popularity_score": float(row["popularity_score"]),
        }
    )


if __name__ == "__main__":
    app.run(host=MOCK_API_HOST, port=MOCK_API_PORT)
