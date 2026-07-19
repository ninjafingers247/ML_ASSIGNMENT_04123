"""
Data preparation and EDA for the RecoMart interactions dataset.

Joins reviews -> orders -> customers -> order_items -> products into a single
user-item interaction table, handles missing/duplicate interactions, encodes
categorical attributes, normalizes numeric variables, and produces EDA plots
(rating distribution, item popularity long-tail, user-item sparsity).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, EDA_PLOTS_DIR  # noqa: E402
from ingestion.common import get_logger  # noqa: E402
from validation.validate_data import load_all_tables  # noqa: E402

logger = get_logger("prepare_eda", "preparation.log")


def build_interactions(tables: dict) -> pd.DataFrame:
    reviews = tables["order_reviews"].drop_duplicates(subset=["review_id"])
    orders = tables["orders"][["order_id", "customer_id", "order_purchase_timestamp"]]
    customers = tables["customers"][["customer_id", "customer_unique_id", "customer_state"]]
    order_items = tables["order_items"][["order_id", "product_id", "price", "freight_value"]]
    products = tables["products"][["product_id", "product_category_name"]]
    category_translation = tables["category_translation"]

    before = len(reviews)
    df = reviews.merge(orders, on="order_id", how="inner")
    df = df.merge(customers, on="customer_id", how="inner")
    df = df.merge(order_items, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(category_translation, on="product_category_name", how="left")

    missing_interactions = df["review_score"].isnull().sum() + df["product_id"].isnull().sum()
    df = df.dropna(subset=["review_score", "product_id", "customer_unique_id"])
    logger.info(
        "Built interactions: %d reviews -> %d joined rows -> %d after dropping %d incomplete",
        before, before, len(df), missing_interactions,
    )

    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")
    return df


def encode_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cat_encoder = LabelEncoder()
    df["category_encoded"] = cat_encoder.fit_transform(df["product_category_name_english"])

    state_encoder = LabelEncoder()
    df["customer_state_encoded"] = state_encoder.fit_transform(df["customer_state"])

    scaler = MinMaxScaler()
    df[["price_norm", "freight_value_norm"]] = scaler.fit_transform(
        df[["price", "freight_value"]]
    )

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    ts_numeric = df["order_purchase_timestamp"].astype("int64")
    df["timestamp_norm"] = (ts_numeric - ts_numeric.min()) / (ts_numeric.max() - ts_numeric.min())

    logger.info("Encoded %d categories, %d states; normalized price/freight/timestamp",
                df["category_encoded"].nunique(), df["customer_state_encoded"].nunique())
    return df


def plot_rating_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    df["review_score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Rating (review_score) distribution")
    ax.set_xlabel("review_score")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "rating_distribution.png", dpi=120)
    plt.close(fig)


def plot_item_popularity(df: pd.DataFrame):
    counts = df["product_id"].value_counts().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(counts) + 1), counts.values, color="#DD8452")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Item popularity long-tail")
    ax.set_xlabel("item rank (log)")
    ax.set_ylabel("interaction count (log)")
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "item_popularity_longtail.png", dpi=120)
    plt.close(fig)


def plot_sparsity(df: pd.DataFrame) -> float:
    n_users = df["customer_unique_id"].nunique()
    n_items = df["product_id"].nunique()
    n_interactions = len(df)
    density = n_interactions / (n_users * n_items)

    top_users = df["customer_unique_id"].value_counts().head(60).index
    top_items = df["product_id"].value_counts().head(60).index
    sub = df[df["customer_unique_id"].isin(top_users) & df["product_id"].isin(top_items)]
    pivot = (
        sub.pivot_table(index="customer_unique_id", columns="product_id",
                         values="review_score", aggfunc="mean")
        .reindex(index=top_users, columns=top_items)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.notna().astype(int).values, cmap="Greens", aspect="auto")
    ax.set_title(f"User-item interaction sparsity (top 60x60 sample)\nGlobal density={density:.6f}")
    ax.set_xlabel("items")
    ax.set_ylabel("users")
    fig.colorbar(im, ax=ax, label="has interaction")
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "sparsity_heatmap.png", dpi=120)
    plt.close(fig)

    logger.info("Sparsity: %d users x %d items, %d interactions, density=%.6f",
                n_users, n_items, n_interactions, density)
    return density


def run_preparation() -> Path:
    tables = load_all_tables()
    df = build_interactions(tables)
    df = encode_and_normalize(df)

    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_rating_distribution(df)
    plot_item_popularity(df)
    density = plot_sparsity(df)

    keep_cols = [
        "customer_unique_id", "product_id", "review_score",
        "product_category_name_english", "category_encoded",
        "customer_state", "customer_state_encoded",
        "price", "price_norm", "freight_value", "freight_value_norm",
        "order_purchase_timestamp", "timestamp_norm",
    ]
    prepared = df[keep_cols].reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "interactions_prepared.parquet"
    prepared.to_parquet(out_path, index=False)

    logger.info(
        "Preparation complete: %d prepared interaction rows, %d users, %d items, density=%.6f -> %s",
        len(prepared), prepared["customer_unique_id"].nunique(),
        prepared["product_id"].nunique(), density, out_path,
    )
    return out_path


if __name__ == "__main__":
    run_preparation()
