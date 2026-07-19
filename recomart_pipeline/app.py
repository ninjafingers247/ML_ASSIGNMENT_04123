"""
RecoMart recommendation demo — a thin visual layer over the pipeline in this
repo. Pick a customer, see their purchase history, and compare what the
collaborative (SVD) and content-based (cosine KNN) models recommend, plus the
final hybrid ranking served by models/infer.py.

Run with: streamlit run app.py   (from inside recomart_pipeline/)
"""
import pandas as pd
import streamlit as st

from config import PROCESSED_DIR, FEATURES_DIR, MLFLOW_TRACKING_URI
import models.infer as infer

st.set_page_config(page_title="RecoMart Recommendations", layout="wide", page_icon="🛒")


@st.cache_resource
def ensure_models_loaded():
    infer._load()
    return True


@st.cache_data
def load_prepared_interactions():
    return pd.read_parquet(PROCESSED_DIR / "interactions_prepared.parquet")


@st.cache_data
def load_item_metadata():
    prepared = load_prepared_interactions()
    price_cat = prepared.groupby("product_id").agg(
        category=("product_category_name_english", "first"),
        avg_price=("price", "mean"),
    )
    item_feat = pd.read_parquet(FEATURES_DIR / "item_features.parquet").set_index("product_id")
    meta = price_cat.join(
        item_feat[["item_interaction_count", "item_avg_rating", "api_popularity_score", "api_sentiment_score"]],
        how="left",
    )
    return meta


@st.cache_data
def load_user_options():
    df = pd.read_parquet(FEATURES_DIR / "user_features.parquet")
    return df.sort_values("user_interaction_count", ascending=False).reset_index(drop=True)


@st.cache_data
def load_model_metrics():
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name("recomart_recommender")
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs([experiment.experiment_id], order_by=["attribute.start_time DESC"])
    rows, seen = [], set()
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        if name in seen:
            continue
        seen.add(name)
        rows.append({"model": name, **run.data.metrics})
    return pd.DataFrame(rows)


def get_user_history(user_id: str) -> pd.DataFrame:
    prepared = load_prepared_interactions()
    hist = prepared[prepared["customer_unique_id"] == user_id][
        ["product_id", "product_category_name_english", "price", "review_score", "order_purchase_timestamp"]
    ].sort_values("order_purchase_timestamp")
    return hist.rename(columns={
        "product_category_name_english": "category",
        "review_score": "rating_given",
        "order_purchase_timestamp": "purchased_at",
    })


def render_item_cards(item_ids: list, item_meta: pd.DataFrame, notes: dict = None):
    if not item_ids:
        st.info("No recommendations available for this user under this strategy "
                 "(e.g. no strongly-rated items to base content similarity on).")
        return
    notes = notes or {}
    cols = st.columns(5)
    for i, item_id in enumerate(item_ids):
        with cols[i % 5]:
            st.markdown(f"**#{i + 1}**")
            st.caption(f"`{item_id[:16]}…`")
            if item_id in item_meta.index:
                m = item_meta.loc[item_id]
                st.write(f"📦 {m['category']}")
                st.write(f"💰 R$ {m['avg_price']:.2f}")
                if pd.notna(m["item_avg_rating"]):
                    st.write(f"⭐ {m['item_avg_rating']:.1f} avg rating")
                if pd.notna(m["api_popularity_score"]):
                    st.write(f"🔥 popularity {m['api_popularity_score']:.2f}")
            if item_id in notes:
                st.info(notes[item_id])
            st.divider()


def main():
    ensure_models_loaded()
    item_meta = load_item_metadata()
    user_feat = load_user_options()

    st.title("🛒 RecoMart Recommendation Demo")
    st.caption(
        "Visual layer over the DM4ML pipeline: Olist e-commerce data → validation → "
        "feature store → SVD collaborative filtering + content-based (cosine KNN) models."
    )

    user_feat = user_feat.copy()
    user_feat["label"] = user_feat.apply(
        lambda r: f"{r['customer_unique_id'][:12]}…  ({int(r['user_interaction_count'])} purchase(s), "
                  f"avg {r['user_avg_rating']:.1f}★)",
        axis=1,
    )

    with st.sidebar:
        st.header("Choose a customer")
        selected_label = st.selectbox("Customer", user_feat["label"], index=0)
        selected_user = user_feat.loc[user_feat["label"] == selected_label, "customer_unique_id"].iloc[0]
        k = st.slider("Number of recommendations", 3, 20, 10)

        st.divider()
        st.caption(
            f"Dataset: {len(user_feat):,} users · {len(item_meta):,} items · "
            "matrix density ≈ 0.0000326 (very sparse — see report for why that "
            "makes content-based filtering outperform pure collaborative filtering here)."
        )

        with st.expander("📊 Latest MLflow model metrics"):
            metrics_df = load_model_metrics()
            if metrics_df.empty:
                st.write("No MLflow runs found yet — run `models/train_collaborative.py` "
                         "and `models/train_content_based.py` first.")
            else:
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Customer profile")
        urow = user_feat.loc[user_feat["customer_unique_id"] == selected_user].iloc[0]
        st.metric("Total purchases", int(urow["user_interaction_count"]))
        st.metric("Average rating given", f"{urow['user_avg_rating']:.2f} / 5")
        st.metric("Average spend", f"R$ {urow['user_avg_spend']:.2f}")
    with col2:
        st.subheader("Purchase history")
        st.dataframe(get_user_history(selected_user), use_container_width=True, hide_index=True)

    st.divider()
    tab_hybrid, tab_cf, tab_cb = st.tabs([
        "🔀 Hybrid (served recommendation)", "🤝 Collaborative filtering", "🏷️ Content-based",
    ])

    with tab_hybrid:
        st.caption("What `models/infer.py` actually serves: content-based ranked first "
                   "(it performs better on this sparse dataset), collaborative filling in the rest.")
        hybrid_items = infer.recommend(selected_user, k, "hybrid")
        render_item_cards(hybrid_items, item_meta)

    with tab_cf:
        st.caption("Matrix-factorization (SVD): ranks items by predicted rating from latent "
                   "user/item factors learned across all users.")
        cf_detailed = infer.recommend_collaborative_detailed(selected_user, k)
        notes = {iid: f"predicted rating ≈ {score:.2f}" for iid, score in cf_detailed}
        render_item_cards([iid for iid, _ in cf_detailed], item_meta, notes)
        if cf_detailed:
            chart_df = pd.DataFrame(cf_detailed, columns=["product_id", "predicted_score"]).set_index("product_id")
            st.bar_chart(chart_df)

    with tab_cb:
        st.caption("Content-based (cosine KNN): ranks items by attribute similarity to "
                   "products this customer already rated ≥4★.")
        cb_detailed = infer.recommend_content_detailed(selected_user, k)
        notes = {
            iid: f"similarity {sim:.2f} — because you liked `{because[:12]}…`"
            for iid, sim, because in cb_detailed
        }
        render_item_cards([iid for iid, _, _ in cb_detailed], item_meta, notes)
        if cb_detailed:
            chart_df = pd.DataFrame(
                [(iid, sim) for iid, sim, _ in cb_detailed], columns=["product_id", "similarity"]
            ).set_index("product_id")
            st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
