# 1. Problem Formulation

## Business problem

RecoMart, an e-commerce startup, wants a personalized product recommendation
engine to increase customer engagement and cross-selling. Without
personalization, customers browse a generic catalog and RecoMart misses
conversion opportunities that competitors with recommendation engines capture.
The business problem: **given a customer's purchase and rating history, product
catalog metadata, and external product-signal feeds, recommend a ranked list of
products the customer is likely to purchase and rate highly**, to raise
conversion rate and average order value.

## Data sources and attributes

| Source | Type | Key attributes |
|---|---|---|
| Vendor CSV batch feed (Olist Brazilian e-commerce data, real dataset used to stand in for RecoMart's own transactional export) | Batch, CSV | `customers` (user id, location), `orders` (order status, purchase timestamp), `order_items` (product, seller, price, freight), `order_reviews` (1-5 `review_score`, comment text), `order_payments`, `products` (category, physical attributes), `sellers`, category name translation |
| Product-signals REST API (local mock standing in for an external sentiment/popularity provider) | Near-real-time, polled | per-product `popularity_score` (order-volume based), `sentiment_score` (lexicon polarity over review text), `avg_price` |

Entity model: **users** = `customer_unique_id`, **items** = `product_id`,
**ratings** = `review_score` (1-5), **transactions** = `order_items` +
`order_payments`.

## Expected pipeline outputs

1. **Clean datasets for EDA** — `data/processed/interactions_prepared.parquet`, with EDA plots in `preparation/eda_plots/` (rating distribution, item-popularity long tail, user-item sparsity).
2. **Engineered features** for collaborative/content-based models — `data/features/*.parquet` + `features/recomart_features.db`, versioned and served through the Feast feature store.
3. **Deployable recommendation model and inference interface** — collaborative (matrix-factorization/SVD) and content-based (category/attribute cosine-similarity) models, tracked in MLflow, exposed via `models/infer.py:recommend(user_id, k, strategy)`.

## Evaluation metrics

- **Precision@K** — fraction of the top-K recommended items that the user actually goes on to interact with.
- **Recall@K** — fraction of the user's held-out relevant items that appear in the top-K recommendations.
- **NDCG@K** — rank-sensitive version of the above, rewarding relevant items appearing earlier in the list.

K=10 is used throughout (`models/evaluate.py`), evaluated via a leave-one-out
split: for every user with 2+ rated purchases, one interaction is held out as
the ground-truth test target and the rest are used for training
(`models/data_split.py`).
