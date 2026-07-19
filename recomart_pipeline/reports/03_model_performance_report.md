# 3. Model Performance Report

## Dataset characteristics feeding the models

- 94,698 users (`customer_unique_id`), 32,660 items (`product_id`), 100,730 rated
  interactions after cleaning/dedup — interaction matrix density ≈ **0.0000326**
  (highly sparse, as is typical for e-commerce where most customers purchase
  once). Only 5,025 users (~5.3%) have 2+ rated purchases and can be evaluated
  under a leave-one-out split (`models/data_split.py`); the rest still
  contribute collaborative signal during training.

## Models trained

### Collaborative filtering — Matrix Factorization (Truncated SVD)
`models/train_collaborative.py`: builds the sparse user x item rating matrix
(mean-centered), decomposes it with `scipy.sparse.linalg.svds` (20 latent
factors), and scores unseen items as `global_mean + U_u . (Σ V^T)_i`.

### Content-based filtering — Category/Attribute Cosine KNN
`models/train_content_based.py`: represents each product as one-hot category +
normalized physical attributes (weight/dimensions), and recommends items nearest
(cosine similarity) to a user's previously highly-rated (≥4) products.

## Results (Precision@10 / Recall@10 / NDCG@10, leave-one-out, n=5,025 evaluated users)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Collaborative filtering (SVD) | 0.00129 | 0.01294 | 0.00922 |
| Content-based (cosine KNN)    | **0.01067** | **0.10667** | **0.06470** |

Both runs are tracked in MLflow (`mlruns/mlflow.db`, experiment
`recomart_recommender`) with parameters (latent factors, neighbor count,
similarity threshold, matrix density) and metrics logged per run, plus the
serialized model artifact attached to each run.

## Discussion

Content-based filtering clearly outperforms pure collaborative filtering on
this dataset — by roughly 8x on Precision/Recall and 7x on NDCG. This is a
direct consequence of the extreme sparsity: with ~95% of users making only one
purchase, there's rarely enough co-rated signal between users for matrix
factorization to find meaningful latent structure, whereas content-based
filtering only needs the target item's own attributes to find similar products,
so it degrades far more gracefully under sparsity and cold-start conditions.
This motivates the `models/infer.py` hybrid strategy (content-based ranked
first, collaborative filling in remaining slots) as the practical inference
default, and points to **hybrid re-ranking or session-based/sequential models**
as the natural next step (see Conclusion & Future Scope).

## Inference interface

`models/infer.py` exposes `recommend(user_id, k=10, strategy="hybrid"|"collaborative"|"content")`,
loading both serialized models and returning a ranked list of `product_id`s —
this is RecoMart's deployable recommendation interface.
