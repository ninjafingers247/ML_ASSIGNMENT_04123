# End-to-End Data Management Pipeline for a Recommendation System

**Course**: Data Management for Machine Learning (Merged - AIMLCZG529/DSECLZG529), S2-25
**Assignment**: Assignment I (EC1, 20 Marks)

## Team Member Details

> _TODO: fill in before submission_

| Name | BITS ID |
|---|---|
| _TODO_ | _TODO_ |
| _TODO_ | _TODO_ |

## Problem Statement

RecoMart, an e-commerce startup, needs a data-driven recommendation engine to
personalize the shopping experience and improve conversion / cross-selling. The
platform's raw signal — clickstream, purchase history, product catalog, and
external sentiment/popularity feeds — must be ingested, validated, transformed
into features, and used to train and serve recommendation models, continuously
and reproducibly. Full detail in `01_problem_formulation.md`.

## Objectives

1. Ingest at least two distinct data sources (batch CSV + a near-real-time REST API) with retry/error-handling and audit logging.
2. Land raw data in a partitioned, structured lake and validate its quality automatically.
3. Clean, encode, and engineer features suitable for both collaborative and content-based recommendation.
4. Serve those features through a versioned feature store, usable identically for training and inference.
5. Version raw/processed/feature data with full lineage.
6. Train and evaluate at least one recommendation model with tracked experiments, and expose a simple inference interface.
7. Orchestrate the entire pipeline as a single, automated, monitored flow.

## Methodology / Pipeline

```
                    ┌─────────────────────┐        ┌───────────────────────────┐
                    │  Olist CSV batch    │        │ Mock product-signals API  │
                    │  (customers,orders, │        │ (popularity / sentiment)  │
                    │  items,reviews,...)  │        │  near-real-time polling   │
                    └──────────┬──────────┘        └─────────────┬─────────────┘
                               │  retry+backoff, logging          │ retry+backoff, logging
                               ▼                                  ▼
                    ┌──────────────────────────── raw data lake ─────────────────────────┐
                    │  data/raw/<source>/<type>/<ingestion_ts>/...  (+ JSON manifests)    │
                    └──────────────────────────────────┬──────────────────────────────────┘
                                                        ▼
                                          validation (nulls/dupes/schema/range/referential)
                                                        ▼
                                     preparation (cleaning, encoding, normalization, EDA)
                                                        ▼
                                    feature engineering (SQL warehouse + parquet feature tables)
                                                        ▼
                                    Feast feature store (offline: training / online: inference)
                                                        ▼
                       model training: collaborative (SVD) + content-based (cosine KNN), MLflow-tracked
                                                        ▼
                                inference interface: models/infer.py recommend(user_id, k, strategy)

  DVC versions source_data/ + every stage's output, with full lineage in dvc.lock.
  Prefect's `recomart_pipeline_flow` (orchestration/prefect_flow.py) runs the whole chain end-to-end.
```

## Implementation Details

- **Dataset**: Olist Brazilian E-Commerce (real data, official company GitHub
  mirror) — 94,698 users, 32,660 items, 100,730 rated interactions after
  cleaning. `customer_unique_id`=user, `product_id`=item, `review_score`
  (1-5)=rating, `order_items`/`order_payments`=transactions.
- **Ingestion** (`ingestion/`): batch CSV ingestion with a retry/backoff
  decorator and per-file sha256 manifest logging; a local Flask mock API
  (`mock_api_server.py`) serving per-product popularity/sentiment, polled by
  `ingest_api_source.py` with the same retry/backoff pattern — both log to
  `logs/ingestion.log`.
- **Validation** (`validation/validate_data.py`): schema conformance, null
  checks, duplicate-key checks, `review_score` range (1-5), price/payment
  non-negativity, and referential integrity across all 8 tables — surfaced 814
  duplicate `review_id`s in the real data, documented in
  `02_data_quality_report.md`.
- **Preparation/EDA** (`preparation/clean_and_eda.py`): joins reviews → orders →
  customers → order_items → products, label-encodes category/state, min-max
  normalizes price/freight/timestamp, and plots rating distribution, item
  popularity long-tail, and user-item sparsity (density ≈ 0.0000326).
- **Feature engineering** (`features/`): user activity frequency, avg
  rating per user/item, item co-occurrence (7,096 pairs found), plus the API's
  popularity/sentiment signals — written to both a SQLite warehouse
  (`schema.sql`) and parquet.
- **Feature store** (`feature_store/feature_repo/`): Feast, local provider,
  file offline store + sqlite online store; `demo_retrieval.py` proves both
  point-in-time historical retrieval (training) and online retrieval
  (inference) work against the same registry.
- **Versioning** (`versioning/VERSIONING.md`): DVC (`dvc init --subdir`),
  local-directory remote, `dvc.yaml` pipeline (`ingest_csv → validate`,
  `ingest_csv → prepare → build_features`), `dvc repro`/`dvc dag` verified.
- **Model training** (`models/`): SVD-based collaborative filtering and
  cosine-KNN content-based filtering, both MLflow-tracked (sqlite backend),
  evaluated via leave-one-out Precision@10/Recall@10/NDCG@10.
- **Orchestration** (`orchestration/prefect_flow.py`): a Prefect flow chaining
  every stage above (including spinning the mock API server up/down), with
  per-task retries; full successful run captured in `logs/pipeline_run.log`.

## Results and Output Screenshots

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Collaborative filtering (SVD) | 0.00129 | 0.01294 | 0.00922 |
| Content-based (cosine KNN) | **0.01067** | **0.10667** | **0.06470** |

Full discussion in `03_model_performance_report.md`. EDA plots:
`preparation/eda_plots/rating_distribution.png`,
`preparation/eda_plots/item_popularity_longtail.png`,
`preparation/eda_plots/sparsity_heatmap.png`. Data quality findings:
`02_data_quality_report.md`. DVC pipeline DAG and Prefect flow execution log
are reproducible via the commands in the top-level `README.md`.

> _TODO: paste in screenshots of the Feast retrieval output, MLflow UI runs
> list, and the Prefect flow's console output when assembling the final PDF._

## Conclusion and Future Scope

The pipeline demonstrates a complete, reproducible path from raw multi-source
data to a served recommendation model, using the same real-tool stack
(Prefect/MLflow/Feast/DVC) an industry team would run, entirely in local mode.
The clearest finding is that this catalog's extreme sparsity (~95% single-
purchase users) makes pure collaborative filtering weak on its own —
content-based filtering outperforms it by roughly 7-8x on every ranking metric
— which is exactly the kind of insight this pipeline is meant to surface early.

**Future scope**: (1) a hybrid/re-ranking model that blends both signals with
learned weights instead of the current simple interleave; (2) session-based or
sequential models to better exploit clickstream-style signals as those become
available; (3) replacing the mock product-signals API with a real external
sentiment/popularity provider; (4) promoting the local Feast/MLflow/DVC setup
to their hosted/server modes for multi-user collaboration; (5) scheduled,
continuously-running Prefect deployments instead of one-shot manual runs.

## Deliverables

- **Video Walkthrough (5-10 min)**: `[Google Drive link — TODO]`
- **Source code, datasets, trained models, documentation (.zip)**: `[Google Drive link — TODO]`

(Both links to be set to "Anyone with a BITS ID" access before submission.)
