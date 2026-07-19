# RecoMart End-to-End Data Management Pipeline

Assignment I ("Data Management for Machine Learning", BITS Pilani S2-25):
an end-to-end data pipeline for a fictional e-commerce recommender, RecoMart,
covering ingestion, storage, validation, preparation, feature engineering, a
feature store, versioning, model training, and orchestration.

Built against the real **Olist Brazilian E-Commerce** dataset (customers,
orders, order items, reviews with genuine 1-5 ratings, products, sellers,
payments) — see `reports/01_problem_formulation.md` for why, and how it maps
onto RecoMart's users/items/ratings/transactions.

## Setup

```bash
cd recomart_pipeline
python3 -m venv .venv-linux
.venv-linux/bin/pip install -r requirements.txt
```

## Visual demo (Streamlit)

```bash
.venv-linux/bin/streamlit run app.py
```

Pick a customer from the sidebar and see their purchase history alongside three
tabs: the hybrid recommendation actually served by `models/infer.py`, the raw
collaborative-filtering ranking (with predicted-rating scores), and the raw
content-based ranking (with similarity scores and "because you liked X"
explanations). Requires `models/train_collaborative.py` and
`models/train_content_based.py` to have been run at least once (they save the
`.joblib` model files this app loads).

## Run everything (orchestrated)

```bash
.venv-linux/bin/python orchestration/prefect_flow.py
```

This starts the local mock product-signals API, runs batch CSV ingestion,
near-real-time API ingestion, validation, preparation/EDA, feature
engineering, Feast apply+materialize, and both model trainings — end to end,
with retries and logging — then tears the mock API back down. Full run log:
`logs/pipeline_run.log`.

## Run stages individually

```bash
# 1. Ingestion
.venv-linux/bin/python ingestion/mock_api_server.py &        # keep running in background
.venv-linux/bin/python ingestion/ingest_csv_sources.py
.venv-linux/bin/python ingestion/ingest_api_source.py --sample-size 300

# 2. Validation
.venv-linux/bin/python validation/validate_data.py

# 3. Preparation / EDA
.venv-linux/bin/python preparation/clean_and_eda.py

# 4. Feature engineering
.venv-linux/bin/python features/build_features.py

# 5. Feature store
cd feature_store/feature_repo
../../.venv-linux/bin/feast apply
../../.venv-linux/bin/feast materialize 2016-01-01T00:00:00 2030-01-01T00:00:00
cd ../..
.venv-linux/bin/python feature_store/demo_retrieval.py

# 6. Model training + evaluation
.venv-linux/bin/python models/train_collaborative.py
.venv-linux/bin/python models/train_content_based.py
.venv-linux/bin/python models/infer.py <customer_unique_id> --k 10 --strategy hybrid

# 7. DVC pipeline (equivalent to steps 1-4 above, with lineage tracking)
.venv-linux/bin/dvc repro
.venv-linux/bin/dvc dag

# View MLflow runs
.venv-linux/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Raw data lake layout (Task 3: storage structure)

```
data/raw/<source>/<type>/<ingestion_timestamp>/<file>
  olist_csv_batch/customers/20260719_031941/olist_customers_dataset.csv
  olist_csv_batch/orders/20260719_031941/olist_orders_dataset.csv
  ...
  olist_csv_batch/_manifests/manifest_20260719_031941.json
  product_signals_api/products/20260719_032016/product_signals.jsonl
  product_signals_api/_manifests/manifest_20260719_032016.json
```

Partitioned by **source** (which ingestion pipeline produced it), **type**
(logical table/entity), and **ingestion timestamp** (one immutable partition
per run) — so every run is independently auditable and nothing is overwritten.
A JSON manifest per run records per-file row counts and sha256 checksums.

## Project layout

```
recomart_pipeline/
├── app.py                   # Streamlit visual demo (pick a customer, see recommendations)
├── source_data/            # upstream vendor CSV drop (DVC-tracked)
├── data/raw/                # partitioned raw lake (DVC-tracked via dvc.yaml)
├── data/processed/          # cleaned/prepared interactions (DVC-tracked)
├── data/features/           # engineered feature tables, feeds Feast (DVC-tracked)
├── ingestion/                # batch CSV + mock REST API ingestion, retry/logging
├── validation/                # data quality checks -> reports/02_data_quality_report.md
├── preparation/               # cleaning, encoding, normalization, EDA plots
├── features/                   # SQL schema, feature engineering, feature dictionary
├── feature_store/feature_repo/ # Feast entities/feature views, local registry+online store
├── versioning/                 # DVC workflow documentation
├── models/                     # collaborative + content-based training, evaluation, inference
├── orchestration/               # Prefect flow chaining every stage
├── reports/                     # problem formulation, data quality, model performance, final report
└── logs/                        # ingestion/validation/preparation/features/training/pipeline logs
```

## Tooling (all local/offline, no cloud accounts needed)

- **Prefect** — orchestration
- **MLflow** (sqlite backend) — experiment tracking
- **Feast** (local provider, file offline store, sqlite online store) — feature store
- **DVC** (local-directory remote) — data versioning and lineage
