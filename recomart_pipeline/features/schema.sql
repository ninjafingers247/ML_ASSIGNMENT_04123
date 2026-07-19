-- RecoMart feature warehouse schema (SQLite)
-- Populated by features/build_features.py from the prepared interactions dataset
-- and the near-real-time product-signals API ingestion.

CREATE TABLE IF NOT EXISTS interactions (
    customer_unique_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    review_score REAL NOT NULL,
    price_norm REAL,
    freight_value_norm REAL,
    timestamp_norm REAL,
    event_timestamp TEXT NOT NULL,
    PRIMARY KEY (customer_unique_id, product_id)
);

CREATE TABLE IF NOT EXISTS user_features (
    customer_unique_id TEXT PRIMARY KEY,
    user_interaction_count INTEGER NOT NULL,
    user_avg_rating REAL NOT NULL,
    user_avg_spend REAL NOT NULL,
    event_timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_features (
    product_id TEXT PRIMARY KEY,
    item_interaction_count INTEGER NOT NULL,
    item_avg_rating REAL NOT NULL,
    category_encoded INTEGER,
    product_category_name_english TEXT,
    api_popularity_score REAL,
    api_sentiment_score REAL,
    api_avg_price REAL,
    event_timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_cooccurrence (
    product_id_a TEXT NOT NULL,
    product_id_b TEXT NOT NULL,
    cooccurrence_count INTEGER NOT NULL,
    PRIMARY KEY (product_id_a, product_id_b)
);
