"""
Feast feature definitions for RecoMart: user and item entities, backed by the
parquet feature tables produced in features/build_features.py, so training and
inference can both retrieve versioned, point-in-time-correct features.
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

from config_paths import USER_FEATURES_PATH, ITEM_FEATURES_PATH, INTERACTIONS_PATH

customer = Entity(name="customer_unique_id", join_keys=["customer_unique_id"])
product = Entity(name="product_id", join_keys=["product_id"])

user_features_source = FileSource(
    name="user_features_source",
    path=str(USER_FEATURES_PATH),
    timestamp_field="event_timestamp",
)

item_features_source = FileSource(
    name="item_features_source",
    path=str(ITEM_FEATURES_PATH),
    timestamp_field="event_timestamp",
)

interactions_source = FileSource(
    name="interactions_source",
    path=str(INTERACTIONS_PATH),
    timestamp_field="event_timestamp",
)

user_features_view = FeatureView(
    name="user_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="user_interaction_count", dtype=Int64),
        Field(name="user_avg_rating", dtype=Float32),
        Field(name="user_avg_spend", dtype=Float32),
    ],
    online=True,
    source=user_features_source,
)

item_features_view = FeatureView(
    name="item_features",
    entities=[product],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="item_interaction_count", dtype=Int64),
        Field(name="item_avg_rating", dtype=Float32),
        Field(name="category_encoded", dtype=Int64),
        Field(name="product_category_name_english", dtype=String),
        Field(name="api_popularity_score", dtype=Float32),
        Field(name="api_sentiment_score", dtype=Float32),
        Field(name="api_avg_price", dtype=Float32),
    ],
    online=True,
    source=item_features_source,
)

interactions_view = FeatureView(
    name="interactions",
    entities=[customer, product],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="review_score", dtype=Float32),
        Field(name="price_norm", dtype=Float32),
        Field(name="freight_value_norm", dtype=Float32),
        Field(name="timestamp_norm", dtype=Float32),
    ],
    online=True,
    source=interactions_source,
)
