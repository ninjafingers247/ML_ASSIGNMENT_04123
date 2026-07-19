"""Leave-one-out train/test split for implicit+explicit recommendation evaluation."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES_DIR, RANDOM_SEED  # noqa: E402


def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_DIR / "interactions.parquet")


def leave_one_out_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """
    For every user with >=2 interactions, holds out one rated item as the test
    target; users with a single interaction stay entirely in train (they still
    contribute collaborative signal, but can't be evaluated on a hidden item).
    """
    test_rows = (
        df.groupby("customer_unique_id", group_keys=False)
        .apply(
            lambda g: g.sample(1, random_state=seed) if len(g) >= 2 else g.iloc[0:0],
            include_groups=False,
        )
    )
    test_rows["customer_unique_id"] = df.loc[test_rows.index, "customer_unique_id"]
    train_df = df.drop(test_rows.index)
    test_holdout = test_rows.groupby("customer_unique_id")["product_id"].apply(set).to_dict()
    return train_df.reset_index(drop=True), test_holdout
