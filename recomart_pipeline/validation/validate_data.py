"""
Data profiling and validation over the most recently ingested raw batch.

Runs pandas-based checks (missing values, duplicate keys, schema conformance,
range/format checks, referential integrity across tables) and writes a
human-readable data quality report to reports/02_data_quality_report.md.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_LAKE_DIR, REPORTS_DIR  # noqa: E402
from ingestion.common import get_logger  # noqa: E402

logger = get_logger("validate_data", "validation.log")

EXPECTED_SCHEMAS = {
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
    "order_reviews": ["review_id", "order_id", "review_score"],
    "order_payments": ["order_id", "payment_sequential", "payment_type", "payment_value"],
    "products": ["product_id", "product_category_name"],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "category_translation": ["product_category_name", "product_category_name_english"],
}

PRIMARY_KEYS = {
    "customers": "customer_id",
    "orders": "order_id",
    "order_reviews": "review_id",
    "products": "product_id",
    "sellers": "seller_id",
}


def _latest_partition(source: str, type_name: str) -> Path:
    type_dir = RAW_LAKE_DIR / source / type_name
    partitions = sorted(p for p in type_dir.iterdir() if p.is_dir())
    if not partitions:
        raise FileNotFoundError(f"No ingested partitions found for {source}/{type_name}")
    return partitions[-1]


def _load_latest_csv(type_name: str) -> pd.DataFrame:
    partition = _latest_partition("olist_csv_batch", type_name)
    csv_files = list(partition.glob("*.csv"))
    return pd.read_csv(csv_files[0])


def load_all_tables() -> dict:
    return {t: _load_latest_csv(t) for t in EXPECTED_SCHEMAS}


def check_schema(name: str, df: pd.DataFrame) -> list:
    issues = []
    expected = EXPECTED_SCHEMAS[name]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing expected columns: {missing_cols}")
    return issues


def check_missing_values(df: pd.DataFrame) -> dict:
    null_counts = df.isnull().sum()
    return {col: int(n) for col, n in null_counts.items() if n > 0}


def check_duplicates(name: str, df: pd.DataFrame) -> int:
    pk = PRIMARY_KEYS.get(name)
    if pk is None or pk not in df.columns:
        return 0
    return int(df.duplicated(subset=[pk]).sum())


def check_range(tables: dict) -> dict:
    issues = {}

    scores = tables["order_reviews"]["review_score"]
    bad_scores = int(((scores < 1) | (scores > 5)).sum())
    issues["review_score_out_of_1_5_range"] = bad_scores

    prices = tables["order_items"]["price"]
    issues["negative_or_zero_price"] = int((prices <= 0).sum())

    freight = tables["order_items"]["freight_value"]
    issues["negative_freight_value"] = int((freight < 0).sum())

    payments = tables["order_payments"]["payment_value"]
    issues["negative_payment_value"] = int((payments < 0).sum())

    return issues


def check_referential_integrity(tables: dict) -> dict:
    issues = {}

    order_ids = set(tables["orders"]["order_id"])
    issues["order_items_orphaned_order_id"] = int(
        (~tables["order_items"]["order_id"].isin(order_ids)).sum()
    )
    issues["order_reviews_orphaned_order_id"] = int(
        (~tables["order_reviews"]["order_id"].isin(order_ids)).sum()
    )
    issues["order_payments_orphaned_order_id"] = int(
        (~tables["order_payments"]["order_id"].isin(order_ids)).sum()
    )

    product_ids = set(tables["products"]["product_id"])
    issues["order_items_orphaned_product_id"] = int(
        (~tables["order_items"]["product_id"].isin(product_ids)).sum()
    )

    customer_ids = set(tables["customers"]["customer_id"])
    issues["orders_orphaned_customer_id"] = int(
        (~tables["orders"]["customer_id"].isin(customer_ids)).sum()
    )

    return issues


def run_validation() -> Path:
    logger.info("Loading latest raw partitions for validation")
    tables = load_all_tables()

    report_lines = [
        "# RecoMart Data Quality Report",
        f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n",
        "## 1. Schema conformance\n",
    ]

    all_ok = True
    for name, df in tables.items():
        issues = check_schema(name, df)
        status = "PASS" if not issues else "FAIL"
        all_ok = all_ok and not issues
        report_lines.append(f"- **{name}** ({len(df)} rows, {len(df.columns)} cols): {status}"
                             + (f" — {issues}" if issues else ""))
        logger.info("Schema check %s: %s", name, status)

    report_lines.append("\n## 2. Missing values (non-zero columns only)\n")
    for name, df in tables.items():
        nulls = check_missing_values(df)
        if nulls:
            report_lines.append(f"- **{name}**: {nulls}")
        else:
            report_lines.append(f"- **{name}**: no missing values")

    report_lines.append("\n## 3. Duplicate primary keys\n")
    for name, df in tables.items():
        dup_count = check_duplicates(name, df)
        report_lines.append(f"- **{name}**: {dup_count} duplicate `{PRIMARY_KEYS.get(name, '')}` rows")
        logger.info("Duplicate check %s: %d duplicates", name, dup_count)

    report_lines.append("\n## 4. Range / format checks\n")
    range_issues = check_range(tables)
    for check, count in range_issues.items():
        report_lines.append(f"- **{check}**: {count} violating rows")
        logger.info("Range check %s: %d violations", check, count)

    report_lines.append("\n## 5. Referential integrity\n")
    ref_issues = check_referential_integrity(tables)
    for check, count in ref_issues.items():
        report_lines.append(f"- **{check}**: {count} orphaned rows")
        logger.info("Referential integrity check %s: %d orphaned rows", check, count)

    total_violations = sum(range_issues.values()) + sum(ref_issues.values()) + sum(
        check_duplicates(n, d) for n, d in tables.items()
    )
    report_lines.append(f"\n## Summary\n\nTotal rows checked: {sum(len(d) for d in tables.values())}. "
                         f"Total violating rows across all checks: {total_violations}. "
                         f"Schema conformance: {'PASS' if all_ok else 'FAIL'}.\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "02_data_quality_report.md"
    report_path.write_text("\n".join(report_lines))

    summary_json = {
        "schema_ok": all_ok,
        "range_issues": range_issues,
        "referential_issues": ref_issues,
        "total_violations": total_violations,
    }
    (REPORTS_DIR / "02_data_quality_summary.json").write_text(json.dumps(summary_json, indent=2))

    logger.info("Validation complete. Report written to %s", report_path)
    return report_path


if __name__ == "__main__":
    run_validation()
