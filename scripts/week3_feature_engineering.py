"""
Week 3 - Feature Engineering
Project: Invoice Payment Delay & Cash-Flow Intelligence Platform
Dataset: Orvians Technologies official invoice dataset

Run from the project root:
    py scripts/week3_feature_engineering.py

Important:
- The available target is is_overdue (overdue-status classification).
- The data does NOT support a paid-late regression target: paid invoices are
  never late, and unpaid payment_date values are placeholders.
- Cancelled invoices are excluded from the modelling dataset.
- payment_status, payment_date and is_overdue are never predictors.
- Customer history features are calculated using strictly earlier invoice dates.
- Validation: invoices dated in 2023.
- Final holdout: the original 2024-2025 test dataset.
- Test labels are retained only for final evaluation, never used to build features.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "datasets/processed/cleaning/train_cleaned.parquet"
TEST_PATH = ROOT / "datasets/processed/cleaning/test_cleaned.parquet"
OUT_DIR = ROOT / "datasets/processed/week3"
REPORT_DIR = ROOT / "reports/week3"

RANDOM_STATE = 42
SMOOTHING_PRIOR = 20
GLOBAL_PRIOR_RATE = None


def require_files():
    missing = [str(p) for p in (TRAIN_PATH, TEST_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Week 1 parquet file(s) not found:\n"
            + "\n".join(missing)
            + "\nRun this script from the official project folder and check the paths."
        )


def load_data():
    print("Loading official Week 1 parquet datasets...")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    required = {
        "invoice_id", "invoice_date", "due_date", "customer_id",
        "vendor_id", "product_category", "quantity", "unit_price",
        "discount_percent", "tax_percent", "subtotal", "discount_amount",
        "tax_amount", "shipping_cost", "total_amount", "currency",
        "payment_method", "payment_status", "payment_date",
        "order_channel", "sales_rep", "is_overdue"
    }
    for label, frame in (("train", train), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} dataset is missing columns: {sorted(missing)}")
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
        frame["due_date"] = pd.to_datetime(frame["due_date"], errors="coerce")

    print(f"Raw train: {train.shape}; raw test: {test.shape}")
    return train, test


def make_calendar_features(frame):
    out = frame.copy()
    out["invoice_year"] = out["invoice_date"].dt.year
    out["invoice_month"] = out["invoice_date"].dt.month
    out["invoice_quarter"] = out["invoice_date"].dt.quarter
    out["invoice_day"] = out["invoice_date"].dt.day
    out["invoice_weekday"] = out["invoice_date"].dt.dayofweek
    out["invoice_is_month_start"] = out["invoice_date"].dt.is_month_start.astype("int8")
    out["invoice_is_month_end"] = out["invoice_date"].dt.is_month_end.astype("int8")
    out["payment_term_days"] = (out["due_date"] - out["invoice_date"]).dt.days
    out["log_total_amount"] = np.log1p(out["total_amount"].clip(lower=0))
    out["log_quantity"] = np.log1p(out["quantity"].clip(lower=0))
    out["discount_amount_ratio"] = np.divide(
        out["discount_amount"], out["subtotal"],
        out=np.zeros(len(out), dtype="float64"),
        where=out["subtotal"].to_numpy() != 0
    )
    return out


def add_past_customer_history(train_model, test_model):
    """
    Only count prior customer invoices whose due_date is strictly earlier
    than the current invoice_date. This is more conservative than simply
    using earlier invoice dates because a prior invoice may not yet have
    reached its due date.

    Test history is built from training-period records only. Test labels
    never update the history table.
    """
    global GLOBAL_PRIOR_RATE
    GLOBAL_PRIOR_RATE = float(train_model["is_overdue"].mean())

    history_source = train_model[
        ["customer_id", "due_date", "invoice_id", "is_overdue", "total_amount"]
    ].copy()

    # One row per customer and due date; same-due-date records are not
    # available as history to one another.
    daily = (
        history_source.groupby(["customer_id", "due_date"], observed=True)
        .agg(
            day_invoice_count=("invoice_id", "size"),
            day_overdue_count=("is_overdue", "sum"),
            day_amount_sum=("total_amount", "sum"),
        )
        .reset_index()
        .sort_values(["due_date", "customer_id"])
    )

    grp = daily.groupby("customer_id", sort=False)
    daily["cum_invoice_count"] = grp["day_invoice_count"].cumsum()
    daily["cum_overdue_count"] = grp["day_overdue_count"].cumsum()
    daily["cum_amount_sum"] = grp["day_amount_sum"].cumsum()

    # merge_asof requires both frames sorted globally by the as-of key.
    left_train = train_model[
        ["customer_id", "invoice_date"]
    ].copy()
    left_train["_row_id"] = np.arange(len(left_train))
    left_train = left_train.sort_values(["invoice_date", "customer_id"])

    right = daily[
        ["customer_id", "due_date", "cum_invoice_count",
         "cum_overdue_count", "cum_amount_sum"]
    ].sort_values(["due_date", "customer_id"])

    train_hist = pd.merge_asof(
        left_train,
        right,
        left_on="invoice_date",
        right_on="due_date",
        by="customer_id",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id")

    train_hist["past_invoice_count"] = train_hist["cum_invoice_count"].fillna(0)
    train_hist["past_overdue_count"] = train_hist["cum_overdue_count"].fillna(0)
    train_hist["past_amount_sum"] = train_hist["cum_amount_sum"].fillna(0)
    train_hist["past_avg_invoice_amount"] = np.divide(
        train_hist["past_amount_sum"],
        train_hist["past_invoice_count"],
        out=np.full(len(train_hist), np.nan, dtype="float64"),
        where=train_hist["past_invoice_count"].to_numpy() > 0
    )
    train_hist["past_overdue_rate_smoothed"] = (
        train_hist["past_overdue_count"] + SMOOTHING_PRIOR * GLOBAL_PRIOR_RATE
    ) / (train_hist["past_invoice_count"] + SMOOTHING_PRIOR)

    train_hist["has_past_customer_history"] = (
        train_hist["past_invoice_count"] > 0
    ).astype("int8")

    train_model = train_model.copy()
    history_columns = [
        "past_invoice_count", "past_overdue_count",
        "past_avg_invoice_amount", "past_overdue_rate_smoothed",
        "has_past_customer_history"
    ]
    for col in history_columns:
        train_model[col] = train_hist[col].to_numpy()

    # Apply the same maturity rule to test invoices using training records only.
    left_test = test_model[["customer_id", "invoice_date"]].copy()
    left_test["_row_id"] = np.arange(len(left_test))
    left_test = left_test.sort_values(["invoice_date", "customer_id"])

    test_hist = pd.merge_asof(
        left_test,
        right,
        left_on="invoice_date",
        right_on="due_date",
        by="customer_id",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id")

    test_hist["past_invoice_count"] = test_hist["cum_invoice_count"].fillna(0)
    test_hist["past_overdue_count"] = test_hist["cum_overdue_count"].fillna(0)
    test_hist["past_amount_sum"] = test_hist["cum_amount_sum"].fillna(0)
    test_hist["past_avg_invoice_amount"] = np.divide(
        test_hist["past_amount_sum"],
        test_hist["past_invoice_count"],
        out=np.full(len(test_hist), np.nan, dtype="float64"),
        where=test_hist["past_invoice_count"].to_numpy() > 0
    )
    test_hist["past_overdue_rate_smoothed"] = (
        test_hist["past_overdue_count"] + SMOOTHING_PRIOR * GLOBAL_PRIOR_RATE
    ) / (test_hist["past_invoice_count"] + SMOOTHING_PRIOR)
    test_hist["has_past_customer_history"] = (
        test_hist["past_invoice_count"] > 0
    ).astype("int8")

    test_model = test_model.copy()
    for col in history_columns:
        test_model[col] = test_hist[col].to_numpy()

    # Neutral fallback for customers with no matured historical invoices.
    fallback_amount = float(train_model["total_amount"].mean())
    for frame in (train_model, test_model):
        frame["past_invoice_count"] = frame["past_invoice_count"].fillna(0)
        frame["past_overdue_count"] = frame["past_overdue_count"].fillna(0)
        frame["past_avg_invoice_amount"] = frame["past_avg_invoice_amount"].fillna(
            fallback_amount
        )
        frame["past_overdue_rate_smoothed"] = (
            frame["past_overdue_rate_smoothed"].fillna(GLOBAL_PRIOR_RATE)
        )
        frame["has_past_customer_history"] = (
            frame["has_past_customer_history"].fillna(0).astype("int8")
        )

    return train_model, test_model


def prepare_frames(train, test):
    # Cancelled records are not meaningful examples for overdue-status modelling.
    train_model = train.loc[train["payment_status"] != "Cancelled"].copy()
    test_model = test.loc[test["payment_status"] != "Cancelled"].copy()

    # Target is already provided; preserve it as the label only.
    train_model["target_is_overdue"] = train_model["is_overdue"].astype("int8")
    test_model["target_is_overdue"] = test_model["is_overdue"].astype("int8")

    train_model = make_calendar_features(train_model)
    test_model = make_calendar_features(test_model)

    train_model, test_model = add_past_customer_history(train_model, test_model)
    return train_model, test_model


def get_feature_columns(frame):
    # Deliberately exclude IDs, direct outcome fields, and raw date columns.
    # customer_id is retained as a categorical predictor; high-cardinality
    # encoding/model choice is handled in the next modelling stage.
    categorical = [
        "customer_id", "vendor_id", "product_category", "currency",
        "payment_method", "order_channel", "sales_rep",
        "customer_country", "customer_state", "customer_city"
    ]
    numeric = [
        "quantity", "unit_price", "discount_percent", "tax_percent",
        "subtotal", "discount_amount", "tax_amount", "shipping_cost",
        "total_amount", "invoice_year", "invoice_month", "invoice_quarter",
        "invoice_day", "invoice_weekday", "invoice_is_month_start",
        "invoice_is_month_end", "payment_term_days", "log_total_amount",
        "log_quantity", "discount_amount_ratio", "past_invoice_count",
        "past_overdue_count", "past_avg_invoice_amount",
        "past_overdue_rate_smoothed", "has_past_customer_history"
    ]
    categorical = [c for c in categorical if c in frame.columns]
    numeric = [c for c in numeric if c in frame.columns]
    return categorical, numeric


def validate_no_leakage(train_model, test_model):
    forbidden = {
        "payment_status", "payment_date", "is_overdue",
        "target_is_overdue", "invoice_id", "invoice_number"
    }
    cat, num = get_feature_columns(train_model)
    features = set(cat + num)
    bad = features & forbidden
    if bad:
        raise AssertionError(f"Leakage columns found in predictors: {sorted(bad)}")

    if train_model["invoice_date"].max() >= test_model["invoice_date"].min():
        raise AssertionError("Expected test dates to be strictly later than train dates.")

    if train_model["invoice_date"].isna().any() or test_model["invoice_date"].isna().any():
        raise AssertionError("Missing invoice_date found after Week 1 cleaning.")

    if train_model["target_is_overdue"].nunique() != 2:
        raise AssertionError("Training target does not contain both classes.")

    # Check same-day records all share identical past count/rate values.
    # (The daily aggregation and merge should guarantee this.)
    dup = train_model.groupby(
        ["customer_id", "invoice_date"], observed=True
    )["past_invoice_count"].nunique().max()
    if dup > 1:
        raise AssertionError("Same-day customer records have inconsistent history.")

    return cat, num


def write_outputs(train_model, test_model, categorical, numeric):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Chronological split within training data: 2019-2022 train, 2023 validation.
    fit_mask = train_model["invoice_date"] < pd.Timestamp("2023-01-01")
    val_mask = (
        (train_model["invoice_date"] >= pd.Timestamp("2023-01-01"))
        & (train_model["invoice_date"] < pd.Timestamp("2024-01-01"))
    )
    fit = train_model.loc[fit_mask].copy()
    val = train_model.loc[val_mask].copy()

    if fit.empty or val.empty:
        raise ValueError("Chronological split produced an empty train or validation set.")

    # Model-ready data: identifiers/dates retained for traceability, but the
    # explicit predictor list is written separately to prevent accidental use.
    fit.to_parquet(OUT_DIR / "train_features.parquet", index=False)
    val.to_parquet(OUT_DIR / "validation_features.parquet", index=False)
    test_model.to_parquet(OUT_DIR / "test_features.parquet", index=False)

    feature_spec = {
        "target": "target_is_overdue",
        "task": "binary classification (overdue-status prediction)",
        "categorical_features": categorical,
        "numeric_features": numeric,
        "all_predictors": categorical + numeric,
        "excluded_predictors": [
            "payment_status", "payment_date", "is_overdue",
            "target_is_overdue", "invoice_id", "invoice_number"
        ],
        "validation_period": "2023-01-01 through 2023-12-31",
        "final_test_period": "2024-01-01 through 2025-12-31",
        "history_rule": (
            "Training features use only earlier invoice dates; same-day records "
            "are excluded. Test customer history uses labelled training-period "
            "records only; test labels never update history."
        ),
        "regression_target": None,
        "regression_target_note": (
            "No valid paid-late regression target can be constructed from this "
            "dataset because paid invoices have no positive delay and unpaid "
            "payment_date values are placeholders."
        )
    }
    with open(OUT_DIR / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2)

    split_summary = pd.DataFrame([
        {
            "split": "train",
            "rows": len(fit),
            "start_date": str(fit["invoice_date"].min().date()),
            "end_date": str(fit["invoice_date"].max().date()),
            "overdue_rate_percent": round(100 * fit["target_is_overdue"].mean(), 4),
        },
        {
            "split": "validation",
            "rows": len(val),
            "start_date": str(val["invoice_date"].min().date()),
            "end_date": str(val["invoice_date"].max().date()),
            "overdue_rate_percent": round(100 * val["target_is_overdue"].mean(), 4),
        },
        {
            "split": "final_test",
            "rows": len(test_model),
            "start_date": str(test_model["invoice_date"].min().date()),
            "end_date": str(test_model["invoice_date"].max().date()),
            "overdue_rate_percent": round(100 * test_model["target_is_overdue"].mean(), 4),
        },
    ])
    split_summary.to_csv(REPORT_DIR / "week3_split_summary.csv", index=False)

    # Dataset-level feature missingness report.
    missing = pd.DataFrame({
        "feature": categorical + numeric,
        "train_missing": [
            int(fit[c].isna().sum()) for c in categorical + numeric
        ],
        "validation_missing": [
            int(val[c].isna().sum()) for c in categorical + numeric
        ],
        "test_missing": [
            int(test_model[c].isna().sum()) for c in categorical + numeric
        ],
    })
    missing.to_csv(REPORT_DIR / "week3_feature_missingness.csv", index=False)

    # A concise reproducibility report.
    report = [
        "WEEK 3 FEATURE ENGINEERING REPORT",
        "=================================",
        f"Train modelling rows after excluding Cancelled: {len(train_model):,}",
        f"Test modelling rows after excluding Cancelled: {len(test_model):,}",
        f"Training split rows (2019-2022): {len(fit):,}",
        f"Validation split rows (2023): {len(val):,}",
        f"Final test rows (2024-2025): {len(test_model):,}",
        f"Global training overdue rate: {100 * train_model['target_is_overdue'].mean():.3f}%",
        f"Categorical predictors ({len(categorical)}): {', '.join(categorical)}",
        f"Numeric predictors ({len(numeric)}): {', '.join(numeric)}",
        "",
        "LEAKAGE CONTROLS",
        "- Excluded payment_status and payment_date from predictors.",
        "- Excluded is_overdue and target_is_overdue from predictors.",
        "- Customer history uses only prior invoices with due_date earlier than the current invoice_date.",
        "- Invoices not yet due are excluded from historical customer aggregates.",
        "- Test history uses training labels only; test labels do not update history.",
        "- Original 2024-2025 test period remains held out.",
        "",
        "TARGET LIMITATION",
        "Only overdue-status classification is supported.",
        "A valid paid-late regression target is not supported by the supplied labels.",
        "Do not describe this task as predicting actual payment delay in days.",
        "",
        "OUTPUTS",
        "- datasets/processed/week3/train_features.parquet",
        "- datasets/processed/week3/validation_features.parquet",
        "- datasets/processed/week3/test_features.parquet",
        "- datasets/processed/week3/feature_spec.json",
        "- reports/week3/week3_split_summary.csv",
        "- reports/week3/week3_feature_missingness.csv",
    ]
    (REPORT_DIR / "week3_feature_engineering_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nWEEK 3 COMPLETE")
    print(split_summary.to_string(index=False))
    print(f"\nCategorical predictors: {len(categorical)}")
    print(f"Numeric predictors: {len(numeric)}")
    print(f"\nOutputs saved under: {OUT_DIR}")
    print(f"Reports saved under: {REPORT_DIR}")


def main():
    require_files()
    train, test = load_data()
    train_model, test_model = prepare_frames(train, test)
    categorical, numeric = validate_no_leakage(train_model, test_model)
    write_outputs(train_model, test_model, categorical, numeric)


if __name__ == "__main__":
    main()
