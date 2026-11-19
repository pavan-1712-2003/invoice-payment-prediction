
import pandas as pd

df = pd.read_parquet(
    "data/invoice_currency_history_features.parquet"
)

target = "overdue_status"

features = [
    "previous_invoice_count",
    "previous_overdue_count",
    "previous_overdue_rate",
    "previous_avg_invoice_amount",
    "days_since_previous_invoice",
    "previous_currency_invoice_count",
    "previous_currency_avg_amount"
]

print("=" * 60)
print("HISTORICAL FEATURE ANALYSIS")
print("=" * 60)

print("\n1. Correlation with target")

print(
    df[features + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(ascending=False)
    .round(4)
)

print("\n2. Target rate by previous overdue rate")

df["history_group"] = pd.cut(
    df["previous_overdue_rate"],
    bins=[-0.001, 0, 0.10, 0.25, 0.50, 1.0],
    labels=[
        "0%",
        "0-10%",
        "10-25%",
        "25-50%",
        "50-100%"
    ],
    include_lowest=True
)

print(
    df.groupby(
        "history_group",
        observed=False
    )[target]
    .agg(["count", "mean"])
    .round(4)
)

print("\n3. Target rate by previous invoice count")

df["history_count_group"] = pd.cut(
    df["previous_invoice_count"],
    bins=[-1, 0, 2, 5, 10, 30],
    labels=[
        "0",
        "1-2",
        "3-5",
        "6-10",
        "11+"
    ]
)

print(
    df.groupby(
        "history_count_group",
        observed=False
    )[target]
    .agg(["count", "mean"])
    .round(4)
)

print("\nANALYSIS COMPLETED")

