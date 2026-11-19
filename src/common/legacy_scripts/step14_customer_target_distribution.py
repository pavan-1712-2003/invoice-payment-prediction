
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

print("=" * 60)
print("CUSTOMER TARGET DISTRIBUTION ANALYSIS")
print("=" * 60)

customer_stats = (
    df.groupby("customer_id")
    .agg(
        invoice_count=("overdue_status", "count"),
        overdue_count=("overdue_status", "sum"),
        overdue_rate=("overdue_status", "mean")
    )
)

print("\n1. Customer overdue-rate distribution")
print(
    customer_stats["overdue_rate"]
    .describe()
    .round(4)
)

print("\n2. Customers grouped by overdue rate")

customer_stats["rate_group"] = pd.cut(
    customer_stats["overdue_rate"],
    bins=[-0.001, 0, 0.10, 0.20, 0.30, 0.50, 1.0],
    labels=[
        "0%",
        "0-10%",
        "10-20%",
        "20-30%",
        "30-50%",
        "50-100%"
    ],
    include_lowest=True
)

print(
    customer_stats["rate_group"]
    .value_counts()
    .sort_index()
)

print("\n3. Customer invoice-count distribution")
print(
    customer_stats["invoice_count"]
    .describe()
    .round(2)
)

print("\n4. Customers with highest overdue rates")
print(
    customer_stats.sort_values(
        "overdue_rate",
        ascending=False
    ).head(15).to_string()
)

print("\n5. Overall overdue rate")
print(round(df["overdue_status"].mean() * 100, 2), "%")

print("\nANALYSIS COMPLETED")
