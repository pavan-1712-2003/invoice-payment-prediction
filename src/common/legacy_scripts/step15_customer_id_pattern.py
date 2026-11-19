
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

# Extract numeric portion of customer ID
df["customer_number"] = (
    df["customer_id"]
    .str.extract(r"(\d+)")
    .astype(int)
)

# Divide customer IDs into 10 groups
df["customer_id_group"] = pd.qcut(
    df["customer_number"],
    q=10,
    duplicates="drop"
)

print("=" * 60)
print("CUSTOMER ID TARGET PATTERN")
print("=" * 60)

print("\n1. Overdue rate by customer ID group")

print(
    df.groupby(
        "customer_id_group",
        observed=False
    )["overdue_status"]
    .agg(["count", "mean"])
    .assign(
        overdue_percent=lambda x: x["mean"] * 100
    )
    .round(3)
)

print("\n2. Correlation between customer number and target")

print(
    round(
        df["customer_number"].corr(
            df["overdue_status"]
        ),
        5
    )
)

print("\nANALYSIS COMPLETED")
