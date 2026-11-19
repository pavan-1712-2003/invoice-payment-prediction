
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

columns = [
    "quantity",
    "unit_price",
    "discount_percent",
    "tax_percent",
    "subtotal",
    "discount_amount",
    "tax_amount",
    "shipping_cost",
    "total_amount",
    "payment_terms_days"
]

print("=" * 60)
print("NUMERICAL FEATURE VS TARGET ANALYSIS")
print("=" * 60)

for col in columns:
    result = (
        df.groupby(
            pd.qcut(df[col], q=10, duplicates="drop")
        )["overdue_status"]
        .agg(["count", "mean"])
    )

    result["overdue_percent"] = result["mean"] * 100

    print(f"\nFEATURE: {col}")
    print(
        result[["count", "overdue_percent"]]
        .round(3)
        .to_string()
    )

print("\nANALYSIS COMPLETED")
