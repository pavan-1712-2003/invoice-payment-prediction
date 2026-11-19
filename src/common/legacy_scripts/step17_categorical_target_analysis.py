
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

categorical_columns = [
    "customer_country",
    "customer_state",
    "customer_city",
    "product_category",
    "currency",
    "payment_method",
    "order_channel",
    "payment_terms_days"
]

print("=" * 60)
print("CATEGORICAL TARGET ANALYSIS")
print("=" * 60)

for col in categorical_columns:

    print("\n" + "=" * 50)
    print("FEATURE:", col)

    result = (
        df.groupby(col, observed=False)["overdue_status"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    result["overdue_percent"] = result["mean"] * 100

    print(
        result[["count", "overdue_percent"]]
        .round(3)
        .to_string()
    )

print("\nANALYSIS COMPLETED")
