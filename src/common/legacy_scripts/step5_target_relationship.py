
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

# Create invoice amount groups
df["amount_group"] = pd.qcut(
    df["total_amount"],
    q=10,
    duplicates="drop"
)

print("OVERDUE RATE BY INVOICE AMOUNT:")
print(
    df.groupby("amount_group", observed=False)["overdue_status"]
    .agg(["count", "mean"])
    .round(4)
)

# Quantity groups
df["quantity_group"] = pd.qcut(
    df["quantity"],
    q=5,
    duplicates="drop"
)

print("\nOVERDUE RATE BY QUANTITY:")
print(
    df.groupby("quantity_group", observed=False)["overdue_status"]
    .agg(["count", "mean"])
    .round(4)
)

# Correlation with numerical features
print("\nNUMERICAL FEATURE CORRELATION WITH TARGET:")
print(
    df[
        [
            "overdue_status",
            "total_amount",
            "quantity",
            "unit_price",
            "payment_terms_days"
        ]
    ].corr()["overdue_status"].round(4)
)
