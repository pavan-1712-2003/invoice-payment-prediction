
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

print("OVERDUE RATE BY PAYMENT TERMS:")
print(
    df.groupby("payment_terms_days")["overdue_status"]
    .agg(["count", "mean"])
    .round(3)
)

print("\nOVERDUE RATE BY COUNTRY:")
print(
    df.groupby("customer_country")["overdue_status"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
    .round(3)
)

print("\nOVERDUE RATE BY PRODUCT CATEGORY:")
print(
    df.groupby("product_category")["overdue_status"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
    .round(3)
)

print("\nOVERDUE RATE BY INVOICE YEAR:")
df["invoice_year"] = df["invoice_date"].dt.year

print(
    df.groupby("invoice_year")["overdue_status"]
    .agg(["count", "mean"])
    .round(3)
)
