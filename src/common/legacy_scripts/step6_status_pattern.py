
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

print("OVERDUE RATE BY CUSTOMER")

customer_stats = (
    df.groupby("customer_id")["overdue_status"]
    .agg(["count", "sum", "mean"])
)

print(customer_stats.describe().round(4))

print("\nCUSTOMERS WITH HIGHEST OVERDUE RATE:")
print(
    customer_stats.sort_values(
        "mean", ascending=False
    ).head(10).round(4)
)

print("\nOVERDUE RATE BY INVOICE MONTH:")
print(
    df.groupby("invoice_month")["overdue_status"]
    .agg(["count", "mean"])
    .round(4)
)

print("\nOVERDUE RATE BY INVOICE WEEKDAY:")
print(
    df.groupby("invoice_weekday")["overdue_status"]
    .agg(["count", "mean"])
    .round(4)
)
