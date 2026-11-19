
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

df["invoice_date"] = pd.to_datetime(df["invoice_date"])

df["invoice_year"] = df["invoice_date"].dt.year
df["invoice_month"] = df["invoice_date"].dt.month

print("=" * 60)
print("TEMPORAL TARGET ANALYSIS")
print("=" * 60)

print("\n1. Overdue rate by year")

print(
    df.groupby("invoice_year")["overdue_status"]
    .agg(["count", "mean"])
    .assign(
        overdue_percent=lambda x: x["mean"] * 100
    )
    .round(3)
)

print("\n2. Overdue rate by year and month")

monthly = (
    df.groupby(
        ["invoice_year", "invoice_month"]
    )["overdue_status"]
    .agg(["count", "mean"])
)

monthly["overdue_percent"] = monthly["mean"] * 100

print(monthly.round(3).to_string())

print("\n3. Overall overdue rate")
print(round(df["overdue_status"].mean() * 100, 3), "%")

print("\nANALYSIS COMPLETED")
