
import pandas as pd

print("Loading dataset...")

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

# Keep original data unchanged
df["invoice_date"] = pd.to_datetime(df["invoice_date"])

print("Creating customer daily history...")

# Aggregate invoices by customer and invoice date
daily = (
    df.groupby(["customer_id", "invoice_date"])
    .agg(
        day_invoice_count=("invoice_id", "count"),
        day_overdue_count=("overdue_status", "sum"),
        day_total_amount=("total_amount", "sum")
    )
    .reset_index()
    .sort_values(["customer_id", "invoice_date"])
)

grouped = daily.groupby("customer_id")

# Cumulative totals minus current date's invoices
daily["previous_invoice_count"] = (
    grouped["day_invoice_count"].cumsum()
    - daily["day_invoice_count"]
)

daily["previous_overdue_count"] = (
    grouped["day_overdue_count"].cumsum()
    - daily["day_overdue_count"]
)

previous_amount = (
    grouped["day_total_amount"].cumsum()
    - daily["day_total_amount"]
)

# Historical overdue rate
daily["previous_overdue_rate"] = (
    daily["previous_overdue_count"]
    / daily["previous_invoice_count"].replace(0, pd.NA)
).fillna(0)

# Historical average invoice amount
daily["previous_avg_invoice_amount"] = (
    previous_amount
    / daily["previous_invoice_count"].replace(0, pd.NA)
).fillna(0)

# Days since previous invoice date
daily["days_since_previous_invoice"] = (
    grouped["invoice_date"].diff().dt.days
)

# Merge historical features into original invoice records
history_features = [
    "customer_id",
    "invoice_date",
    "previous_invoice_count",
    "previous_overdue_count",
    "previous_overdue_rate",
    "previous_avg_invoice_amount",
    "days_since_previous_invoice"
]

df = df.merge(
    daily[history_features],
    on=["customer_id", "invoice_date"],
    how="left"
)

output_path = "data/invoice_customer_history_features.parquet"

df.to_parquet(output_path, index=False)

print("\nCUSTOMER HISTORY FEATURES CREATED")
print("Dataset shape:", df.shape)

print("\nNew feature summary:")
print(df[history_features[2:]].describe().round(2))

print("\nMissing values:")
print(df[history_features[2:]].isnull().sum())

print("\nSaved:", output_path)
