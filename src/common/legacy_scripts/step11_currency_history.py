
import pandas as pd

print("Loading dataset...")

df = pd.read_parquet(
    "data/invoice_customer_history_features.parquet"
)

df["invoice_date"] = pd.to_datetime(df["invoice_date"])

print("Calculating currency-aware customer history...")

daily = (
    df.groupby(
        ["customer_id", "currency", "invoice_date"]
    )
    .agg(
        daily_count=("invoice_id", "count"),
        daily_amount=("total_amount", "sum")
    )
    .reset_index()
    .sort_values(
        ["customer_id", "currency", "invoice_date"]
    )
)

grouped = daily.groupby(
    ["customer_id", "currency"]
)

daily["previous_currency_invoice_count"] = (
    grouped["daily_count"].cumsum()
    - daily["daily_count"]
)

previous_amount = (
    grouped["daily_amount"].cumsum()
    - daily["daily_amount"]
)

daily["previous_currency_avg_amount"] = (
    previous_amount
    / daily["previous_currency_invoice_count"].replace(0, pd.NA)
).fillna(0).astype(float)

features = [
    "customer_id",
    "currency",
    "invoice_date",
    "previous_currency_invoice_count",
    "previous_currency_avg_amount"
]

df = df.merge(
    daily[features],
    on=["customer_id", "currency", "invoice_date"],
    how="left"
)

output = "data/invoice_currency_history_features.parquet"

df.to_parquet(output, index=False)

print("\nCURRENCY HISTORY FEATURES CREATED")
print("Dataset shape:", df.shape)

print("\nNew feature summary:")
print(
    df[
        [
            "previous_currency_invoice_count",
            "previous_currency_avg_amount"
        ]
    ].describe().round(2)
)

print("\nSaved:", output)
