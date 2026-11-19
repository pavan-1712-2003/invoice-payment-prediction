
import pandas as pd

df = pd.read_parquet(
    "data/invoice_currency_history_features.parquet"
)

print("=" * 60)
print("PAYMENT LOGIC VALIDATION")
print("=" * 60)

# Compare payment status with payment date
print("\n1. PAYMENT STATUS DISTRIBUTION")
print(df["payment_status"].value_counts())

print("\n2. PAYMENT DATE EQUALS DUE DATE")
print(
    df.groupby("payment_status")
    .apply(
        lambda x: (x["payment_date"] == x["due_date"]).sum(),
        include_groups=False
    )
)

# Calculate actual date difference
df["actual_delay_days"] = (
    df["payment_date"] - df["due_date"]
).dt.days

print("\n3. ACTUAL DELAY BY PAYMENT STATUS")
print(
    df.groupby("payment_status")["actual_delay_days"]
    .agg(["count", "min", "max", "mean"])
    .round(2)
)

print("\n4. POSITIVE DELAY RECORDS BY STATUS")
print(
    df.assign(
        positive_delay=df["actual_delay_days"] > 0
    )
    .groupby("payment_status")["positive_delay"]
    .sum()
)

print("\n5. PAYMENT DATE MISSING VALUES")
print(df["payment_date"].isna().sum())

print("\nPAYMENT LOGIC CHECK COMPLETED")
