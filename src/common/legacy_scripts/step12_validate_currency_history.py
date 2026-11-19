
import pandas as pd

df = pd.read_parquet(
    "data/invoice_currency_history_features.parquet"
)

print("=" * 55)
print("CURRENCY HISTORY VALIDATION")
print("=" * 55)

print("\n1. Negative previous currency invoice counts:")
print(
    (df["previous_currency_invoice_count"] < 0).sum()
)

print("\n2. Negative previous currency average amounts:")
print(
    (df["previous_currency_avg_amount"] < 0).sum()
)

print("\n3. Records with zero previous currency invoices:")
first = df[
    df["previous_currency_invoice_count"] == 0
]

print("Records:", len(first))
print(
    "Non-zero previous averages:",
    (first["previous_currency_avg_amount"] != 0).sum()
)

print("\n4. Same-day history consistency:")

same_day = (
    df.groupby(
        ["customer_id", "currency", "invoice_date"]
    )["previous_currency_invoice_count"]
    .nunique()
)

print(
    "Groups with inconsistent history:",
    (same_day > 1).sum()
)

print("\n5. Currency distribution:")
print(df["currency"].value_counts())

print("\nVALIDATION COMPLETED")
