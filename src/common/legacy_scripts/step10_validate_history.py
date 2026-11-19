
import pandas as pd

df = pd.read_parquet(
    "data/invoice_customer_history_features.parquet"
)

print("=" * 55)
print("CUSTOMER HISTORY VALIDATION")
print("=" * 55)

# Check that counts are non-negative
print("\n1. Negative previous invoice counts:")
print((df["previous_invoice_count"] < 0).sum())

print("\n2. Negative previous overdue counts:")
print((df["previous_overdue_count"] < 0).sum())

# Overdue count cannot exceed previous invoice count
print("\n3. Overdue count greater than invoice count:")
print(
    (
        df["previous_overdue_count"]
        > df["previous_invoice_count"]
    ).sum()
)

# Historical rate must be between 0 and 1
print("\n4. Invalid historical overdue rates:")
print(
    (
        (df["previous_overdue_rate"] < 0)
        | (df["previous_overdue_rate"] > 1)
    ).sum()
)

# First invoice should have zero previous invoices
print("\n5. Records with zero previous invoices:")
first = df[df["previous_invoice_count"] == 0]

print("Records:", len(first))
print(
    "Non-zero previous overdue counts:",
    (first["previous_overdue_count"] != 0).sum()
)

# Same-day records should have identical historical counts
print("\n6. Same-day history consistency:")
same_day = (
    df.groupby(["customer_id", "invoice_date"])
    ["previous_invoice_count"]
    .nunique()
)

print("Groups with inconsistent history:", (same_day > 1).sum())

print("\nVALIDATION COMPLETED")
