
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

# Sort invoices chronologically
df = df.sort_values(
    ["customer_id", "invoice_date"]
)

# Count earlier invoices for each customer
df["previous_invoice_count"] = (
    df.groupby("customer_id").cumcount()
)

print("PREVIOUS INVOICE COUNT DISTRIBUTION:")
print(
    df["previous_invoice_count"].describe()
)

print("\nINVOICES WITH NO PREVIOUS CUSTOMER HISTORY:")
print(
    (df["previous_invoice_count"] == 0).sum()
)

print("\nINVOICES WITH AT LEAST 3 PREVIOUS INVOICES:")
print(
    (df["previous_invoice_count"] >= 3).sum()
)

print("\nINVOICES WITH AT LEAST 5 PREVIOUS INVOICES:")
print(
    (df["previous_invoice_count"] >= 5).sum()
)
