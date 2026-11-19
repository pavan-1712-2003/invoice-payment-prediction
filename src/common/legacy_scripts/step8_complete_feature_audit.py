
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

target = "overdue_status"

# Features available in the working dataset
exclude = [
    "invoice_id", "invoice_number",
    "customer_name", "customer_email",
    "vendor_id", "vendor_name",
    "product_id", "product_name",
    "sales_rep", "payment_status",
    "payment_date", "is_overdue",
    target
]

X = df.drop(columns=exclude, errors="ignore")

print("=" * 60)
print("COMPLETE FEATURE QUALITY AUDIT")
print("=" * 60)

print("\n1. DATASET SHAPE")
print("Rows:", len(df))
print("Features:", X.shape[1])

print("\n2. MISSING VALUES")
missing = X.isnull().sum()
print(missing[missing > 0] if missing.sum() else "No missing values")

print("\n3. DUPLICATE ROWS")
print("Duplicates:", df.duplicated().sum())

print("\n4. NUMERICAL FEATURES")
numeric = X.select_dtypes(include="number")
print(numeric.describe().T.to_string())

print("\n5. CATEGORICAL FEATURES")
categorical = X.select_dtypes(include=["object", "category"])
for col in categorical.columns:
    print(
        col,
        "| Unique:", categorical[col].nunique(),
        "| Missing:", categorical[col].isnull().sum()
    )

print("\n6. CONSTANT FEATURES")
constant = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]
print(constant if constant else "No constant features")

print("\n7. DATE FEATURES")
for col in X.select_dtypes(include=["datetime64[ns]"]).columns:
    print(col, "|", X[col].min(), "to", X[col].max())

print("\n8. TARGET DISTRIBUTION")
print(df[target].value_counts())
print(df[target].value_counts(normalize=True).mul(100).round(2))

print("\n9. NUMERICAL CORRELATION WITH TARGET")
print(
    numeric.join(df[target])
    .corr(numeric_only=True)[target]
    .sort_values(ascending=False)
    .round(4)
)

print("\n10. CUSTOMER HISTORY AVAILABILITY")
history = (
    df.sort_values(["customer_id", "invoice_date"])
      .groupby("customer_id")
      .cumcount()
)
print("No previous invoices:", (history == 0).sum())
print("At least 3 previous:", (history >= 3).sum())
print("At least 5 previous:", (history >= 5).sum())

print("\nAUDIT COMPLETED")
