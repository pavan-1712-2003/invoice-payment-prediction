
import pandas as pd
import numpy as np

file_path = "data/invoice_currency_history_features.parquet"

df = pd.read_parquet(file_path)

print("=" * 65)
print("FINAL DATASET AUDIT")
print("=" * 65)

# 1. Dataset size
print("\n1. DATASET SHAPE")
print(df.shape)

# 2. Target distribution
print("\n2. TARGET DISTRIBUTION")
print(df["overdue_status"].value_counts())
print("\nTarget percentages:")
print(
    (df["overdue_status"].value_counts(normalize=True) * 100).round(2)
)

# 3. Missing values
print("\n3. MISSING VALUES")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# 4. Duplicate records
print("\n4. DUPLICATE RECORDS")
print("Duplicate rows:", df.duplicated().sum())

# 5. Engineered feature availability
history_features = [
    "previous_invoice_count",
    "previous_overdue_count",
    "previous_overdue_rate",
    "previous_avg_invoice_amount",
    "days_since_previous_invoice",
    "previous_currency_invoice_count",
    "previous_currency_avg_amount"
]

print("\n5. ENGINEERED HISTORY FEATURES")

for col in history_features:
    if col in df.columns:
        print(f"\n{col}")
        print(df[col].describe().round(3))
    else:
        print(f"\nMISSING FEATURE: {col}")

# 6. Potential leakage columns
print("\n6. POTENTIAL LEAKAGE COLUMNS")

leakage_columns = [
    "payment_status",
    "payment_date",
    "is_overdue"
]

for col in leakage_columns:
    print(col, "PRESENT" if col in df.columns else "NOT PRESENT")

# 7. Target relationship with history features
print("\n7. HISTORY FEATURE CORRELATION WITH TARGET")

for col in history_features:
    if col in df.columns:
        correlation = df[col].corr(df["overdue_status"])
        print(f"{col}: {correlation:.6f}")

# 8. Final feature count
print("\n8. FEATURE COUNT")

excluded = [
    "overdue_status",
    "payment_status",
    "payment_date",
    "is_overdue"
]

features = [col for col in df.columns if col not in excluded]

print("Total columns:", len(df.columns))
print("Potential feature columns:", len(features))
print("\nFeature names:")
print(features)

print("\n" + "=" * 65)
print("FINAL AUDIT COMPLETED")
print("=" * 65)
