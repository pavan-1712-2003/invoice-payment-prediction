
import pandas as pd
import gc
from pathlib import Path

# Project paths
BASE_DIR = Path(r"C:\invioce_data")
DATA_DIR = BASE_DIR / "datasets" / "week3" / "engineered"
OUTPUT_DIR = BASE_DIR / "datasets" / "week3" / "feature_selection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "training": DATA_DIR / "training_engineered.parquet",
    "validation": DATA_DIR / "validation_engineered.parquet",
    "test": DATA_DIR / "test_engineered.parquet"
}

TARGET = "is_overdue"

# Columns unavailable or inappropriate as model inputs
EXCLUDED_COLUMNS = [
    "invoice_id",
    "invoice_number",
    "customer_email",
    "customer_name",
    "payment_status",
    "payment_date",
    "is_overdue"
]

print("=" * 55)
print("WEEK 3 - FEATURE SELECTION")
print("=" * 55)

# Read training data to identify feature columns
df = pd.read_parquet(FILES["training"])

if TARGET not in df.columns:
    raise ValueError("Target column is missing.")

feature_columns = [
    col for col in df.columns
    if col not in EXCLUDED_COLUMNS
]

# Verify target distribution
print("\n1. TARGET DISTRIBUTION")
print(df[TARGET].value_counts())
print("\nTarget percentages:")
print((df[TARGET].value_counts(normalize=True) * 100).round(2))

# Verify excluded columns
print("\n2. EXCLUDED COLUMNS")
print(EXCLUDED_COLUMNS)

# Verify selected features
print("\n3. SELECTED FEATURES")
print("Total selected features:", len(feature_columns))

for col in feature_columns:
    print(col, ":", df[col].dtype)

# Verify no leakage columns remain
assert TARGET not in feature_columns
assert "payment_status" not in feature_columns
assert "payment_date" not in feature_columns
assert "invoice_id" not in feature_columns
assert "invoice_number" not in feature_columns

print("\n4. LEAKAGE CHECK")
print("Target excluded from inputs: PASS")
print("Payment status excluded: PASS")
print("Payment date excluded: PASS")
print("Invoice identifiers excluded: PASS")

del df
gc.collect()

# Check all three datasets and save feature lists
for name, file_path in FILES.items():

    df = pd.read_parquet(file_path)

    missing_features = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name}: Missing features {missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(f"{name}: Target column missing")

    # Save selected features and target separately
    selected_df = df[feature_columns + [TARGET]]

    output_file = OUTPUT_DIR / f"{name}_selected.parquet"
    selected_df.to_parquet(output_file, index=False)

    print(f"\n{name.upper()} DATASET")
    print("Original shape:", df.shape)
    print("Selected shape:", selected_df.shape)
    print("Saved:", output_file)

    del selected_df
    del df
    gc.collect()

# Save feature list
feature_list = pd.DataFrame({
    "feature_name": feature_columns
})

feature_list.to_csv(
    OUTPUT_DIR / "selected_features.csv",
    index=False
)

print("\n" + "=" * 55)
print("FEATURE SELECTION COMPLETED")
print("=" * 55)
print("Selected features:", len(feature_columns))
print("Target:", TARGET)
print("Output folder:", OUTPUT_DIR)
