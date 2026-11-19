
import pandas as pd

# Load the original dataset
df = pd.read_parquet(
    "data/invoice_eda_dataset.parquet"
)

print("Original dataset shape:", df.shape)

# Keep only Paid and Overdue invoices
df_model = df[
    df["payment_status"].isin(["Paid", "Overdue"])
].copy()

# Create classification target
df_model["overdue_status"] = (
    df_model["payment_status"] == "Overdue"
).astype(int)

# Check target distribution
print("\nTarget distribution:")
print(df_model["overdue_status"].value_counts())

print("\nTarget percentages:")
print(
    df_model["overdue_status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Save separate working dataset
df_model.to_parquet(
    "data/invoice_classification_working.parquet",
    index=False
)

print("\nNew dataset shape:", df_model.shape)
print("Dataset saved successfully.")
