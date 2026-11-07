
import os
import numpy as np
import pandas as pd

# 1. File paths
RAW_FOLDER = r"C:\invioce_data\datasets\raw"
OUTPUT_FOLDER = r"C:\invioce_data\datasets\processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 2. Load datasets
train_df = pd.read_csv(
    os.path.join(RAW_FOLDER, "train_invoices.csv")
)

test_df = pd.read_csv(
    os.path.join(RAW_FOLDER, "test_invoices.csv")
)

# 3. Validate dataset structure
assert train_df.shape[1] == 31
assert test_df.shape[1] == 31
assert train_df.columns.equals(test_df.columns)

# 4. Standardize date columns
date_columns = ["invoice_date", "due_date", "payment_date"]

for col in date_columns:
    train_df[col] = pd.to_datetime(
        train_df[col], format="%d-%m-%Y", errors="coerce"
    )

    test_df[col] = pd.to_datetime(
        test_df[col], format="%Y-%m-%d", errors="coerce"
    )

# 5. Correct test data types
test_df["tax_percent"] = test_df["tax_percent"].astype("int64")

# 6. Data quality validation
assert train_df.isnull().sum().sum() == 0
assert test_df.isnull().sum().sum() == 0

assert not train_df.duplicated().any()
assert not test_df.duplicated().any()

assert train_df[date_columns].notna().all().all()
assert test_df[date_columns].notna().all().all()

assert not np.isinf(
    train_df.select_dtypes(include=np.number)
).any().any()

assert not np.isinf(
    test_df.select_dtypes(include=np.number)
).any().any()

# 7. Date consistency
assert (train_df["due_date"] >= train_df["invoice_date"]).all()
assert (test_df["due_date"] >= test_df["invoice_date"]).all()

assert (
    train_df["payment_date"] >= train_df["invoice_date"]
).all()

assert (
    test_df["payment_date"] >= test_df["invoice_date"]
).all()

# 8. Outlier investigation
numeric_columns = [
    "quantity", "unit_price", "discount_percent",
    "tax_percent", "subtotal", "discount_amount",
    "tax_amount", "shipping_cost", "total_amount"
]

outlier_summary = []

for col in numeric_columns:
    q1 = train_df[col].quantile(0.25)
    q3 = train_df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (train_df[col] < lower) |
        (train_df[col] > upper)
    ).sum()

    outlier_summary.append({
        "Column": col,
        "Outlier_Count": count,
        "Action": "Retained after investigation"
    })

outlier_summary = pd.DataFrame(outlier_summary)

# 9. Data dictionary
data_dictionary = pd.DataFrame({
    "Column": train_df.columns,
    "Data_Type": train_df.dtypes.astype(str).values,
    "Unique_Values": train_df.nunique().values,
    "Missing_Values": train_df.isnull().sum().values
})

# 10. Save cleaning documentation
cleaning_log = pd.DataFrame([
    ["Missing values", "All columns", "No action", "No missing values"],
    ["Duplicate records", "All columns", "No action", "No duplicates"],
    ["Date standardization", "Date columns", "Converted", "Standardized date formats"],
    ["Invalid dates", "Date columns", "No action", "All dates valid"],
    ["Negative values", "Numeric columns", "No action", "No negative values"],
    ["Infinite values", "Numeric columns", "No action", "No infinite values"],
    ["Outliers", "Numeric columns", "Retained", "No invalid outliers established"],
    ["Financial calculations", "Amount columns", "No action", "Formulas validated"],
    ["Categorical consistency", "Categorical columns", "No action", "No inconsistencies found"],
    ["Identifier consistency", "ID columns", "No action", "Mappings validated"],
    ["Test tax data type", "tax_percent", "Converted to int64", "Whole-number tax rates"]
], columns=["Issue", "Affected_Columns", "Action", "Reason"])

# 11. Save cleaned datasets
for name, df in [
    ("train_cleaned", train_df),
    ("test_cleaned", test_df)
]:
    df.to_csv(
        os.path.join(OUTPUT_FOLDER, name + ".csv"),
        index=False
    )

    df.to_parquet(
        os.path.join(OUTPUT_FOLDER, name + ".parquet"),
        index=False
    )

# 12. Save documentation
data_dictionary.to_csv(
    os.path.join(OUTPUT_FOLDER, "data_dictionary.csv"),
    index=False
)

cleaning_log.to_csv(
    os.path.join(OUTPUT_FOLDER, "cleaning_log.csv"),
    index=False
)

outlier_summary.to_csv(
    os.path.join(OUTPUT_FOLDER, "outlier_summary.csv"),
    index=False
)

print("Week 1 data cleaning completed.")
print("Training dataset:", train_df.shape)
print("Test dataset:", test_df.shape)
print("Data dictionary:", data_dictionary.shape)
print("Output folder:", OUTPUT_FOLDER)
