
import pandas as pd
from pathlib import Path
import gc

# Input and output folders
input_folder = Path(r"C:\invioce_data\datasets\week3")

output_folder = input_folder / "engineered"
output_folder.mkdir(parents=True, exist_ok=True)

# Datasets to process
datasets = {
    "training": "training_data.parquet",
    "validation": "validation_data.parquet",
    "test": "test_data.parquet"
}

def create_features(df):

    # Convert dates
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["due_date"] = pd.to_datetime(df["due_date"])

    # Date features
    df["invoice_year"] = df["invoice_date"].dt.year
    df["invoice_month"] = df["invoice_date"].dt.month
    df["invoice_quarter"] = df["invoice_date"].dt.quarter
    df["invoice_day"] = df["invoice_date"].dt.day

    df["invoice_day_of_week"] = (
        df["invoice_date"].dt.day_of_week
    )

    df["invoice_is_weekend"] = (
        df["invoice_day_of_week"] >= 5
    ).astype("int8")

    # Payment terms
    df["payment_terms_days"] = (
        df["due_date"] - df["invoice_date"]
    ).dt.days

    # Financial ratios
    df["discount_ratio"] = (
        df["discount_amount"] / df["subtotal"]
    ).where(df["subtotal"] > 0, 0)

    df["tax_ratio"] = (
        df["tax_amount"] / df["subtotal"]
    ).where(df["subtotal"] > 0, 0)

    df["shipping_ratio"] = (
        df["shipping_cost"] / df["subtotal"]
    ).where(df["subtotal"] > 0, 0)

    return df


# Process one dataset at a time
for name, filename in datasets.items():

    print(f"\nProcessing {name} dataset...")

    input_file = input_folder / filename
    output_file = output_folder / f"{name}_engineered.parquet"

    df = pd.read_parquet(input_file)

    original_rows = len(df)
    original_columns = len(df.columns)

    df = create_features(df)

    # Basic validation
    print("Rows:", len(df))
    print("Original columns:", original_columns)
    print("New columns:", len(df.columns))

    print(
        "Missing values in new features:",
        df[
            [
                "invoice_year",
                "invoice_month",
                "invoice_quarter",
                "invoice_day",
                "invoice_day_of_week",
                "invoice_is_weekend",
                "payment_terms_days",
                "discount_ratio",
                "tax_ratio",
                "shipping_ratio"
            ]
        ].isnull().sum().sum()
    )

    assert len(df) == original_rows

    # Save engineered dataset
    df.to_parquet(output_file, index=False)

    print("Saved:", output_file)

    del df
    gc.collect()

print("\nWeek 3 basic feature engineering completed.")

print("\n8. SHIPPING RATIO BY CURRENCY")

df = pd.read_parquet(
    output_folder / "training_engineered.parquet"
)

print(
    df.groupby("currency")["shipping_ratio"]
    .agg(["count", "mean", "median", "max"])
    .round(3)
    .sort_values("mean", ascending=False)
)

print("\n9. HIGH SHIPPING RATIO BY PRODUCT CATEGORY")

high_shipping = df[df["shipping_ratio"] > 5]
print(high_shipping["product_category"].value_counts())

print("\n10. SHIPPING RATIO BY ORDER CHANNEL")

print(
    df.groupby("order_channel")["shipping_ratio"]
    .agg(["count", "mean", "median", "max"])
    .round(3)
    .sort_values("mean", ascending=False)
)

del df, high_shipping
gc.collect()

print("\nWeek 3 feature engineering and shipping analysis completed.")