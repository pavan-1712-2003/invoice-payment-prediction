
import pandas as pd

df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

print("FEATURE DATA TYPES:")
print(df.dtypes.to_string())

print("\nUNIQUE VALUES IN CATEGORICAL COLUMNS:")

categorical_columns = [
    "customer_city",
    "customer_state",
    "customer_country",
    "product_category",
    "currency",
    "payment_method",
    "order_channel"
]

for col in categorical_columns:
    print(f"\n{col}:")
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts().head(10).to_string())

print("\nDATE RANGE:")
print("Invoice date:", df["invoice_date"].min(),
      "to", df["invoice_date"].max())
print("Due date:", df["due_date"].min(),
      "to", df["due_date"].max())
