
import pandas as pd

# Load working dataset
df = pd.read_parquet(
    "data/invoice_classification_working.parquet"
)

# Target column
target = "overdue_status"

# Columns that must not be used as model inputs
drop_columns = [
    "invoice_id",
    "invoice_number",
    "customer_name",
    "customer_email",
    "vendor_id",
    "vendor_name",
    "product_id",
    "product_name",
    "sales_rep",
    "payment_status",
    "payment_date",
    "is_overdue"
]

# Separate target and features
X = df.drop(columns=drop_columns + [target])
y = df[target]

print("Feature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)

print("\nSelected input features:")
print(X.columns.tolist())

print("\nExcluded columns:")
print(drop_columns)

print("\nMissing values in features:")
print(X.isnull().sum().sum())
