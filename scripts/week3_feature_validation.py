import pandas as pd
from pathlib import Path

folder = Path(
    r"C:\invioce_data\datasets\week3\engineered"
)

df = pd.read_parquet(
    folder / "training_engineered.parquet"
)

features = [
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

print("FEATURE VALIDATION")
print("=" * 50)

print("\n1. Feature ranges")
print(df[features].describe().round(3))

print("\n2. Weekend values")
print(df["invoice_is_weekend"].value_counts())

print("\n3. Payment terms")
print(df["payment_terms_days"].value_counts().head(10))

print("\n4. Invalid date features")

print(
    "Invalid month:",
    (~df["invoice_month"].between(1, 12)).sum()
)

print(
    "Invalid quarter:",
    (~df["invoice_quarter"].between(1, 4)).sum()
)

print(
    "Invalid weekday:",
    (~df["invoice_day_of_week"].between(0, 6)).sum()
)

print("\n5. Financial ratio checks")

print(
    "Negative discount ratios:",
    (df["discount_ratio"] < 0).sum()
)

print(
    "Negative tax ratios:",
    (df["tax_ratio"] < 0).sum()
)

print(
    "Negative shipping ratios:",
    (df["shipping_ratio"] < 0).sum()
)

print("\n6. Missing values")
print(df[features].isnull().sum())

print("\nValidation inspection completed.")

print("\n7. HIGH SHIPPING RATIO INVESTIGATION")

columns = [
    "invoice_id",
    "subtotal",
    "shipping_cost",
    "total_amount",
    "currency",
    "shipping_ratio"
]

high_shipping = df.nlargest(
    20,
    "shipping_ratio"
)[columns]

print(high_shipping.to_string(index=False))

print("\nShipping ratio above 1:")
print((df["shipping_ratio"] > 1).sum())

print("\nShipping ratio above 5:")
print((df["shipping_ratio"] > 5).sum())

print("\nShipping ratio above 10:")
print((df["shipping_ratio"] > 10).sum())


print("\n8. SHIPPING RATIO BY CURRENCY")
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

print("\n11. DETAILED HIGH SHIPPING RATIO RECORDS")

columns = [
    "invoice_id",
    "product_name",
    "product_category",
    "quantity",
    "unit_price",
    "subtotal",
    "shipping_cost",
    "total_amount",
    "currency",
    "order_channel",
    "shipping_ratio"
]

print(
    df.nlargest(20, "shipping_ratio")[columns].to_string(index=False)
)


print("\n12. SHIPPING COST VS SUBTOTAL ANALYSIS")

print("Invoices with subtotal below 20:",
      (df["subtotal"] < 20).sum())

print("Invoices with shipping cost above 50:",
      (df["shipping_cost"] > 50).sum())

print("Invoices satisfying both conditions:",
      ((df["subtotal"] < 20) &
       (df["shipping_cost"] > 50)).sum())

print("\nShipping cost statistics for invoices with subtotal below 20:")

print(
    df.loc[df["subtotal"] < 20, "shipping_cost"]
    .describe()
    .round(2)
)
