
import pandas as pd

df = pd.read_parquet(
    "data/invoice_currency_history_features.parquet"
)

report = {
    "total_records": len(df),
    "total_columns": len(df.columns),
    "paid_records": int((df["payment_status"] == "Paid").sum()),
    "overdue_records": int((df["payment_status"] == "Overdue").sum()),
    "overdue_with_payment_equal_due": int(
        (
            (df["payment_status"] == "Overdue") &
            (df["payment_date"] == df["due_date"])
        ).sum()
    ),
    "paid_with_positive_delay": int(
        (
            (df["payment_status"] == "Paid") &
            (df["payment_date"] > df["due_date"])
        ).sum()
    ),
    "overdue_with_positive_delay": int(
        (
            (df["payment_status"] == "Overdue") &
            (df["payment_date"] > df["due_date"])
        ).sum()
    ),
    "missing_payment_dates": int(df["payment_date"].isna().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
}

result = pd.DataFrame(
    report.items(),
    columns=["metric", "value"]
)

result.to_csv(
    "reports/final_data_validation_summary.csv",
    index=False
)

print(result.to_string(index=False))
print("\nValidation summary saved successfully.")
