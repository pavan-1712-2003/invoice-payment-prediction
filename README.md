# Invoice Analytics & Payment Behaviour Analysis

## Project Overview

This project analyzes invoice transaction data to understand payment patterns, customer behaviour, invoice aging, and cash-flow-related trends.

The project focuses on exploratory data analysis (EDA), data quality validation, feature engineering, and identifying limitations that affect future payment-delay modelling.

**Important:** The current dataset supports invoice analytics and exploratory payment-status analysis. It does not contain reliable historical late-payment examples for building a genuine future payment-delay prediction model.

## Dataset Overview

| Dataset | Records | Columns |
|---|---:|---:|
| Training data | 1,048,575 | 31 |
| Test data | 2,000,000 | 31 |
| EDA dataset | 1,048,575 | 37 |

The EDA dataset contains additional date-related and engineered features.

The original datasets and processed Parquet files are excluded from this public repository because of their size.

## EDA Activities

- Data inspection and validation
- Missing-value and duplicate checks
- Univariate and bivariate analysis
- Customer payment behaviour analysis
- Monthly and weekday trend analysis
- Invoice aging analysis
- Correlation analysis
- Feature engineering
- Data quality and target leakage checks
- Customer and currency history analysis

## Key Findings

1. Approximately 68.04% of training invoices are marked as Paid.
2. Approximately 16.99% are Pending, 9.95% are Overdue, and 5.02% are Cancelled.
3. Among paid invoices, 96.03% are recorded as paid before the due date.
4. No positive payment delays were observed among paid invoices.
5. Invoice amount has almost no correlation with payment delay in this dataset.
6. Payment terms show a negative relationship with payment delay.
7. Monthly and weekday payment-delay averages are relatively stable.
8. Invoice volume varies slightly across calendar months.
9. The aging analysis indicates that most outstanding records fall into the 90+ day bucket.

## Important Data Limitations

- Non-paid invoices have payment dates equal to their due dates. These dates may be placeholders and should not be treated as actual payment dates.
- No positive payment delays were observed among paid invoices. Therefore, the dataset does not provide meaningful examples of historical late payments.
- The `is_overdue` feature is derived from payment status and can cause target leakage.
- `payment_date` is future information when making predictions at invoice creation time.
- The dataset contains multiple currencies. Monetary totals should not be combined without currency normalization.
- The aging analysis uses 29 February 2024 as its reference date. It is not a verified current outstanding-invoice snapshot.

**Modelling limitation:** A separate, validated dataset containing actual invoice dates, due dates, and payment datesâ€”including genuine late paymentsâ€”is required before building a meaningful payment-delay prediction model.

## Project Deliverables

- `notebooks/analytics/Invoice_Payment_Prediction_EDA_Cleaned.ipynb` â€” Cleaned EDA notebook.
- `reports/eda/Invoice_Payment_Prediction_EDA_Report.pdf` â€” EDA report.
- `reports/eda/` â€” 14 EDA visualizations.
- `reports/analytics/final_data_validation_summary.csv` â€” Data validation summary.
- `data/sample/invoice_eda_sample.csv` â€” Public sample dataset.
- `src/common/legacy_scripts/` â€” Historical analysis and validation scripts retained for reference.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Parquet
- CSV

## Repository Structure

```text
invoice-payment-prediction/
|
|-- data/
|   |-- sample/
|       |-- invoice_eda_sample.csv
|
|-- notebooks/
|   |-- analytics/
|       |-- Invoice_Payment_Prediction_EDA_Cleaned.ipynb
|
|-- reports/
|   |-- analytics/
|   |   |-- final_data_validation_summary.csv
|   |
|   |-- eda/
|       |-- 14 visualization images
|       |-- Invoice_Payment_Prediction_EDA_Report.pdf
|
|-- src/
|   |-- common/
|       |-- legacy_scripts/
|           |-- Historical analysis scripts
|
|-- README.md
|-- .gitignore
```

## How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Open the EDA notebook using Jupyter Notebook.
4. Obtain the full dataset separately if you want to reproduce the original analysis.
5. Update the notebook's dataset paths to match your local files.
6. Run the notebook cells.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn pyarrow jupyter
```

## Data Availability

The full raw and processed datasets are not included in this repository. A sample CSV is provided for demonstration.

The full processed Parquet dataset is required to reproduce the original analysis exactly.

## Project Status

**Completed:** Data cleaning, exploratory data analysis, feature investigation, validation, and reporting.

**Future work:** Identify and validate a genuine historical payment-delay dataset before developing a reliable prediction model.
