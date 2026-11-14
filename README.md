# Invoice Payment Prediction – Exploratory Data Analysis

## Project Overview

This project explores invoice transaction data to understand payment behaviour, identify patterns, and prepare the dataset for machine learning.

The analysis focuses on payment status, invoice amounts, payment delays, payment terms, customer behaviour, monthly trends, and outstanding invoice aging.

## Dataset Overview

| Dataset | Records | Columns |
|---|---:|---:|
| Training data | 1,048,575 | 31 |
| Test data | 2,000,000 | 31 |
| EDA dataset | 1,048,575 | 37 |

The EDA dataset includes six additional date-related features.

## EDA Activities

- Data inspection and validation
- Missing-value and duplicate checks
- Univariate analysis
- Bivariate analysis
- Customer payment behaviour analysis
- Time-based trend analysis
- Invoice aging analysis
- Correlation analysis
- Feature engineering
- Data quality and leakage checks

## Key Findings

1. Approximately 68.04% of training invoices are marked as Paid.
2. Approximately 16.99% are Pending, 9.95% are Overdue, and 5.02% are Cancelled.
3. Among paid invoices, 96.03% are recorded as paid before the due date.
4. No positive payment delays were observed in the paid records.
5. Invoice amount has almost no correlation with payment delay in this dataset.
6. Payment terms show a negative relationship with payment delay.
7. Monthly and weekday payment-delay averages are relatively stable.
8. Invoice volume varies slightly across calendar months.
9. The dataset's aging analysis indicates that most outstanding records fall into the 90+ day bucket.

## Important Data Limitations

- Non-paid invoices have payment dates equal to their due dates. These dates may be placeholders and should not be treated as actual payment dates.
- No positive payment delays were observed among paid invoices. Therefore, this dataset does not support a meaningful analysis of historical late-payment behaviour.
- The `is_overdue` feature is derived from payment status and may cause target leakage.
- `payment_date` is future information for predictions made at invoice creation time.
- The dataset contains multiple currencies. Monetary totals should not be interpreted as a single-currency business total.
- The aging analysis uses 29 February 2024 as its reference date. It is a dataset-based analysis, not a verified current outstanding-invoice snapshot.

## Project Deliverables

- `Invoice_Payment_Prediction_EDA.ipynb` – EDA notebook with 14 visualizations.
- `Invoice_Payment_Prediction_EDA_Report.pdf` – EDA report.
- `invoice_eda_sample.csv` – Sample dataset for public sharing.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Parquet / CSV

## Repository Structure

```text
invoice-payment-prediction/
│
├── data/
│   └── invoice_eda_sample.csv
│
├── notebooks/
│   └── Invoice_Payment_Prediction_EDA.ipynb
│
├── reports/
│   └── Invoice_Payment_Prediction_EDA_Report.pdf
│
├── README.md
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Install the required Python libraries.
3. Open the EDA notebook in Jupyter Notebook.
4. Update the dataset path to point to your local full EDA Parquet file, or adapt the notebook to use the public CSV sample.
5. Run the notebook cells.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn pyarrow jupyter
```

## Note

The full datasets are not included in this public repository because of their size. A sample CSV is provided for reference. The full Parquet dataset is required to reproduce the original analysis exactly.

## Project Status

Week 2 – Exploratory Data Analysis and reporting completed. Further modelling should account for the data limitations documented above.
