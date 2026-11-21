# Invoice Payment Delay & Cash-Flow Intelligence Platform

## 1. Project Overview

This project focuses on invoice analytics, customer payment behaviour, data quality validation, feature engineering, and overdue-status classification.

The project uses the invoice datasets provided for the Orvians Technologies internship and follows a chronological data-processing workflow.

## 2. Dataset Overview

| Dataset | Records | Columns |
|---|---:|---:|
| Original training dataset | 1,048,575 | 31 |
| Original test dataset | 2,000,000 | 31 |

The datasets cover:
- Training invoices: 2019–2023
- Test invoices: 2024–2025

The original and processed Parquet datasets are excluded from this public repository because of their size.

## 3. Week 1 — Data Cleaning and Validation

Completed:
- Data type validation and date conversion
- Missing-value and duplicate checks
- Invalid date and financial-value checks
- Data dictionary generation
- Cleaning log generation
- Outlier analysis
- CSV and Parquet output generation

## 4. Week 2 — Exploratory Data Analysis

Completed:
- Invoice payment-status analysis
- Invoice amount distribution
- Customer payment behaviour analysis
- Invoice aging analysis
- Country and monthly trends
- Correlation analysis
- Data quality and target investigation

### Important findings

- Approximately 68.04% of training invoices are marked Paid.
- Approximately 16.99% are Pending.
- Approximately 9.95% are Overdue.
- Approximately 5.02% are Cancelled.
- No positive payment delays were observed among paid invoices.
- Payment dates for unpaid invoices contain placeholder-like values and should not be interpreted as actual settlements.

## 5. Week 3 — Feature Engineering

The official Week 3 workflow uses the same Orvians invoice dataset from Weeks 1 and 2.

### Prediction objective

Classify the supplied overdue-status label (`is_overdue`).

This is a classification task using the dataset's existing overdue-status labels. It should not be interpreted as a verified prediction of actual payment delay in days.

### Feature engineering

Created feature groups include:

- **Time features:** Invoice year, month, quarter, day, weekday, month-start and month-end indicators.
- **Payment-term features:** Days between invoice date and due date.
- **Amount features:** Log-transformed invoice amount, quantity, and financial ratios.
- **Customer history:** Previous invoice count, previous overdue count, historical average invoice amount, and smoothed historical overdue rate.
- **Categorical features:** Customer, vendor, product category, currency, payment method, order channel, sales representative, and customer location.

Customer history features use prior invoices with due dates earlier than the current invoice date. Test history uses training records only.

### Chronological data splitting

| Split | Period | Records |
|---|---|---:|
| Training | 2019–2022 | 797,101 |
| Validation | 2023 | 198,865 |
| Final test | 2024–2025 | 1,899,611 |

Cancelled invoices are excluded from modelling.

### Leakage controls

- Excluded `payment_status` and `payment_date` from model predictors.
- Excluded the target from model predictors.
- Used past-only customer history features.
- Kept the final test period separate.
- Documented feature definitions and missingness.

### Target limitation

The supplied dataset does not contain positive examples of completed late payments among paid invoices. Therefore, it does not support a genuine paid-late regression target or reliable payment-delay-in-days prediction.

The overdue-status classifier must be evaluated as a classification of the supplied label, with its data limitations clearly acknowledged.

## 6. Week 3 Deliverables

- `scripts/week3_feature_engineering.py`
- `datasets/processed/week3/feature_spec.json`
- `reports/week3/week3_feature_engineering_report.txt`
- `reports/week3/week3_feature_missingness.csv`
- `reports/week3/week3_split_summary.csv`

Generated Parquet feature datasets are retained outside Git because of their size.

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyArrow
- Parquet
- CSV
- Git and GitHub

## 8. Project Status

| Week | Work | Status |
|---|---|---|
| Week 1 | Data cleaning and validation | Completed |
| Week 2 | Exploratory data analysis | Completed |
| Week 3 | Feature engineering and leakage checks | Completed |
| Week 4 | Baseline model development | Pending |

## 9. Limitations and Future Work

- The supplied overdue-status label is not independently verified as a final payment outcome.
- The dataset does not support genuine paid-late regression.
- Historical source-system availability timestamps have not been verified.
- Stable overdue rates across time require further investigation.
- Model performance and predictive usefulness must be established through validation.

The project is intended as an applied machine-learning engineering exercise. Production deployment requires stronger validation of target semantics and source-data availability.