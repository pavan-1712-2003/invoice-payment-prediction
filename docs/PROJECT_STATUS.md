# Invoice Payment Delay & Cash-Flow Intelligence

## Project Status

### Week 1 — Data Cleaning and Validation
Status: Completed and verified on the new PC.

- Training dataset: 2,500,000 rows × 31 columns.
- Test dataset: 2,000,000 rows × 31 columns.
- Missing values: 0 in both datasets.
- Duplicate invoice IDs: 0.
- Invalid dates: 0.
- Invalid numeric values: 0.
- Financial formula checks passed for subtotal, discount, tax on discounted subtotal, and total amount.
- Cleaned CSV and Parquet datasets generated and verified.

### Week 2 — Exploratory Data Analysis
Status: EDA execution completed.

- Training overdue invoices: 249,714 (9.99%).
- Training non-overdue invoices: 2,250,286 (90.01%).
- Payment status matches the target is_overdue exactly.
- No positive payment delays were found among paid invoices.
- EDA charts, summaries, and PDF report generated.

### Important Dataset Limitation
The target is_overdue is an exact match for payment_status == Overdue. This means the current target represents invoice overdue status in the dataset, not independently verified future payment behaviour.

For an invoice-time prediction model, exclude payment_status and payment_date because they are post-outcome fields.

The dataset does not support reliable late-payment delay regression because no positive payment delays were observed among paid invoices.

### Week 3 — Feature Engineering
Status: Existing script requires leakage review and correction before final execution.

### Week 4 Onward
Not yet verified as completed.
