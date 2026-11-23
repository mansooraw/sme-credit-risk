# Key findings from exploratory analysis and feature engineering

Author: Sultan Zayed Ahmed Saeed Alkindi
Produced by: notebooks/03_eda_feature_engineering.ipynb
Data: training split of data/processed/loans_clean.parquet (716,761 loans, default rate 17.56%); the ablation in finding 12 is scored on the 179,191 test loans.

1. Loan term is the strongest single signal. Loans of 13 to 36 months default at 53.0%, while loans of 240 months or more, normally secured on real estate, default at 1.7% (against 20.8% for shorter loans). TermYears has mutual information of 0.259 nats, four times any other feature, so its role must be checked for possible post-outcome recording in notebooks 04 and 05.
2. Default follows the lending cycle and is affected by right-censoring. The rate rose from 3.0% for 1995 approvals to 44.4% for 2007 approvals and fell to 2.2% for 2013, the last fall being mostly because recent loans had not yet had time to fail when the data was extracted.
3. Sector risk differs by a factor of more than three: real estate and rental and leasing defaults at 28.8% against 8.3% for mining, quarrying, and oil and gas extraction. Loans with no industry code default at 8.4% because almost all were approved in 2000 or earlier.
4. Small loans are the risky ones: 27.4% of loans under USD 25,000 are charged off, against 8.7% of loans between USD 500,000 and USD 1 million.
5. Lender location matters more than borrower age. Loans from a bank in another state default at 25.4% against 10.5% from a bank in the same state, while new businesses default only slightly more than existing ones (18.8% against 17.1%).
6. The guaranteed share reflects the loan programme rather than the borrower: loans with a 50 to 59% guarantee default at 26.4% and those with 90% or more at 6.3%.
7. The identity of the lender is highly informative. The largest lenders (10,000 or more training loans) have a default rate of 25.6% against 9.5% for banks with 100 to 999 loans, and among the 81 banks with 1,000 or more loans the rate ranges from 0.0% to 73.4%.
8. The recession effect is hidden in the overall numbers. Across all loans, being active during December 2007 to June 2009 changes the default rate by only 0.3 points, but loans of 60 months or less that ran through the recession defaulted at 53.4% against 28.4% for other loans of that length.
9. The raw RevLineCr code T, merged into Y during cleaning, defaults at 43.0% against 25.5% for Y, with a gap of 19.6 percentage points within the same approval year (weighted by T loans). It is kept as its own level of the RevLineCr feature.
10. Being a new business matters in some industries only: in accommodation and food services new businesses default at 25.5% against 19.5% for existing ones, while across all loans the gap is under 2 points.
11. Unknown codes carry signal: UrbanRural Unknown loans default at 7.2% and LowDoc Unknown loans at 26.0%, against an overall rate of 17.6%, which supports keeping Unknown as its own category.
12. Feature engineering adds measurable value. In a default LightGBM model the test ROC-AUC rises from 0.9778 (raw cleaned columns) to 0.9806 (final feature set) and PR-AUC from 0.9139 to 0.9231. Without the bank encodings PR-AUC is 0.9162; with RevLineCr T merged into Y it is 0.9227. The ten features with the highest mutual information are TermYears, ApprovalYear, BankDefaultRate, BankSize, Portion, RealEstate, JobsPerDollar, DisbursementLagDays, UrbanRural, RetainedJob.

Figures: docs/figures/fig03_*.png. Tables: docs/feature_value_tests.csv and docs/feature_ablation.csv.
