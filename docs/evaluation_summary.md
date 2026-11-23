# Evaluation summary

Author: Mansoor Mohamed Mansoor Saeed Alwahedi
Produced by: notebooks/04_model_selection.ipynb and notebooks/05_tuning_evaluation.ipynb
Data: 716,761 training and 179,191 test loans (stratified 80/20 split of notebook 03, default rate 17.56%); deployment feature set of 23 input columns.

1. Deployment feature set. Removing ApprovalYear, ApprovalMonth and DisbursementLagDays, which cannot be valued for a new application and partly encode right-censoring, lowers default LightGBM's 5-fold CV ROC-AUC from 0.9801 to 0.9748 and CV PR-AUC from 0.9241 to 0.9081. Every model after this point uses the deployment set.
2. Term sensitivity. Without TermYears and RealEstate the CV ROC-AUC falls to 0.8379 and PR-AUC to 0.5799. Term is agreed at approval and is kept, but much of the performance rests on it (see limitations in docs/model_card.md).
3. Model comparison (5-fold CV ROC-AUC, deployment features): dummy 0.5000, logistic regression 0.8634, random forest 0.9551 (200,000-loan subsample), histogram gradient boosting 0.9729, LightGBM 0.9748. LightGBM was chosen; the table is in docs/model_comparison.csv.
4. Tuning. RandomizedSearchCV (25 candidates, 3 folds) raised LightGBM's 3-fold CV ROC-AUC from 0.9747 to 0.9802 and its test ROC-AUC from 0.9748 to 0.9807 (PR-AUC 0.9068 to 0.9262). The logistic regression's best C was 0.1 (test ROC-AUC 0.8655); the tuned random forest reached 0.9722. Details in docs/tuning_before_after.csv.
5. Final test performance of the tuned LightGBM: ROC-AUC 0.9807, PR-AUC 0.9262, Brier score 0.0351; at the business threshold of 0.22, recall on defaults 0.9263, precision 0.7817, F1 0.8479, accuracy 0.9416, balanced accuracy 0.9356. SMART objective 1 (ROC-AUC at least 0.80, recall at least 0.70) is met.
6. Calibration. Isotonic calibration changed the test Brier score from 0.03515 to 0.03717 and ROC-AUC from 0.9807 to 0.9796; it was not adopted under the rule fixed beforehand (at least 1% lower Brier, at most 0.001 lower AUC).
7. Threshold. Chosen on out-of-fold training predictions as the threshold with the highest net value among those approving at least 75% of loans: 0.22. On the test split it approves 79.2% of loans, with an observed default rate of 1.64% among approved loans (expected from the model 1.88%).
8. Business value on the test split (realised outcomes): net value USD 1,413.5 m for the LightGBM policy against USD 733.7 m for approve-all and USD 1,094.6 m for the logistic comparison model at its own threshold of 0.62 (approval rate 77.4%).
9. SMART objective 2. Charge-off loss falls from USD 864.4 m (approve-all) to USD 94.4 m, a loss avoided of USD 770.0 m or 89.1% (target at least 25% at an approval rate of at least 75%: met). The logistic model reduces the loss by 62.2%.
10. Risk bands: Low below 0.10, Medium 0.10 to 0.22, High 0.22 to 0.50, Very high 0.50 or more. Observed test default rates Low 1.0%, Medium 13.9%, High 34.0%, Very high 88.8%; shares of loans Low 75.1%, Medium 4.1%, High 4.0%, Very high 16.8%.
11. Explainability. The ten features with the highest mean absolute SHAP value are TermYears, BankDefaultRate, Portion, Sector, RealEstate, LogDisbursement, BankSize, SameStateBank, BankState, Recession.
12. Error analysis. Among sectors with at least 1,000 test loans the lowest ROC-AUC is 0.953 (Unknown). False negatives: 2,320 approved loans that defaulted, with a loss of USD 94.4 m; the costliest 10% of them carry 46.3% of that loss.
13. Temporal validation (train FY 2005 or earlier, 652,382 loans; test FY 2006 or later, 243,570 loans): ROC-AUC 0.9676 (0.0130 below the random split), recall 0.9189 (0.0074 below), observed default rate 33.25% against a mean prediction of 29.02%.
14. Artefacts. models/final_model.joblib is 3.25 MB and the whole models/ folder 3.78 MB, well under the 100 MB limit.

Tables: docs/model_comparison.csv, docs/feature_set_sensitivity.csv, docs/class_weight_comparison.csv, docs/tuning_before_after.csv, docs/threshold_sweep.csv, docs/policy_comparison.csv, docs/risk_bands.csv, docs/error_analysis.csv, docs/temporal_validation.csv, docs/permutation_importance_logistic.csv. Figures: docs/figures/fig04_*.png and docs/figures/fig05_*.png.
