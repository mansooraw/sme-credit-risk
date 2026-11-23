# Model card: CreditLens SME default-risk model

Owner: Mansoor Mohamed Mansoor Saeed Alwahedi (modelling and evaluation), CreditLens team, HCT CDS4003
Model file: models/final_model.joblib (scikit-learn Pipeline, 3.25 MB). Metadata: models/model_metadata.json.
Version date: 2026-11-23

## Intended use

The model estimates the probability that a guaranteed SME loan will be charged off, from information available when the credit decision is made. It is a decision-support tool for a bank's SME credit desk: it ranks applications, places them in a risk band, suggests approve, refer or decline, and shows the expected value of the loan under the bank's cost assumptions together with the features that drove the score. The final decision stays with a credit officer. The model is not intended for consumer lending, for loans outside a guarantee scheme, for setting interest rates, or as the only basis for declining an applicant.

## Training data

US Small Business Administration 7(a) loans approved between 1962 and 2014 (Li, Mickel and Taylor, 2018, via Kaggle, CC BY-SA 4.0), cleaned in notebook 02 to 895,952 loans with a known outcome. The model was trained on the stratified 80% training split (716,761 loans, default rate 17.56%) and tested on the remaining 179,191 loans. Default means the loan status is charged off. Data are from the United States; using the model for a UAE bank would require validation and recalibration on local loans before any live use.

## Features

23 inputs: 15 numeric (including 0/1 indicators), 7 categorical and the lending bank, which the pipeline turns into BankSize and an out-of-fold BankDefaultRate learned from training loans only. ApprovalYear, ApprovalMonth and DisbursementLagDays were removed because they cannot be valued for a new application and partly measure right-censoring (notebook 04). Leakage columns (ChgOffDate, ChgOffPrinGr, BalanceGross), identifiers, City and Zip are never used.

| Feature | Meaning |
|---|---|
| `NoEmp` | Number of employees |
| `CreateJob` | Jobs to be created |
| `RetainedJob` | Jobs retained |
| `Portion` | Guaranteed share |
| `TermYears` | Term (years) |
| `LogDisbursement` | Log of disbursed amount |
| `LoanPerEmployee` | Loan per employee (USD) |
| `JobsPerDollar` | Jobs per USD 100,000 |
| `DisbursedToApprovedRatio` | Disbursed to approved ratio |
| `RealEstate` | Real-estate term (20 years or more) |
| `Recession` | Active during the 2007-2009 recession |
| `NewBusiness` | New business |
| `IsFranchise` | Franchise |
| `SameStateBank` | Lender in the same state |
| `DisbursementDateMissing` | Disbursement date unknown |
| `Sector` | Industry sector |
| `RevLineCr` | Revolving line of credit |
| `LowDoc` | LowDoc programme |
| `UrbanRural` | Urban or rural |
| `NewExist` | Business age group |
| `State` | Borrower state |
| `BankState` | Bank state |
| `Bank` | Lending bank |

## Performance (test split, 179,191 loans)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9807 |
| PR-AUC | 0.9262 |
| Brier score | 0.0351 |
| Recall on defaults at threshold 0.22 | 0.9263 |
| Precision at threshold | 0.7817 |
| F1 at threshold | 0.8479 |
| Accuracy / balanced accuracy | 0.9416 / 0.9356 |
| 3-fold CV ROC-AUC (training split) | 0.9802 (sd 0.0002) |
| Temporal validation ROC-AUC (train FY up to 2005, test FY 2006 on) | 0.9676 |
| Temporal validation recall | 0.9189 |

The interpretable comparison model (logistic regression, C = 0.1) reaches a test ROC-AUC of 0.8655.

Most important features by mean absolute SHAP value (log-odds):

| Rank | Feature | Mean absolute SHAP |
|---|---|---|
| 1 | `TermYears` | 2.178 |
| 2 | `BankDefaultRate` | 0.595 |
| 3 | `Portion` | 0.429 |
| 4 | `Sector` | 0.233 |
| 5 | `RealEstate` | 0.223 |
| 6 | `LogDisbursement` | 0.180 |
| 7 | `BankSize` | 0.160 |
| 8 | `SameStateBank` | 0.153 |
| 9 | `BankState` | 0.136 |
| 10 | `Recession` | 0.127 |

## Threshold, risk bands and decision rule

Cost assumptions: a repaid loan earns 5% of the disbursed amount; a charged-off loan loses 50% of the unguaranteed principal plus a 5% workout cost; a declined loan earns and loses nothing. The threshold of 0.22 maximises net value on out-of-fold training predictions subject to an approval rate of at least 75%.

Approve when the probability of default is below 0.22 (Low and Medium bands); refer to manual review when it is from 0.22 to below 0.50 (High band); decline when it is 0.50 or more (Very high band).

| Band | Probability of default | Decision | Share of test loans | Observed default rate |
|---|---|---|---|---|
| Low | 0.00 to 0.10 | Approve | 75.1% | 1.0% |
| Medium | 0.10 to 0.22 | Approve | 4.1% | 13.9% |
| High | 0.22 to 0.50 | Refer | 4.0% | 34.0% |
| Very high | 0.50 to 1.00 | Decline | 16.8% | 88.8% |

On the test split this policy approves 79.2% of loans, earns a realised net value of USD 1,413.5 m against USD 733.7 m for approving every loan, and cuts charge-off loss by 89.1%.

## Limitations

* Loan term dominates. Without TermYears and RealEstate the cross-validated ROC-AUC falls from 0.975 to 0.838. Term is agreed at approval, but we cannot rule out that for some charged-off loans the recorded term was shortened after the event; if so, part of the measured performance would not be available in live use.
* Right-censoring and drift. Loans approved shortly before 2014 had little time to default, and the default rate moved from about 3% to 44% across approval years. Trained on loans up to FY 2005, the model scores later loans with ROC-AUC 0.968 and predicts a mean default rate of 29.0% against 33.3% observed.
* The data are US loans from 1962 to 2014 under SBA programme rules; guarantee shares, lender behaviour and the economy differ in the UAE and in 2026.
* The cost matrix is a simplification: it ignores the time value of money, partial recoveries beyond the fixed loss rate and the value of referred loans that are approved after review.
* Recession is a scenario input: the model learned it from the 2007 to 2009 downturn, so a different kind of downturn may affect loans differently.

## Ethics and fairness

The data contain no personal characteristics of the owners (no gender, age, ethnicity or nationality), so the model cannot use them directly. Borrower state, bank state, sector and urban or rural location are used, and geography can act as a proxy for protected characteristics or penalise whole regions; across the ten largest states the test ROC-AUC ranges from 0.974 to 0.989 and the observed default rate from 12.1% to 27.0%. BankDefaultRate means that a borrower's score depends on the lender's past book, which the borrower does not control. New businesses and small loans receive higher scores on average, which could reduce access to credit for the firms a guarantee scheme exists to help. We therefore recommend that the model is used to refer rather than automatically decline borderline cases, that declines are reviewed by a person and explained with the SHAP drivers, and that approval rates are monitored by region, sector and business age.

## Monitoring plan

* Monthly: distribution of the input features and of the predicted probabilities against the training data (population stability index above 0.25 triggers a review), approval rate and band mix.
* Quarterly: observed default rate against predicted probability by approval quarter for loans with at least 24 months of history, ROC-AUC and recall on matured loans, and the same figures by state, sector and loan size.
* Annually, or when a monitored ROC-AUC falls by more than 0.05 or calibration error exceeds 5 percentage points in any band: retrain on recent matured loans, re-choose the threshold with the bank's current cost assumptions and repeat the temporal validation.
* Any change to the guarantee programme, the cost assumptions or the feature definitions requires a new model version and an updated model card.
