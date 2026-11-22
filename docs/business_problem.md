# Business Problem: CreditLens

**CRISP-DM phase:** Business Understanding (Milestone 1)
**Course:** CDS4003 Data Science for Business Application
**Lead:** Mansoor Mohamed Mansoor Saeed Alwahedi

## 1. Business context

Small and medium enterprises (SMEs) find it harder than large firms to borrow, because they have short trading histories and little collateral. Governments respond with loan-guarantee schemes. Under the US Small Business Administration (SBA) 7(a) programme, a commercial bank makes the loan and the SBA guarantees a portion of it. If the borrower defaults, the SBA covers the guaranteed amount and the bank absorbs the rest (Li, Mickel and Taylor, 2018). In our data the guaranteed share has a median of 75% and a mean of 70.9% of the approved amount, so the bank still carries a quarter or more of the principal on a typical loan.

The same mechanism exists in the UAE. Emirates Development Bank (EDB) runs a credit guarantee scheme in which it guarantees part of the SME loans that partner commercial banks extend, and the Khalifa Fund for Enterprise Development finances and supports Emirati-owned SMEs. A UAE bank lending under such a scheme faces the same question as a US bank lending under 7(a): which applicants are likely to repay, and how much is lost when they do not. No public UAE dataset of loan-level SME outcomes exists, so we use the SBA records, which are the largest public dataset of guaranteed SME loans, and frame the result as a tool that a UAE bank's SME credit desk could adopt after retraining on its own portfolio.

## 2. Decision to support

At application time, the credit officer has to decide whether to **approve**, **refer for manual review** or **decline** a guaranteed SME loan, and needs an estimate of the expected loss if the loan is approved. CreditLens supports this decision with a default probability, a risk band, a recommended action and an explanation of the main factors behind the score. The officer keeps the final decision.

The economics are asymmetric. A charged-off loan costs the bank the unguaranteed principal plus collection and administration cost. A wrongly declined good borrower costs the bank its interest margin and costs the economy the jobs that the loan would have created or retained. We make this explicit with the cost matrix used throughout the project.

| Decision | Loan repaid | Loan defaults |
|---|---|---|
| Approve | +5% of disbursed amount (net interest margin) | minus 50% of unguaranteed principal, minus 5% of disbursed amount (workout cost) |
| Decline | 0 | 0 |

## 3. Stakeholders

| Stakeholder | What they gain |
|---|---|
| SME credit officers | A consistent first assessment of each application and a clear reason for the score, so review time goes to the borderline cases |
| Bank credit risk management | Lower expected charge-off loss and a portfolio view of how the approval threshold changes risk and return |
| Guarantee scheme operators (SBA, EDB) | Fewer claims on the guarantee and evidence that guaranteed lending reaches viable firms |
| SME applicants | Faster and more consistent decisions, and a stated basis for a decline or referral |
| Regulators and internal audit | A documented, explainable model with stated assumptions and known limitations |

## 4. SMART objectives

1. Build a default-risk classifier with ROC-AUC of at least 0.80 and recall on defaulted loans of at least 0.70 on a held-out test set, by Week 13.
2. Reduce expected charge-off loss on the held-out portfolio by at least 25% compared with the approve-everything policy, at an approval rate of at least 75%, using an explicit cost matrix, by Week 14.
3. Deploy a public Streamlit app that returns a scored decision with an explanation in under 3 seconds per loan, by Week 14.

## 5. Expected outcomes

We expect to deliver a cleaned and documented SBA dataset, a tuned classifier saved as one scikit-learn pipeline, a decision threshold that maximises portfolio net value at an approval rate of at least 75%, and a public Streamlit app with single-loan and batch scoring, a threshold explorer and a model card.

## 6. Why machine learning

Banks have traditionally screened SME loans with manual scorecards and fixed rules, such as a minimum time trading. These are transparent, but each rule looks at one or two variables, the cut-offs are set by judgement, and interactions such as a new business in a cyclical sector borrowing during a downturn are hard to express.

The SBA data contains 897,167 loans with a known outcome, which is enough to learn these interactions from evidence. A supervised model ranks every applicant on a continuous probability scale, so the bank can move the threshold to match its risk appetite instead of rewriting rules. Gradient-boosted trees handle mixed input types and non-linear effects, logistic regression gives an interpretable benchmark, and SHAP breaks each score into the contribution of each input, which keeps the transparency of a scorecard.

## 7. Machine learning task

| Item | Definition |
|---|---|
| Task | Binary classification with probability output |
| Unit of analysis | One SBA-guaranteed loan |
| Target | `Default` = 1 if `MIS_Status` is `CHGOFF`, 0 if `P I F` |
| Excluded rows | 1,997 loans with missing `MIS_Status` (outcome unknown) |
| Class balance | 157,558 defaults (17.56%) and 739,609 paid in full (82.44%) |
| Split | 80/20 stratified on `Default`, `RANDOM_STATE = 42` |

**Leakage rule.** `ChgOffDate`, `ChgOffPrinGr` and `BalanceGross` are only known after the outcome and are never used as features. `MIS_Status` is used only to build the label. `LoanNr_ChkDgt` and `Name` are identifiers and are never features. `DisbursementDate` and `DisbursementGross` are known at disbursement and may be used. Bank-level default rates are computed on the training fold only.

## 8. Success criteria

| Criterion | Threshold | Measured in |
|---|---|---|
| ROC-AUC on held-out test set | at least 0.80 | 05_tuning_evaluation.ipynb |
| Recall on defaulted loans | at least 0.70 | 05_tuning_evaluation.ipynb |
| Reduction in expected charge-off loss versus approve-everything | at least 25% | 05_tuning_evaluation.ipynb |
| Approval rate at the chosen threshold | at least 75% | 05_tuning_evaluation.ipynb |
| App response time per loan | under 3 seconds | Deployed Streamlit app |

## 9. Risks

**Class imbalance.** Only 17.56% of labelled loans defaulted. Predicting "paid in full" for everyone is 82.44% accurate and useless, so we judge models on ROC-AUC, PR-AUC, recall on defaults and net value.

**Right-censoring.** The data ends in 2014. Loans approved in the final years can only appear as paid in full if they were repaid early, while defaults appear quickly. The default rate is 34.2% for fiscal years 2005 to 2007 but only 6.6% for 2011 to 2014, which reflects censoring as well as credit quality. We handle this with a temporal validation in notebook 05.

**Data age and policy change.** The loans were approved between 1962 and 2014 under changing SBA rules, guarantee shares and economic conditions, including the 2007 to 2009 recession. Patterns learnt from this period may not hold for current lending.

**Fairness and ethics.** Automated credit decisions can reproduce historical patterns that disadvantage certain regions, sectors or new businesses. The data holds no protected personal attributes, but location and sector can act as proxies. We report error rates by sector and state and keep a human officer responsible for every decline.

**Explainability.** Banking regulators expect lenders to explain credit decisions to applicants and supervisors. Every score in the app therefore comes with a per-loan explanation, and the model card states the assumptions and limits.

**Transfer to the UAE.** The model learns US borrower behaviour. A UAE bank would need to retrain and validate it on local data before relying on it.

## Reference

Li, M., Mickel, A. and Taylor, S. (2018) '"Should This Loan be Approved or Denied?": A Large Dataset with Class Assignment Guidelines', *Journal of Statistics Education*, 26(1), pp. 55-66. doi:10.1080/10691898.2018.1434342.
