# CreditLens: SME Loan Default Risk Scoring

CreditLens is a decision-support tool for a bank's SME credit desk lending under a government loan-guarantee scheme. Under such a scheme the guarantor covers part of each loan and the bank carries the rest, so a charged-off loan costs the bank the unguaranteed principal plus workout cost, while a wrongly declined good borrower costs the bank its interest margin and costs the economy the jobs the loan would have supported. Using 899,164 US Small Business Administration (SBA) 7(a) loan records, we build a classifier that estimates the probability of default at application time and recommends whether to approve, refer for manual review or decline, together with the expected value of the loan and an explanation of the score. The same mechanism operates in the UAE through the Emirates Development Bank credit guarantee scheme and the Khalifa Fund, and the tool is framed for a UAE bank that would retrain it on its own portfolio.

Course project for CDS4003 Data Science for Business Application (HCT), organised around the six CRISP-DM phases.

**Live app:** `<streamlit-app-url>`

## Team

| Member | Role | GitHub |
|---|---|---|
| Mansoor Mohamed Mansoor Saeed Alwahedi (team lead) | Business understanding, modelling and tuning (notebooks 04, 05), report integration, repository admin | [mansooraw](https://github.com/mansooraw) |
| Salem Adel Salem Alshamlan Almarri | Data collection, data understanding and cleaning (notebooks 01, 02), data dictionary, dataset search evidence | `<github-username>` |
| Sultan Zayed Ahmed Saeed Alkindi | EDA and feature engineering (notebook 03), Streamlit app and deployment, figures | `<github-username>` |

## Objectives

1. Build a default-risk classifier with ROC-AUC of at least 0.80 and recall on defaulted loans of at least 0.70 on a held-out test set, by Week 13.
2. Reduce expected charge-off loss on the held-out portfolio by at least 25% compared with the approve-everything policy, at an approval rate of at least 75%, using an explicit cost matrix, by Week 14.
3. Deploy a public Streamlit app that returns a scored decision with an explanation in under 3 seconds per loan, by Week 14.

The full problem statement is in [docs/business_problem.md](docs/business_problem.md).

## Dataset

| Item | Detail |
|---|---|
| Name | "Should This Loan be Approved or Denied?" (`SBAnational.csv`) |
| Source | Kaggle, uploaded by Mirbek Toktogaraev: https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied |
| Original provenance | US SBA 7(a) loan records, published with Li, M., Mickel, A. and Taylor, S. (2018) *Journal of Statistics Education*, 26(1), 55-66, https://doi.org/10.1080/10691898.2018.1434342 |
| Licence | CC BY-SA 4.0 (Kaggle dataset); CC BY 4.0 (paper) |
| Size | 899,164 rows, 27 columns, 179 MB |
| Target | `Default` = 1 if `MIS_Status` is `CHGOFF`, 0 if `P I F`; 17.56% of the 897,167 labelled loans defaulted |

The raw file exceeds GitHub's 100 MB limit and is not committed. `python src/download_data.py` downloads it with `kagglehub` (no API key needed) into `data/raw/`. A 5,000-row sample is committed in `data/sample/`. The search that led to this dataset is documented in [docs/dataset_search_evidence.md](docs/dataset_search_evidence.md) and every column is described in [docs/data_dictionary.md](docs/data_dictionary.md).

## Repository structure

```
sme-credit-risk/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  SBAnational.csv (git-ignored, see download step)
│   ├── processed/            cleaned and feature parquet files
│   └── sample/               sba_sample_5000.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_feature_engineering.ipynb
│   ├── 04_model_selection.ipynb
│   └── 05_tuning_evaluation.ipynb
├── src/
│   ├── config.py             paths, constants, cost assumptions
│   ├── download_data.py      fetches the raw data from Kaggle
│   ├── cleaning.py
│   ├── features.py
│   ├── modelling.py
│   └── evaluation.py
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   └── assets/
├── models/                   final_model.joblib, model_metadata.json, feature_schema.json
├── docs/
│   ├── business_problem.md
│   ├── dataset_search_evidence.md
│   ├── data_dictionary.md
│   ├── cleaning_log.md
│   ├── model_card.md
│   ├── milestones.md
│   ├── search_evidence/
│   └── figures/
└── reports/                  group report and individual assessments
```

## CRISP-DM mapping

| CRISP-DM phase | Where |
|---|---|
| Business Understanding | `docs/business_problem.md` |
| Data Understanding | `docs/dataset_search_evidence.md`, `docs/data_dictionary.md`, `notebooks/01_data_understanding.ipynb` |
| Data Preparation | `notebooks/02_data_cleaning.ipynb`, `notebooks/03_eda_feature_engineering.ipynb`, `docs/cleaning_log.md` |
| Modeling | `notebooks/04_model_selection.ipynb` |
| Evaluation | `notebooks/05_tuning_evaluation.ipynb` |
| Deployment | `app/`, `docs/model_card.md` |

## How to run

Requires Python 3.11 or later (the deployed app runs on Python 3.12). From the repository root:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the pinned dependencies
pip install -r requirements.txt

# 3. Download the raw data into data/raw/
python src/download_data.py

# 4. Run the notebooks in order (01 to 05), interactively or headless
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_understanding.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_tuning_evaluation.ipynb

# 5. Start the app
streamlit run app/streamlit_app.py
```

All paths are resolved through `src/config.py`, and `RANDOM_STATE = 42` is used throughout, so the results are reproducible on any machine.

## Results summary

> **Placeholder.** This table will be completed from `docs/model_comparison.csv` and `models/model_metadata.json` after modelling and tuning (notebooks 04 and 05). No results have been recorded yet.

| Model | ROC-AUC | PR-AUC | Recall (defaults) | Precision (defaults) | Approval rate | Reduction in expected loss vs approve-everything |
|---|---|---|---|---|---|---|
| Dummy (prior) baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| Logistic regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Random forest | TBD | TBD | TBD | TBD | TBD | TBD |
| HistGradientBoosting | TBD | TBD | TBD | TBD | TBD | TBD |
| LightGBM (tuned, final) | TBD | TBD | TBD | TBD | TBD | TBD |

| Objective | Target | Achieved |
|---|---|---|
| ROC-AUC on test set | at least 0.80 | TBD |
| Recall on defaults | at least 0.70 | TBD |
| Reduction in expected loss | at least 25% at an approval rate of at least 75% | TBD |
| App response time | under 3 seconds per loan | TBD |

## Licence and citation

The data are used under CC BY-SA 4.0. Please cite the original source:

Li, M., Mickel, A. and Taylor, S. (2018) '"Should This Loan be Approved or Denied?": A Large Dataset with Class Assignment Guidelines', *Journal of Statistics Education*, 26(1), pp. 55-66. doi:10.1080/10691898.2018.1434342.
