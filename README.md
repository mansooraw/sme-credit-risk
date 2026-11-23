# CreditLens: SME Loan Default Risk Scoring

CreditLens is a decision-support tool for a bank's SME credit desk lending under a government loan-guarantee scheme. Under such a scheme the guarantor covers part of each loan and the bank carries the rest, so a charged-off loan costs the bank the unguaranteed principal plus workout cost, while a wrongly declined good borrower costs the bank its interest margin and costs the economy the jobs the loan would have supported. Using 899,164 US Small Business Administration (SBA) 7(a) loan records, we build a classifier that estimates the probability of default at application time and recommends whether to approve, refer for manual review or decline, together with the expected value of the loan and an explanation of the score. The same mechanism operates in the UAE through the Emirates Development Bank credit guarantee scheme and the Khalifa Fund, and the tool is framed for a UAE bank that would retrain it on its own portfolio.

Course project for CDS4003 Data Science for Business Application (HCT), organised around the six CRISP-DM phases.

**Live app:** https://creditlens-sme.streamlit.app

## Team

| Member | Role | GitHub |
|---|---|---|
| Mansoor Mohamed Mansoor Saeed Alwahedi (team lead) | Business understanding, modelling and tuning (notebooks 04, 05), report integration, repository admin | [mansooraw](https://github.com/mansooraw) |
| Salem Adel Salem Alshamlan Almarri | Data collection, data understanding and cleaning (notebooks 01, 02), data dictionary, dataset search evidence | shared repository (see note) |
| Sultan Zayed Ahmed Saeed Alkindi | EDA and feature engineering (notebook 03), Streamlit app and deployment, figures | shared repository (see note) |

The team works in this single repository on the team lead's account. Each commit covers one stage of the project and its message names the content; the header of every notebook and document names its author.

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

Test split: 179,191 held-out loans (20% stratified); cross-validation on the 716,761 training loans. Full tables are in `docs/model_comparison.csv`, `docs/tuning_before_after.csv` and `docs/policy_comparison.csv`; the interpretation is in `docs/evaluation_summary.md` and `docs/model_card.md`.

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Recall (defaults) | Precision (defaults) | Brier score |
|---|---|---|---|---|---|---|
| Dummy (prior) baseline | 0.500 | 0.500 | 0.176 | 0.000 | 0.000 | 0.145 |
| Logistic regression | 0.863 | 0.865 | 0.628 | 0.820 | 0.432 | 0.151 |
| Random forest | 0.955 | 0.958 | 0.865 | 0.679 | 0.890 | 0.056 |
| HistGradientBoosting | 0.973 | 0.974 | 0.908 | 0.803 | 0.883 | 0.041 |
| LightGBM (default settings) | 0.975 | 0.975 | 0.907 | 0.807 | 0.877 | 0.041 |
| **LightGBM (tuned, final)** | **0.980** | **0.981** | **0.926** | **0.926** at 0.22 | **0.782** at 0.22 | **0.035** |

Recall and precision for the comparison models are at a 0.5 cut-off; the final model is reported at its business threshold of 0.22, chosen on out-of-fold training predictions to maximise portfolio net value with the approval rate kept at 75% or above. Tuning raised the LightGBM test ROC-AUC from 0.9748 to 0.9807 and cut the Brier score from 0.0412 to 0.0352.

| Policy on the test portfolio | Approval rate | Default rate among approved | Net value (USD m) | Charge-off loss (USD m) |
|---|---|---|---|---|
| Approve every loan | 100% | 17.6% | 733.7 | 864.4 |
| Logistic regression at its own threshold (0.62) | 77.4% | 6.5% | 1,094.6 | 327.1 |
| LightGBM (tuned) at 0.22 | 79.2% | 1.64% | 1,413.5 | 94.4 |

| Objective | Target | Achieved |
|---|---|---|
| ROC-AUC on test set | at least 0.80 | 0.981 |
| Recall on defaults | at least 0.70 | 0.926 at the 0.22 threshold |
| Reduction in charge-off loss | at least 25% at an approval rate of at least 75% | 89.1% at a 79.2% approval rate (USD 770 m avoided on the test portfolio) |
| App response time | under 3 seconds per loan | about 0.2 seconds per loan once the model is cached (about 1 second including the explanation and similar-loan search) |

Temporal validation (train on fiscal years up to 2005, test on 2006 onwards) gives ROC-AUC 0.968 (against 0.981 on the random split) and shows the model under-predicts the 2006 to 2008 vintages, which is why the model card asks for monitoring and periodic retraining.

## App

CreditLens is deployed as a multipage Streamlit app (CRISP-DM Deployment phase) that works on a desktop screen and on a phone. The entry point is `app/streamlit_app.py`. Shared loading, cleaning, scoring, expected-value, explanation and chart code is in `app/utils.py` and reuses `src/cleaning.py` and `src/features.py`, so a loan entered in the app goes through exactly the same steps as the training data. `app/analytics.py` holds the pure computations behind the interactive pages (similar loans, objective thresholds, Monte Carlo, stress test, the game and shareable case links) and is unit-tested on its own; `app/experience.py` holds the presentation layer (dark mode, the animated hero, the auto-play strip and the guided tour). The visual design is in `app/assets/theme.css` and `app/assets/theme_dark.css`, and every chart uses one Plotly template defined in `app/utils.py` that follows the dark-mode toggle.

**Run locally** from the repository root, after installing `requirements.txt`:

```bash
streamlit run app/streamlit_app.py
```

The app loads the final artefacts from `models/`. If they are not present it falls back to an interim development model in `app/_dev/` (git-ignored), which `python app/build_assets.py interim` creates, and shows a small notice. The same script rebuilds the committed files in `app/assets/`: cleaning parameters, sample files and templates (`raw`, `samples`) and the aggregate files that let the explorer pages run without `data/processed` (`aggregates`, documented in [docs/deployment_guide.md](docs/deployment_guide.md), section 6). No number in the app is typed in: each comes from the model metadata, a committed aggregate or a computation on the scored data.

The sidebar groups the pages in four sections and keeps the threshold, the dark-mode choice and the drill-down filters while you move between pages. Charts with a "click to filter" hint are clickable, every filter also has a tap-friendly selector, and every tile has an "i" control that explains how its number is computed.

| Section | Page | What it shows |
|---|---|---|
| Decide | Home | Full-width hero with an animated flow of loans passing a scoring gate in the model's approve, refer and decline proportions, glass metric tiles that count up, a Watch it score strip cycling through three applicants, a guided tour of five pages, how it works, the decision rules and the cost matrix |
| Decide | Score a loan | Validated form with example cases and shareable links (`?case=`); a short scoring sequence, PD gauge, band, decision, expected value, SHAP drivers, an applicant radar against the training portfolio, the ten most similar training loans and their outcomes, a what-if panel, and a decision memo plus a print-ready (PDF) memo |
| Decide | Compare applicants | Two or three applicants in an editable table that rescores on every change, a PD bar chart, side-by-side gauges, the top drivers per applicant, a recommendation table, Copy one applicant into another and a shareable link (`?a=&b=&c=`) |
| Decide | Beat the model | A game on five real held-out loans: approve or decline, the card turns over to show the model's PD and the real outcome, a running score and a final summary of accuracy and value under the cost matrix |
| Decide | Batch scoring | CSV upload in two layouts, validation, cleaning and scoring; summary tiles, a clickable PD histogram that filters the table, band and decision donuts, a sortable and filterable table whose amount, guarantee, term and sector cells can be edited and rescored, the scored CSV and a portfolio memo |
| Portfolio | Portfolio and threshold | Threshold slider and objective chips (maximum net value, best with 75% approved, stop 90% of defaults), animated net-value curve, confusion matrix, a clickable state map that filters the sector treemap |
| Portfolio | Stress test | Recession on every loan, loss given default, margin and guarantee shift applied to the 20,000-loan test sample; animated KPI deltas, net value by threshold before and after, expected loss by sector and the break-even LGD of each sector |
| Portfolio | Monte Carlo | 1,000 simulated portfolio years at the chosen threshold (each approved loan defaults with its PD): histogram with the 5th, 50th and 95th percentiles, the chance of a loss-making year and a fan chart across thresholds |
| Explore | Market explorer | Animated map of default rates by state and approval year, sector bars, a sector, business age and outcome sunburst, a term by guarantee heatmap and a loan-volume timeline with the recession shaded; click a state or sector to filter everything, with an insight line under each chart |
| Explore | Bank league | Searchable league table of 1,245 lenders, a clickable default rate against volume scatter and a profile card with an approvals sparkline |
| Model | Model lab | ROC, precision-recall and calibration curves for the final and comparison models with a threshold marker (click the ROC curve to set it), the tuning slope chart and why LightGBM was chosen, computed from the metadata |
| Model | Model insights | Metrics, calibration, ROC, business results, global SHAP importance, model comparison, tuning and temporal validation |
| Model | Model card and about | Model card, team, course, data provenance, licence and how to add the app to a phone home screen |

Smoke tests for every page and the interactive logic run with `python -m pytest tests/test_app_smoke.py -q`.

| Desktop | Phone |
|---|---|
| ![Home](docs/figures/app_home.png) | ![Home on a phone](docs/figures/app_mobile_home.png) |
| ![Score a loan](docs/figures/app_score_loan.png) | ![Score a loan on a phone](docs/figures/app_mobile_score_loan.png) |
| ![Compare applicants](docs/figures/app_compare.png) | ![Compare applicants on a phone](docs/figures/app_mobile_compare.png) |
| ![Beat the model](docs/figures/app_beat_model.png) | ![Beat the model on a phone](docs/figures/app_mobile_beat_model.png) |
| ![Batch scoring](docs/figures/app_batch_scoring.png) | ![Batch scoring on a phone](docs/figures/app_mobile_batch_scoring.png) |
| ![Portfolio and threshold](docs/figures/app_portfolio_threshold.png) | ![Portfolio and threshold on a phone](docs/figures/app_mobile_portfolio_threshold.png) |
| ![Stress test](docs/figures/app_stress_test.png) | ![Stress test on a phone](docs/figures/app_mobile_stress_test.png) |
| ![Monte Carlo](docs/figures/app_monte_carlo.png) | ![Monte Carlo on a phone](docs/figures/app_mobile_monte_carlo.png) |
| ![Market explorer](docs/figures/app_market_explorer.png) | ![Market explorer on a phone](docs/figures/app_mobile_market_explorer.png) |
| ![Bank league](docs/figures/app_bank_league.png) | ![Bank league on a phone](docs/figures/app_mobile_bank_league.png) |
| ![Model lab](docs/figures/app_model_lab.png) | ![Model lab on a phone](docs/figures/app_mobile_model_lab.png) |
| ![Model insights](docs/figures/app_model_insights.png) | ![Model insights on a phone](docs/figures/app_mobile_model_insights.png) |
| ![Model card](docs/figures/app_model_card.png) | ![Model card on a phone](docs/figures/app_mobile_model_card.png) |

## Licence and citation

The data are used under CC BY-SA 4.0. Please cite the original source:

Li, M., Mickel, A. and Taylor, S. (2018) '"Should This Loan be Approved or Denied?": A Large Dataset with Class Assignment Guidelines', *Journal of Statistics Education*, 26(1), pp. 55-66. doi:10.1080/10691898.2018.1434342.
