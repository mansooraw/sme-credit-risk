# Project Milestones

The project runs over five milestones from Week 10 to Week 14, one for each CRISP-DM phase or pair of phases. Each milestone has a lead from the role split agreed at the start, and the files listed are the evidence that the milestone was delivered. Status is updated at the end of each week.

| Milestone | Week | CRISP-DM phase | Lead | Delivered | Evidence in the repository | Status |
|---|---|---|---|---|---|---|
| M1 Business understanding | 10 | Business Understanding | Mansoor Mohamed Mansoor Saeed Alwahedi | Business context, decision to support, stakeholders, three SMART objectives, ML task definition with target and leakage rule, success criteria and risks; repository set up with shared configuration | `docs/business_problem.md`, `src/config.py`, `README.md` | Complete |
| M2 Data collection and understanding | 11 | Data Understanding | Salem Adel Salem Alshamlan Almarri | Dataset search across six sources with a dated log, requirement check, provenance and licence, download script, 5,000-row sample, data dictionary with observed quality issues, profiling notebook | `docs/dataset_search_evidence.md`, `docs/search_evidence/`, `docs/data_dictionary.md`, `src/download_data.py`, `data/sample/sba_sample_5000.csv`, `notebooks/01_data_understanding.ipynb` | Documents complete; notebook 01 in progress |
| M3 Data preparation | 12 | Data Preparation | Salem Adel Salem Alshamlan Almarri (cleaning), Sultan Zayed Ahmed Saeed Alkindi (EDA and features) | Cleaning module with before/after quality table, cleaned parquet file, EDA figures, engineered features with univariate tests, mutual information and ablation | `src/cleaning.py`, `notebooks/02_data_cleaning.ipynb`, `docs/cleaning_log.md`, `src/features.py`, `notebooks/03_eda_feature_engineering.ipynb`, `docs/figures/`, `models/feature_schema.json` | Planned |
| M4 Modelling and evaluation | 13 | Modeling, Evaluation | Mansoor Mohamed Mansoor Saeed Alwahedi | Five models compared on identical folds, class-weighting test, tuning, cost-based threshold, SHAP explanations, error analysis and temporal validation, saved final pipeline | `src/modelling.py`, `src/evaluation.py`, `notebooks/04_model_selection.ipynb`, `notebooks/05_tuning_evaluation.ipynb`, `docs/model_comparison.csv`, `models/final_model.joblib`, `models/model_metadata.json` | Planned |
| M5 Deployment | 14 | Deployment | Sultan Zayed Ahmed Saeed Alkindi | Public Streamlit app with single-loan scoring, batch scoring, threshold explorer, model insights and model card; pinned requirements | `app/streamlit_app.py`, `app/pages/`, `docs/model_card.md`, `requirements.txt` | Planned |

## Link to the SMART objectives

| Objective | Deadline | Milestone that delivers it |
|---|---|---|
| 1. Classifier with ROC-AUC of at least 0.80 and recall on defaults of at least 0.70 on the held-out test set | Week 13 | M4 |
| 2. At least 25% lower expected charge-off loss than approve-everything, at an approval rate of at least 75% | Week 14 | M4, reported in M5 |
| 3. Public Streamlit app returning a scored decision with an explanation in under 3 seconds per loan | Week 14 | M5 |

## Working arrangements

Mansoor, as team lead, administers the single shared repository on his GitHub account and integrates the group report. Each member writes the files they lead; the lead commits each stage in build order with a message that names its content, and the header of every notebook and document names its author, so the repository shows who did what. Results quoted in the report are taken from the saved tables in `docs/` rather than retyped.
