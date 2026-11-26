# Deployment guide: CreditLens on Streamlit Community Cloud

**CRISP-DM phase:** Deployment
**Author:** Sultan Zayed Ahmed Saeed Alkindi

This guide explains how we deploy the CreditLens app directly from the GitHub repository to Streamlit Community Cloud, what we check after each deployment and how we update the app.

## 1. Before deploying

The app reads everything it needs from the repository, so the following must be committed on the `main` branch:

| File or folder | Why the app needs it |
|---|---|
| `app/streamlit_app.py`, `app/pages/` (13 pages), `app/utils.py`, `app/analytics.py`, `app/ui.py`, `app/experience.py`, `app/assets/` | The app itself: shared loading and scoring (`utils.py`), the pure computations behind the interactive pages (`analytics.py`: shareable case links, similar loans, objective thresholds, Monte Carlo, stress test, game logic) and the presentation layer (`experience.py`: dark mode, hero animation, auto-play strip, guided tour) |
| `app/assets/` | Theme (`theme.css`, `theme_dark.css`), logos, cleaning parameters, CSV templates, the 50-loan sample files and `test_sample_scored.csv` (20,000 held-out loans used by the portfolio, stress test, Monte Carlo, game and model pages, because `data/processed/` is not in the repository) |
| `app/assets/aggregates/` | Small summaries of the full data for the Market explorer, Bank league and Score a loan pages (section 6) |
| `src/` | Cleaning and feature functions shared with the notebooks |
| `models/final_model.joblib`, `models/model_metadata.json`, `models/feature_schema.json`, `models/shap_global.csv` | The model, its threshold, metrics and the form schema |
| `requirements.txt` | Pinned library versions installed by Streamlit Cloud |
| `.streamlit/config.toml` | Theme and server settings |

`app/_dev/` holds an interim development model and is git-ignored. If the files in `models/` are missing, the deployed app cannot find a model and shows an error, so we confirm that the final model files are committed (`git ls-files models`) before deploying. Everything under `models/` must stay under 100 MB in total.

We test locally first from the repository root:

```bash
python -m pytest tests/test_app_smoke.py -q
streamlit run app/streamlit_app.py
```

## 2. Deploy

1. Push the latest commit to the `main` branch of the GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub. The first time, authorise Streamlit to read the repository (for a private repository, grant access to it explicitly).
3. Click **Create app** (shown as **New app** on older versions of the dashboard) and choose to deploy from an existing GitHub repository.
4. Fill in the form:
   - **Repository:** `mansooraw/sme-credit-risk`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** choose a short subdomain, for example `creditlens`.
5. Open **Advanced settings** and set **Python version** to **3.12**. No secrets are needed: the app uses no API keys or passwords.
6. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes; the log is visible on the right of the page.
7. Copy the app URL into `README.md` and into the report cover page. The live app is at https://creditlens-sme.streamlit.app.

Streamlit Cloud runs the app from the repository root, so `app/streamlit_app.py` adds the repository root to `sys.path` and `from src import ...` works without any extra configuration.

## 3. What to check after deploying

| Check | Expected result |
|---|---|
| Overview opens | The glass page bar shows the model, the held-out loan count and the approve-below pill; the title block, the figure strip, the chart, the decision policy panel and the three example applications appear; no "interim development model" notice (that notice means the app did not find `models/final_model.joblib`) |
| Navigation | The sidebar groups the pages under Decide, Portfolio, Explore and Model; the active page is the emerald pill |
| Sidebar status card | Says "Model in service" and shows the final model name, training date and thresholds from `model_metadata.json` |
| Score a loan | The borderline example is scored on arrival: the result panel shows the probability, decision and band badges and the three values; the drivers card lists the top inputs and reports a scoring time under 3 seconds; the what-if sheet opens; both memos download and Share link gives a link that reopens the case |
| Compare applicants | Three applicants appear side by side with the best one tinted; Edit the inputs shows the table and editing a cell rescores it; Copy copies one applicant into another |
| Decision exercise | Approve or Decline reveals the model's probability and the recorded outcome; after five loans the panel shows the round summary, and New round draws five new loans |
| Validation | Enter a guaranteed amount above the approved amount: the app refuses to score and explains why |
| Batch scoring | Switch on the 50-loan sample file: 50 rows pass validation; the figure strip, the histogram and the portfolio summary panel appear; clicking a histogram bar filters the table; Edit rows shows the table and editing a term rescores that loan; the scored CSV and the portfolio memo download |
| Portfolio and threshold | The objective rows and the slider update the figures, the curve, the confusion matrix panel, the state map and the sector treemap (the map outline is fetched by the browser from the Plotly CDN); clicking a state filters the treemap |
| Stress test, Monte Carlo | The sliders rescore the sample and update the tiles and charts within a second or two |
| Market explorer, Bank league | Run animates the map; clicking a state or a bar filters the figures and the top states panel; clicking a lender bar selects that lender and shows its profile |
| Model lab | ROC, precision-recall and calibration curves for both models; clicking the ROC curve moves the threshold |
| Phone | Open the link on a phone: the sidebar is collapsed, tiles sit two to a row and the form and results stack |
| Model insights and Model card | Tables and charts load without errors |
| Logs | **Manage app** (bottom right) shows no tracebacks |

## 4. Updating the app

Streamlit Community Cloud redeploys automatically when a new commit reaches the deployed branch. To update the model we run notebooks 04 and 05, which overwrite the files in `models/`, then rebuild the committed sample with `python app/build_assets.py samples` so its stored probabilities match the new model (and `python app/build_assets.py aggregates` if the cleaned data changed) (the app rescores the sample on load in any case), run the smoke tests and push. Changes to `requirements.txt` trigger a full reinstall. If the app does not pick up a change, use **Manage app**, then **Reboot app**. The Python version can only be changed by deleting and redeploying the app.

## 5. Resource limits

Community Cloud apps share modest resources. At the time of writing each app receives roughly 1 GB of memory (up to about 2.7 GB is quoted as the ceiling), shared CPU, and limited disk; the exact figures are set by Streamlit and can change, so we check the current documentation at https://docs.streamlit.io/deploy/streamlit-community-cloud. Apps that receive no traffic for several days go to sleep and wake up when someone opens the link, which takes up to a minute. Files uploaded to the app are limited to 50 MB in our configuration (`.streamlit/config.toml`).

We keep CreditLens inside these limits by deploying a LightGBM model of a few hundred trees rather than a large random forest, by using the 20,000-loan test sample instead of the 38 MB feature file, and by caching the model and metadata with `st.cache_resource` and the scored portfolio with `st.cache_data`, so they are loaded once per container rather than once per visitor.

## 6. Aggregate files

The explorer pages need statistics of all 895,952 cleaned loans, but `data/processed/` is not in the repository. `python app/build_assets.py aggregates` therefore writes small, precomputed summaries to `app/assets/aggregates/` from `data/processed/loans_features.parquet` and the saved test split. They are read with `st.cache_data`, contain counts and sums only (no identifiers, names or outcomes of single loans except the anonymous reference sample), and each is far below the 5 MB limit we set for committed assets. Rebuild them whenever the cleaned data changes.

| File | Rows (approx.) | Size | Contents | Used by |
|---|---|---|---|---|
| `market_state_sector_year.parquet` | 21,000 | 0.26 MB | Loans, defaults, disbursed and guaranteed USD by borrower state, sector and approval year | Market explorer: animated map, sector bars, timeline |
| `market_state_sector_age.parquet` | 2,400 | 0.03 MB | Loans and defaults by state, sector and business age (New, Existing, Unknown) | Market explorer: sunburst and its insight line |
| `market_state_sector_term_portion.parquet` | 32,000 | 0.21 MB | Loans and defaults by state, sector, term band and guaranteed-share band | Market explorer: term by guarantee heatmap |
| `bank_league.parquet` | 1,245 | 0.08 MB | One row per lender with at least 50 loans: loans, defaults, default rate, average and total disbursed, mean guaranteed share, same-state share, average term, home state, states served, first and last year | Bank league: table, scatter and profile card |
| `bank_years.parquet` | 26,000 | 0.17 MB | Loans by lender and approval year for the same lenders | Bank league: approvals sparkline |
| `reference_sample.parquet` | 50,000 | 0.88 MB | Random sample of training loans (none from the test split) with the numeric model inputs, outcome, disbursed amount, term, sector, state and business age | Score a loan: applicant radar percentiles and the ten most similar loans |
| `manifest.json` | | < 0.01 MB | Build date, source, band labels, minimum loans per lender and the list of files | Captions and band order |

The whole `app/assets/` folder is about 4.7 MB. The scored-sample file `test_sample_scored.csv` is rebuilt with `python app/build_assets.py samples` after the model changes; every page rescored the sample with the loaded model in any case, and all numbers shown come from the metadata, these files or computations on them.

