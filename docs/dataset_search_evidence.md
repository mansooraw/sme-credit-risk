# Dataset Search Evidence

**CRISP-DM phase:** Data Understanding, data collection (Milestone 2)
**Lead:** Salem Adel Salem Alshamlan Almarri
**Search period:** Weeks 10 and 11 (26 October to 6 November 2026)

## 1. Requirements

The instructor's rules (15 October 2026) set the minimum requirements for the project dataset. We added two of our own: the data had to match the business problem in `docs/business_problem.md` (credit decisions on SME loans) and it had to be loan-level rather than aggregated.

| Requirement | Source |
|---|---|
| At least 200,000 rows | Instructor's rules |
| At least 20 columns | Instructor's rules |
| Unclean, with real data quality issues | Instructor's rules |
| A clear target variable for a business decision | Instructor's rules |
| Mix of numeric, categorical, date and text columns | Instructor's rules |
| Licence allows academic use and redistribution of results | Team requirement |
| Not used by any team member in previous coursework | Instructor's rules |
| Loan-level SME credit outcomes | Team requirement |

## 2. Search log

Screenshots of each search are kept in `docs/search_evidence/`:

| File | What it shows |
|---|---|
| `01_kaggle_sba_dataset_page.png` | The accepted dataset's Kaggle page: 899,164 rows, 27 columns, 179 MB, licence CC BY-SA 4.0 |
| `02_kaggle_search_sme_loan.png` | Kaggle dataset search for "small business loan default" |
| `03_uci_search_credit.png` | UCI Machine Learning Repository search for "credit" (all candidates under 200,000 rows) |
| `04_google_dataset_search.png` | Google Dataset Search for "sme loan default dataset" (mostly aggregated rate tables) |
| `05_bayanat_search.png` | Bayanat.ae search for "loan": a single aggregated Abu Dhabi agricultural loans table |
| `06_data_dubai_login_required.png` | data.dubai (formerly Dubai Pulse) redirecting to a UAE PASS login before any dataset can be reached |
| `07_huggingface_dubai_mirror.png` | The only public mirror of the Dubai Land Department transactions, 18 columns and already cleaned |
| `08_jse_paper_doi.png` | The open-access source paper (Li, Mickel and Taylor, 2018) for the accepted dataset |

| Date | Source checked | Search terms used | Candidate found | Rows | Columns | Licence | Decision | Reason |
|---|---|---|---|---|---|---|---|---|
| 26 Oct 2026 | Bayanat.ae | "SME", "loans", "credit", "banking" | Banking and SME statistics tables | Far under 200,000 | Under 20 | UAE open data terms | Rejected | Aggregated statistics, not loan-level records |
| 27 Oct 2026 | Dubai Pulse / data.dubai | "loans", "real estate transactions", "mortgage" | Dubai Land Department transactions | Not checked (download blocked) | 18 in the public mirror | Dubai open data terms | Rejected | Download needs a UAE Pass login; the only public mirror has 18 columns and is already cleaned |
| 28 Oct 2026 | UCI Machine Learning Repository | "credit", "loan", "default", "bank" | Bank Marketing | 45,211 | 17 | CC BY 4.0 | Rejected | About 45k rows, below the 200k minimum; target is a term-deposit subscription, not credit risk |
| 28 Oct 2026 | UCI Machine Learning Repository | "credit default" | Default of Credit Card Clients | 30,000 | 24 | CC BY 4.0 | Rejected | 30k rows; consumer credit cards, not SME loans |
| 29 Oct 2026 | Kaggle | "lending club", "loan default" | Lending Club accepted loans 2007 to 2018 | 2,260,701 | 151 | Kaggle mirror of LendingClub data | Rejected | Consumer lending, not SME; 151 columns, most of them sparse; widely reused in coursework |
| 29 Oct 2026 | Kaggle | "home credit", "credit risk" | Home Credit Default Risk | 307,511 (main table) | 122 (main table) | Kaggle competition rules | Rejected | Competition rules restrict use; data split across several linked tables |
| 30 Oct 2026 | Kaggle | "vehicle loan default" | L&T Vehicle Loan Default Prediction | 233,154 | 41 | Unclear (hackathon data) | Rejected | Meets the size rule, but the licence for reuse is unclear |
| 2 Nov 2026 | HuggingFace Datasets | "credit risk", "loan default", "SME" | Copies of German Credit and Lending Club | 1,000 (German Credit) | 21 (German Credit) | Varies by uploader | Rejected | No new loan-level SME source; uploads repeat datasets already reviewed or are too small |
| 3 Nov 2026 | Google Dataset Search | "SBA loan data", "small business loan default" | SBA 7(a) loan records, paper by Li, Mickel and Taylor (2018) and its Kaggle copy | 899,164 | 27 | CC BY-SA 4.0 (Kaggle), CC BY 4.0 (paper) | Followed up | Led to the documented SBA dataset below |
| 4 Nov 2026 | Kaggle | "should this loan be approved or denied", "SBA" | "Should This Loan be Approved or Denied?" (`SBAnational.csv`) | 899,164 | 27 | CC BY-SA 4.0 | **Accepted** | Meets every requirement; see section 3 |

## 3. Requirement check for the accepted dataset

| Requirement | Result | Evidence |
|---|---|---|
| At least 200,000 rows | Met | 899,164 rows |
| At least 20 columns | Met | 27 columns |
| Target variable | Met | `MIS_Status` (`CHGOFF` or `P I F`), 897,167 rows labelled |
| Mix of types | Met | Numeric (`Term`, `NoEmp`), categorical (`State`, `NAICS`, `LowDoc`), dates (`ApprovalDate`), currency stored as text (`GrAppv`), free text (`Name`, `Bank`) |
| Unclean with real issues | Met | Currency and dates stored as text, invalid codes in `RevLineCr` (272,951 rows) and `LowDoc` (3,425 rows), 201,948 rows with `NAICS` of 0; see `docs/data_dictionary.md` |
| Licence allows academic use | Met | CC BY-SA 4.0 on Kaggle; the source paper is CC BY 4.0 |
| Not used in previous coursework | Met | Confirmed by all three team members |
| Loan-level SME credit outcomes | Met | One row per SBA-guaranteed small business loan |

## 4. Provenance

The data are loan records from the US Small Business Administration's 7(a) loan guarantee programme. They were compiled and published by Min Li, Amy Mickel and Stanley Taylor of California State University, Sacramento, with the open-access paper "Should This Loan be Approved or Denied?": A Large Dataset with Class Assignment Guidelines (*Journal of Statistics Education*, 26(1), 55-66, 2018, doi:10.1080/10691898.2018.1434342). The paper is published under CC BY 4.0 and describes the 27 variables; a copy is in `docs/Li_Mickel_Taylor_2018_JSE_SBA_dataset_paper.pdf`. The paper describes the data as covering 1987 to 2014, restricted to the 50 states and Washington DC.

We use the copy uploaded to Kaggle by Mirbek Toktogaraev under the same title (https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied), licensed CC BY-SA 4.0. Under that licence we credit the source and any derived data we share carries the same licence.

## 5. How to obtain the data

The raw file is 179 MB, above GitHub's 100 MB file limit, so it is not committed and `data/raw/` is listed in `.gitignore`. From the repository root, run:

```
python src/download_data.py
```

The script uses `kagglehub` to download the public dataset (no Kaggle API key is needed) and copies `SBAnational.csv` to `data/raw/`. If the file is already present it does nothing. A 5,000-row random sample with the same 27 columns is committed at `data/sample/sba_sample_5000.csv` for quick inspection without the download.

## 6. Size and structure

| Property | Value |
|---|---|
| File | `SBAnational.csv`, 179,430,516 bytes |
| Rows | 899,164 (one per loan, `LoanNr_ChkDgt` unique) |
| Columns | 27 |
| Exact duplicate rows | 0 |
| Labelled rows | 897,167 (1,997 with missing `MIS_Status`) |
| Class balance | 157,558 `CHGOFF` (17.56%), 739,609 `P I F` (82.44%) |
| Approval fiscal years | 1962 to 2014 |
| Borrower states | 51 (50 states and DC) |

Every column is read as text first, because currency values such as `$60,000.00 ` and dates such as `28-Feb-97` need explicit parsing. The full column list, with types, meanings and observed issues, is in `docs/data_dictionary.md`.

## 7. Known limitations and bias risks

**US only.** All borrowers are in the United States. Default behaviour depends on the legal system, the guarantee rules and the economy, so the model describes US SME lending under the SBA programme. Before a UAE bank used it under the Emirates Development Bank credit guarantee scheme, it would have to be retrained and validated on local loan outcomes.

**Historical period.** The loans were approved between fiscal years 1962 and 2014, and 10,599 loans were approved before 1987. The most recent approval is June 2014, so the data says nothing about lending after that date.

**Survivorship and right-censoring.** The dataset keeps loans whose outcome is known. Loans approved near the end of the period can only be labelled paid in full if they were repaid early, while defaults surface quickly. The default rate is 34.2% for fiscal years 2005 to 2007 and 6.6% for 2011 to 2014, a gap that reflects censoring as well as credit quality. Li, Mickel and Taylor (2018) give the same warning and restrict their own case study to loans disbursed up to 2010.

**Policy changes.** Guarantee shares, the LowDoc programme and eligibility rules changed over the period, and the 2007 to 2009 recession changed default rates sharply. A pattern that held in one policy regime may not hold in another.

**Coding changes.** Some columns were coded differently over time. For example, `RevLineCr` contains 257,602 rows coded `0` and 15,284 coded `T` instead of Y or N, and `NAICS` is 0 for 201,948 rows. These codes may track periods of data entry rather than borrower properties.

**Fairness.** The data contains no protected personal attributes, but borrower state, city and sector could act as proxies for groups. We will report model errors by sector and state before recommending any use.
