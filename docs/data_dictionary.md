# Data Dictionary

**CRISP-DM phase:** Data Understanding (Milestone 2)
**Lead:** Salem Adel Salem Alshamlan Almarri
**Source file:** `data/raw/SBAnational.csv` (899,164 rows, 27 columns)

Meanings follow Table 1(a) of Li, Mickel and Taylor (2018). All counts below were computed on the full raw file, read with every column as text and only empty cells treated as missing. "Raw type" is how the value is stored in the CSV, followed by the dtype pandas infers with default settings. Dates were parsed with format `%d-%b-%y`, treating two-digit years above 26 as 19xx, as set out in the cleaning decisions. Notebook 01 must reproduce these counts.

## 1. Raw columns

| # | Column | Raw type (pandas default) | Meaning | Example | Observed issues | Role |
|---|---|---|---|---|---|---|
| 1 | `LoanNr_ChkDgt` | Integer text (int64) | Loan identifier with check digit, primary key | `1000014003` | 0 missing; 899,164 unique values, no duplicates | Identifier |
| 2 | `Name` | Text (object) | Borrower name | `ABC HOBBYCRAFT` | 3 missing; 779,586 unique; used with `City`, `State`, `ApprovalDate`, `GrAppv` to find 1,266 near-duplicate rows | Identifier |
| 3 | `City` | Text (object) | Borrower city | `EVANSVILLE` | 30 missing; 32,581 unique values, too many to encode | Dropped (used in near-duplicate check) |
| 4 | `State` | Two-letter code (object) | Borrower state | `IN` | 14 missing; 51 codes (50 states and DC) | Feature |
| 5 | `Zip` | Integer text (int64) | Borrower zip code | `47711` | 0 missing but 283 rows are 0; 94,590 values are not five digits because leading zeros were lost | Dropped (used to fill missing `State`) |
| 6 | `Bank` | Text (object) | Name of the lending bank | `FIFTH THIRD BANK` | 1,559 missing; 5,802 unique banks | Source of engineered features (`BankSize`, `BankDefaultRate`) |
| 7 | `BankState` | Two-letter code (object) | State of the lending bank | `OH` | 1,566 missing; 56 codes, including `AN`, `EN`, `GU`, `PR`, `VI`, which are not borrower states | Feature |
| 8 | `NAICS` | Integer text (int64) | North American Industry Classification System code (first two digits give the sector) | `451120` | 0 missing but 201,948 rows are 0 (industry unknown); 1,312 unique codes | Source of `Sector` |
| 9 | `ApprovalDate` | Date text `dd-Mon-yy` (object) | Date the SBA commitment was issued | `28-Feb-97` | 0 missing; two-digit years 00 to 99, so the century must be fixed; after the fix the range is 7 Dec 1961 to 25 Jun 2014 | Source of `ApprovalYear`, `ApprovalMonth`, `DisbursementLagDays` |
| 10 | `ApprovalFY` | Year text (object, mixed) | Fiscal year of the commitment | `1997` | 0 missing; 18 rows read `1976A`; range 1962 to 2014 after stripping non-digits | Temporal validation split only |
| 11 | `Term` | Integer (int64) | Loan term in months | `84` | 0 missing; 810 rows are 0 (impossible); min 0, median 84, max 569 | Feature |
| 12 | `NoEmp` | Integer (int64) | Number of business employees | `4` | 0 missing; 6,631 rows are 0; 4 rows are the placeholder 9999; 885 rows above the 99.9th percentile (300); max 9,999 | Feature |
| 13 | `NewExist` | Code (float64) | 1 = existing business, 2 = new business (two years or less) | `2` | 136 missing; 1,034 rows coded 0 (invalid) | Feature |
| 14 | `CreateJob` | Integer (int64) | Number of jobs created | `0` | 0 missing; 629,248 rows are 0; 870 rows above the 99.9th percentile (200); max 8,800 | Feature |
| 15 | `RetainedJob` | Integer (int64) | Number of jobs retained | `0` | 0 missing; 440,403 rows are 0; 900 rows above the 99.9th percentile (276.7); max 9,500 | Feature |
| 16 | `FranchiseCode` | Integer text (int64) | Franchise code; 00000 or 00001 means no franchise | `1` | 0 missing; 847,389 rows are 0 or 1 (no franchise, two codes for one meaning); 51,775 rows hold a franchise code; 2,768 unique values | Source of `IsFranchise` |
| 17 | `UrbanRural` | Code (int64) | 1 = urban, 2 = rural, 0 = undefined | `0` | 0 missing; 323,167 rows coded 0 (undefined) | Feature |
| 18 | `RevLineCr` | Code (object) | Revolving line of credit, Y or N | `N` | 4,528 missing; 272,951 rows hold codes other than Y or N: `0` (257,602), `T` (15,284) and 65 rows spread over `1`, `R`, `` ` ``, `2`, `C`, `,`, `3`, `7`, `A`, `5`, `.`, `4`, `-`, `Q` | Feature |
| 19 | `LowDoc` | Code (object) | LowDoc loan programme (one-page application), Y or N | `Y` | 2,582 missing; 3,425 rows hold invalid codes: `0` (1,491), `C` (758), `S` (603), `A` (497), `R` (75), `1` (1) | Feature |
| 20 | `ChgOffDate` | Date text `dd-Mon-yy` (object) | Date the loan was declared in default | `24-Jun-91` | 736,465 missing; present for 4,927 `P I F` loans and absent for 47 `CHGOFF` loans; one value (`22-Oct-26`) falls after the end of the data | Leakage (post-outcome) |
| 21 | `DisbursementDate` | Date text `dd-Mon-yy` (object) | Date the loan was disbursed | `28-Feb-99` | 2,368 missing; 809 rows before `ApprovalDate` (135 by more than 30 days); one value after 2014 (`14-Apr-20`) and two before 1960; median lag after approval 59 days | Source of `DisbursementLagDays`, `Recession` |
| 22 | `DisbursementGross` | Currency text `$60,000.00 ` (object) | Amount disbursed | `$60,000.00 ` | 0 missing; stored as text with `$`, commas and a trailing space; 196 rows are 0; max 11,446,325; 65,967 rows above twice `GrAppv` and 4,132 below half of it | Feature |
| 23 | `BalanceGross` | Currency text (object) | Gross amount outstanding | `$0.00 ` | 0 missing; 899,150 rows are 0 and only 14 are positive (max 996,262) | Leakage (post-outcome) |
| 24 | `MIS_Status` | Text (object) | Loan status: `CHGOFF` = charged off, `P I F` = paid in full | `P I F` | 1,997 missing; 157,558 `CHGOFF` and 739,609 `P I F` | Label (creates `Default`) |
| 25 | `ChgOffPrinGr` | Currency text (object) | Charged-off principal amount | `$0.00 ` | 0 missing; 162,012 rows positive, of which 4,884 are `P I F` loans and 257 have missing status; max 3,512,596 | Leakage (post-outcome) |
| 26 | `GrAppv` | Currency text (object) | Gross amount of loan approved by the bank | `$60,000.00 ` | 0 missing; min 200, max 5,472,000 | Feature |
| 27 | `SBA_Appv` | Currency text (object) | SBA's guaranteed amount of the approved loan | `$48,000.00 ` | 0 missing; min 100, max 5,472,000; 0 rows above `GrAppv`; guaranteed share median 0.75, mean 0.709 | Feature (and source of `Portion`) |

### Notes for notebooks 01 and 02

- No exact duplicate rows exist. The near-duplicate rule in the brief (same `Name`, `City`, `State`, `ApprovalDate`, `GrAppv`) flags 1,266 rows beyond the first occurrence.
- After dropping the 1,997 rows with missing status, 897,167 labelled rows remain, with a default rate of 17.56%.
- Default rate by code group, on labelled rows, gives the evidence needed for the `RevLineCr` and `LowDoc` mapping decisions: `RevLineCr` N 14.6% (419,252 rows), `0` 15.0% (257,431), Y 25.4% (200,660), `T` 43.1% (15,239), missing 2.5% (4,520); `LowDoc` N 18.7% (780,997), Y 9.0% (110,171), missing 38.7% (2,578), `0` 3.4% (1,490), `C` 9.9% (758), `S` 44.8% (603), `A` 32.1% (495), `R` 33.8% (74).
- Default rate by approval fiscal year band: up to 1989 27.9%, 1990 to 1994 2.5%, 1995 to 1999 6.0%, 2000 to 2004 14.1%, 2005 to 2007 34.2%, 2008 to 2010 29.9%, 2011 to 2014 6.6%. The low rate in the final band is consistent with right-censoring.
- 152,157 loans have `Term` of 240 months or more (`RealEstate` = 1).

## 2. Engineered features

These are defined in section 6 of the project brief and implemented in `src/features.py`. Notebook 03 must use these exact names and formulas.

| Feature | Formula | Type | Rationale |
|---|---|---|---|
| `Portion` | `SBA_Appv / GrAppv` | Float in (0, 1] | Share of the loan guaranteed by the SBA; sets the bank's exposure on default and enters the cost matrix |
| `RealEstate` | `Term >= 240` | Binary | Loans of 20 years or more are normally backed by real estate, which reduces loss on default |
| `Recession` | 1 if the period from `DisbursementDate` to `DisbursementDate + Term` months overlaps 2007-12-01 to 2009-06-30; `ApprovalDate` is used when `DisbursementDate` is missing | Binary | Loans active during the Great Recession faced a weaker economy |
| `NewBusiness` | 1 if `NewExist` = 2 (new business), 0 if existing or unknown | Binary | Businesses of two years or less have shorter track records |
| `IsFranchise` | 1 if `FranchiseCode` not in {0, 1} | Binary | Franchises operate a tested business model |
| `SameStateBank` | 1 if `BankState == State` (0 when `State` is Unknown) | Binary | A local lender may know the borrower and market better |
| `Sector` | First two digits of `NAICS` mapped to the 20 sector names in `src/config.py` (31-33 Manufacturing, 44-45 Retail trade, 48-49 Transportation and warehousing; 0 becomes Unknown) | Categorical | Default rates differ widely between industries |
| `LoanPerEmployee` | `DisbursementGross / max(NoEmp, 1)` | Float | Loan size relative to firm size, a proxy for indebtedness |
| `TermYears` | `Term / 12` | Float | Term in an interpretable unit |
| `ApprovalYear` | Year of `ApprovalDate` after the century fix | Integer | Captures the lending cycle and policy period |
| `ApprovalMonth` | Month of `ApprovalDate` | Integer (1 to 12) | Seasonality in applications |
| `DisbursementLagDays` | `DisbursementDate - ApprovalDate` in days; 0 when `DisbursementDate` is missing | Integer | Delays between approval and disbursement may signal problems |
| `DisbursementDateMissing` | 1 if `DisbursementDate` is missing (2,171 loans) and the approval date was used instead | Binary | Records the fallback used for `Recession` and `DisbursementLagDays` |
| `JobsPerDollar` | `(CreateJob + RetainedJob) / DisbursementGross * 1e5` | Float | Jobs supported per 100,000 dollars disbursed |
| `LogDisbursement` | `log(1 + DisbursementGross)` | Float | Reduces the right skew of loan amounts |
| `DisbursedToApprovedRatio` | `DisbursementGross / GrAppv` | Float | Large gaps between approved and disbursed amounts may indicate changes in the loan |
| `LowDoc` | Y, N, Unknown | Categorical | Low-documentation applications carry less information |
| `RevLineCr` | Y, N (including raw `0`), T, Unknown. The raw code `T`, merged into Y in cleaning, is kept as its own level because it defaults at 43.0% against 25.5% for Y in the training data and stays higher within approval years (notebook 03, section 5.3) | Categorical | Revolving credit lines behave differently from term loans |
| `UrbanRural` | Urban, Rural, Unknown | Categorical | Local economic conditions |
| `NewExist` | Existing, New, Unknown | Categorical | Business age, with the unknown code kept as its own level |
| `State` | Borrower state, missing filled from `Zip` or Unknown | Categorical | Regional economic differences |
| `BankState` | Bank state, missing as Unknown | Categorical | Regional lender effects |
| `BankSize` | Number of loans per `Bank` in the training data; 0 for a bank not seen in training | Integer (frequency encoding) | Lender scale and experience with SBA lending |
| `BankDefaultRate` | Default rate per `Bank`, smoothed towards the global rate with weight 20 loans, computed out of fold (5 folds) on the training data only by `features.BankEncoder`; unseen banks get the global training rate | Float (target encoding) | Lender underwriting quality; computing it on the training fold only stops test labels leaking into the feature |

The cleaned columns `NoEmp`, `CreateJob` and `RetainedJob` are also model inputs. `Term`, `DisbursementGross`, `GrAppv`, `SBA_Appv`, `NAICS` and `FranchiseCode` enter only through the engineered features above (the full list is `FEATURE_COLUMNS` in `src/features.py` and `models/feature_schema.json`).

## Reference

Li, M., Mickel, A. and Taylor, S. (2018) '"Should This Loan be Approved or Denied?": A Large Dataset with Class Assignment Guidelines', *Journal of Statistics Education*, 26(1), pp. 55-66. doi:10.1080/10691898.2018.1434342.
