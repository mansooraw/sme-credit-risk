# Cleaning log

Date: 22 November 2026
Author: Salem Adel Salem Alshamlan Almarri
Produced by: notebooks/02_data_cleaning.ipynb using src/cleaning.py
Input: data/raw/SBAnational.csv (899,164 rows, 27 columns)
Output: data/processed/loans_clean.parquet (895,952 rows, 35 columns)

## Summary

3,212 rows were removed (0.36% of the raw file): 1,997 with unknown outcome and 1,215 near-duplicates. The default rate of the clean table is 17.56%. Missing cells outside ChgOffDate fell from 14,783 to 2,454. All invalid codes and impossible values counted in the raw file are zero after cleaning; flagged logical inconsistencies are kept on purpose.

## Issues and decisions

| No. | Issue | Decision | Reason | Rows or cells affected |
|---|---|---|---|---|
| 1 | Currency columns stored as text | Strip $, commas and spaces; convert to float | Amounts must be numeric for every later calculation | 5 columns, all rows |
| 2 | Dates as dd-Mon-yy text with two-digit years | Explicit format and century rule (years above the current two digits are 19xx) | Python's default rule put 9 approvals from the 1960s in the 2060s | 3 columns, all non-empty values |
| 3 | ApprovalFY text with '1976A' | Strip non-digits, store as integer | Fiscal year is a number | 18 rows |
| 4 | MIS_Status missing | Drop rows; create Default (1 = CHGOFF, 0 = P I F) | Outcome unknown; imputing it would invent the label | 1,997 rows removed |
| 5 | DisbursementGross equal to 0 | No separate rule | All such loans had unknown status and were removed with them | 196 rows (within the rows above) |
| 6 | Near-duplicate loans (Name, City, State, ApprovalDate, GrAppv) | Keep the first of each group | Prevents one borrower decision appearing in both training and test data | 1,215 rows removed |
| 7 | RevLineCr codes 0, T, 16 rare codes and missing | 0 to N, T to Y, others to Unknown; raw code kept in RevLineCr_raw | 0 matches N and T matches Y on amount, term, guarantee share and redraw pattern (notebook 01) | 257,602 coded 0, 15,284 coded T, 4,582 Unknown after cleaning |
| 8 | LowDoc codes 0, C, S, A, R, 1 and missing | Y and N kept, others to Unknown | No code matches Y or N; unknown group has a distinct default rate | 5,989 Unknown after cleaning |
| 9 | NewExist 0 and missing | 1 Existing, 2 New, others Unknown | Unknown group defaults far less than either valid group | 1,161 Unknown after cleaning |
| 10 | UrbanRural 0 (undefined) | 1 Urban, 2 Rural, 0 Unknown | Unknown group (mostly older loans) defaults at a third of the urban rate; mode imputation would hide this | 322,565 Unknown after cleaning |
| 11 | Dates outside 1960 to 2014 | Set to missing, flag date_out_of_range | Typing errors outside the period covered | 4 rows |
| 12 | Disbursement more than 30 days before approval | Keep, flag disb_before_approval | Other fields look normal; date may be recorded at another stage | 133 rows flagged |
| 13 | SBA_Appv above GrAppv | Cap at GrAppv | Guarantee cannot exceed the loan; safeguard for app inputs | 0 rows |
| 14 | Zip equal to 0 | Set to missing | Not a valid ZIP code | 283 cells |
| 15 | ChgOffDate disagrees with MIS_Status | Keep MIS_Status as label, flag chgoff_mismatch | Status is the reported outcome; ChgOffDate is post-outcome and never a feature | 4,970 rows flagged |
| 16 | NoEmp 9999 and above the 99.9th percentile | Set to missing, impute NAICS sector median, flag noemp_imputed (cap 300) | Placeholder and impossible head counts; firm size depends on sector | 884 cells |
| 17 | Term equal to 0 months | Impute median Term of the loan-size band, flag term_imputed | Keeps 802 mostly defaulted loans; flag is post-outcome and must not be a feature | 802 cells |
| 18 | CreateJob, RetainedJob extremes | Winsorise at 99.9th percentile (200 and 280) | Thousands of jobs per small loan are entry errors | 870 CreateJob and 894 RetainedJob values capped |
| 19 | State missing | Fill from three-digit ZIP prefix when one state has at least 95% of 20 or more loans, else Unknown | Bank state differs from borrower state for out-of-state lending | 11 filled, 2 Unknown |
| 20 | Name, City, Bank, BankState missing | Fill with Unknown | Missing text is not a reason to drop a loan | Name 3, City 30, Bank 1,559, BankState 1,566 |
| 21 | NAICS 0, FranchiseCode 0 and 1 | No change in cleaning | Handled as Unknown sector and IsFranchise in notebook 03 | no cells changed |
| 22 | Post-outcome columns | Kept in the file, excluded from features via config.LEAKAGE_COLUMNS | Needed for checks only | ChgOffDate, ChgOffPrinGr, BalanceGross, MIS_Status |

## Rows and cells changed by each step

| Step | Function | What it does | Rows before | Rows after | Rows removed | Cells changed |
|---|---|---|---|---|---|---|
| 1 | parse_types | Currency, dates (century fix) and ApprovalFY parsed | 899,164 | 899,164 | 0 | 7,353,643 |
| 2 | add_target | Loans with unknown MIS_Status dropped, Default added | 899,164 | 897,167 | 1,997 | 0 |
| 3 | remove_near_duplicates | Near-duplicate loans removed | 897,167 | 895,952 | 1,215 | 0 |
| 4 | clean_codes | RevLineCr, LowDoc, NewExist, UrbanRural recoded | 895,952 | 895,952 | 0 | 2,074,891 |
| 5 | apply_logical_checks | Impossible dates, Zip 0, SBA_Appv cap, flags | 895,952 | 895,952 | 0 | 287 |
| 6 | handle_outliers | NoEmp, Term imputed; job counts winsorised | 895,952 | 895,952 | 0 | 3,450 |
| 7 | fill_missing_categoricals | State from ZIP prefix; other text to Unknown | 895,952 | 895,952 | 0 | 3,063 |

Cells changed counts every value that differs from the previous step, including type conversions and relabelling (for example 1 to Urban), so it is larger than the number of loans whose meaning changed.

## Quality before and after

| Group | Check | Raw file | Clean file | Change |
|---|---|---|---|---|
| Size | Rows | 899,164 | 895,952 | -3,212 |
| Size | Columns | 27 | 35 | 8 |
| Missing | Missing cells (original 27 columns) | 751,248 | 736,160 | -15,088 |
| Missing | Missing cells (% of original cells) | 3.094 | 3.043 | -0.051 |
| Missing | Missing cells excluding ChgOffDate | 14,783 | 2,454 | -12,329 |
| Missing | Rows with unknown MIS_Status | 1,997 | 0 | -1,997 |
| Format | Numeric or date columns stored as text | 9 | 0 | -9 |
| Invalid codes | RevLineCr: values outside the code list | 272,951 | 0 | -272,951 |
| Invalid codes | LowDoc: values outside the code list | 3,425 | 0 | -3,425 |
| Invalid codes | NewExist: values outside the code list | 1,034 | 0 | -1,034 |
| Invalid codes | UrbanRural: values outside the code list | 323,167 | 0 | -323,167 |
| Invalid codes | ApprovalFY: non-numeric values | 18 | 0 | -18 |
| Impossible values | NoEmp equal to placeholder 9999 | 4 | 0 | -4 |
| Impossible values | Term equal to 0 months | 810 | 0 | -810 |
| Impossible values | Zip equal to 0 | 283 | 0 | -283 |
| Impossible values | SBA_Appv above GrAppv | 0 | 0 | 0 |
| Impossible values | DisbursementGross equal to 0 | 196 | 0 | -196 |
| Impossible values | Dates outside 1960-2014 | 4 | 0 | -4 |
| Extreme values | NoEmp above 99.9th percentile (300) | 885 | 0 | -885 |
| Extreme values | CreateJob above 99.9th percentile (200) | 870 | 0 | -870 |
| Extreme values | RetainedJob above 99.9th percentile (280) | 894 | 0 | -894 |
| Logical | Disbursed over 30 days before approval | 135 | 133 | -2 |
| Logical | Paid in full but ChgOffDate present | 4,927 | 4,922 | -5 |
| Logical | Charged off but ChgOffDate missing | 47 | 48 | 1 |
| Duplicates | Exact duplicate rows | 0 | 0 | 0 |
| Duplicates | Duplicate rows ignoring LoanNr_ChkDgt | 128 | 0 | -128 |
| Duplicates | Near-duplicates (Name, City, State, ApprovalDate, GrAppv) | 1,266 | 0 | -1,266 |

The same table is saved as docs/quality_before_after.csv.
