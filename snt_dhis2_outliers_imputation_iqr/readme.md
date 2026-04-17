# SNT DHIS2 Outliers Imputation (IQR) Pipeline

This pipeline applies a Tukey-style fence to formatted DHIS2 routine indicators: for each facility and indicator it compares monthly values to quartiles computed over the whole series, scaled by a configurable IQR multiplier. Detected outliers are imputed or blanked in derivative wide tables, then packaged for the shared outliers dataset and optional database export.

## Parameters

* **`deviation_iqr`** (float, Optional):
  * **Name:** IQR multiplier
  * **Description:** Multiplier *m* applied to the classic inter-quartile range **Q3 − Q1**: a month is flagged when **`VALUE`** lies outside **[Q1 − m×(Q3 − Q1), Q3 + m×(Q3 − Q1)]`** (values computed with ceiling-rounded quartiles in the notebook).
  * **Choices/Default:** Default: `1.5`.
* **`push_db`** (bool, Optional):
  * **Name:** Push outliers table to DB
  * **Description:** When true, publishes the detection Parquet to the `outliers_detected` table.
  * **Choices/Default:** Default: `true`.

## Functionality Overview

1. Initialise paths, validate `SNT_config.json`, and run `code/snt_dhis2_outliers_imputation_iqr.ipynb` with `ROOT_PATH` and `DEVIATION_IQR` when not in report-only mode.
2. Ingest `{COUNTRY_CODE}_routine.parquet` from **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`** and expand to long form at **ADM1 × ADM2 × OU × PERIOD/YEAR/MONTH × INDICATOR**.
3. Deduplicate on **ADM1_ID, ADM2_ID, OU_ID, PERIOD, YEAR, MONTH, INDICATOR**.
4. For each **(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)** group, compute ceiling-rounded mean, median, SD, MAD, Q1, and Q3 across all months.
5. Derive **`iqr = (Q3 − Q1) × DEVIATION_IQR`**, fences **`Q1 − iqr`** and **`Q3 + iqr`**, and a flag column named **`OUTLIER_IQR{DEVIATION_IQR_NAME}`** where `DEVIATION_IQR_NAME` replaces decimal points in the multiplier (for example `1.5` becomes `1_5`).
6. Impute flagged observations using the **three-month centred moving average** along the **OU × INDICATOR** timeline, then build wide imputed and removed routine extracts.
7. Write detection, imputed, and removed Parquet files to `data/dhis2/outliers_imputation/`, save parameters JSON, upload everything to **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**, optionally push `outliers_detected`, and execute the reporting notebook.

## Inputs

* **`configuration/SNT_config.json`**.
* **`{COUNTRY_CODE}_routine.parquet`** from `DHIS2_DATASET_FORMATTED`.

## Outputs

* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_detected.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_imputed.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_removed.parquet`**
* **Pipeline parameters JSON** (`save_pipeline_parameters` output in the same directory).
* **Dataset:** files attached to **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**.

> **Notes for the Data Analyst:**
>
> - **`OUTLIER_IQR{m}`**: Boolean flag whose suffix reflects the configured multiplier (dots turned into underscores in the column name).
> - **`q1` / `q3`**: Ceiling-rounded quartiles of **`VALUE`** across all months for the **OU_ID × INDICATOR** group; fences widen proportionally to **`deviation_iqr`**.
