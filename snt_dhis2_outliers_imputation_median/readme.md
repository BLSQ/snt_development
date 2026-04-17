# SNT DHIS2 Outliers Imputation (Median) Pipeline

This pipeline flags implausible spikes in formatted DHIS2 routine data using a median ± *k*× median absolute deviation (MAD) rule defined separately for each facility and indicator over time. It writes detection, imputed, and outlier-removed Parquet products, registers them on the outliers imputation dataset, and can mirror the detection extract into the workspace relational database.

## Parameters

* **`deviation_median`** (int, Optional):
  * **Name:** Number of MAD around the median
  * **Description:** Width of the acceptance band around the median, expressed as a multiple of MAD (`mad(..., constant = 1)`) computed from all non-missing monthly values for each organisation unit and indicator.
  * **Choices/Default:** Default: `3`.
* **`push_db`** (bool, Optional):
  * **Name:** Push outliers table to DB
  * **Description:** When true, loads `{COUNTRY_CODE}_routine_outliers_detected.parquet` into the `outliers_detected` table.
  * **Choices/Default:** Default: `true`.

## Functionality Overview

1. Prepare pipeline and output directories under the workspace, load and validate `configuration/SNT_config.json`, and read the country code.
2. When the main stage runs, inject `ROOT_PATH` and `DEVIATION_MEDIAN` into `code/snt_dhis2_outliers_imputation_median.ipynb` and execute it with the R kernel.
3. Load `{COUNTRY_CODE}_routine.parquet` from **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`**, restrict to configured indicators, and reshape to long format at **ADM1 × ADM2 × OU × monthly period × INDICATOR** granularity.
4. Remove duplicates keyed by **ADM1_ID, ADM2_ID, OU_ID, PERIOD, YEAR, MONTH, INDICATOR**.
5. Within each **(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)** group across months, compute ceiling-rounded median, mean, SD, MAD, Q1, and Q3 for use in detection.
6. Flag outliers where `VALUE` falls outside **median ± `DEVIATION_MEDIAN` × MAD**, emitting `OUTLIER_MEDIAN{DEVIATION_MEDIAN}MAD`.
7. Impute flagged months with the **three-month centred moving average** helper used in the mean pipeline (applied along the monthly **OU × INDICATOR** series), then materialise wide imputed and removed routine tables.
8. Write the three standard Parquet filenames under `data/dhis2/outliers_imputation/`, persist run parameters as JSON, validate freshness, push files to **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**, optionally push to `outliers_detected`, and render `reporting/snt_dhis2_outliers_imputation_median_report.ipynb`.

## Inputs

* **`configuration/SNT_config.json`**.
* **`{COUNTRY_CODE}_routine.parquet`** from the formatted DHIS2 dataset (`DHIS2_DATASET_FORMATTED`).

## Outputs

* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_detected.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_imputed.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_removed.parquet`**
* **Pipeline parameters JSON** from `save_pipeline_parameters` next to the Parquet outputs.
* **Dataset registration:** **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`** (`add_files_to_dataset`).

> **Notes for the Data Analyst:**
>
> - **`OUTLIER_MEDIAN{k}MAD`**: Outlier flag using median ± k×MAD with *k* from `deviation_median`; MAD uses `constant = 1` (not the default 1.4826 scale).
> - **`median` / `mad`**: Group-level statistics summarising the entire **OU_ID × INDICATOR** monthly series; each month inherits the same bounds for comparison against **`VALUE`**.
