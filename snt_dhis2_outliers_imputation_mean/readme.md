# SNT DHIS2 Outliers Imputation (Mean) Pipeline

This pipeline detects extreme values in formatted DHIS2 routine time series using a mean ± *k*× standard deviation rule at the facility–indicator level, then builds imputed and outlier-removed versions of the same routine table. It publishes the results and a parameter snapshot to the configured OpenHEXA dataset and optionally loads the detection table into the workspace database for the outliers explorer.

## Parameters

* **`deviation_mean`** (int, Optional):
  * **Name:** Number of SD around the mean
  * **Description:** Half-width of the acceptance interval around the series mean, expressed as a multiple of the standard deviation computed over all monthly observations for each organisation unit and indicator.
  * **Choices/Default:** Default: `3`.
* **`push_db`** (bool, Optional):
  * **Name:** Push outliers table to DB
  * **Description:** When true, pushes `{COUNTRY_CODE}_routine_outliers_detected.parquet` to the workspace table `outliers_detected` for downstream Shiny or SQL use.
  * **Choices/Default:** Default: `true`.

## Functionality Overview

1. Resolve workspace paths, ensure `pipelines/snt_dhis2_outliers_imputation_mean` and `data/dhis2/outliers_imputation` exist, and load `configuration/SNT_config.json` with validation.
2. When the main computation stage runs, pass **`ROOT_PATH`** and **`DEVIATION_MEAN`** into **`code/snt_dhis2_outliers_imputation_mean.ipynb`** via Papermill.
3. In the notebook, load `{COUNTRY_CODE}_routine.parquet` from the OpenHEXA dataset named in `SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`, pivot configured DHIS2 indicators to long form, and keep rows at **ADM1 × ADM2 × facility (OU) × monthly period (PERIOD / YEAR / MONTH) × INDICATOR** resolution.
4. Collapse duplicate keys at **ADM1_ID, ADM2_ID, OU_ID, PERIOD, YEAR, MONTH, INDICATOR** before analysis.
5. Compute descriptive statistics **within each (ADM1_ID, ADM2_ID, OU_ID, INDICATOR) group** (entire facility–indicator series across months): ceiling-rounded mean, median, SD, MAD, Q1, and Q3; each monthly row retains these group-level values.
6. Flag outliers where `VALUE` lies outside **mean ± `DEVIATION_MEAN` × SD**, producing a boolean column named with the pattern `OUTLIER_MEAN{DEVIATION_MEAN}SD`.
7. Apply the shared imputation helper: for flagged months, replace values using a **three-month centred moving average** (ceiling of the mean of non-missing neighbours) along the **OU × INDICATOR** monthly series, then reshape to wide routine layouts for imputed and removed variants.
8. Enrich the detection export with administrative names, write `{COUNTRY_CODE}_routine_outliers_detected.parquet`, `{COUNTRY_CODE}_routine_outliers_imputed.parquet`, and `{COUNTRY_CODE}_routine_outliers_removed.parquet` under `data/dhis2/outliers_imputation/`.
9. Save a JSON parameter record alongside those outputs, verify the three Parquet files were updated in the current run, upload them plus the parameter file to the OpenHEXA dataset whose id is `SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`, and optionally push the detection Parquet to `outliers_detected`.
10. Execute the reporting notebook `reporting/snt_dhis2_outliers_imputation_mean_report.ipynb` to refresh static outputs.

## Inputs

* **`configuration/SNT_config.json`**: country code, administration labels, DHIS2 indicator definitions, and dataset identifiers.
* **`{COUNTRY_CODE}_routine.parquet`**: formatted wide routine table from the `DHIS2_DATASET_FORMATTED` dataset (latest version resolved by the notebook helpers).

## Outputs

* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_detected.parquet`**: long-format detection table (period, geography ids, indicator, value, outlier flag, method-specific columns, names).
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_imputed.parquet`**: wide routine table with imputed values for flagged cells.
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_removed.parquet`**: wide routine table with flagged values set to missing.
* **Pipeline parameters JSON**: file produced by `save_pipeline_parameters` in the same folder (exact name assigned at run time).
* **OpenHEXA upload**: the three Parquet outputs and the parameters JSON are registered on the dataset referenced by **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`** via `add_files_to_dataset`.

> **Notes for the Data Analyst:**
>
> - **`OUTLIER_MEAN{k}SD`**: Boolean flag for the mean-based rule using the configured *k* (`deviation_mean`); column name embeds *k*.
> - **`VALUE`**: Monthly indicator count or rate input from the formatted routine file; compared to **mean ± k×SD** within each **OU_ID × INDICATOR** group across all periods in the series.
> - **`mean` / `sd`**: Ceiling-rounded group statistics over the full facility–indicator history used to form the bounds; unusually low or high months relative to that history are flagged.
