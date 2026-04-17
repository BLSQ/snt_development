# SNT DHIS2 Outliers Imputation (Magic Glasses) Pipeline

Magic Glasses runs a staged outlier workflow on formatted DHIS2 routine data: median ± MAD screens at two strictness levels, optionally followed by seasonal decomposition rules when “complete” mode is selected. The pipeline clears stale Parquet artifacts before each compute run, exports the standard detection / imputed / removed trio, and registers them on the shared outliers dataset.

## Parameters

* **`mode`** (str, Optional):
  * **Name:** Detection mode
  * **Description:** **`partial`** runs MAD-based stages only (MAD15 then MAD10). **`complete`** adds seasonal detection passes (seasonal5 then seasonal3) on the remaining rows, which is far slower.
  * **Choices/Default:** `partial`, `complete`. Default: `partial`.
* **`push_db`** (bool, Optional):
  * **Name:** Push to Shiny database
  * **Description:** When true, loads the detection Parquet into `outliers_detected` for the Shiny outliers explorer.
  * **Choices/Default:** Default: `false` (unlike the classic mean/median/IQR pipelines).

## Functionality Overview

1. Normalise `mode`, reject unknown strings, log whether **`RUN_MAGIC_GLASSES_COMPLETE`** is true, and warn when complete mode is chosen because seasonal work can take hours.
2. Optionally pull notebooks from the repository, then ensure pipeline and `data/dhis2/outliers_imputation` directories exist and load `SNT_config.json`.
3. When not in report-only mode, delete any existing `{COUNTRY_CODE}_routine_outliers_*.parquet` in the output folder to avoid publishing stale files.
4. Pass **`ROOT_PATH`**, **`RUN_MAGIC_GLASSES_COMPLETE`**, fixed MAD thresholds (**15**, **10**), seasonal thresholds (**5**, **3**), and **`SEASONAL_WORKERS = 1`** into `code/snt_dhis2_outliers_imputation_magic_glasses.ipynb`.
5. Load `{COUNTRY_CODE}_routine.parquet` from **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`**, melt configured indicators to long form at **ADM1 × ADM2 × OU × PERIOD/YEAR/MONTH × INDICATOR**, and drop duplicate keys on that grain.
6. Run **`run_magic_glasses_outlier_detection`**: compute median and MAD with `constant = 1` **by YEAR, OU_ID, and INDICATOR** for MAD passes; MAD15 runs on all rows, MAD10 on rows not flagged by MAD15, then merge into **`OUTLIER_MAD15_MAD10`**. When complete, run seasonal detectors sequentially on rows still not flagged, producing **`OUTLIER_SEASONAL5_SEASONAL3`**.
7. Export via **`export_magic_glasses_outputs`**: attach **`OUTLIER_MAGIC_GLASSES_PARTIAL`** and (if complete) **`OUTLIER_MAGIC_GLASSES_COMPLETE`**, choose the active flag column, write `{COUNTRY_CODE}_routine_outliers_detected.parquet` with **`OUTLIER_DETECTED`**, **`OUTLIER_METHOD`**, and a **`DATE`** column; build imputed wide data using a **three-month centred `frollapply` mean (ceiling)** on non-outlier values **by (ADM1_ID, ADM2_ID, OU_ID, INDICATOR)**; build removed wide data by nulling flagged values.
8. After successful outputs, save the expanded parameter JSON (including injected thresholds), upload the three Parquet files plus JSON to **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**, optionally push `outliers_detected`, and render `reporting/snt_dhis2_outliers_imputation_magic_glasses_report.ipynb`.

## Inputs

* **`configuration/SNT_config.json`**.
* **`{COUNTRY_CODE}_routine.parquet`** from `DHIS2_DATASET_FORMATTED` (all indicators declared under `DHIS2_INDICATOR_DEFINITIONS`).

## Outputs

* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_detected.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_imputed.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_removed.parquet`**
* **Pipeline parameters JSON** (saved after output validation; includes `RUN_MAGIC_GLASSES_COMPLETE`, MAD and seasonal constants, worker count).
* **Dataset:** **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**.

> **Notes for the Data Analyst:**
>
> - **`OUTLIER_MAGIC_GLASSES_PARTIAL`**: Result after MAD15 → MAD10 chaining (any row still flagged remains an outlier in partial mode exports).
> - **`OUTLIER_MAGIC_GLASSES_COMPLETE`**: Present only in complete mode; incorporates seasonal passes after MAD filtering.
> - **`OUTLIER_DETECTED` / `OUTLIER_METHOD`**: Final boolean used for imputation/removal and textual label (**`MAGIC_GLASSES_PARTIAL`** or **`MAGIC_GLASSES_COMPLETE`**).
> - **MAD stages**: Median and MAD are computed **per calendar YEAR × OU × INDICATOR**, while imputation smoothing runs **along monthly PERIOD order within OU × INDICATOR**.
