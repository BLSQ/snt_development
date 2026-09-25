# SNT DHIS2 Outliers Imputation (Mean) Pipeline

The **SNT DHIS2 Outliers Imputation (Mean)** pipeline flags outliers in the formatted DHIS2 routine data with a mean ± *k* × standard deviation rule computed over each facility × indicator series. It builds a detection table plus imputed and removed versions of the routine data, publishes them to **`DHIS2_OUTLIERS_IMPUTATION`**, optionally loads the detection table into the workspace database, and runs the Mean reporting notebook.

## Parameters

* **`deviation_mean`** (int, Optional):
  * **Name:** Number of SD around the mean
  * **Description:** *k*, the half-width of the accepted interval around the series mean, in standard deviations. A value outside **mean ± *k* × SD** is flagged as an outlier.
  * **Default:** `3`.
* **`push_db`** (bool, Optional):
  * **Name:** Push outliers table to DB
  * **Description:** When true, loads **`[COUNTRY_CODE]_routine_outliers_detected.parquet`** into the workspace database table **`outliers_detected`** (for the Shiny outliers explorer), replacing whatever the last outliers pipeline pushed there.
  * **Default:** `true`.

`run_report_only` and `pull_scripts` behave as in the other SNT pipelines. In report-only mode nothing is computed or published: only the reporting notebook runs, on the files already in the datasets.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`**, resolve **`COUNTRY_CODE`**, and create `pipelines/snt_dhis2_outliers_imputation_mean/` and `data/dhis2/outliers_imputation/` if missing.
2. **Detection and imputation** (skipped when `run_report_only`): run **`code/snt_dhis2_outliers_imputation_mean.ipynb`** with **`ROOT_PATH`** and **`DEVIATION_MEAN`**. The notebook:
   1. Loads **`[COUNTRY_CODE]_routine.parquet`** from **`DHIS2_DATASET_FORMATTED`** and stops if a configured indicator column is missing.
   2. Reshapes it to long format at **facility (`OU_ID`) × month (`PERIOD`) × `INDICATOR`**, and removes rows duplicated on that key (first row kept; logged).
   3. Computes the ceiling-rounded **mean** and **SD** of `VALUE` **per `ADM1_ID` × `ADM2_ID` × `OU_ID` × `INDICATOR`, over the whole series** (all months, missing values ignored).
   4. Flags values outside **mean ± `DEVIATION_MEAN` × SD**; values that cannot be evaluated (missing value or SD) are not flagged.
   5. **Imputation:** replaces each flagged value with a centred 3-month moving mean (see Notes). **Removal:** sets each flagged value to missing.
   6. Writes the detected, imputed and removed tables with the standard outliers formatters from `code/snt_utils.r` (see Outputs).
3. **Output check:** stop with an error if any of the three Parquet files is missing or was not rewritten during this run. All outliers imputation pipelines write the same filenames, so this prevents publishing a file left by an earlier run of another method.
4. **Publish:** save the pipeline parameters JSON (the injected notebook parameters) and upload the three Parquet files plus that JSON to **`DHIS2_OUTLIERS_IMPUTATION`**.
5. **Database** (only when `push_db`): push the detection table to **`outliers_detected`**.
6. **Reporting:** run **`reporting/snt_dhis2_outliers_imputation_mean_report.ipynb`**, in every mode including report-only.

## Inputs

* **`[COUNTRY_CODE]_routine.parquet`** on **`DHIS2_DATASET_FORMATTED`**: required; the notebook stops with an `[ERROR]` if it cannot be loaded or lacks a configured indicator column.
* **`configuration/SNT_config.json`** for:
  * **`SNT_CONFIG.COUNTRY_CODE`** (and **`SNT_CONFIG.COUNTRY_NAME`** in the report)
  * **`DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS`**: its keys are the indicators screened
  * **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`** and **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**
* The reporting notebook reads **`[COUNTRY_CODE]_routine_outliers_detected.parquet`** and **`[COUNTRY_CODE]_routine_outliers_imputed.parquet`** back from **`DHIS2_OUTLIERS_IMPUTATION`**, and **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`** for its maps.

## Outputs

**Workspace filesystem**

* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_detected.parquet`**
* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_imputed.parquet`**
* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_removed.parquet`**
* **Pipeline parameters JSON** in the same directory
* Executed notebook under **`pipelines/snt_dhis2_outliers_imputation_mean/papermill_outputs/`**; report outputs under **`pipelines/snt_dhis2_outliers_imputation_mean/reporting/outputs/`** (figures in `figures/`)

**Published to `DHIS2_OUTLIERS_IMPUTATION`** (not in report-only mode)

* The three Parquet files above and the pipeline parameters JSON. No `.csv` twins are written.

**Database** (only when `push_db`)

* Table **`outliers_detected`**, loaded from the detection Parquet.

> **Notes for the Data Analyst:**
>
> - **Last run wins:** all five outliers imputation pipelines publish the same three filenames to **`DHIS2_OUTLIERS_IMPUTATION`**. Downstream uses whichever ran last; check **`OUTLIER_METHOD`** or the parameters JSON to see which method (and which `DEVIATION_MEAN`) produced the files.
> - **Detection table** (long format, one row per facility × month × indicator of the routine data; rows sorted by `ADM1_ID`, `ADM2_ID`, `OU_ID`, `INDICATOR`, `PERIOD`), columns in order:
>   - **`PERIOD`** (integer, `YYYYMM`), **`YEAR`**, **`MONTH`** (integer), **`DATE`** (date, first day of the month)
>   - **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`OU_ID`**, **`OU_NAME`** (character; names unchanged from the routine data), **`INDICATOR`** (character)
>   - **`VALUE`** (double): original reported value.
>   - **`OUTLIER_DETECTED`** (logical, never missing): `TRUE` if outside mean ± *k* × SD. Missing values are never flagged.
>   - **`OUTLIER_METHOD`** (character): `MEAN`. The value of *k* is recorded in the parameters JSON, not in the table.
> - **Imputed and removed tables** (wide routine format, one row per facility × month of the routine data, kept even when all values are missing; rows sorted by `ADM1_ID`, `ADM2_ID`, `OU_ID`, `PERIOD`), columns in order: **`PERIOD`**, **`YEAR`**, **`MONTH`** (integer), **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`OU_ID`**, **`OU_NAME`** (character), then one double column per configured indicator (all missing if the indicator has no data).
>   - **Imputed:** only flagged values are replaced, by the ceiling of the mean of the non-flagged, non-missing values of the previous and next month in the same `OU_ID` × `INDICATOR` series, ordered by `PERIOD`. The value stays missing when both neighbours are flagged or missing, and at the first and last month of a series. Values that were missing in the routine data stay missing.
>   - **Removed:** flagged values are set to missing; every other value, and every row, is kept.
> - **Grain:** facility × month. The mean and SD are computed over each facility × indicator's full history, not per year.
> - **Gaps in `PERIOD`:** the imputation window counts rows, not calendar months, so a series with missing months treats the months on either side of a gap as neighbours.
