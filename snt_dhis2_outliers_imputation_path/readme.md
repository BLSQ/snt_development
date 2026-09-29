# SNT DHIS2 Outliers Imputation (PATH) Pipeline

The **SNT DHIS2 Outliers Imputation (PATH)** pipeline flags unusually high values in the formatted DHIS2 routine data against a trimmed mean + *k* × SD threshold computed over each facility × indicator series, then un-flags likely stock-out periods and replaces the remaining outliers with the trimmed mean. It builds a detection table plus imputed and removed versions of the routine data, publishes them to **`DHIS2_OUTLIERS_IMPUTATION`**, optionally loads the detection table into the workspace database, and runs the PATH reporting notebook.

## Parameters

* **`deviation_mean`** (int, Optional):
  * **Name:** Number of SD around the mean
  * **Description:** *k*, the number of standard deviations above the trimmed mean (`MEAN_80`) beyond which a value is flagged: a value above **`MEAN_80` + *k* × `SD_80`** is an outlier. Only high values are flagged. The same *k* sets the upper limit used by the stock-out rule.
  * **Default:** `10`.
* **`push_db`** (bool, Optional):
  * **Name:** Push outliers table to DB
  * **Description:** When true, loads **`[COUNTRY_CODE]_routine_outliers_detected.parquet`** into the workspace database table **`outliers_detected`** (for the Shiny outliers explorer), replacing whatever the last outliers pipeline pushed there.
  * **Default:** `true`.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`**, resolve **`COUNTRY_CODE`**, and create `pipelines/snt_dhis2_outliers_imputation_path/` and `data/dhis2/outliers_imputation/` if missing.
2. **Detection and imputation** (skipped when `run_report_only`): run **`code/snt_dhis2_outliers_imputation_path.ipynb`** with **`ROOT_PATH`** and **`DEVIATION_MEAN`**. The notebook:
   1. Loads **`[COUNTRY_CODE]_routine.parquet`** from **`DHIS2_DATASET_FORMATTED`** and stops if a configured indicator column is missing.
   2. Reshapes it to long format at **facility (`OU_ID`) × month (`PERIOD`) × `INDICATOR`**, completing every facility with all periods and indicators present in the data (added combinations have a missing `VALUE`), and removes rows duplicated on that key (first row kept; logged as a warning).
   3. Computes, **per `ADM1_ID` × `ADM2_ID` × `OU_ID` × `INDICATOR`, over the whole series**, the ceiling-rounded mean (**`MEAN_80`**) and SD (**`SD_80`**) of the positive values, after dropping the lowest 10% and highest 10% of them (zeros and missing values are excluded, as more likely non-reporting than true zeros).
   4. **Flags** values above **`MEAN_80` + `DEVIATION_MEAN` × `SD_80`**. Values that cannot be evaluated (missing value or `SD_80`) are not flagged, nor are low counts: **`TEST`** or **`PRES`** below 50, **`CONF`** below 10.
   5. **Stock-out exception:** un-flags a flagged **`PRES`** value when, in the same facility and month, **`TEST`** is below its own `MEAN_80` and the `PRES` value is below the `TEST` threshold (`TEST` `MEAN_80` + `DEVIATION_MEAN` × `SD_80`), since a sudden rise in presumed cases with low testing suggests an RDT stock-out.
   6. **Epidemic check:** marks facility-months where **`CONF`** is flagged and either **`TEST`** is flagged or `TEST` ≥ `CONF`. It does not change any flag (see Notes).
   7. **Imputation:** replaces each flagged value with its series' **`MEAN_80`**, then applies the **TEST/CONF reversal**: when the imputed `TEST` is below the imputed `CONF` of the same facility-month while the reported `TEST` was above the reported `CONF`, both are restored to their reported values and un-flagged, so they are not outliers in any output table (the number of reverted values is logged). **Removal:** sets the values still flagged after the reversal to missing.
   8. Writes the detected, imputed and removed tables with the standard outliers formatters from `code/snt_utils.r` (see Outputs).
3. **Output check:** stop with an error if any of the three Parquet files is missing or was not rewritten during this run. All outliers imputation pipelines write the same filenames, so this prevents publishing a file left by an earlier run of another method.
4. **Publish:** save the pipeline parameters JSON (the injected notebook parameters) and upload the three Parquet files plus that JSON to **`DHIS2_OUTLIERS_IMPUTATION`**.
5. **Database** (only when `push_db`): push the detection table to **`outliers_detected`**.
6. **Reporting:** run **`reporting/snt_dhis2_outliers_imputation_path_report.ipynb`**, in every mode including report-only. It is currently a placeholder and reads no data.

## Inputs

* **`[COUNTRY_CODE]_routine.parquet`** on **`DHIS2_DATASET_FORMATTED`**: required; the notebook stops with an `[ERROR]` if it cannot be loaded or lacks a configured indicator column.
* **`configuration/SNT_config.json`** for:
  * **`SNT_CONFIG.COUNTRY_CODE`**
  * **`DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS`**: its keys are the indicators screened. They must include **`TEST`** and **`CONF`** (used by the epidemic check and the reversal; the notebook fails without them); **`PRES`** is used by the stock-out exception.
  * **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`** and **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**

## Outputs

**Workspace filesystem**

* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_detected.parquet`**
* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_imputed.parquet`**
* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_removed.parquet`**
* **Pipeline parameters JSON** in the same directory
* Executed notebook under **`pipelines/snt_dhis2_outliers_imputation_path/papermill_outputs/`**; report outputs under **`pipelines/snt_dhis2_outliers_imputation_path/reporting/outputs/`**

**Published to `DHIS2_OUTLIERS_IMPUTATION`** (not in report-only mode)

* The three Parquet files above and the pipeline parameters JSON. No `.csv` twins are written.

**Database** (only when `push_db`)

* Table **`outliers_detected`**, loaded from the detection Parquet.

> **Notes for the Data Analyst:**
>
> - **Last run wins:** all five outliers imputation pipelines publish the same three filenames to **`DHIS2_OUTLIERS_IMPUTATION`**. Downstream uses whichever ran last; check **`OUTLIER_METHOD`** or the parameters JSON to see which method (and which `DEVIATION_MEAN`) produced the files.
> - **Detection table** (long format, one row per facility × month × indicator, including the combinations added when completing the series; rows sorted by `ADM1_ID`, `ADM2_ID`, `OU_ID`, `INDICATOR`, `PERIOD`), columns in order:
>   - **`PERIOD`** (integer, `YYYYMM`), **`YEAR`**, **`MONTH`** (integer), **`DATE`** (date, first day of the month)
>   - **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`OU_ID`**, **`OU_NAME`** (character; names unchanged from the routine data), **`INDICATOR`** (character)
>   - **`VALUE`** (double): original reported value.
>   - **`OUTLIER_DETECTED`** (logical, never missing): `TRUE` if flagged after the stock-out exception and the TEST/CONF reversal, i.e. exactly the values replaced in the imputed table and set to missing in the removed table. Missing values are never flagged.
>   - **`OUTLIER_METHOD`** (character): `PATH`. The value of *k* is recorded in the parameters JSON, not in the table.
> - **Imputed and removed tables** (wide routine format, one row per facility × month of the routine data, kept even when all values are missing; rows sorted by `ADM1_ID`, `ADM2_ID`, `OU_ID`, `PERIOD`), columns in order: **`PERIOD`**, **`YEAR`**, **`MONTH`** (integer), **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`OU_ID`**, **`OU_NAME`** (character), then one double column per configured indicator (all missing if the indicator has no data).
>   - **Imputed:** flagged values are replaced by the series' **`MEAN_80`** (a whole number, as it is ceiling-rounded). Every other value is unchanged.
>   - **Removed:** flagged values are set to missing; every other value, and every row, is kept.
> - **TEST/CONF reversal:** a TEST or CONF value flagged by the rule but restored by the reversal is not an outlier in any of the three tables. Its reported value is kept; the number of such values appears only in the run log.
> - **Epidemic check has no effect:** the rule is evaluated, but it only sets to `TRUE` flags that are already `TRUE`, so no value is un-flagged (or flagged) because of a possible epidemic.
> - **Grain:** facility × month. `MEAN_80` and `SD_80` are computed over each facility × indicator's full history, not per year.
> - **Small series:** a series with fewer than two positive values left after trimming has no `SD_80`, so none of its values is flagged.
> - **Guarded execution:** nothing is skipped silently. The notebook stops with an `[ERROR]` if the routine file cannot be loaded or a configured indicator column is missing; the run stops before publishing if any of the three Parquet files was not rewritten during the run; a failed dataset upload or database push also stops the run. The database push comes after the dataset upload, so a failed push leaves the new files already published.
