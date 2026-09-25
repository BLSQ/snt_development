# SNT DHIS2 Outliers Imputation (Magic Glasses) Pipeline

The **SNT DHIS2 Outliers Imputation (Magic Glasses)** pipeline flags outliers in the formatted DHIS2 routine data with a staged median ± MAD screen (MAD15 → MAD10), optionally followed by two seasonal passes (seasonal5 → seasonal3) in complete mode. It builds a detection table plus imputed and removed versions of the routine data, publishes them to **`DHIS2_OUTLIERS_IMPUTATION`**, and runs the Magic Glasses reporting notebook.

## Parameters

* **`mode`** (str, Optional):
  * **Name:** Detection mode
  * **Description:** Detection passes to run.
    * `partial`: MAD15 then MAD10 only. Fast (about 7 minutes, per the UI help text).
    * `complete`: `partial`, then seasonal5 then seasonal3 on the values not yet flagged. Can take several hours; the pipeline logs a warning when selected.
  * **Choices:** `partial`, `complete`. The value is trimmed and lower-cased before use; any other value stops the run with a `ValueError`.
  * **Default:** `partial`.
* **`push_db`** (bool, Optional):
  * **Name:** Push to Shiny database
  * **Description:** When true, loads **`[COUNTRY_CODE]_routine_outliers_detected.parquet`** into the workspace database table **`outliers_detected`** (for the Shiny outliers explorer), replacing whatever the last outliers pipeline pushed there.
  * **Default:** `false`.

`run_report_only` and `pull_scripts` behave as in the other SNT pipelines. In report-only mode nothing is computed or published: only the reporting notebook runs, on the files already in the dataset.

## Functionality Overview

1. **Mode:** Normalise **`mode`**, reject unknown values, and set **`RUN_MAGIC_GLASSES_COMPLETE`** (`true` for `complete`).
2. **Configuration:** Load and validate **`SNT_config.json`**, resolve **`COUNTRY_CODE`**, and create `pipelines/snt_dhis2_outliers_imputation_magic_glasses/` and `data/dhis2/outliers_imputation/` if missing.
3. **Detection and imputation** (skipped when `run_report_only`): run **`code/snt_dhis2_outliers_imputation_magic_glasses.ipynb`** with **`ROOT_PATH`**, **`RUN_MAGIC_GLASSES_COMPLETE`** and the fixed thresholds **`DEVIATION_MAD15 = 15`**, **`DEVIATION_MAD10 = 10`**, **`DEVIATION_SEASONAL5 = 5`**, **`DEVIATION_SEASONAL3 = 3`** (the thresholds are not exposed as parameters). The notebook:
   1. Loads **`[COUNTRY_CODE]_routine.parquet`** from **`DHIS2_DATASET_FORMATTED`** and stops if a configured indicator column is missing.
   2. Reshapes it to long format at **facility (`OU_ID`) × month (`PERIOD`) × `INDICATOR`**, and removes rows duplicated on that key (first row kept; logged).
   3. **MAD15 → MAD10:** flags values outside `median ± k × MAD` (`MAD` with `constant = 1`), computed **per `YEAR` × `OU_ID` × `INDICATOR`**. MAD10 runs only on the values MAD15 did not flag.
   4. **Seasonal5 → seasonal3** (complete mode only): for each **`OU_ID` × `INDICATOR`** monthly series, flags values whose residual from `forecast::tsclean()`, scaled by the series MAD, is at least `k`. Runs only on the values not flagged by the MAD passes, in parallel on (available cores − 1) workers.
   5. Combines the passes into one flag, **`OUTLIER_DETECTED`**, and writes the detected, imputed and removed tables (see Outputs).
4. **Output check:** stop with an error if any of the three Parquet files is missing or was not rewritten during this run. All outliers imputation pipelines write the same filenames, so this prevents publishing a file left by an earlier run of another method.
5. **Publish:** save the pipeline parameters JSON (the injected notebook parameters) and upload the three Parquet files plus that JSON to **`DHIS2_OUTLIERS_IMPUTATION`**.
6. **Database** (only when `push_db`): push the detection table to **`outliers_detected`**.
7. **Reporting:** run **`reporting/snt_dhis2_outliers_imputation_magic_glasses_report.ipynb`**, in every mode including report-only.

## Inputs

* **`[COUNTRY_CODE]_routine.parquet`** on **`DHIS2_DATASET_FORMATTED`**: required; the notebook stops with an `[ERROR]` if it cannot be loaded or lacks a configured indicator column.
* **`configuration/SNT_config.json`** for:
  * **`SNT_CONFIG.COUNTRY_CODE`** (and **`SNT_CONFIG.COUNTRY_NAME`** in the report)
  * **`DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS`**: its keys are the indicators screened
  * **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`** and **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**
* The reporting notebook reads **`[COUNTRY_CODE]_routine_outliers_detected.parquet`** back from **`DHIS2_OUTLIERS_IMPUTATION`**.

## Outputs

**Workspace filesystem**

* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_detected.parquet`**
* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_imputed.parquet`**
* **`data/dhis2/outliers_imputation/[COUNTRY_CODE]_routine_outliers_removed.parquet`**
* **Pipeline parameters JSON** in the same directory
* Executed notebook under **`pipelines/snt_dhis2_outliers_imputation_magic_glasses/papermill_outputs/`**; report outputs under **`pipelines/snt_dhis2_outliers_imputation_magic_glasses/reporting/outputs/`**

**Published to `DHIS2_OUTLIERS_IMPUTATION`** (not in report-only mode)

* The three Parquet files above and the pipeline parameters JSON. No `.csv` twins are written.

**Database** (only when `push_db`)

* Table **`outliers_detected`**, loaded from the detection Parquet.

> **Notes for the Data Analyst:**
>
> - **Last run wins:** all five outliers imputation pipelines publish the same three filenames to **`DHIS2_OUTLIERS_IMPUTATION`**. Downstream uses whichever ran last; check **`OUTLIER_METHOD`** or the parameters JSON to see which method produced the files.
> - **Detection table** (long format, one row per facility × month × indicator of the routine data; rows sorted by `ADM1_ID`, `ADM2_ID`, `OU_ID`, `INDICATOR`, `PERIOD`), columns in order:
>   - **`PERIOD`** (integer, `YYYYMM`), **`YEAR`**, **`MONTH`** (integer), **`DATE`** (date, first day of the month)
>   - **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`OU_ID`**, **`OU_NAME`** (character; names unchanged from the routine data), **`INDICATOR`** (character)
>   - **`VALUE`** (double): original reported value.
>   - **`OUTLIER_DETECTED`** (logical, never missing): `TRUE` if flagged by any pass of the selected mode. Missing values are never flagged.
>   - **`OUTLIER_METHOD`** (character): `MAGIC_GLASSES_PARTIAL` or `MAGIC_GLASSES_COMPLETE`. A file holds one mode only; comparing the two modes needs two runs.
> - **Imputed and removed tables** (wide routine format, one row per facility × month of the routine data, kept even when all values are missing; rows sorted by `ADM1_ID`, `ADM2_ID`, `OU_ID`, `PERIOD`), columns in order: **`PERIOD`**, **`YEAR`**, **`MONTH`** (integer), **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`OU_ID`**, **`OU_NAME`** (character), then one double column per configured indicator (all missing if the indicator has no data).
>   - **Imputed:** flagged values are replaced by a centred 3-row moving mean (rounded up) of the non-flagged values in the same `OU_ID` × `INDICATOR` series, ordered by `PERIOD`. The value stays missing when both neighbours are flagged or missing, and at the first and last month of a series.
>   - **Removed:** flagged values are set to missing; rows are kept.
> - **Grain:** facility × month. MAD statistics are computed per calendar year; seasonal detection and imputation run along the whole monthly series of each facility × indicator.
> - **Gaps in `PERIOD`:** seasonal detection and the imputation window treat consecutive rows as consecutive months. A series with missing months logs a warning in complete mode, and its seasonal flags and imputed values may be misaligned.
