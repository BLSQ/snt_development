# SNT DHIS2 Outliers Imputation (PATH) Pipeline

The PATH pipeline targets a small set of programme indicators (typically testing, confirmation, and prescription volumes) and flags month-to-month spikes relative to a robust central trend. It applies epidemiological exception rules for suspected stock-outs and epidemics, imputes flagged points with a pre-computed baseline, and publishes the same three-file contract as the other DHIS2 outlier pipelines.

## Parameters

* **`deviation_mean`** (int, Optional):
  * **Name:** Number of SD around the mean
  * **Description:** Intended multiplier *k* for the PATH spike rule **MEAN_80 + k × SD_80**. `pipeline.py` passes this value to Papermill as **`DEVIATION_MEAN`**, while the R notebook reads **`MEAN_DEVIATION`** (default **10** when unset). Until parameter names are aligned, the live threshold in R follows **`MEAN_DEVIATION`**, not necessarily this OpenHEXA field.
  * **Choices/Default:** Default: `10`.
* **`push_db`** (bool, Optional):
  * **Name:** Push outliers table to DB
  * **Description:** When true, loads `{COUNTRY_CODE}_routine_outliers_detected.parquet` into `outliers_detected`.
  * **Choices/Default:** Default: `true`.

## Functionality Overview

1. Create pipeline and `data/dhis2/outliers_imputation` folders, load `SNT_config.json`, and execute `code/snt_dhis2_outliers_imputation_path.ipynb` with `ROOT_PATH` and `DEVIATION_MEAN` when the main stage runs.
2. Load `{COUNTRY_CODE}_routine.parquet` from **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`** (helper `load_routine_data`), retaining only columns needed for PATH plus the indicator names listed in `DHIS2_INDICATOR_DEFINITIONS`.
3. Pivot to long format and complete missing **PERIOD × INDICATOR** combinations per **ADM1 × ADM2 × OU**, yielding monthly **OU × INDICATOR** rows.
4. Remove duplicates at **ADM1_ID, ADM2_ID, OU_ID, PERIOD, INDICATOR** via `remove_path_duplicates`.
5. Restrict to **VALUE > 0**, rank values within each **(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)** group, keep the middle **80%** (ranks strictly between 10th and 90th percentile by count), and summarise **MEAN_80** and **SD_80** (ceiling-rounded mean and SD of that trimmed sample). Join those statistics back to every monthly row.
6. Flag **`OUTLIER_TREND`** when **`VALUE > MEAN_80 + MEAN_DEVIATION × SD_80`**, forcing false when values or **SD_80** are missing, and suppressing flags below indicator-specific low-count floors (**TEST**/**PRES** < 50, **CONF** < 10).
7. Build **`possible_stockout`** and **`possible_epidemic`** exception tables (`detect_possible_stockout`, `detect_possible_epidemic`) and merge corrections into **`OUTLIER_TREND_01`** and **`OUTLIER_TREND_02`** (`build_path_clean_outliers`).
8. Impute months still flagged in the final correction column by replacing values with **`MEAN_80`** (`impute_path_outliers`), including reversal logic when imputation would break TEST vs CONF ordering.
9. Reshape to wide imputed and removed routine tables (removed sets flagged cells to missing), write the three Parquet outputs, save parameters JSON, upload to **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`**, optionally push `outliers_detected`, and run `reporting/snt_dhis2_outliers_imputation_path_report.ipynb`.

## Inputs

* **`configuration/SNT_config.json`** (including `DHIS2_INDICATOR_DEFINITIONS` subset used as PATH indicators).
* **`{COUNTRY_CODE}_routine.parquet`** from `DHIS2_DATASET_FORMATTED`.

## Outputs

* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_detected.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_imputed.parquet`**
* **`data/dhis2/outliers_imputation/{COUNTRY_CODE}_routine_outliers_removed.parquet`**
* **Pipeline parameters JSON** from `save_pipeline_parameters`.
* **Dataset:** **`SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION`** via `add_files_to_dataset`.

> **Notes for the Data Analyst:**
>
> - **`OUTLIER_TREND`**: Initial spike flag when **`VALUE > MEAN_80 + MEAN_DEVIATION × SD_80`** after the central 80% positive-value trim (see **`deviation_mean`** parameter note for **`MEAN_DEVIATION`** wiring).
> - **`OUTLIER_TREND_01` / `OUTLIER_TREND_02`**: Sequentially adjusted flags after stock-out and epidemic heuristics.
> - **`MEAN_80` / `SD_80`**: Robust location and scale from the **central 80%** of positive monthly **`VALUE`**s for each **OU × INDICATOR** (excluding zeros before ranking).
