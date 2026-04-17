# SNT DHIS2 Reporting Rate (Dataset) Pipeline

This pipeline computes **monthly** DHIS2 **dataset-level** reporting rates at **ADM2** resolution by combining routine metadata with pre-computed **actual** and **expected** report counts from DHIS2, then aggregating and deduplicating at facility and district level. Results are written to **`data/dhis2/reporting_rate/`**, registered in the **`DHIS2_REPORTING_RATE`** dataset, and summarized again in the reporting notebook.

## Parameters

* **`routine_data_choice`** (String, Required):
  * **Name:** Routine data source
  * **Description:** Selects which routine Parquet the notebook loads: formatted raw routine, outlier-imputed routine, or outlier-removed routine.
  * **Choices/Default:** `raw` (**`{COUNTRY_CODE}_routine.parquet`** on **`DHIS2_DATASET_FORMATTED`**), `imputed` (**`{COUNTRY_CODE}_routine_outliers_imputed.parquet`** on **`DHIS2_OUTLIERS_IMPUTATION`**), `outliers_removed` (**`{COUNTRY_CODE}_routine_outliers_removed.parquet`** on **`DHIS2_OUTLIERS_IMPUTATION`**). Default: `imputed`.

## Functionality Overview

1. Load **`configuration/SNT_config.json`** and resolve the routine filename implied by **`routine_data_choice`**.
2. Verify the chosen routine file exists in the appropriate source dataset; exit early with a warning if it is missing (for example before outliers imputation has been run).
3. Run **`pipelines/snt_dhis2_reporting_rate_dataset/code/snt_dhis2_reporting_rate_dataset.ipynb`** with **`SNT_ROOT_PATH`** and **`ROUTINE_FILE`**.
4. Save the Papermill parameter record alongside outputs.
5. Upload **`{COUNTRY_CODE}_reporting_rate_dataset`** tables and the parameter file to **`DHIS2_REPORTING_RATE`**.
6. Run the dataset reporting notebook to refresh charts and tables in **`reporting/outputs/`**.

## Inputs

* **`configuration/SNT_config.json`**: Dataset identifiers (formatted routine versus outliers imputation) and **`COUNTRY_CODE`**.
* **`{COUNTRY_CODE}_routine.parquet`** or **`{COUNTRY_CODE}_routine_outliers_imputed.parquet`** or **`{COUNTRY_CODE}_routine_outliers_removed.parquet`**: Monthly facility-level routine table (columns include **`PERIOD`** in **`YYYYMM`** form, **`ADM2_ID`**, and indicators as configured).
* **`{COUNTRY_CODE}_reporting.parquet`**: DHIS2 reporting extract with actual and expected report counts by **`OU_ID`** and **`PERIOD`**, restricted in the notebook to datasets declared in the configuration.

## Outputs

* **`{COUNTRY_CODE}_reporting_rate_dataset.parquet`** and **`.csv`**: District-month reporting rates (**`YEAR`**, **`MONTH`**, **`ADM2_ID`**, **`REPORTING_RATE`**, plus supporting admin labels as produced by the notebook).
* **Pipeline parameters JSON** under **`data/dhis2/reporting_rate/`**.
* **Papermill output** under **`papermill_outputs/`** and **report outputs** under **`reporting/outputs/`**.

> **Notes for the Data Analyst:**
>
> - **`Temporal and spatial resolution`**: Reporting rates are computed per **calendar month** (**`PERIOD`** / **`YEAR`**–**`MONTH`**) and summarized to **ADM2**, per the main notebook.
> - **`routine_data_choice`**: **`imputed`** and **`outliers_removed`** require successful runs of the outliers pipeline; **`raw`** reads only from the formatted routine dataset.
> - **`Dataset selection`**: When several DHIS2 products exist, the notebook keeps only datasets listed in **`SNT_config.json`** and deduplicates **`OU_ID`** by **`PERIOD`** using the highest **`ACTUAL_REPORTS`** when safe to do so.
> - **`Period alignment`**: Mismatched years or months between routine and reporting extracts can bias downstream metrics; review overlap checks in the executed notebook.
