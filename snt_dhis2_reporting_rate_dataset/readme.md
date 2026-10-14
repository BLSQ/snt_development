# SNT DHIS2 Reporting Rate (Dataset) Pipeline

The **SNT DHIS2 Reporting Rate (Dataset)** pipeline computes **monthly** DHIS2 reporting rates at **ADM2** resolution as **`ACTUAL_REPORTS / EXPECTED_REPORTS`**, from the reporting extract formatted by the DHIS2 formatting pipeline. It publishes the district-month table to **`DHIS2_REPORTING_RATE`** and runs the reporting notebook.

## Parameters

This pipeline takes no domain parameters; it only has the standard `run_report_only` and `pull_scripts` flags.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`**, read **`COUNTRY_CODE`** and the dataset identifiers.
2. **Input check:** Verify that **`[COUNTRY_CODE]_reporting.parquet`** exists on **`DHIS2_DATASET_FORMATTED`**. If it is missing, log a warning and **stop without failing**: nothing is computed, published or reported.
3. **Product UID check:** Fail the run with an error if **`SNT_CONFIG.REPORTING_RATE_PRODUCT_UID`** is missing, empty or holds only blank values.
4. **Parameters:** Save the pipeline parameters JSON (records **`ROOT_PATH`**) to **`data/dhis2/reporting_rate/`**.
5. **Computation notebook:** Run **`pipelines/snt_dhis2_reporting_rate_dataset/code/snt_dhis2_reporting_rate_dataset.ipynb`**:
   1. **Load** **`[COUNTRY_CODE]_reporting.parquet`** from **`DHIS2_DATASET_FORMATTED`** and detect its grain: **facility level** when an **`OU_ID`** column is present (dataset-based extract), **already ADM2** when it is absent (indicator-based extract).
   2. **Filter by product:** Keep only the rows whose **`PRODUCT_UID`** is listed in **`REPORTING_RATE_PRODUCT_UID`**, when all the listed UIDs are present in the data; otherwise log a warning and keep all products.
   3. **Pivot** **`PRODUCT_METRIC`** into **`ACTUAL_REPORTS`** and **`EXPECTED_REPORTS`** columns.
   4. **Deduplicate (facility level only):** When a facility appears in several datasets for the same period, keep the row with the highest **`ACTUAL_REPORTS`**, provided all duplicated values are 0 or 1; otherwise log a warning and keep the duplicates. Skipped for ADM2-level data.
   5. **NER only, facility level only:** Convert **`ACTUAL_REPORTS`** and **`EXPECTED_REPORTS`** values above 1 to 1 (pre-aggregated HOP hospital datasets). Skipped, with a warning, for ADM2-level data.
   6. **Aggregate:** Sum **`ACTUAL_REPORTS`** and **`EXPECTED_REPORTS`** **by `ADM2_ID` and `PERIOD`**, then compute **`REPORTING_RATE`**.
   7. **Export** **`[COUNTRY_CODE]_reporting_rate_dataset.parquet`** and **`.csv`** to **`data/dhis2/reporting_rate/`**.
6. **Output check:** Fail the run if the parquet or CSV is missing or was not written during this run, so files left over from a previous run are never published.
7. **Publish:** Upload the parquet, CSV and parameters JSON to **`DHIS2_REPORTING_RATE`**.
8. **Reporting:** Run **`pipelines/snt_dhis2_reporting_rate_dataset/reporting/snt_dhis2_reporting_rate_dataset_report.ipynb`**, which reads the published table back from **`DHIS2_REPORTING_RATE`**.

## Inputs

* **`[COUNTRY_CODE]_reporting.parquet`** on **`DHIS2_DATASET_FORMATTED`** — required; produced by the DHIS2 formatting pipeline from the extract's reporting data. Its grain depends on how reporting rates are configured for extraction (**`DHIS2_DATA_DEFINITIONS.DHIS2_REPORTING_RATES`**):
  * **`REPORTING_DATASETS`**: one row per facility (**`OU_ID`**) × period × dataset metric.
  * **`REPORTING_INDICATORS`**: one row per district (**`ADM2_ID`**) × period × indicator, already aggregated by DHIS2; no **`OU_ID`** / **`OU_NAME`** columns.
* **`configuration/SNT_config.json`** for **`SNT_CONFIG.COUNTRY_CODE`**, **`SNT_CONFIG.REPORTING_RATE_PRODUCT_UID`** (required) and the dataset identifiers **`DHIS2_DATASET_FORMATTED`** and **`DHIS2_REPORTING_RATE`**.
* Reporting notebook only: **`[COUNTRY_CODE]_shapes.geojson`** on **`DHIS2_DATASET_FORMATTED`** (maps) and the **`REPORTING_RATE.SCALE`** breaks in **`configuration/SNT_metadata.json`** (colour categories).

## Outputs

**Workspace filesystem**

* **`data/dhis2/reporting_rate/[COUNTRY_CODE]_reporting_rate_dataset.parquet`** and **`.csv`**.
* **Pipeline parameters JSON** in the same directory.
* **Report figures** in **`pipelines/snt_dhis2_reporting_rate_dataset/reporting/outputs/figures/`**: line-point plot, heatmap, monthly map and yearly-mean map, named **`[COUNTRY_CODE]_reporting_rate_dataset_adm2_{linepoint|heatmap|map|map_year}_[PRODUCT_UIDs].png`**. Written, not published.

**Published to `DHIS2_REPORTING_RATE`**

* **`[COUNTRY_CODE]_reporting_rate_dataset.parquet`** and **`.csv`**.
* The pipeline parameters JSON.

> **Notes for the Data Analyst:**
>
> - **Grain:** one row per **`ADM2_ID`** × **`YEAR`** × **`MONTH`**, for the district-months present in the reporting extract. Columns: **`YEAR`**, **`MONTH`**, **`ADM2_ID`**, **`REPORTING_RATE`** (no **`PERIOD`**, no names).
> - **`REPORTING_RATE`**: district-month **`sum(ACTUAL_REPORTS) / sum(EXPECTED_REPORTS)`**, a proportion — a ratio of sums, not an average of facility rates.
>   - There is no guard on the denominator: a district-month with **`EXPECTED_REPORTS` = 0** gets **`NaN`** (or **`Inf`** if actual reports are positive).
>   - Values above 1 are kept; the notebook logs a warning when any value falls outside [0, 1].
> - **`REPORTING_RATE_PRODUCT_UID`** must list UIDs from the extraction mode actually used: dataset UIDs for **`REPORTING_DATASETS`** (it may be a subset of the extracted datasets), or both indicator UIDs (actual and expected reports) for **`REPORTING_INDICATORS`**. If a listed UID is absent from the data, no filtering is applied and all products are kept.
> - **Guarded execution:** a missing **`[COUNTRY_CODE]_reporting.parquet`** ends the run early as a success, with only a warning in the log. Facility-level steps (deduplication, NER HOP conversion) are skipped when the data has no **`OU_ID`**.
> - **Downstream use:** **`snt_dhis2_incidence`** left-joins this table onto its own routine district-months by **`ADM2_ID`**, **`YEAR`**, **`MONTH`**, so district-months missing here get a missing reporting rate there.
