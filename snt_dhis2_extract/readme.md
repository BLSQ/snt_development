# SNT DHIS2 Extract Pipeline

The **SNT DHIS2 Extract** pipeline pulls raw DHIS2 analytics, population, organisation-unit hierarchy (“pyramid”), facility geometries (“shapes”), and reporting-rate inputs for a configured period. Processing is implemented in **`pipeline.py`** (Polars/Pandas, OpenHexa DHIS2 toolbox); optional **Niger**-specific pyramid adjustments use a transformation notebook referenced from that file.

---

## Parameters

* **`dhis2_connection`** (DHIS2Connection, Required):
  * **Name:** DHIS2 connection
  * **Description:** Workspace connection used to instantiate the DHIS2 client for all API downloads.
* **`start`** (Integer, Required):
  * **Name:** Period (start)
  * **Description:** First period in **`YYYYMM`** inclusive.
* **`end`** (Integer, Required):
  * **Name:** Period (end)
  * **Description:** Last period in **`YYYYMM`** inclusive.
* **`overwrite`** (Boolean, Optional):
  * **Name:** Overwrite
  * **Description:** When **`true`**, clears matching raw period files before re-downloading where implemented.
  * **Choices/Default:** Default: **`true`**.

---

## Functionality Overview

1. **Orchestration (`pipeline.py`):** Validates the **`YYYYMM`** period range, then runs sequential download and merge tasks (population, shapes, pyramid, analytics, reporting) using the OpenHexa DHIS2 toolbox and Polars/Pandas. Country-specific logic includes Burkina Faso pyramid filtering, Niger organisation-unit group retrieval, and optional **`NER_pyramid_format.ipynb`** execution via Papermill when configured.
2. **Organisation units:** Downloads the full pyramid, applies config-driven filters (e.g. BFA), optional NER transforms, then saves a truncated pyramid at **`ANALYTICS_ORG_UNITS_LEVEL`**.
3. **Population:** Yearly pulls for indicators/data elements defined under **`POPULATION_DEFINITIONS`**, merged to **`[COUNTRY_CODE]_dhis2_raw_population.parquet`**.
4. **Shapes:** Exports **ADM2**-level geometries and admin names to **`[COUNTRY_CODE]_dhis2_raw_shapes.parquet`**.
5. **Analytics:** Downloads configured data elements per month for org units at **`ANALYTICS_ORG_UNITS_LEVEL`**, enriches with metadata, merges periods into **`[COUNTRY_CODE]_dhis2_raw_analytics.parquet`**.
6. **Reporting rates:** Either dataset-based or indicator-based downloads per **`DHIS2_REPORTING_RATES`** config, merged to **`[COUNTRY_CODE]_dhis2_raw_reporting.parquet`**.
7. **Dataset upload:** Creates a new dataset version (when files exist) under **`DHIS2_DATASET_EXTRACTS`** with raw Parquet artifacts and a parameters JSON snapshot.
8. **Reporting:** Runs **`snt_dhis2_extract_report.ipynb`** (HTML report when execution succeeds).

---

## Inputs

* **`SNT_config.json`**: Country code, org-unit levels, **`DHIS2_DATA_DEFINITIONS`**, dataset IDs, and reporting-rate definitions.
* **Live DHIS2 instance** reachable via **`dhis2_connection`**.

---

## Outputs

* **`data/dhis2/extracts_raw/routine_data/[COUNTRY_CODE]_dhis2_raw_analytics.parquet`**
* **`data/dhis2/extracts_raw/population_data/[COUNTRY_CODE]_dhis2_raw_population.parquet`**
* **`data/dhis2/extracts_raw/shapes_data/[COUNTRY_CODE]_dhis2_raw_shapes.parquet`**
* **`data/dhis2/extracts_raw/pyramid_data/[COUNTRY_CODE]_dhis2_raw_pyramid.parquet`**
* **`data/dhis2/extracts_raw/reporting_data/[COUNTRY_CODE]_dhis2_raw_reporting.parquet`**
* **Parameters JSON** under `extracts_raw/`
* **Published copies** of the above (where present) plus parameters to **`DHIS2_DATASET_EXTRACTS`**

---
