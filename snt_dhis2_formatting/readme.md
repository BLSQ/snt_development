# SNT DHIS2 Formatting Pipeline

The **SNT DHIS2 Formatting** pipeline converts raw DHIS2 extract Parquet files from the extract dataset into analysis-ready **routine**, **population**, **shapes**, **pyramid**, and **reporting** tables and vectors. It publishes the formatted artifacts to **`DHIS2_DATASET_FORMATTED`** and runs the formatting reporting notebook for quality review.

## Parameters

None. `pipeline.py` does not define user-configurable domain parameters; standard OpenHEXA orchestration flags are omitted from this document per project conventions.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`**, resolve **`COUNTRY_CODE`**, and resolve dataset identifiers for extracts and formatted outputs.
2. **Shapes (ADM2 geometry):** When **`[COUNTRY_CODE]_dhis2_raw_shapes.parquet`** exists on **`DHIS2_DATASET_EXTRACTS`**, run **`pipelines/snt_dhis2_formatting/code/snt_dhis2_formatting_shapes.ipynb`** to produce **`[COUNTRY_CODE]_shapes.geojson`** (and any companion outputs defined in that notebook).
3. **Pyramid (facility metadata):** When the raw pyramid extract exists, run **`snt_dhis2_formatting_pyramid.ipynb`** to write **`[COUNTRY_CODE]_pyramid.parquet`** and **`.csv`**.
4. **Routine (analytics):** When **`[COUNTRY_CODE]_dhis2_raw_analytics.parquet`** exists, run **`snt_dhis2_formatting_routine.ipynb`** to write **`[COUNTRY_CODE]_routine.parquet`** and **`.csv`** (monthly facility-level rows with administrative keys as produced by the notebook).
5. **Population:** When **`[COUNTRY_CODE]_dhis2_raw_population.parquet`** exists, run **`snt_dhis2_formatting_population.ipynb`** to write **`[COUNTRY_CODE]_population.parquet`** and **`.csv`**.
6. **Reporting rates:** When **`[COUNTRY_CODE]_dhis2_raw_reporting.parquet`** exists, run **`snt_dhis2_formatting_reporting_rates.ipynb`** to write **`[COUNTRY_CODE]_reporting.parquet`** and **`.csv`**.
7. **Publish:** Save pipeline parameters JSON, upload all produced formatted files plus the parameter file to **`DHIS2_DATASET_FORMATTED`** via **`add_files_to_dataset`** (steps whose upstream file is missing are skipped).
8. **Reporting:** Run **`snt_dhis2_formatting_report.ipynb`** to refresh static reporting outputs.

## Inputs

* **`[COUNTRY_CODE]_dhis2_raw_analytics.parquet`**, **`[COUNTRY_CODE]_dhis2_raw_population.parquet`**, **`[COUNTRY_CODE]_dhis2_raw_shapes.parquet`**, **`[COUNTRY_CODE]_dhis2_raw_pyramid.parquet`**, **`[COUNTRY_CODE]_dhis2_raw_reporting.parquet`** on **`DHIS2_DATASET_EXTRACTS`** (each is optional; missing inputs skip the corresponding formatting step).
* **`configuration/SNT_config.json`** for **`COUNTRY_CODE`** and **`SNT_DATASET_IDENTIFIERS`**.

## Outputs

* **`data/dhis2/extracts_formatted/[COUNTRY_CODE]_routine.parquet`** and **`.csv`**
* **`data/dhis2/extracts_formatted/[COUNTRY_CODE]_population.parquet`** and **`.csv`**
* **`data/dhis2/extracts_formatted/[COUNTRY_CODE]_shapes.geojson`**
* **`data/dhis2/extracts_formatted/[COUNTRY_CODE]_pyramid.parquet`** and **`.csv`**
* **`data/dhis2/extracts_formatted/[COUNTRY_CODE]_reporting.parquet`** and **`.csv`**
* **Pipeline parameters JSON** in the same formatted data directory
* Copies of the above artifacts on the **`DHIS2_DATASET_FORMATTED`** OpenHEXA dataset
* **Report outputs** under **`pipelines/snt_dhis2_formatting/reporting/outputs/`**

> **Notes for the Data Analyst:**
>
> - **`Routine` resolution**: The formatted routine product is **monthly** at **facility (OU)** level with **`ADM1_ID`**, **`ADM2_ID`**, and **`PERIOD`** / **`YEAR`** / **`MONTH`** as produced by the routine notebook (see that notebook for the authoritative column list).
> - **`Shapes`**: **`[COUNTRY_CODE]_shapes.geojson`** supplies **ADM2** geometries and identifiers used by downstream SNT pipelines.
> - **Guarded execution**: Each formatting stage is gated on **`dataset_file_exists`** for its raw extract; do not expect outputs for stages whose raw Parquet was never published to **`DHIS2_DATASET_EXTRACTS`**.
