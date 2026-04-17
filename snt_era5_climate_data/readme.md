# SNT ERA5 Climate Data Pipeline

The **SNT ERA5 Climate Data** pipeline synchronizes **ERA5-Land** reanalysis from the Copernicus Climate Data Store (CDS), masks and aggregates gridded fields to **ADM2** boundaries from formatted DHIS2 shapes, and publishes monthly climate summaries for SNT while emitting daily, weekly, and DHIS2-style epidemic-week tables on disk.

---

## Parameters

* **`start_date`** (String, Required):
  * **Name:** Start date
  * **Description:** Start of the extraction window in **`YYYY-MM-DD`** format. The pipeline normalizes this to the **first day of that month** before syncing.
  * **Default:** `2018-01-01`.
* **`end_date`** (String, Optional):
  * **Name:** End date
  * **Description:** End of the extraction window. If omitted, defaults to the **last day of the month before the current UTC month**. If provided, it is adjusted to the **last day of the previous calendar month** when that rule applies (see `to_last_day_previous_month` in **`pipeline.py`**).
* **`cds_connection`** (CustomConnection, Required):
  * **Name:** Climate data store
  * **Description:** OpenHEXA custom connection holding the CDS API credential (read from **`api_key`** or fallback **`key`**). Used to build the ERA5 **`Client`** against **`https://cds.climate.copernicus.eu/api`**.

---

## Functionality Overview

1. **Boundaries:** Loads **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`** via **`get_file_from_dataset`**, derives CDS **`area`** bounds (N/W/S/E integers with padding).
2. **Sync:** For each variable in **`ERA5_VARIABLES`** (currently **`total_precipitation`**), downloads missing GRIB slices via **`prepare_requests`** / **`retrieve_requests`**, converts to a per-variable **zarr** store under `data/era5/raw/`, and validates non-empty samples.
3. **Daily SNT table:** **`build_daily_snt`** masks the raster to ADM2 polygons, builds daily **`mean`**, **`min`**, **`max`** (precipitation uses accumulated-time semantics and millimetre scaling in output), and adds **`week`**, **`period_month`**, **`epi_week`** using DHIS2 week helpers.
4. **Rollups:** **`aggregate_daily_snt`** produces weekly, epidemic-weekly, and monthly aggregates (precipitation uses **sum** over days; other variables would use **mean**).
5. **On-disk outputs:** Writes **`[COUNTRY_CODE]_{variable}_daily|weekly|epi_weekly|monthly.parquet`** under `data/era5/aggregate/{variable}/`.
6. **Dataset upload:** **`add_files_to_dataset`** publishes **monthly** Parquet files (plus parameters JSON) to **`ERA5_DATASET_CLIMATE`**—matching the historical aggregate pipeline upload scope.
7. **Reporting:** Runs **`snt_era5_climate_data_report.ipynb`** regardless of extraction mode.

---

## Inputs

* **`SNT_config.json`**: **`COUNTRY_CODE`**, **`DHIS2_DATASET_FORMATTED`**, **`ERA5_DATASET_CLIMATE`**.
* **`cds_connection`**: Valid CDS API key for programmatic access.
* **Formatted shapes** file in the DHIS2 formatted dataset.

---

## Outputs

* **Zarr stores:** `data/era5/raw/{variable}.zarr` (rebuilt each run for deterministic outputs).
* **Parquet aggregates:** under `data/era5/aggregate/{variable}/` for daily, weekly, epidemic-weekly, and monthly resolutions.
* **Published monthly** Parquet (per configured variable) and **parameters JSON** on **`ERA5_DATASET_CLIMATE`**.
* **Report notebook** outputs under `pipelines/snt_era5_climate_data/reporting/outputs/`.

---
