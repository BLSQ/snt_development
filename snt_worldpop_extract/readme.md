# SNT WorldPop Extract Pipeline

The **SNT WorldPop Extract** pipeline downloads gridded WorldPop population rasters for the configured country, aggregates them to **ADM2** zones from formatted DHIS2 shapes, optionally combines UN-adjusted totals, and publishes outputs to the workspace WorldPop dataset.

---

## Parameters

* **`year`** (Integer, Optional):
  * **Name:** Year
  * **Description:** Calendar year of the WorldPop constrained PPP raster to request.
  * **Default:** `2020`.
* **`overwrite`** (Boolean, Optional):
  * **Name:** Overwrite
  * **Description:** When **`true`**, replaces existing downloaded rasters if present; when **`false`**, skips download if files already exist.
  * **Choices/Default:** Default: **`false`**.

---

## Functionality Overview

1. **Download:** Uses `WorldPopClient` to fetch **`[COUNTRY_CODE]_worldpop_ppp_{year}.tif`** and, for years up to 2020, optionally **`[COUNTRY_CODE]_worldpop_ppp_{year}_UNadj.tif`**.
2. **Zonal statistics:** Loads **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`** via **`get_file_from_dataset`**, filters invalid geometries, aligns CRS to the raster, and runs **`rasterstats.zonal_stats`** (`sum`, `count`) per ADM2 for each available raster.
3. **Formatting:** Merges constrained and UN-adjusted aggregates into **`[COUNTRY_CODE]_worldpop_population.csv`** / **`.parquet`** with a **`YEAR`** column and **`POPULATION_UNADJ`** when the UN-adjusted aggregation exists.
4. **Publication:** Saves parameters JSON and calls **`add_files_to_dataset`** for **`WORLDPOP_DATASET_EXTRACT`** (population tables, `.tif` rasters, parameters).
5. **Reporting:** Runs `snt_worldpop_extract_report.ipynb`.

---

## Inputs

* **`SNT_config.json`**: **`COUNTRY_CODE`** and **`WORLDPOP_DATASET_EXTRACT`**, **`DHIS2_DATASET_FORMATTED`** identifiers.
* **`[COUNTRY_CODE]_shapes.geojson`** from the formatted DHIS2 dataset.

---

## Outputs

* **`data/worldpop/raw/[COUNTRY_CODE]_worldpop_ppp_{year}.tif`** (and optional **`_UNadj.tif`**)
* **`data/worldpop/aggregations/*.parquet`** / **`.csv`** per raster stem
* **`data/worldpop/population/[COUNTRY_CODE]_worldpop_population.parquet`** and **`.csv`**
* **Dataset version** on **`WORLDPOP_DATASET_EXTRACT`** with the above plus parameters JSON

---
