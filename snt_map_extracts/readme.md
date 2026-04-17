# SNT Map Extracts Pipeline

The **SNT Map Extracts** pipeline downloads Malaria Atlas Project (MAP) rasters for a fixed set of malaria and intervention indicators, computes **ADM2** zonal statistics (mean and optional population-weighted summaries), and publishes long-format tables for downstream assembly.

---

## Parameters

* **`pop_raster_selection`** (File, Optional):
  * **Name:** Population raster selection (.tif)
  * **Description:** Population **`.tif`** used for population-weighted metrics and total-population denominators; must exist on disk when provided.
  * **Default:** `None` (unweighted branch only).
* **`target_year`** (String, Required):
  * **Name:** Target Year
  * **Description:** Target calendar year passed to MAP downloads (e.g. **`2022`**); the MAP client may fall back when a layer is unavailable.
  * **Choices/Default:** Required string (no default in **`pipeline.py`**).

---

## Functionality Overview

1. **Indicator set (code):** Downloads MAP rasters under categories **Malaria** (`Pf_Parasite_Rate`, `Pf_Mortality_Rate`, `Pf_Incidence_Rate`) and **Interventions** (ITN access/use, IRS coverage, antimalarial effective treatment).
2. **Shapes:** Loads **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**, drops null/empty geometries, then retrieves rasters clipped to the country extent.
3. **Zonal stats:** For each raster band (**Data**, **LCI**, **UCI**, **GRAY_INDEX** when present), computes polygon means; optionally aligns the metric grid to the population raster and adds **`population_weighted`**.
4. **Output layout:** Writes long-format **`[COUNTRY_CODE]_map_data.parquet`** / **`.csv`** under `data/map/formatted/{country}/` with uppercase columns including **`METRIC_CATEGORY`**, **`METRIC_NAME`**, **`STATISTIC`**, **`VALUE`**, **`YEAR`**, **`VERSION`**.
5. **Dataset upload:** Publishes parquet, CSV, and parameters JSON to **`SNT_MAP_EXTRACTS`**.
6. **Logging:** Writes timestamped log files under `pipelines/snt_map_extracts/logs/`.
7. **Reporting:** Runs `snt_map_extracts_report.ipynb`.

---

## Inputs

* **`SNT_config.json`**: **`COUNTRY_CODE`**, **`DHIS2_DATASET_FORMATTED`**, **`SNT_MAP_EXTRACTS`**.
* **Optional population raster** file path from **`pop_raster_selection`**.

---

## Outputs

* **`data/map/formatted/[COUNTRY_CODE]/[COUNTRY_CODE]_map_data.parquet`**
* **`data/map/formatted/[COUNTRY_CODE]/[COUNTRY_CODE]_map_data.csv`**
* **Cached rasters** under `data/map/raster_files/[COUNTRY_CODE]/`
* **Published files** on **`SNT_MAP_EXTRACTS`** (parquet, csv, parameters)
* **Report outputs** under `pipelines/snt_map_extracts/reporting/outputs/`

---
