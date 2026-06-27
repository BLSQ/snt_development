# SNT Map Extracts Pipeline

The **SNT Map Extracts** pipeline downloads Malaria Atlas Project (MAP) rasters for a fixed set of malaria and intervention indicators, computes **ADM2** zonal statistics (mean and optional population-weighted summaries of malaria-related health indicators and intervention metrics), and publishes long-format aggregated result tables for downstream assembly.


## Parameters

- `**pop_raster_selection**` (File, Optional):
  - **Name:** Population raster selection (.tif)
  - **Description:** Population `**.tif**` used for population-weighted metrics and total-population denominators; must exist on disk when provided.
  - **Default:** `None` (unweighted branch only).
- `**year_start**` (Integer, Required):
  - **Name:** Start Year
  - **Description:** Start calendar year passed to MAP downloads (e.g. `**2020**`); the MAP client may fall back when a layer is unavailable.
  - **Choices/Default:** Required integer (no default in `**pipeline.py**`).
- `**year_end**` (Integer, Required):
  - **Name:** End Year
  - **Description:** End calendar year passed to MAP downloads (e.g. `**2022**`); the MAP client may fall back when a layer is unavailable.
  - **Choices/Default:** Required integer (no default in `**pipeline.py**`).


## Functionality Overview

1. **Indicator set (code):** Downloads MAP rasters under categories **Malaria** (`Pf_Parasite_Rate`, `Pf_Mortality_Rate`, `Pf_Incidence_Rate`) and **Interventions** (ITN access/use, IRS coverage, antimalarial effective treatment).
2. **Shapes:** Loads **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**, drops null/empty geometries, then retrieves rasters clipped to the country extent.
3. **Zonal stats:** For each raster band (**Data**, **LCI**, **UCI**, **GRAY_INDEX** when present), computes polygon means; optionally aligns the metric grid to the population raster and adds **`population_weighted`**.
4. **Output layout:** Writes long-format **`[COUNTRY_CODE]_map_data.parquet`** / **`.csv`** under `data/map/formatted/{country}/` with uppercase columns including **`METRIC_CATEGORY`**, **`METRIC_NAME`**, **`STATISTIC`**, **`VALUE`**, **`YEAR`**, **`VERSION`**.
5. **Dataset upload:** Publishes parquet, CSV, and parameters JSON to **`SNT_MAP_EXTRACTS`**.
6. **Logging:** Writes timestamped log files under `pipelines/snt_map_extracts/logs/`.
7. **Reporting:** Runs `snt_map_extracts_report.ipynb` to create visualizations of each indicator, for the most recent year available (`**year_end**`).


## Inputs

* **`SNT_config.json`**: **`COUNTRY_CODE`**, **`DHIS2_DATASET_FORMATTED`**
* **Optional population raster** file path from **`pop_raster_selection`**.


## Outputs

* **`data/map/formatted/[COUNTRY_CODE]/[COUNTRY_CODE]_map_data.parquet`**
* **`data/map/formatted/[COUNTRY_CODE]/[COUNTRY_CODE]_map_data.csv`**
* **Cached rasters** under `data/map/raster_files/[COUNTRY_CODE]/`
* **Published files** on **`SNT_MAP_EXTRACTS`** (parquet, csv, parameters)
* **Report outputs** under `pipelines/snt_map_extracts/reporting/outputs/`

> **Notes for the Data Analyst:**
> - Logic
>    - Validates input periods and loads SNT configuration
>    - Loads geographic boundary data (shapes) from the dataset, validates geometries
>    - Defines indicators to extract: Malaria metrics (parasite rate, mortality, incidence) and Interventions (net access, IRS coverage, antimalarial treatment)
>    - For each year in the range:
>       - Downloads/retrieves WorldPop population rasters using ISO country codes
>       - Builds map statistics by intersecting health indicators with geographic shapes
>    - Aggregates data using population weighting
>    - Uploads results to a DHIS2 dataset
>    - Runs a reporting notebook to visualize results