# SNT Map Extracts Pipeline

The **SNT Map Extracts** pipeline downloads Malaria Atlas Project (MAP) rasters for a fixed set of malaria and intervention indicators, computes **ADM2** zonal statistics (mean and optional population-weighted summaries of malaria-related health indicators and intervention metrics), and publishes long-format aggregated result tables for downstream assembly.


## Parameters

- **`year_start`** (Integer, Required):
  - **Name:** Year start
  - **Description:** Start year of indicators selection (e.g. **`2022`**). Also used as the start of the WorldPop population period (valid range 2015-2030).
- **`year_end`** (Integer, Required):
  - **Name:** Year end
  - **Description:** End year of indicators selection (e.g. **`2023`**). Also used as the end of the WorldPop population period (valid range 2015-2030).
- **`run_report_only`** (Boolean, Default `False`):
  - **Name:** Run reporting only
  - **Description:** Skips extraction/aggregation and only (re)executes the reporting notebook against previously published outputs.
- **`pull_scripts`** (Boolean, Default `False`):
  - **Name:** Pull Scripts
  - **Description:** Pulls the latest reporting notebook (`snt_map_extracts_report.ipynb`) from the scripts repository before running.


## Functionality Overview

1. **Indicator set (code):** Downloads MAP rasters under categories **Malaria** (`Pf_Parasite_Rate`, `Pf_Mortality_Rate`, `Pf_Incidence_Rate`) and **Interventions** (ITN access/use, IRS coverage, antimalarial effective treatment).
2. **Shapes:** Loads **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**, drops organisation units with null or empty geometries.
3. **For each year between `year_start` and `year_end`:**
   - **Population:** Looks for an existing WorldPop raster under `data/worldpop/rasters/`; if missing, downloads it via the WorldPop API. Generates an ADM2 total-population table from the raster (saved under `data/map/aggregated_populations/`).
   - **Rasters:** Downloads MAP rasters for each indicator, clipped to the country extent, cached under `data/map/raster_files/[COUNTRY_CODE]/`.
   - **Zonal stats:** For each raster band (**Data**, **LCI**, **UCI**, **GRAY_INDEX** when present), computes polygon means; when the population raster is available, aligns the metric grid to it and adds a **`population_weighted`** column.
   - Writes a long-format **`[COUNTRY_CODE]_map_data_[YEAR].parquet`** / **`.csv`** with uppercase columns including **`METRIC_CATEGORY`**, **`METRIC_NAME`**, **`STATISTIC`**, **`VALUE`**, **`YEAR`**, **`VERSION`**.
4. **Dataset upload:** Publishes all per-year parquet/CSV outputs plus the parameters JSON to **`SNT_MAP_EXTRACTS`**.
5. **Reporting:** Runs `snt_map_extracts_report.ipynb` (or a country-specific variant, e.g. `snt_map_extracts_report_[COUNTRY_CODE].ipynb`, when present) to create visualizations of each indicator.


## Inputs

* **`SNT_config.json`**: **`COUNTRY_CODE`**, **`DHIS2_DATASET_FORMATTED`**, **`SNT_MAP_EXTRACTS`**


## Outputs

* **`data/map/formatted/[COUNTRY_CODE]_map_data_[YEAR].parquet`** (one file per year in range)
* **`data/map/formatted/[COUNTRY_CODE]_map_data_[YEAR].csv`**
* **`data/map/aggregated_populations/[COUNTRY_CODE]_worldpop_population_[YEAR].parquet`**
* **Cached MAP rasters** under `data/map/raster_files/[COUNTRY_CODE]/`
* **Cached WorldPop rasters** under `data/worldpop/rasters/`
* **Published files** on **`SNT_MAP_EXTRACTS`** (parquet, csv, parameters)
* **Report outputs** under `pipelines/snt_map_extracts/reporting/outputs/`

> **Notes for the Data Analyst:**
> - Logic
>    - Validates input periods and loads SNT configuration
>    - Loads geographic boundary data (shapes) from the dataset, validates geometries
>    - Defines indicators to extract: Malaria metrics (parasite rate, mortality, incidence) and Interventions (net access, IRS coverage, antimalarial treatment)
>    - For each year in the range:
>       - Downloads/retrieves WorldPop population rasters using ISO country codes, and builds a population table
>       - Builds map statistics by intersecting health indicators with geographic shapes
>       - Aggregates data using population weighting, when a population raster is available
>    - Uploads results to the **`SNT_MAP_EXTRACTS`** dataset
>    - Runs a reporting notebook to visualize results