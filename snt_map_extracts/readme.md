# SNT Map Extracts Pipeline

The **SNT Map Extracts** pipeline downloads Malaria Atlas Project (MAP) rasters for a fixed set of malaria and intervention indicators, computes per-**ADM2** medians and population-weighted medians for each year of the selected period, and publishes them as one wide table, **`[COUNTRY_CODE]_map_data.parquet`** / **`.csv`**, to **`SNT_MAP_EXTRACTS`**. It then runs the MAP reporting notebook.

## Parameters

* **`year_start`** (Integer, Required):
  * **Name:** Year start
  * **Description:** First year of the extraction. Also the first year for which a WorldPop population raster is used for the population weighting. WorldPop rasters are available for 2015–2030: a year outside that range logs a warning at start, then stops the run when its population raster is downloaded (see **Guarded execution**).
  * **Default:** none.
* **`year_end`** (Integer, Required):
  * **Name:** Year end
  * **Description:** Last year of the extraction, and the year shown in the report's maps. Must be greater than or equal to **`year_start`**, otherwise the run fails before any computation.
  * **Default:** none.

## Functionality Overview

1. **Validation:** Fail if **`year_start`** > **`year_end`**. Log a warning if either year is outside 2015–2030 (WorldPop availability); the run continues.
2. **Configuration:** Load and validate **`SNT_config.json`** and read **`COUNTRY_CODE`**.
3. **Shapes:** Load **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`** and drop organisation units with a null or empty geometry. The run fails if no shape is left. Steps 4–7 are skipped when **`run_report_only`** is set.
4. **Parameters:** Save **`[COUNTRY_CODE]_parameters.json`** under **`data/map/`**.
5. **For each year from `year_start` to `year_end`:**
   1. **Population:** Use the WorldPop raster cached under **`data/worldpop/rasters/`** (**`[country_code]_pop_[YEAR]_*.tif`**, lowercase country code), or download it from WorldPop. The run fails if the download fails, including for a year outside 2015–2030. Write an ADM2 total-population table to **`data/map/aggregated_populations/`**.
   2. **Rasters:** Download the MAP raster of each indicator for the shapes' extent into **`data/map/raster_files/[COUNTRY_CODE]/`**. The MAP release (version) is selected per indicator for the target year, so indicators can come from different releases. An indicator that fails to download is logged as an error and skipped.
   3. **Zonal statistics:** For each raster band (**`Data`**, **`LCI`**, **`UCI`**, or **`GRAY_INDEX`** for single-band rasters), compute the ADM2 median, plus the population-weighted median when a population raster is available. **`Data`** and **`GRAY_INDEX`** give the **`MEDIAN`** statistic.
   4. Write the long per-year table **`data/map/formatted/[COUNTRY_CODE]_map_data_[YEAR].parquet`** / **`.csv`**, one row per ADM2 × indicator × **`STATISTIC`** (**`MEDIAN`**, **`LCI`**, **`UCI`**). A year with no raster or no valid statistic writes nothing.
6. **SNT format:** Keep only the **`MEDIAN`** statistic and combine the per-year tables into one wide table, one row per ADM2 × **`YEAR`**, with a value column and a **`_POP_W`** column per indicator. Both columns of each indicator are multiplied by a factor fixed in the pipeline code (**`conversions`** in **`format_map_indicators`**) and rounded to 2 decimals (see **Units** below). The **`LCI`** / **`UCI`** rows stay in the per-year tables only. Write **`data/map/formatted_snt/[COUNTRY_CODE]_map_data.parquet`** / **`.csv`**. The run fails if no per-year table could be read, or if these files were not written during this run.
7. **Dataset publication:** Upload **`[COUNTRY_CODE]_map_data.parquet`**, **`[COUNTRY_CODE]_map_data.csv`** and **`[COUNTRY_CODE]_parameters.json`** to **`SNT_MAP_EXTRACTS`**.
8. **Reporting:** Run **`snt_map_extracts_report.ipynb`** (also in report-only mode), or **`snt_map_extracts_report_[COUNTRY_CODE].ipynb`** when that country-specific variant exists in the workspace. The run fails if the report fails.
   * The report reads **`[COUNTRY_CODE]_map_data.parquet`** and **`[COUNTRY_CODE]_parameters.json`** from the latest version of **`SNT_MAP_EXTRACTS`**, and the shapes from **`DHIS2_DATASET_FORMATTED`**.
   * For each indicator, it draws two ADM2 maps side by side for **`YEAR_END`**: the median and the population-weighted median, on a shared colour scale. An indicator with no value that year is not mapped.

## Inputs

* **`[COUNTRY_CODE]_shapes.geojson`** on **`DHIS2_DATASET_FORMATTED`** — required; produced by **`snt_dhis2_formatting`**.
* **MAP rasters** downloaded from the Malaria Atlas Project — required per indicator; a missing indicator is skipped.
* **WorldPop population rasters** (2015–2030) — cached under **`data/worldpop/rasters/`**, a cache shared with the other WorldPop-based pipelines, or downloaded from WorldPop. A failed download stops the run.
* **`configuration/SNT_config.json`**: **`SNT_CONFIG.COUNTRY_CODE`**, and under **`SNT_DATASET_IDENTIFIERS`**: **`DHIS2_DATASET_FORMATTED`** and **`SNT_MAP_EXTRACTS`**.

## Outputs

**Workspace filesystem**

* **`data/map/formatted_snt/[COUNTRY_CODE]_map_data.parquet`** and **`.csv`** — the wide table
* **`data/map/formatted/[COUNTRY_CODE]_map_data_[YEAR].parquet`** and **`.csv`** — one long table per year (not published), with all three statistics (**`MEDIAN`**, **`LCI`**, **`UCI`**), each with its value (**`VALUE`**) and population-weighted value (**`POPULATION_WEIGHTED`**)
* **`data/map/aggregated_populations/[COUNTRY_CODE]_worldpop_population_[YEAR].parquet`** — ADM2 total population per year (not published, not used by the rest of the pipeline)
* **`data/map/[COUNTRY_CODE]_parameters.json`**
* **Cached MAP rasters** under **`data/map/raster_files/[COUNTRY_CODE]/`** and **WorldPop rasters** under **`data/worldpop/rasters/`**
* **Report outputs** under **`pipelines/snt_map_extracts/reporting/outputs/`**, maps as **`figures/[COUNTRY_CODE]_map_[INDICATOR]_[YEAR].png`**

**Published to `SNT_MAP_EXTRACTS`**

* **`[COUNTRY_CODE]_map_data.parquet`**
* **`[COUNTRY_CODE]_map_data.csv`**
* **`[COUNTRY_CODE]_parameters.json`** (keys in uppercase: **`YEAR_START`**, **`YEAR_END`**, …)

> **Notes for the Data Analyst:**
>
> - **Grain:** ADM2 × **`YEAR`**, one row each, across all years of the period.
> - **Key columns:** **`YEAR`**, **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**.
> - **Statistic:** values are the **`MEDIAN`** statistic (from the **`Data`** / **`GRAY_INDEX`** band, the indicator's main estimate); MAP's credible-interval bounds (**`LCI`** / **`UCI`**) are not published, but remain available in the per-year tables under **`data/map/formatted/`**.
> - **Indicator columns:** **`PF_PARASITE_RATE`**, **`PF_INCIDENCE_RATE`**, **`PF_MORTALITY_RATE`**, **`INSECTICIDE_TREATED_NET_ACCESS`**, **`INSECTICIDE_TREATED_NET_USE_RATE`**, **`IRS_COVERAGE`**, **`ANTIMALARIAL_EFFECTIVE_TREATMENT`**, each followed by its **`_POP_W`** column.
>   - **`[INDICATOR]`**: median of the raster pixels covering the ADM2, weighted by the share of each pixel inside the ADM2.
>   - **`[INDICATOR]_POP_W`**: median of the same pixels, weighted by pixel coverage × WorldPop population (resampled onto the MAP grid; pixels without population weigh 0). Empty when the year's population raster could not be used (no CRS, or unreadable).
>   - Near borders, a MAP pixel (about 5 km) can extend into a neighbouring country. Those pixels have no WorldPop population in this country, so they affect **`[INDICATOR]`** but not **`[INDICATOR]_POP_W`**. For border districts, prefer **`_POP_W`**.
> - **Units** (same for an indicator and its **`_POP_W`** column), from MAP's native values times a factor fixed in the pipeline code, rounded to 2 decimals:
>   - **`PF_PARASITE_RATE`**, **`INSECTICIDE_TREATED_NET_ACCESS`**, **`INSECTICIDE_TREATED_NET_USE_RATE`**, **`IRS_COVERAGE`**, **`ANTIMALARIAL_EFFECTIVE_TREATMENT`**: percentage (× 100).
>   - **`PF_INCIDENCE_RATE`**: cases per 1,000 people per year (× 1,000).
>   - **`PF_MORTALITY_RATE`**: deaths per 100,000 people per year (× 100,000).
>   - The per-year tables under **`data/map/formatted/`** keep MAP's native, unscaled values.
> - **MAP version:** not stored in the output. Indicators of the same year can come from different MAP releases.
> - **Guarded execution:**
>   - A MAP indicator that fails to download is skipped with an error in the log; its columns stay empty.
>   - A year whose rasters cannot be downloaded or aggregated is skipped; its rows are missing from the table.
>   - A WorldPop raster that cannot be used (no CRS, or unreadable) leaves the **`_POP_W`** columns empty for that year. A WorldPop download that fails, including for a year outside 2015–2030, stops the run.
