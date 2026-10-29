# SNT Healthcare Access Pipeline

This pipeline estimates, for each **ADM2**, the share of the population living within a **5 km straight-line buffer** around active **FOSA** (health facility) points. It rasterizes districts on the **population grid**, buffers facilities in a metric CRS, zonalizes population totals and covered counts, and exports **`{COUNTRY_CODE}_population_covered_health`** tables to **`data/healthcare_access/`** and the **`SNT_HEALTHCARE_ACCESS`** dataset.

## Parameters

* **`input_fosa_file`** (File, Optional):
  * **Name:** Optional FOrmation SAnitaire (FOSA) location file (.csv)
  * **Description:** Optional upload of health facility coordinate file (.csv). When omitted or unusable, the notebook uses the DHIS2 pyramid coordinates (**`{COUNTRY_CODE}_pyramid.parquet`** from **`DHIS2_DATASET_FORMATTED`**).
  * **Choices/Default:** Default: **`None`**.

* **`wpop_year`** (Integer, Mandatory):
  * **Name:** Year
  * **Description:** Reference year for the WorldPop population raster, downloaded if not already present. Must be between 2015 and 2030; the run stops otherwise.
  * **Choices/Default:** No default (required).


## Functionality Overview

1. Validation:
  - If an **`input_fosa_file`** is supplied: check that it exists and has a **`.csv`** extension. A failed check is logged as a warning and the run continues; the notebook then falls back to the DHIS2 pyramid.
  - Stop the run if **`wpop_year`** is outside 2015–2030.
2. Load **`configuration/SNT_config.json`**, determine **`COUNTRY_CODE`**, and retrieve the **ADM2** shapes (**`{COUNTRY_CODE}_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**), dropping units with missing or empty geometry; the run stops if no valid shapes remain.
3. Save the parameters JSON, download the WorldPop raster for **`wpop_year`** into **`data/worldpop/rasters/`** if it is not already there (the run stops if it cannot be retrieved), and run **`pipelines/snt_healthcare_access/code/snt_healthcare_access.ipynb`** via Papermill with **`INPUT_FOSA_FILE`** and **`WORLDPOP_YEAR`**.
4. Inside the notebook: load **ADM2** polygons, facility coordinates, population raster, build buffers in the metric CRS used in the notebook (see notebook for EPSG), rasterize inclusion masks, and compute zonal sums of total versus covered population per **`ADM2_ID`**.
5. Upload **`{COUNTRY_CODE}_population_covered_health`** parquet and CSV plus the parameters JSON to **`SNT_HEALTHCARE_ACCESS`**; the run fails if that dataset identifier is missing from the configuration.
6. Execute **`snt_healthcare_access_report.ipynb`**.

## Inputs

* **`configuration/SNT_config.json`**: Country code and dataset identifiers for DHIS2 extracts.
* **Facility coordinates**: Either **`input_fosa_file`** or **`{COUNTRY_CODE}_pyramid.parquet`** from **`DHIS2_DATASET_FORMATTED`**. For **NER** only, **`data/dhis2/extracts_raw/organisation_unit_groups/NER_organisation_unit_groups.parquet`** (if present) is also read to drop closed facilities.
* **Population raster**: The corresponding population data, **`{country_code.lower()}_pop_{wpop_year}_CN_100m_*.tif`**, under **`data/worldpop/rasters/`**.
* **ADM2 polygons**: **`{COUNTRY_CODE}_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**.

## Outputs

* **`{COUNTRY_CODE}_population_covered_health.parquet`** and **`.csv`**: one row per **ADM2** with the ADM1/ADM2 IDs and names, total population (**`POP_TOTAL`**), covered population (**`POP_COVERED`**) and the coverage percentage (**`PCT_HEALTH_ACCESS`**, 0–100).
* **Pipeline parameters JSON** saved under **`data/healthcare_access/`**.
* **Intermediate files** under **`pipelines/snt_healthcare_access/intermediate_results/`**: the filtered facility points and coverage buffers (**`{COUNTRY_CODE}_FOSA_filtered.gpkg`**, **`{COUNTRY_CODE}_health_coverage_buffers.gpkg`**, both read by the reporting notebook) and the covered-population raster (**`.pop_covered_healthcare.tif`**).
* **Reporting artefacts** under **`reporting/outputs/`**.

> **Notes for the Data Analyst:**
>
> - **`Spatial resolution`**: Analyses inherit the **native cell size** of the population GeoTIFF; **ADM2** summaries come from **zonal statistics** on that grid.
> - **Radius**: The 5000 m (5 km) radius around health facilities describes **Euclidean** buffers around facility points; results are not network travel times.
> - **`input_fosa_file`**: If an optional FOSA location file is supplied, it must be a readable .csv with numeric **`LATITUDE`** and **`LONGITUDE`** columns (column names are matched case-insensitively); otherwise the notebook logs the reason and falls back to **`{COUNTRY_CODE}_pyramid.parquet`** from **`DHIS2_DATASET_FORMATTED`**.
> - **`Closed facilities`**: Facilities with a **`CLOSED_DATE`** are dropped before buffering, as are facilities outside the country boundaries or without coordinates. **NER** has extra name- and group-based filters (see the notebook).
> - **`CRS choice`**: Buffer operations use a fixed metric CRS inside the notebook; very large countries may need methodological review for edge districts.
