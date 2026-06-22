# SNT Healthcare Access Pipeline

This pipeline estimates, for each **ADM2**, the share of the population living within a configurable **straight-line buffer** around active **FOSA** (health facility) points. It rasterizes districts on the **population grid**, buffers facilities in a metric CRS, zonalizes population totals and covered counts, and exports **`{COUNTRY_CODE}_population_covered_health`** tables to **`data/healthcare_access/`** and the **`SNT_HEALTHCARE_ACCESS`** dataset when configured.

## Parameters

* **`input_fosa_file`** (File, Optional):
  * **Name:** Optional FOrmation SAnitaire (FOSA) location file (.csv)
  * **Description:** Optional upload of health facility coordinate file (.csv). When omitted, the notebook uses DHIS2 pyramid coordinates from the configured formatted dataset.
  * **Choices/Default:** Default: **`None`**.

* **`wpop_year`** (Integer, Mandatory):
  * **Name:** Reference year for WorldPop population data.
  * **Description:** Year for the population raster data to download from WorldPop. Should be between 2015 and 2030.
  * **Choices/Default:** Default: **`None`**.


## Functionality Overview

1. Validation:
  - If an **`input_fosa_file`** exists: validate custom extension (must be **`.csv`**).
  - Validate the **`wpop_year`** to download WorldPop population raster data.
2. Load **`configuration/SNT_config.json`**, determine **`COUNTRY_CODE`**, and run **`pipelines/snt_healthcare_access/code/snt_healthcare_access.ipynb`** via Papermill with **`FOSA_FILE`** and **`wpop_year`**.
3. Inside the notebook: load **ADM2** polygons, facility coordinates, population raster, build buffers in the metric CRS used in the notebook (see notebook for EPSG), rasterize inclusion masks, and compute zonal sums of total versus covered population per **`ADM2_ID`**.
4. Save parameter JSON and upload **`{COUNTRY_CODE}_population_covered_health`** parquet and CSV plus parameters to **`SNT_HEALTHCARE_ACCESS`** when the dataset identifier is present.
5. Execute **`snt_healthcare_access_report.ipynb`**.

## Inputs

* **`configuration/SNT_config.json`**: Country code and dataset identifiers for DHIS2 extracts.
* **Facility coordinates**: Either **`input_fosa_file`** or **`{COUNTRY_CODE}_pyramid.parquet`** (and related tables as implemented in the notebook).
* **Population raster**: The corresponding population data, **`{country_code.lower()}_pop_{wpop_year}_CN_100m_*.tif`**, under **`data/worldpop/rasters/`**.
* **ADM2 polygons**: **`{COUNTRY_CODE}_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**.

## Outputs

* **`{COUNTRY_CODE}_population_covered_health.parquet`** and **`.csv`**: **ADM2**-level totals, covered population counts, and derived coverage fractions.
* **Pipeline parameters JSON** saved under **`data/healthcare_access/`**.
* **Intermediate rasters** (if written during notebook execution) and **reporting artefacts** under **`reporting/outputs/`**.

> **Notes for the Data Analyst:**
>
> - **`Spatial resolution`**: Analyses inherit the **native cell size** of the population GeoTIFF; **ADM2** summaries come from **zonal statistics** on that grid.
> - **Radius**: The 5000m meters around health facilities describe **Euclidean** buffers around facility points; results are not network travel times.
> - **`input_fosa_file`**: If an optional FOSA location file is supplied, it must be .csv and must have the mandatory **`LATITUDE`** and **`LONGITUDE`** columns; failing that, the pipeline defaults to using DHIS2 data from **`DHIS2_DATASET_FORMATTED`**
> - **`CRS choice`**: Buffer operations use a fixed metric CRS inside the notebook; very large countries may need methodological review for edge districts.
