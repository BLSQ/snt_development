# SNT Healthcare Access Pipeline

This pipeline estimates, for each **ADM2**, the share of the population living within a configurable **straight-line buffer** around active **FOSA** (health facility) points. It rasterizes districts on the **population grid**, buffers facilities in a metric CRS, zonalizes population totals and covered counts, and exports **`{COUNTRY_CODE}_population_covered_health`** tables to **`data/healthcare_access/`** and the **`SNT_HEALTHCARE_ACCESS`** dataset when configured.

## Parameters

* **`input_fosa_file`** (File, Optional):
  * **Name:** FOSA location file (.csv)
  * **Description:** Optional upload of facility coordinates. When omitted, the notebook uses DHIS2 pyramid coordinates from the configured formatted dataset.
  * **Choices/Default:** Default: **`None`**.

* **`input_radius_meters`** (Integer, Optional):
  * **Name:** Radius around FOSA (meters)
  * **Description:** Buffer distance for access calculations (converted to kilometres in logs).
  * **Choices/Default:** Default: **`5000`**.

* **`input_pop_file`** (File, Optional):
  * **Name:** Population raster file (.tif)
  * **Description:** Optional population GeoTIFF. When omitted, **`pipeline.py`** searches **`data/worldpop/raw/`** for the newest **`{COUNTRY_CODE}_worldpop_ppp_{YEAR}.tif`** (excluding **`_UNadj`** variants).
  * **Choices/Default:** Default: **`None`**.

* **`input_shapes_file`** (File, Optional):
  * **Name:** Shapes file (.geojson)
  * **Description:** Optional **ADM2** polygons. When omitted, the notebook loads **`{COUNTRY_CODE}_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**.
  * **Choices/Default:** Default: **`None`**.

## Functionality Overview

1. Resolve population raster path (user upload versus latest WorldPop candidate) and validate custom shape extensions (must be **`.geojson`**).
2. Load **`configuration/SNT_config.json`**, determine **`COUNTRY_CODE`**, and run **`pipelines/snt_healthcare_access/code/snt_healthcare_access.ipynb`** via Papermill with **`FOSA_FILE`**, **`RADIUS_METERS`**, **`POP_FILE`**, and **`SHAPES_FILE`**.
3. Inside the notebook: load **ADM2** polygons, facility coordinates, population raster, build buffers in the metric CRS used in the notebook (see notebook for EPSG), rasterize inclusion masks, and compute zonal sums of total versus covered population per **`ADM2_ID`**.
4. Save parameter JSON and upload **`{COUNTRY_CODE}_population_covered_health`** parquet and CSV plus parameters to **`SNT_HEALTHCARE_ACCESS`** when the dataset identifier is present.
5. Execute **`snt_healthcare_access_report.ipynb`**.

## Inputs

* **`configuration/SNT_config.json`**: Country code and dataset identifiers for DHIS2 extracts.
* **Facility coordinates**: Either **`input_fosa_file`** or **`{COUNTRY_CODE}_pyramid.parquet`** (and related tables as implemented in the notebook).
* **Population raster**: **`input_pop_file`** or the latest **`{COUNTRY_CODE}_worldpop_ppp_*.tif`** under **`data/worldpop/raw/`**.
* **ADM2 polygons**: **`input_shapes_file`** or **`{COUNTRY_CODE}_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**.

## Outputs

* **`{COUNTRY_CODE}_population_covered_health.parquet`** and **`.csv`**: **ADM2**-level totals, covered population counts, and derived coverage fractions.
* **Pipeline parameters JSON** saved under **`data/healthcare_access/`**.
* **Intermediate rasters** (if written during notebook execution) and **reporting artefacts** under **`reporting/outputs/`**.

> **Notes for the Data Analyst:**
>
> - **`Spatial resolution`**: Analyses inherit the **native cell size** of the population GeoTIFF; **ADM2** summaries come from **zonal statistics** on that grid.
> - **`input_radius_meters`**: Describes **Euclidean** buffers around facility points; results are not network travel times.
> - **`input_shapes_file`**: Custom boundaries risk **`ADM2_ID`** mismatches versus DHIS2 pyramid extracts, producing missing coverage for affected districts.
> - **`input_pop_file`**: When no raster is supplied and none exists locally, the computation step cannot run; place an extract in **`data/worldpop/raw/`** or upload a raster explicitly.
> - **`CRS choice`**: Buffer operations use a fixed metric CRS inside the notebook; very large countries may need methodological review for edge districts.
