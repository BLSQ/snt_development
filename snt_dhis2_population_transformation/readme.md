# SNT DHIS2 Population Transformation Pipeline

This pipeline takes the formatted DHIS2 population table from the standard formatting dataset and applies three optional transformation stages: (1) scaling the `POPULATION` column to a national reference total, (2) creating disaggregated population columns from user-supplied proportions and/or an uploaded CSV, and (3) projecting population figures across years using a growth rate. It writes refreshed population Parquet and CSV extracts and attaches them to the population-transformation dataset.

Only the **`POPULATION`** column extracted from DHIS2 is used as the basis for all transformations; any other population indicators present in the source data are ignored.

## Parameters

### Part 1 — Population adjustment

* **`tot_pop_reference`** (int, Optional):
  * **Name:** Population reference
  * **Description:** Total national population used to scale DHIS2 population data. When provided, population values are adjusted proportionally so that the national total matches this reference.
  * **Example:** `1000000` for a total population of 1 million.
  * **Default:** `None` (no adjustment applied).
* **`tot_pop_reference_year`** (int, Optional):
  * **Name:** Population year reference
  * **Description:** Year in the population data to which **`tot_pop_reference`** applies. Must be available in the data. If omitted or invalid, defaults to the latest available year.
  * **Example:** `2025`.
  * **Default:** `None` (latest available year used).

### Part 2 — Disaggregation

* **`pop_under_5`** (float, Optional): Proportion of total population aged under 5 (e.g. `0.17` for 17%).
* **`pop_pregnant_women`** (float, Optional): Proportion of total population that are pregnant women (e.g. `0.05` for 5%).
* **`pop_0_1_y`** (float, Optional): Proportion aged 0–1 years (e.g. `0.04`).
* **`pop_1_2_y`** (float, Optional): Proportion aged 1–2 years (e.g. `0.03`).
* **`pop_5_10_y`** (float, Optional): Proportion aged 5–10 years (e.g. `0.06`).
* **`pop_5_36_m`** (float, Optional): Proportion aged 5–36 months (e.g. `0.06`).
* **`disaggregation_file`** (File, Optional):
  * **Name:** Use disaggregation proportions (.csv)
  * **Description:** User-uploaded CSV with ADM2-level disaggregation proportions. Columns in the CSV that match a disaggregation already computed from the parameters above will **overwrite** those values; unmatched columns are added as new disaggregation columns.
  * **Default:** `None` (no file).

### Part 3 — Projections

* **`growth_factor`** (float, Optional):
  * **Name:** Projection growth rate
  * **Description:** Annual growth rate used to project population figures into past and future years (e.g. `0.03` for 3%).
  * **Default:** `None` (no projection applied).
* **`growth_reference_year`** (int, Optional):
  * **Name:** Projection reference year
  * **Description:** Base year from which projections are calculated. Must be available in the population data. If omitted or invalid, defaults to the latest available year.
  * **Default:** `None` (latest available year used).

### Utility

* **`run_report_only`** (bool, Optional): When `true`, skips all transformation steps and only executes the reporting notebook. Default: `false`.
* **`pull_scripts`** (bool, Optional): When `true`, pulls the latest code and report notebooks from the repository before running. Default: `false`.

## Functionality Overview

1. Ensure `pipelines/snt_dhis2_population_transformation` and `data/dhis2/population_transformed` exist; optionally pull code/report notebooks from the repository.
2. Load and validate `configuration/SNT_config.json`, read **`COUNTRY_CODE`**, and abort early with an error if a `disaggregation_file` was supplied but the path does not exist on disk.
3. Retrieve the years available in the DHIS2 formatted population data; abort if no data is found.
4. **Part 1 — Adjustment:** if **`tot_pop_reference`** is provided, resolve the reference year (defaulting to the latest available if omitted or invalid) and pass both values to the notebook to scale the **`POPULATION`** column proportionally.
5. **Part 2 — Disaggregation:** apply the proportion parameters to compute named disaggregation columns from **`POPULATION`**. If **`disaggregation_file`** is also provided, CSV proportions are applied after the parameter-based columns: any column name that matches an existing disaggregation will overwrite it; new columns are appended.
6. **Part 3 — Projection:** if **`growth_factor`** is provided, resolve the reference year and project the **`POPULATION`** column backward and forward from that year using the annual growth rate.
7. Persist all parameters to a JSON file via **`save_pipeline_parameters`** in `data/dhis2/population_transformed/`.
8. **`dhis2_population_transformation`** checks whether **`{COUNTRY_CODE}_population.parquet`** exists in **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`**; if not, it logs a warning and skips the notebook entirely (no new outputs in that run).
9. When data exist, run `code/snt_dhis2_population_transformation.ipynb` with **`SNT_ROOT_PATH`** and the parameters above.
10. Write **`{COUNTRY_CODE}_population.parquet`** and **`{COUNTRY_CODE}_population.csv`** to `data/dhis2/population_transformed/`, upload them with the parameters JSON to **`SNT_DATASET_IDENTIFIERS.DHIS2_POPULATION_TRANSFORMATION`**, and execute `reporting/snt_dhis2_population_transformation_report.ipynb`.

## Inputs

* **`configuration/SNT_config.json`**: administration labels and dataset identifiers.
* **`{COUNTRY_CODE}_population.parquet`** from **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`**: only the **`POPULATION`** column is used; other indicators are ignored.
* **Optional `disaggregation_file`**: user CSV with ADM2-level proportion columns.

## Outputs

* **`data/dhis2/population_transformed/{COUNTRY_CODE}_population.parquet`**
* **`data/dhis2/population_transformed/{COUNTRY_CODE}_population.csv`**
* **Pipeline parameters JSON** from `save_pipeline_parameters` in the same folder.
* **Dataset:** files registered on **`SNT_DATASET_IDENTIFIERS.DHIS2_POPULATION_TRANSFORMATION`**.

> **Notes for the Data Analyst:**
>
> - **Source column:** Only **`POPULATION`** from the DHIS2 extract is used. Any other population indicators in the source file are dropped.
> - **Part 1 scaling:** When **`tot_pop_reference`** is set, the **`POPULATION`** column is replaced with proportionally scaled values anchored to the reference total for the specified year.
> - **Part 2 precedence:** Disaggregation columns are first computed from the proportion parameters; if a **`disaggregation_file`** is provided, its columns overwrite any matching parameter-derived columns. Provide the CSV only when you need ADM2-level variation in proportions.
> - **Part 3 projections:** Projection uses the resolved **`growth_reference_year`** as the base. If the year is not provided or not found in the data, the pipeline falls back to the latest available year with a warning.
> - **Reference year fallback:** Both **`tot_pop_reference_year`** and **`growth_reference_year`** default silently to the latest available year when omitted or invalid — check the run logs to confirm which year was actually used.
