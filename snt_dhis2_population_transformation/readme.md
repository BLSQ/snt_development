# SNT DHIS2 Population Transformation Pipeline

This pipeline takes the formatted DHIS2 population table from the standard formatting dataset, optionally rescales totals to a national reference, projects counts across years using configured growth assumptions, and adds disaggregation columns from configuration and/or an uploaded CSV. It writes refreshed population Parquet and CSV extracts and attaches them to the population-transformation dataset.

## Parameters

* **`adjust_population`** (bool, Optional):
  * **Name:** Adjust using population totals
  * **Description:** When true, the notebook attempts to scale facility populations so national totals match **`TOTAL_POPULATION_REF`** from `DHIS2_DATA_DEFINITIONS.POPULATION_DEFINITIONS`, replacing **`POPULATION`** with scaled values when a reference is present.
  * **Choices/Default:** Default: `false`.
* **`disaggregation_file`** (File, Optional):
  * **Name:** Use disaggregation proportions (.csv)
  * **Description:** Optional user-uploaded CSV of **ADM2-level proportions** merged in the notebook to create additional population columns (each new column is **`POPULATION` × proportion**, rounded).
  * **Choices/Default:** Default: `None` (no file).

## Functionality Overview

1. Ensure `pipelines/snt_dhis2_population_transformation` and `data/dhis2/population_transformed` exist; optionally pull code/report notebooks from the repository.
2. Load and validate `configuration/SNT_config.json`, read **`COUNTRY_CODE`**, and abort early with an error if a `disaggregation_file` argument was supplied but the path does not exist.
3. When the main computation stage runs, merge notebook parameters **`ADJUST_TOTALS`** (from **`adjust_population`**), optional **`DISAGGREGATION_FILE`**, and persist them with **`save_pipeline_parameters`** into **`data/dhis2/population_transformed/`**.
4. **`dhis2_population_transformation`** checks whether **`{COUNTRY_CODE}_population.parquet`** exists in **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`**; if not, it logs a warning and skips the notebook entirely (no new outputs in that run).
5. When data exist, run `code/snt_dhis2_population_transformation.ipynb` with **`SNT_ROOT_PATH`** plus the parameters above.
6. In the notebook, load the formatted population Parquet, keep metadata columns **`YEAR`, ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID`**, and treat remaining numeric columns as population indicators at **ADM2 organisation-unit × YEAR** grain.
7. Optionally compute **year-level scaling factors** against **`TOTAL_POPULATION_REF`** and replace **`POPULATION`** with scaled counts when enabled and configured.
8. Project **all population indicator columns** backward and forward from the resolved **`REFERENCE_YEAR`** using **`GROWTH_FACTOR`**, then bind reference, backward, and forward slices so each **ADM2 unit × YEAR** is populated across the target year span.
9. Multiply **`POPULATION`** by each factor in **`POPULATION_DISAGGREGATIONS`** from config to append named disaggregation columns; if **`DISAGGREGATION_FILE`** is set, merge proportions from CSV via **`add_population_disaggregations`**.
10. Write **`{COUNTRY_CODE}_population.parquet`** and **`{COUNTRY_CODE}_population.csv`** to `data/dhis2/population_transformed/`, upload them with the parameters JSON to **`SNT_DATASET_IDENTIFIERS.DHIS2_POPULATION_TRANSFORMATION`**, and execute `reporting/snt_dhis2_population_transformation_report.ipynb`.

## Inputs

* **`configuration/SNT_config.json`**: `POPULATION_DEFINITIONS` (growth factor, reference year, disaggregation factors, optional total reference), administration labels, dataset ids.
* **`{COUNTRY_CODE}_population.parquet`** from **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`** (latest file via `load_dataset_file` in the notebook).
* **Optional `disaggregation_file`**: user CSV with ADM2 proportion columns for extra stratifiers.

## Outputs

* **`data/dhis2/population_transformed/{COUNTRY_CODE}_population.parquet`**
* **`data/dhis2/population_transformed/{COUNTRY_CODE}_population.csv`**
* **Pipeline parameters JSON** from `save_pipeline_parameters` in the same folder.
* **Dataset:** files registered on **`SNT_DATASET_IDENTIFIERS.DHIS2_POPULATION_TRANSFORMATION`** (`add_files_to_dataset`; key read with `.get`, so misconfiguration yields `None` at runtime).

> **Notes for the Data Analyst:**
>
> - **`POPULATION`**: Primary total population column; may be overwritten by national scaling when **`adjust_population`** is enabled and **`TOTAL_POPULATION_REF`** is defined.
> - **`POPULATION_SCALED`**: Intermediate column created during scaling before values are copied back into **`POPULATION`**.
> - **Projection years**: Derived from **`REFERENCE_YEAR`**, **`GROWTH_FACTOR`**, and the years present in the source extract; overlapping years are handled when building backward and forward stacks.
> - **Disaggregation columns**: Named from config or CSV headers; values are rounded whole counts proportional to **`POPULATION`** after prior steps.
