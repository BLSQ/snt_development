# SNT DHIS2 Population Transformation Pipeline

This pipeline takes a population table — either the formatted DHIS2 population from **`DHIS2_DATASET_FORMATTED`** or a user-provided population table from **`SNT_POPULATION_USER_PROVIDED`** — and applies three optional transformation stages: (1) scaling the population columns to a national reference total, (2) creating disaggregated population columns from proportion parameters and/or an uploaded CSV, and (3) projecting population figures across years using a growth rate. It publishes the transformed population Parquet and CSV to **`DHIS2_POPULATION_TRANSFORMATION`**.

All columns of the source table are kept. **`POPULATION`** is the basis for the scaling factor and for all computed disaggregations; population indicators already present in the source (e.g. disaggregations in a user-provided table) are carried through and scaled with it.

## Parameters

### Population source

* **`pop_source`** (str, Required):
  * **Name:** Population source
  * **Description:** Source of the population data to transform.
  * **Choices:**
    * `DHIS2` — **`{COUNTRY_CODE}_population.parquet`** from **`DHIS2_DATASET_FORMATTED`**.
    * `User-provided` — **`{COUNTRY_CODE}_population.parquet`** from **`SNT_POPULATION_USER_PROVIDED`**.
  * **Default:** `DHIS2`.

### Part 1 — Population adjustment

* **`tot_pop_reference`** (int, Optional):
  * **Name:** Part 1: Population reference
  * **Description:** Total national population used to scale the population data. When provided, all population columns are multiplied by **`tot_pop_reference` / total `POPULATION` of the reference year**, so that the reference year total matches this value.
  * **Example:** `1000000` for a total population of 1 million.
  * **Default:** `None` (no adjustment applied).
* **`tot_pop_reference_year`** (int, Optional):
  * **Name:** Part 1: Population year reference
  * **Description:** Year in the population data to which **`tot_pop_reference`** applies. Must be available in the data. If omitted or not available, the latest available year is used (with a warning).
  * **Example:** `2025`.
  * **Default:** `None` (latest available year used).

### Part 2 — Disaggregation

Each proportion is applied to **`POPULATION`** (after scaling) and written to the column of the same name in uppercase. The same proportion is used for every ADM2 and every year.

* **`pop_under_5`** (float, Optional): Proportion of the total population aged under 5 (e.g. `0.17` for 17%). Column **`POP_UNDER_5`**.
* **`pop_pregnant_women`** (float, Optional): Proportion of the total population that are pregnant women (e.g. `0.05` for 5%). Column **`POP_PREGNANT_WOMEN`**.
* **`pop_0_1_y`** (float, Optional): Proportion aged 0–1 years (e.g. `0.04`). Column **`POP_0_1_Y`**.
* **`pop_1_2_y`** (float, Optional): Proportion aged 1–2 years (e.g. `0.03`). Column **`POP_1_2_Y`**.
* **`pop_5_10_y`** (float, Optional): Proportion aged 5–10 years (e.g. `0.06`). Column **`POP_5_10_Y`**.
* **`pop_5_36_m`** (float, Optional): Proportion aged 5–36 months (e.g. `0.06`). Column **`POP_5_36_M`**.
* **`pop_50_plus`** (float, Optional): Proportion aged 50 years and above (e.g. `0.06`). Column **`POP_50_PLUS`**.
* **`disaggregation_file`** (File, Optional):
  * **Name:** Part 2: Use disaggregation proportions (.csv)
  * **Description:** User-uploaded CSV (comma- or semicolon-separated) with ADM2-level disaggregation proportions, based on **`uploads/{COUNTRY_CODE}_population_disaggregation_template.csv`**. Must contain **`ADM2_ID`** and must not contain **`YEAR`** or **`POPULATION`**. Each non-empty column is computed as **`POPULATION`** × the ADM2 proportion; a column matching a parameter-derived disaggregation **overwrites** it, other columns are added.
  * **Default:** `None` (no file).

### Part 3 — Projections

* **`growth_factor`** (float, Optional):
  * **Name:** Part 3: Projection growth rate
  * **Description:** Annual growth rate used to project all population columns 6 years backward and 6 years forward from the reference year (e.g. `0.03` for 3%).
  * **Default:** `None` (no projection applied).
* **`growth_reference_year`** (int, Optional):
  * **Name:** Part 3: Projection reference year
  * **Description:** Base year from which projections are calculated. Must be available in the population data. If omitted or not available, the latest available year is used (with a warning).
  * **Default:** `None` (latest available year used).

## Functionality Overview

1. Ensure `pipelines/snt_dhis2_population_transformation` and `data/dhis2/population_transformed` exist; optionally pull the code and report notebooks from the repository.
2. Load and validate `configuration/SNT_config.json` and read **`COUNTRY_CODE`**. Steps 3–9 are skipped when **`run_report_only`** is set.
3. Load the population table of the selected **`pop_source`** from its dataset; abort with an error if the dataset, its latest version or the file cannot be found or downloaded, or if the table is empty.
4. If a **`disaggregation_file`** is supplied: abort if the file does not exist on disk, and validate its header — abort if **`ADM2_ID`** is missing or **`YEAR`** / **`POPULATION`** are present.
5. Read the years available in the **`YEAR`** column of the population table (abort if the column is missing), then resolve **`tot_pop_reference_year`** (only if **`tot_pop_reference`** is set) and **`growth_reference_year`** (only if **`growth_factor`** is set) against them.
6. Save the notebook parameters to **`{COUNTRY_CODE}_parameters.json`** via **`save_pipeline_parameters`** in `data/dhis2/population_transformed/`. The source is recorded as **`POPULATION_DATASET_SOURCE`**, the dataset id resolved from **`pop_source`**; the reference years are the resolved values from step 5.
7. Run `code/snt_dhis2_population_transformation.ipynb` with **`SNT_ROOT_PATH`** and the parameters above:
   1. Load **`{COUNTRY_CODE}_population.parquet`** from **`POPULATION_DATASET_SOURCE`**, keeping all columns.
   2. **Part 1 — Adjustment:** if **`tot_pop_reference`** is set, multiply **`POPULATION`** and every existing non-empty numeric population column by **`tot_pop_reference` / total `POPULATION` of the reference year**, for all years.
   3. **Part 2 — Disaggregation (parameters):** for each proportion parameter provided, compute its column as **`POPULATION`** × proportion.
   4. **Part 2 — Disaggregation (file):** if **`disaggregation_file`** is set, join the proportions on **`ADM2_ID`** and compute each valid column as **`POPULATION`** × proportion (see Notes for the validation rules).
   5. **Part 3 — Projection:** if **`growth_factor`** is set, project all population columns backward and forward 6 years from the reference year.
   6. Convert all population columns to integer and write **`{COUNTRY_CODE}_population.parquet`** and **`{COUNTRY_CODE}_population.csv`** to `data/dhis2/population_transformed/`.
8. Check that the Parquet and the CSV were written during this run; abort with an error if either is missing or older than the run (guards against publishing stale files).
9. Publish the Parquet, the CSV and the parameters JSON to **`DHIS2_POPULATION_TRANSFORMATION`**.
10. Run `reporting/snt_dhis2_population_transformation_report.ipynb` (also in report-only mode). It draws one choropleth per population column, faceted by **`YEAR`**, with a subtitle naming the source read from **`POPULATION_DATASET_SOURCE`** in the published parameters JSON (`DHIS2` or `population fournie par l'utilisateur`).

## Inputs

* **`configuration/SNT_config.json`**: **`SNT_CONFIG.COUNTRY_CODE`** and the dataset identifiers **`DHIS2_DATASET_FORMATTED`**, **`SNT_POPULATION_USER_PROVIDED`** and **`DHIS2_POPULATION_TRANSFORMATION`** under **`SNT_DATASET_IDENTIFIERS`**.
* **`pop_source` = `DHIS2`:** **`{COUNTRY_CODE}_population.parquet`** from **`DHIS2_DATASET_FORMATTED`**, produced by **`snt_dhis2_formatting`** (required).
* **`pop_source` = `User-provided`:** **`{COUNTRY_CODE}_population.parquet`** from **`SNT_POPULATION_USER_PROVIDED`**, produced by **`snt_user_population`** (required).
* **Optional `disaggregation_file`**: operator-uploaded CSV with ADM2-level proportion columns. Start from **`uploads/{COUNTRY_CODE}_population_disaggregation_template.csv`**, written by **`snt_dhis2_formatting`**.

## Outputs

Written to the workspace filesystem, in `data/dhis2/population_transformed/`:

* **`{COUNTRY_CODE}_population.parquet`**
* **`{COUNTRY_CODE}_population.csv`**
* **`{COUNTRY_CODE}_parameters.json`** (pipeline parameters, from `save_pipeline_parameters`)

Published to **`DHIS2_POPULATION_TRANSFORMATION`**: the same three files.

Executed notebooks are also written to `pipelines/snt_dhis2_population_transformation/papermill_outputs/` and `reporting/outputs/`, and the report maps to `reporting/outputs/figures/{COUNTRY_CODE}_choropleth_poptransformed_<INDICATOR>.png` (not published).

> **Notes for the Data Analyst:**
>
> - **Grain:** one row per ADM2 × **`YEAR`**, with **`YEAR`**, **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`POPULATION`** and any disaggregation columns.
> - **Data types:** all population columns (**`POPULATION`** and disaggregations) are integers, whichever steps ran.
> - **Source columns:** all columns of the source table are kept, including population indicators already present in a user-provided table.
> - **Part 1 scaling:** the scaling factor is computed from the reference year and applied to all years, so the year-to-year trend is preserved. **`POPULATION`** and every existing disaggregation column with values are scaled by the same factor, so their proportions are preserved. Empty or non-numeric columns are left unchanged and listed in a warning.
> - **Part 2 precedence:** parameter-based disaggregations are computed first; columns from **`disaggregation_file`** then overwrite any matching column. A disaggregation already in the source table is overwritten (with a warning) when a parameter or file column of the same name is provided.
> - **Part 2 file validation:**
>   - A column left completely blank is ignored.
>   - A column with any value outside [0, 1] is reported as an error and ignored — proportions must be written as e.g. `0.17`, not `17`.
>   - A duplicated **`ADM2_ID`** is reported as an error and only its first row is used.
>   - ADM2 units missing from the file get empty values for the file's columns.
> - **Part 3 projections:** the output contains only the reference year and the 6 years before and after it. Existing years in that window are overwritten by the projections (a warning lists the overwritten years); existing years outside it are dropped.
> - **Reference year fallback:** **`tot_pop_reference_year`** and **`growth_reference_year`** fall back to the latest available year when omitted or not available, with a warning — check the run logs to confirm which year was used.
> - **Guarded execution:** a missing or empty population source, a missing or invalid **`disaggregation_file`**, or a notebook run that does not write both outputs all **fail** the run; nothing is skipped silently, and stale files are never published.
> - **Proportion parameters are not range-checked:** unlike the columns of **`disaggregation_file`**, the Part 2 parameters are applied as given — enter `0.17`, not `17`, or the column is 17 times the population.
