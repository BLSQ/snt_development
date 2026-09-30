# SNT User Population Pipeline

The **SNT User Population** pipeline imports a population file supplied by the operator (a CSV filled in from the **`[COUNTRY_CODE]_population_user_template.csv`** template), validates it against the formatted DHIS2 pyramid, and publishes it as **`[COUNTRY_CODE]_population_user.parquet`** and **`.csv`** to **`SNT_POPULATION_USER_PROVIDED`**. It then runs the reporting notebook, which maps each population indicator per year at ADM2 level.

## Parameters

* **`user_file`** (File, Required):
  * **Name:** Upload user population file (.csv)
  * **Description:** The CSV file to import, based on the population user template. A file must be selected even in reporting-only mode, where it is not read. A normal run also stops with an error if the selected file does not exist.
  * **Default:** `None`.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`**, resolve **`COUNTRY_CODE`** and the **`SNT_POPULATION_USER_PROVIDED`** dataset identifier.
2. **Pyramid:** Load **`[COUNTRY_CODE]_pyramid.parquet`** from **`DHIS2_DATASET_FORMATTED`**; stop if it is missing or empty. Extract the unique ADM1/ADM2 names and IDs from the pyramid columns named by **`DHIS2_ADMINISTRATION_1`** / **`DHIS2_ADMINISTRATION_2`** (stop if those columns are not in the pyramid).
3. **Read the user file:** Detect the separator from the header line (comma with decimal point, or semicolon with decimal comma); read as UTF-8 (with or without BOM), falling back to Latin-1. Column names are stripped and uppercased.
4. **Validate** (stops the run only on structural problems):
   * **Stop** if any of **`YEAR`**, **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**, **`POPULATION`** is missing, or if no row has both **`YEAR`** and **`POPULATION`** filled in.
   * **Warn** (the rows are kept as provided) about: columns outside the template (dropped, not published); missing or non-whole-number **`YEAR`**; non-numeric or negative values in **`POPULATION`** / **`POP_*`**; missing **`POPULATION`**; **`ADM2_ID`**s not in the pyramid; pyramid ADM2 units missing for a given **`YEAR`**; ADM1 name/ID or ADM2 name that differ from the pyramid for the same **`ADM2_ID`**; rows sharing the same **`YEAR`** and **`ADM2_ID`**.
   * Round **`POPULATION`** up to an integer.
5. **Save:** Write **`[COUNTRY_CODE]_population_user.parquet`** and **`.csv`** to **`data/user_population/`** (ADM2 × yearly).
6. **Publish:** Save the pipeline parameters JSON and upload the parquet, the CSV and the parameters JSON to **`SNT_POPULATION_USER_PROVIDED`**.
7. **Reporting:** Run **`snt_user_population_report.ipynb`**. It stops with an `[ERROR]` if any ADM1/ADM2 name + ID combination in the published file does not match **`[COUNTRY_CODE]_shapes.geojson`**; otherwise it draws one choropleth per non-empty indicator, faceted by **`YEAR`**. Runs in both normal and reporting-only mode.

## Inputs

* **User population CSV** (the **`user_file`** parameter) — required; not read in reporting-only mode. Built from **`uploads/[COUNTRY_CODE]_population_user_template.csv`**, which **`snt_dhis2_formatting`** writes to the workspace.
* **`[COUNTRY_CODE]_pyramid.parquet`** on **`DHIS2_DATASET_FORMATTED`** — required; the run stops if it is missing.
* **`[COUNTRY_CODE]_shapes.geojson`** on **`DHIS2_DATASET_FORMATTED`** — read by the reporting notebook only, for the map boundaries and the name check.
* **`[COUNTRY_CODE]_population_user.parquet`** on **`SNT_POPULATION_USER_PROVIDED`** — read back by the reporting notebook.
* **`configuration/SNT_config.json`** for **`SNT_CONFIG.COUNTRY_CODE`**, **`SNT_CONFIG.DHIS2_ADMINISTRATION_1`**, **`SNT_CONFIG.DHIS2_ADMINISTRATION_2`**, **`SNT_DATASET_IDENTIFIERS.SNT_POPULATION_USER_PROVIDED`** and **`SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED`**.

## Outputs

**Workspace filesystem**

* **`data/user_population/[COUNTRY_CODE]_population_user.parquet`** and **`.csv`**
* **Pipeline parameters JSON** in the same directory
* **Report outputs** under **`pipelines/snt_user_population/reporting/outputs/`**, including one PNG per indicator in **`figures/[COUNTRY_CODE]_choropleth_population_user_<INDICATOR>.png`** (not published)

**Published to `SNT_POPULATION_USER_PROVIDED`**

* **`[COUNTRY_CODE]_population_user.parquet`**
* **`[COUNTRY_CODE]_population_user.csv`**
* **Pipeline parameters JSON** (records **`user_file`** as the uploaded file path)

In reporting-only mode nothing is written or published; only the report is re-run.

> **Notes for the Data Analyst:**
>
> - **`YEAR`**: integer. Rows with a missing or invalid **`YEAR`** are kept and published empty; the report leaves them out of the maps.
> - **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`**: text, published exactly as typed by the operator. Differences from the pyramid are only warned about, not corrected.
> - **`POPULATION`**: total population, integer, decimals rounded **up**. May be empty for some rows (warned).
> - **`POP_UNDER_5`**, **`POP_PREGNANT_WOMEN`**, **`POP_0_1_Y`**, **`POP_1_2_Y`**, **`POP_5_10_Y`**, **`POP_5_36_M`**, **`POP_50_PLUS`**: optional disaggregations, published as numeric (decimals kept). Columns absent from the file are not added; empty cells stay empty. The report skips any indicator with no values.
> - **Grain:** ADM2 × yearly, one row per **`ADM2_ID`** and **`YEAR`** as provided. Duplicates and missing ADM2 units are warned about, not removed or filled.
> - **Guarded execution:** Data issues never stop the run; they are logged as warnings and the file is published as provided. The shapes name check in the report runs **after** publication, so a file whose names do not match the shapes is already on **`SNT_POPULATION_USER_PROVIDED`** when the report fails.
