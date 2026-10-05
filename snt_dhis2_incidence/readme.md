# SNT DHIS2 Incidence Pipeline

The **SNT DHIS2 Incidence** pipeline estimates monthly malaria case metrics and yearly incidence rates from routine DHIS2 data. It supports crude incidence and adjustments for testing gaps, incomplete reporting, and optional care-seeking behaviour, using the population source selected by the operator. It publishes **`[COUNTRY_CODE]_incidence.parquet`** and **`.csv`** to **`DHIS2_INCIDENCE`**.

---

## Parameters

* **`n1_method`** (String, Required):
  * **Name:** Method for N1 calculations
  * **Description:** Defines how **`N1`** (cases adjusted for testing gaps) is derived: from presumed cases or from suspected minus tested cases.
  * **Choices/Default:** `PRES`, `SUSP-TEST`. Default: `PRES`.
* **`routine_data_choice`** (String, Required):
  * **Name:** Routine data to use
  * **Description:** Which routine dataset variant to use:
    * `raw` — **`[COUNTRY_CODE]_routine.parquet`** from **`DHIS2_DATASET_FORMATTED`**.
    * `raw_without_outliers` — **`[COUNTRY_CODE]_routine_outliers_removed.parquet`** from **`DHIS2_OUTLIERS_IMPUTATION`**.
    * `imputed` — **`[COUNTRY_CODE]_routine_outliers_imputed.parquet`** from **`DHIS2_OUTLIERS_IMPUTATION`**.
  * **Choices/Default:** `raw`, `raw_without_outliers`, `imputed`. Default: `imputed`.
* **`population_selection`** (String, Required):
  * **Name:** Population data selection
  * **Description:** Source of the population used as incidence denominator. In every case the file read is **`[COUNTRY_CODE]_population.parquet`**:
    * `DHIS2` — from **`DHIS2_DATASET_FORMATTED`**, produced by **`snt_dhis2_formatting`**.
    * `User-provided` — from **`SNT_POPULATION_USER_PROVIDED`**, produced by **`snt_user_population`**.
    * `Population-transformed` — from **`DHIS2_POPULATION_TRANSFORMATION`**, produced by **`snt_dhis2_population_transformation`**.
  * **Choices/Default:** `DHIS2`, `User-provided`, `Population-transformed`. Default: `DHIS2`.
* **`disaggregation_selection`** (String, Optional):
  * **Name:** Analyze by Population Group (only if available)
  * **Description:** Optional demographic subset. Mapped to a suffix (`Children Under 5 Years Old` → `UNDER_5`, `Pregnant Women` → `PREGNANT_WOMAN`); the notebook then uses the routine columns **`CONF_<suffix>`**, **`TEST_<suffix>`** and **`SUSP_<suffix>`** (or **`PRES_<suffix>`**) and the population column **`POP_<suffix>`**. The run fails if any of them is missing.
  * **Choices/Default:** `Children Under 5 Years Old`, `Pregnant Women`. Default: `None`.
* **`careseeking_file_path`** (File, Optional):
  * **Name:** Care seeking behaviour (CSB) data file (.csv)
  * **Description:** User-supplied CSB file; if omitted (or unreadable) the notebook loads DHS-based care-seeking; if neither is available, care-seeking adjustment is skipped.
  * **Template:** **`uploads/[COUNTRY_CODE]_careseeking_template.csv`**, written by **`snt_dhis2_formatting`** (pyramid step). It has one row per ADM2 with **`ADM1_NAME`**, **`ADM1_ID`**, **`ADM2_NAME`**, **`ADM2_ID`** and an empty **`CARESEEKING_PCT`** to fill in; it has no **`YEAR`** column, so add one only to provide values per year (see below).
  * **Expected columns:** **`ADM1_ID`** and **`CARESEEKING_PCT`** (0–100, or 0–1 which is rescaled); optional **`ADM2_ID`** (joined at ADM2 instead of ADM1) and optional **`YEAR`**.
    * **Without `YEAR`:** each ADM value is applied to **every** year of the routine data.
    * **With `YEAR`:** each year uses its own value, and years absent from the file get **no** care-seeking adjustment (empty **`INCIDENCE_ADJ_CARESEEKING`**, with a warning listing them). A file covering only one year therefore adjusts only that year — leave out **`YEAR`** to apply a single set of values to all years.
  * **Default:** `None`.

---

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`** and read **`COUNTRY_CODE`**. Steps 2–6 are skipped when **`run_report_only`** is set.
2. **Population check (`pipeline.py`):** Resolve the dataset of the selected **`population_selection`** and check that **`[COUNTRY_CODE]_population.parquet`** exists in its latest version; the run fails before any computation if it does not (or if the dataset identifier is missing from the config).
3. **Parameter mapping:** Save **`N1_METHOD`**, **`ROUTINE_DATA_CHOICE`**, **`POPULATION_DATASET_ID`** (the resolved dataset id), **`DISAGGREGATION_SELECTION`**, **`CARESEEKING_FILE_PATH`** and **`ROOT_PATH`** to **`[COUNTRY_CODE]_parameters.json`** and inject them into the notebook.
4. **Computation:** Run **`pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb`**:
   1. Load the routine data of the selected variant (fails with an `[ERROR]` naming the upstream pipeline if the file is missing), the population from **`POPULATION_DATASET_ID`**, and the reporting rate.
   2. Apply the optional disaggregation to the routine and population columns.
   3. Load care-seeking data: the uploaded file if provided and readable, else the DHS file, else none.
   4. Compute monthly **`TPR`**, **`N1`**, **`N2`** and (with care-seeking data) **`N3`** at ADM2 × monthly, and write **`[COUNTRY_CODE]_monthly_cases.parquet`**.
   5. Aggregate to ADM2 × yearly, join the population on **`ADM2_ID`** + **`YEAR`**, compute the incidence rates, and write **`[COUNTRY_CODE]_incidence.parquet`** and **`.csv`**.
5. **Output check:** Fail if the incidence parquet / CSV were not written during this run.
6. **Dataset publication:** Upload **`[COUNTRY_CODE]_incidence.parquet`**, **`[COUNTRY_CODE]_incidence.csv`** and the parameters JSON to **`DHIS2_INCIDENCE`**.
7. **Reporting:** Run **`snt_dhis2_incidence_report.ipynb`** (also in report-only mode).

---

## Inputs

* **Routine data** (required), per **`routine_data_choice`** — see Parameters for the file and dataset of each choice.
* **Population** (required): **`[COUNTRY_CODE]_population.parquet`** from the dataset of the selected **`population_selection`** — see Parameters.
* **Reporting rate** (required): **`[COUNTRY_CODE]_reporting_rate_dataelement.parquet`** from **`DHIS2_REPORTING_RATE`**, falling back to **`[COUNTRY_CODE]_reporting_rate_dataset.parquet`**; the run fails if neither can be loaded.
* **Care-seeking** (optional): the uploaded **`careseeking_file_path`** CSV (start from **`uploads/[COUNTRY_CODE]_careseeking_template.csv`**, written by **`snt_dhis2_formatting`**), else **`[COUNTRY_CODE]_DHS_ADM1_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet`** from **`DHS_INDICATORS`** (column **`PCT_PUBLIC_CARE`**).
* **`configuration/SNT_config.json`**: **`SNT_CONFIG.COUNTRY_CODE`**, **`SNT_CONFIG.DHIS2_ADMINISTRATION_1`** / **`DHIS2_ADMINISTRATION_2`**, **`DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS`**, and under **`SNT_DATASET_IDENTIFIERS`**: **`DHIS2_DATASET_FORMATTED`**, **`DHIS2_OUTLIERS_IMPUTATION`**, **`SNT_POPULATION_USER_PROVIDED`**, **`DHIS2_POPULATION_TRANSFORMATION`**, **`DHIS2_REPORTING_RATE`**, **`DHS_INDICATORS`** and **`DHIS2_INCIDENCE`**.

---

## Outputs

**Workspace filesystem**

* **`data/dhis2/incidence/[COUNTRY_CODE]_incidence.parquet`** and **`.csv`**
* **`data/dhis2/incidence/[COUNTRY_CODE]_parameters.json`**
* **`pipelines/snt_dhis2_incidence/intermediate_results/[COUNTRY_CODE]_monthly_cases.parquet`** — intermediate ADM2 × monthly table (not published)
* Executed notebooks under **`pipelines/snt_dhis2_incidence/papermill_outputs/`** and **`reporting/outputs/`** (not published)

**Published to `DHIS2_INCIDENCE`**

* **`[COUNTRY_CODE]_incidence.parquet`**
* **`[COUNTRY_CODE]_incidence.csv`**
* **`[COUNTRY_CODE]_parameters.json`** (records **`POPULATION_DATASET_ID`**, the dataset id, rather than the **`population_selection`** label)

---

> **Notes for the Data Analyst:**
>
> - **Grain:** ADM2 × yearly, with **`YEAR`**, the **`ADM*`** columns, **`POPULATION`** and the **`INCIDENCE_*`** columns.
> - **`TPR`**: Test positivity rate, **`CONF`** / **`TEST`**. If **`TEST`** is zero or missing, **`TPR`** is set to `1` to avoid division by zero.
> - **`N1`**: Testing-adjusted cases. With **`SUSP-TEST`**: **`CONF`** + ((**`SUSP`** − **`TEST`**) × **`TPR`**), capping **`TEST`** at **`SUSP`** when needed. With **`PRES`**: **`CONF`** + (**`PRES`** × **`TPR`**).
> - **`N2`**: **`N1`** / **`REPORTING_RATE`**. If the monthly reporting rate is zero, **`N2`** is set to missing.
> - **`N3`**: **`N2`** / (**`CARESEEKING_PCT`** / 100). If care-seeking percent is zero, **`N3`** is missing.
> - **`INCIDENCE_CRUDE`** / **`INCIDENCE_ADJ_TESTING`** / **`INCIDENCE_ADJ_REPORTING`** / **`INCIDENCE_ADJ_CARESEEKING`**: Yearly rates per 1,000 population from **`CONF`**, **`N1`**, **`N2`** and **`N3`**.
>   - **`INCIDENCE_ADJ_CARESEEKING`** is empty when no care-seeking data is available.
>   - When the care-seeking data has no value for an ADM unit or a **`YEAR`** (a warning lists the missing years), every month of that ADM2 × **`YEAR`** has a missing **`N3`**, and the yearly **`N3`** and **`INCIDENCE_ADJ_CARESEEKING`** are left **empty** (not `0`).
>   - When only some months of an ADM2 × **`YEAR`** have a missing **`N3`** (e.g. zero reporting rate or zero care-seeking percent), the yearly **`N3`** is the sum of the available months only.
> - **Population source:** whichever **`population_selection`** is used, the population file must have one row per **`ADM2_ID`** × **`YEAR`**. A duplicated pair (possible in a user-provided file, where duplicates are only warned about) duplicates that row in the incidence table.
> - **`Pregnant Women`** disaggregation: the mapped suffix composes **`POP_PREGNANT_WOMAN`** (singular), while the population pipelines produce **`POP_PREGNANT_WOMEN`** (plural), so this selection currently fails at the population step.
