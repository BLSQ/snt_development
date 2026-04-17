# SNT DHIS2 Incidence Pipeline

The **SNT DHIS2 Incidence** pipeline estimates monthly malaria case metrics and yearly incidence rates from routine DHIS2 data. It supports crude incidence and adjustments for testing gaps, incomplete reporting, and optional care-seeking behaviour.

---

## Parameters

* **`n1_method`** (String, Required):
  * **Name:** Method for N1 calculations
  * **Description:** Defines how **`N1`** (cases adjusted for testing gaps) is derived: from presumed cases or from suspected minus tested cases.
  * **Choices/Default:** `PRES`, `SUSP-TEST`. Default: `PRES`.
* **`routine_data_choice`** (String, Required):
  * **Name:** Routine data to use
  * **Description:** Which formatted routine dataset variant to use (`raw`, after outlier removal, or after imputation).
  * **Choices/Default:** `raw`, `raw_without_outliers`, `imputed`. Default: `imputed`.
* **`use_adjusted_population`** (Boolean, Required):
  * **Name:** Use adjusted population
  * **Description:** If enabled, incidence denominators use adjusted population where configured; otherwise standard formatted population is used.
  * **Default:** `false`.
* **`disaggregation_selection`** (String, Optional):
  * **Name:** Disaggregation selection (only if available)
  * **Description:** Optional demographic subset mapped to disaggregated columns in the notebook (`Children Under 5 Years Old` → `UNDER_5`, `Pregnant Women` → `PREGNANT_WOMAN`).
  * **Default:** `None`.
* **`careseeking_file_path`** (File, Optional):
  * **Name:** Care seeking behaviour (CSB) data file (.csv)
  * **Description:** User-supplied CSB file; if omitted the notebook may load DHS-based care-seeking; if neither is available, care-seeking adjustment is skipped.
  * **Choices/Default:** Default: **`None`**.

---

## Functionality Overview

1. **Parameter mapping (`pipeline.py`):** Maps OpenHEXA parameters to Papermill inputs (**`N1_METHOD`**, **`ROUTINE_DATA_CHOICE`**, **`USE_ADJUSTED_POPULATION`**, **`DISAGGREGATION_SELECTION`**, **`CARESEEKING_FILE_PATH`**, **`ROOT_PATH`**).
2. **Computation:** Runs **`pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb`** when the main computation stage is enabled.
3. **Dataset publication:** Uploads **`[COUNTRY_CODE]_incidence.parquet`**, matching CSV if present, and the saved parameters file to **`DHIS2_INCIDENCE`**.
4. **Reporting:** Executes **`snt_dhis2_incidence_report.ipynb`** for narrative outputs.

---

## Inputs

* **Routine data** (variant per **`routine_data_choice`**): e.g. `[COUNTRY_CODE]_routine.parquet` and related outlier-processed files as produced upstream.
* **Population, reporting rates, optional CSB/DHS** inputs as implemented in the incidence notebook and workspace layout under `data/dhis2/incidence/`.
* **`SNT_config.json`**: Country code and dataset identifiers.

---

## Outputs

* **`[COUNTRY_CODE]_incidence.parquet`** / **`[COUNTRY_CODE]_incidence.csv`**: Final incidence tables uploaded to **`DHIS2_INCIDENCE`** when the computation step runs.
* **`[COUNTRY_CODE]_monthly_cases.parquet`**: Intermediate monthly table used for checks and reporting (from the notebook).
* **Pipeline parameters JSON** next to data outputs.

---

> **Notes for the Data Analyst:**
>
> - **`TPR`**: Test positivity rate, **`CONF`** / **`TEST`**. If **`TEST`** is zero or missing, **`TPR`** is set to `1` to avoid division by zero.
> - **`N1`**: Testing-adjusted cases. With **`SUSP-TEST`**: **`CONF`** + ((**`SUSP`** − **`TEST`**) × **`TPR`**), capping **`TEST`** at **`SUSP`** when needed. With **`PRES`**: **`CONF`** + (**`PRES`** × **`TPR`**).
> - **`N2`**: **`N1`** / **`REPORTING_RATE`**. If the monthly reporting rate is zero, **`N2`** is set to missing.
> - **`N3`**: **`N2`** / (**`CARESEEKING_PCT`** / 100). If care-seeking percent is zero, **`N3`** is missing.
> - **`INCIDENCE_CRUDE`** / **`INCIDENCE_ADJ_*`**: Yearly rates per 1,000 population from crude or adjusted case numerators.

---
