# SNT Quality of Care Pipeline

The **SNT Quality of Care** pipeline computes **district-year** quality-of-care indicators from DHIS2 routine data that has already been processed by an outlier imputation or removal pipeline. It derives testing and treatment rates, malaria admission and death shares, case fatality among admissions, and related counts, then exports maps and tabular products for monitoring.

## Parameters

* **`data_action`** (String, Required):
  * **Name:** Data action
  * **Description:** Selects whether to read outlier-**imputed** routine files or outlier-**removed** routine files from the outliers dataset.
  * **Choices/Default:** `imputed`, `removed`. Default: `imputed`.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`** and read **`COUNTRY_CODE`**.
2. **Routine acquisition:** In **`pipelines/snt_dhis2_quality_of_care/code/snt_dhis2_quality_of_care.ipynb`**, list files in the latest version of **`DHIS2_OUTLIERS_IMPUTATION`**, keep those matching **`[COUNTRY_CODE]_routine_outliers-*_[data_action].parquet`**, and load the **lexicographically greatest** filename (tie-break for multiple imputation methods).
3. **Shapes:** Load **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`** for **ADM2** geometry and **`ADM2_NAME`** merge.
4. **Cleaning:** Coerce configured indicator columns from character (treating **`""`**, **`"-"`**, and missing as NA) to numeric.
5. **Aggregation:** Sum all available indicator columns **by `ADM2_ID` and `YEAR`**, producing one row per **district-year**.
6. **Indicators:** Compute rates (**`testing_rate`**, **`treatment_rate`**, **`case_fatality_rate`**, **`prop_adm_malaria`**, **`prop_malaria_deaths`**) and retain absolute columns (**`non_malaria_all_cause_outpatients`** from **`ALLOUT`**, **`presumed_cases`** from **`PRES`**); rates use **`fifelse`** with **`NA`** when the denominator is zero.
7. **Maps and export:** When the computation notebook runs, write yearly **ADM2** choropleth **`.png`** maps under the pipeline reporting figures path and write **`[COUNTRY_CODE]_quality_of_care_district_year_[data_action].parquet`** and **`.csv`** under **`data/dhis2/quality_of_care/`**.
8. **Orchestration (`pipeline.py`):** Run the computation notebook unless reporting-only mode is enabled at the platform level, save parameters JSON, upload parquet, CSV, and parameters file to **`DHIS2_QUALITY_OF_CARE`** when present, then always run **`snt_dhis2_quality_of_care_report.ipynb`**.

## Inputs

* **`[COUNTRY_CODE]_routine_outliers-*_[data_action].parquet`** on **`DHIS2_OUTLIERS_IMPUTATION`** (latest matching file selected in code).
* **`[COUNTRY_CODE]_shapes.geojson`** on **`DHIS2_DATASET_FORMATTED`**.

## Outputs

* **`[COUNTRY_CODE]_quality_of_care_district_year_[data_action].parquet`** and **`.csv`** in **`data/dhis2/quality_of_care/`** (district-year table).
* **Yearly indicator maps** (e.g. **`testing_rate_2023.png`**; the outpatient count map uses the internal filename prefix **`allout_`**).
* **Pipeline parameters JSON** and copies uploaded to **`DHIS2_QUALITY_OF_CARE`** when the computation step produces files.
* **Reporting artefacts** from **`snt_dhis2_quality_of_care_report.ipynb`**.

> **Notes for the Data Analyst:**
>
> - **`testing_rate`**: **`TEST`** / **`SUSP`** when **`SUSP` > 0**, else missing.
> - **`treatment_rate`**: **`MALTREAT`** / **`CONF`** when **`CONF` > 0**, else missing.
> - **`case_fatality_rate`**: **`MALDTH`** / **`MALADM`** when **`MALADM` > 0**, else missing.
> - **`prop_adm_malaria`**: **`MALADM`** / **`ALLADM`** when **`ALLADM` > 0**, else missing.
> - **`prop_malaria_deaths`**: **`MALDTH`** / **`ALLDTH`** when **`ALLDTH` > 0**, else missing; the notebook also assigns **`prop_deaths_malaria`** as an alias of **`prop_malaria_deaths`**.
> - **`non_malaria_all_cause_outpatients`**: District-year sum of **`ALLOUT`** after the **`ADM2_ID`–`YEAR` aggregation**.
> - **`presumed_cases`**: District-year sum of **`PRES`** after the same aggregation.
