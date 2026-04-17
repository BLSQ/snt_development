# SNT Quality of Care Pipeline

The **SNT Quality of Care** pipeline computes district-year quality-of-care indicators from DHIS2 routine data that has undergone outlier imputation or removal. It calculates key performance metrics such as testing and treatment rates, case fatality, and hospitalization proportions to support health system monitoring.

---

## Parameters

* **`data_action`** (String, Required):
  * **Name:** Data action
  * **Description:** Determines whether the pipeline processes data where outliers have been replaced with imputed values or data where outliers have been removed entirely.
  * **Choices/Default:** `imputed`, `removed`. Default: `imputed`.
* **`run_report_only`** (Boolean, Optional):
  * **Name:** Run reporting only
  * **Description:** When `true`, skips the computation notebook and runs only the reporting notebook (no refreshed district-year outputs or dataset upload from that run).
  * **Default:** `false`.
* **`pull_scripts`** (Boolean, Optional):
  * **Name:** Pull scripts
  * **Description:** When `true`, pulls the latest code and reporting notebooks for this pipeline from the configured repository before execution.
  * **Default:** `false`.

---

## Functionality Overview

1.  **Data Acquisition:** Automatically identifies and loads the latest outlier-processed routine data file from the `DHIS2_OUTLIERS_IMPUTATION` dataset based on the selected **`data_action`**.
2.  **Spatial Metadata Joining:** Loads administrative boundary shapes (`.geojson`) from the `DHIS2_DATASET_FORMATTED` dataset to facilitate mapping and ensure correct **ADM2** naming.
3.  **Data Cleaning:** Standardizes numeric columns by handling missing values, empty strings, and placeholders (e.g., "-") before converting them to numeric types.
4.  **Temporal & Spatial Aggregation:** Sums indicator columns by **District (ADM2)** and **Year** so each row is one district-year.
5.  **Indicator Calculation:** Computes a suite of quality-of-care indicators (Rates and Absolute Counts) using specific DHIS2 data elements.
6.  **Visual Analytics:** Generates yearly distribution maps for each calculated indicator at the **ADM2** level, saved as `.png` files under the pipeline reporting outputs (from the computation notebook when it runs).
7.  **Data Export:** Saves the final aggregated indicators in both `.parquet` and `.csv` formats, saves the pipeline parameter file, and uploads those artifacts to the OpenHEXA **`DHIS2_QUALITY_OF_CARE`** dataset (when outputs exist and the computation step is not skipped).

---

## Inputs

* **`[COUNTRY_CODE]_routine_outliers-*_[data_action].parquet`**: The primary input file containing routine data after outlier treatment.
* **`[COUNTRY_CODE]_shapes.geojson`**: Geospatial boundaries used for mapping and administrative name resolution.

---

## Outputs

* **`[COUNTRY_CODE]_quality_of_care_district_year_[data_action].parquet`**: Processed district-year indicators in Parquet format.
* **`[COUNTRY_CODE]_quality_of_care_district_year_[data_action].csv`**: Processed district-year indicators in CSV format.
* **Yearly Indicator Maps**: Static `.png` maps (e.g., `testing_rate_2023.png`) stored in the pipeline reporting outputs.

---

> **Notes for the Data Analyst:**
>
> * **`testing_rate`**: Calculated as **`TEST`** / **`SUSP`**. Represents the proportion of suspected cases that received a diagnostic test.
> * **`treatment_rate`**: Calculated as **`MALTREAT`** / **`CONF`**. Represents the proportion of confirmed cases that received antimalarial treatment.
> * **`case_fatality_rate`**: Calculated as **`MALDTH`** / **`MALADM`**. Specifically monitors in-hospital malaria deaths relative to malaria admissions.
> * **`prop_adm_malaria`** & **`prop_malaria_deaths`**:
>   * **`prop_adm_malaria`**: Malaria admissions (**`MALADM`**) divided by all-cause admissions (**`ALLADM`**).
>   * **`prop_malaria_deaths`**: Malaria deaths (**`MALDTH`**) divided by all-cause deaths (**`ALLDTH`**).
> * **`non_malaria_all_cause_outpatients`**: Absolute count derived directly from the **`ALLOUT`** column.
> * **`presumed_cases`**: Absolute count derived directly from the **`PRES`** column.
> * **Zero Handling**: For all rate calculations, the pipeline returns `NA` if the denominator is zero to prevent division errors.

---

### Note on earlier drafts

An earlier draft of this README documented an `outlier_imputation_method` parameter. The implementation in `pipeline.py` and the notebook instead picks the **latest** outlier-processed file available in the dataset automatically. The sections above match that behavior.