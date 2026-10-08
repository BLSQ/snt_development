# SNT Quality of Care Pipeline

The **SNT Quality of Care** pipeline computes **district-year (ADM2 × YEAR)** quality-of-care indicators from DHIS2 routine data already processed by an outliers pipeline (outliers imputed or removed). It derives testing, treatment, case-fatality and malaria-share rates plus two absolute counts, publishes the district-year table to **`DHIS2_QUALITY_OF_CARE`**, and runs the quality-of-care reporting notebook.

## Parameters

* **`data_action`** (String, Required):
  * **Name:** Data action
  * **Description:** Selects which outliers-processed routine file to read from **`DHIS2_OUTLIERS_IMPUTATION`**, and sets the suffix of the output filenames.
    * `imputed`: routine data with outliers replaced by imputed values (**`[COUNTRY_CODE]_routine_outliers_imputed.parquet`**).
    * `removed`: routine data with outliers set to missing (**`[COUNTRY_CODE]_routine_outliers_removed.parquet`**).
  * **Choices:** `imputed`, `removed` — there is no option to use raw routine data.
  * **Default:** `imputed`.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`** and read **`COUNTRY_CODE`**.
2. **Parameters:** Save the pipeline parameters JSON to **`data/dhis2/quality_of_care/`**.
3. **Computation notebook:** Run **`pipelines/snt_dhis2_quality_of_care/code/snt_dhis2_quality_of_care.ipynb`**:
   1. **Routine data:** Load **`[COUNTRY_CODE]_routine_outliers_[data_action].parquet`** from the latest version of **`DHIS2_OUTLIERS_IMPUTATION`**; the run stops with an `[ERROR]` if it is missing.
   2. **Shapes:** Load **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`**.
   3. **Cleaning:** Cast the indicator columns **`TEST`**, **`SUSP`**, **`MALTREAT`**, **`CONF`**, **`MALDTH`**, **`MALADM`**, **`ALLADM`**, **`ALLDTH`**, **`ALLOUT`**, **`PRES`** to numeric (`""` and `"-"` become missing), **`YEAR`** to integer and **`ADM2_ID`** to character. Indicator columns absent from the routine file are skipped.
   4. **Aggregation:** Sum the available indicators **by `ADM2_ID` and `YEAR`**, producing one row per district-year.
   5. **Indicators:** Compute each rate only when both its numerator and denominator columns are present, and copy the two absolute counts (see Notes).
   6. **District names:** Join **`ADM2_NAME`** from the shapes by **`ADM2_ID`**.
   7. **Export:** Write **`[COUNTRY_CODE]_quality_of_care_district_year_[data_action].parquet`** and **`.csv`** to **`data/dhis2/quality_of_care/`**.
   8. **Maps:** Save one **ADM2** choropleth PNG per indicator and year to **`pipelines/snt_dhis2_quality_of_care/reporting/outputs/figures/`**. A map that fails is logged as a `[WARNING]` and skipped.
4. **Output check:** Fail the run if the parquet or CSV is missing or was not written during this run, so files left over from a previous run are never published.
5. **Publish:** Upload the parquet, CSV and parameters JSON to **`DHIS2_QUALITY_OF_CARE`**.
6. **Reporting:** Run **`pipelines/snt_dhis2_quality_of_care/reporting/snt_dhis2_quality_of_care_report.ipynb`**.

## Inputs

* **`[COUNTRY_CODE]_routine_outliers_imputed.parquet`** or **`[COUNTRY_CODE]_routine_outliers_removed.parquet`** (per **`data_action`**) on **`DHIS2_OUTLIERS_IMPUTATION`** — required. Produced by whichever outliers imputation pipeline ran last.
* **`[COUNTRY_CODE]_shapes.geojson`** on **`DHIS2_DATASET_FORMATTED`** — required.
* **`configuration/SNT_config.json`** for **`SNT_CONFIG.COUNTRY_CODE`** and the dataset identifiers **`DHIS2_OUTLIERS_IMPUTATION`**, **`DHIS2_DATASET_FORMATTED`** and **`DHIS2_QUALITY_OF_CARE`**.

## Outputs

**Workspace filesystem**

* **`data/dhis2/quality_of_care/[COUNTRY_CODE]_quality_of_care_district_year_[data_action].parquet`** and **`.csv`**.
* **Pipeline parameters JSON** in the same directory.
* **Yearly indicator maps** in **`pipelines/snt_dhis2_quality_of_care/reporting/outputs/figures/`**, named **`[indicator]_[YEAR].png`** (e.g. **`testing_rate_2023.png`**; the outpatient map uses the prefix **`allout_`**). Written, not published.
* **Report outputs** in **`pipelines/snt_dhis2_quality_of_care/reporting/outputs/`**: **`[COUNTRY_CODE]_quality_of_care_summary.parquet`** and **`.csv`** (year-level summary) and **`figures/[COUNTRY_CODE]_quality_of_care_by_year.png`**. Written, not published.

**Published to `DHIS2_QUALITY_OF_CARE`**

* **`[COUNTRY_CODE]_quality_of_care_district_year_[data_action].parquet`** and **`.csv`**.
* The pipeline parameters JSON.

> **Notes for the Data Analyst:**
>
> - **Grain:** one row per **`ADM2_ID`** × **`YEAR`**. Columns: **`ADM2_ID`**, **`YEAR`**, the summed indicator columns present in the routine data, the derived indicators below, and **`ADM2_NAME`** (only if the shapes carry it).
> - **Rates are ratios of district-year sums**, not averages of monthly or facility rates. A rate is missing when its denominator is 0 or missing.
>   - **`TESTING_RATE`**: **`TEST`** / **`SUSP`**.
>   - **`TREATMENT_RATE`**: **`MALTREAT`** / **`CONF`**.
>   - **`CASE_FATALITY_RATE`**: **`MALDTH`** / **`MALADM`** (in-hospital, among malaria admissions).
>   - **`PROP_ADM_MALARIA`**: **`MALADM`** / **`ALLADM`**.
>   - **`PROP_MALARIA_DEATHS`**: **`MALDTH`** / **`ALLDTH`**.
> - **`NON_MALARIA_ALL_CAUSE_OUTPATIENTS`**: district-year sum of **`ALLOUT`**.
> - **`PRESUMED_CASES`**: district-year sum of **`PRES`**.
> - **Missing values sum to 0:** sums ignore missing values, so a district-year where an indicator is missing in every record gets **0**, not missing.
> - **Which outliers method?** **`DHIS2_OUTLIERS_IMPUTATION`** holds the output of whichever outliers pipeline ran last; check the **`[COUNTRY_CODE]_parameters.json`** published beside it to know the method.
> - **Guarded execution:** missing indicator columns are skipped rather than failing the run, so an indicator whose inputs are absent simply does not appear in the output.
> - Stock-out indicators are not implemented (on hold, pending NMDR data).
