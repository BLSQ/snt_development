# SNT DHIS2 Incidence Pipeline

**Description:** The DHIS2 incidence pipeline calculates monthly malaria case metrics and yearly incidence rates based on routine health information systems. Its primary objective is to estimate crude incidence and systematically adjust the data for incomplete testing, incomplete reporting, and, optionally, careseeking behavior to provide a more accurate epidemiological picture.

**Parameters:**
* **`n1_method`** (String, Required):
  * **Name:** Method for N1 calculations.
  * **Description:** Determines the logic used to calculate `N1` (cases adjusted for testing gaps). It can rely on directly reported presumed cases or derive them from the difference between suspected and tested cases.
  * **Choices/Default:** `PRES`, `SUSP-TEST` (Default: `PRES`).
* **`routine_data_choice`** (String, Required):
  * **Name:** Routine data to use.
  * **Description:** Specifies which version of the routine DHIS2 data to use for analysis (raw formatted data, data with outliers removed, or data with imputed values replacing removed outliers).
  * **Choices/Default:** `raw`, `raw_without_outliers`, `imputed` (Default: `imputed`).
* **`use_adjusted_population`** (Boolean, Required):
  * **Name:** Use adjusted population.
  * **Description:** If enabled, the pipeline uses adjusted population data for incidence calculations rather than the standard formatted population dataset.
  * **Choices/Default:** `True`, `False` (Default: `False`).
* **`disaggregation_selection`** (String, Optional):
  * **Name:** Disaggregation selection (only if available).
  * **Description:** Selects a specific demographic subset for incidence computation, mapping internal variables to target indicators (e.g., replacing totals with under-5s or pregnant women). This will only run if the data supports it.
  * **Choices/Default:** `Children Under 5 Years Old`, `Pregnant Women` (Default: `None`).
* **`careseeking_file_path`** (File, Optional):
  * **Name:** Care seeking behaviour (CSB) data file (.csv).
  * **Description:** Path to a custom file containing careseeking behaviour proportions. If none is provided, the pipeline falls back to loading DHS data. If neither is available, the pipeline skips `N3` adjustments entirely.
  * **Choices/Default:** `.csv` File (Default: `None`).

**Functionality Overview:**
1. **Data Loading:** Imports the specified routine DHIS2 dataset, population data at the `ADM2` level, monthly reporting rates, and optional careseeking behavior data.
2. **Disaggregation Mapping:** Replaces generic indicator columns (e.g., `SUSP`, `POPULATION`) with disaggregated equivalents (e.g., `SUSP_UNDER_5`, `POP_UNDER_5`) if a disaggregation parameter is selected.
3. **Monthly Aggregation & TPR Calculation:** Aggregates cases to the `ADM2` x `MONTH` resolution and calculates the Test Positivity Rate (`TPR` = `CONF` / `TEST`).
4. **Testing Adjustment (N1):** Adjusts confirmed cases for testing gaps at the monthly level using either the `SUSP-TEST` or `PRES` methodology.
5. **Reporting Adjustment (N2):** Adjusts `N1` for incomplete monthly reporting by dividing by the `REPORTING_RATE`.
6. **Careseeking Adjustment (N3):** Optionally adjusts `N2` for careseeking behavior using user-provided files or DHS proportions.
7. **Yearly Incidence Aggregation:** Sums the adjusted monthly cases (`CONF`, `N1`, `N2`, `N3`) to the `ADM2` x `YEAR` level and divides by the population to compute incidence rates per 1,000.
8. **Export & Verification:** Runs coherence checks to ensure successive adjustments do not decrease incidence values, then exports final tables.

**Inputs:**
* **Routine Data**: `[COUNTRY_CODE]_routine.parquet` (or its `_outliers_removed` / `_outliers_imputed` variant).
* **Population Data**: `[COUNTRY_CODE]_population.parquet`.
* **Reporting Rates**: `[COUNTRY_CODE]_reporting_rate_dataelement.parquet` (falls back to `_dataset.parquet`).
* **Careseeking Data (Optional)**: `[COUNTRY_CODE]_careseeking_template.csv` or `[COUNTRY_CODE]_DHS_ADM1_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet`.
* **Configuration File**: `SNT_config.json`.

**Outputs:**
* Final incidence datasets: `[COUNTRY_CODE]_incidence.csv` and `[COUNTRY_CODE]_incidence.parquet` exported to the SNT Incidence Dataset.
* Intermediate notebook artifacts: `[COUNTRY_CODE]_monthly_cases.parquet` saved for coherence checks and reporting.
* Saved parameter configurations exported as JSON alongside the outputs.

> **Notes for the Data Analyst:**
> - **`TPR`**: Test Positivity Rate. Calculated as `CONF / TEST`.
>   - If `TEST` is `0` or `NA`, `TPR` is forced to `1` to prevent division by zero errors.
> - **`N1`**: Cases adjusted for testing gaps.
>   - If `n1_method` is `SUSP-TEST`: Computed as `CONF + ((SUSP - TEST) * TPR)`. In this logic, if `TEST` > `SUSP`, `TEST` is capped at the value of `SUSP` to prevent generating negative adjusted cases.
>   - If `n1_method` is `PRES`: Computed as `CONF + (PRES * TPR)`.
> - **`N2`**: Cases adjusted for incomplete reporting.
>   - Calculated as `N1 / REPORTING_RATE`. If the monthly reporting rate is exactly zero, `N2` is set to `NA` (avoiding `Inf`), meaning the annual `N2` aggregate might be underestimated.
> - **`N3`**: Cases adjusted for careseeking behavior.
>   - Calculated as `N2 / (CARESEEKING_PCT / 100)`. If `CARESEEKING_PCT` is zero, `N3` is set to `NA`. 
> - **`INCIDENCE_CRUDE`** & **`INCIDENCE_ADJ_*`**: Final yearly rates calculated per 1,000 population (e.g., `CONF / POPULATION * 1000` or `N1 / POPULATION * 1000`).