# SNT DHIS2 Reporting Rate (Data Element) Pipeline

This pipeline estimates **monthly** reporting rates (routine data completeness) at **ADM2**, from whether each facility reported values for selected malaria data elements. It builds a full **facility × `PERIOD`** grid from the DHIS2 pyramid, flags active and open facilities, and computes unweighted or volume-**weighted** reporting rates. It publishes **`[COUNTRY_CODE]_reporting_rate_dataelement`** to **`DHIS2_REPORTING_RATE`** and runs the reporting notebook.

## Parameters

* **`routine_data_choice`** (String, Required):
  * **Name:** Routine data source
  * **Description:** Selects the routine file used to determine facility activity and volume.
    * `raw`: **`[COUNTRY_CODE]_routine.parquet`** on **`DHIS2_DATASET_FORMATTED`**.
    * `imputed`: **`[COUNTRY_CODE]_routine_outliers_imputed.parquet`** on **`DHIS2_OUTLIERS_IMPUTATION`**.
    * `outliers_removed`: **`[COUNTRY_CODE]_routine_outliers_removed.parquet`** on **`DHIS2_OUTLIERS_IMPUTATION`**.
  * **Choices:** `raw`, `imputed`, `outliers_removed`.
  * **Default:** `imputed`.

* **`activity_indicators`** (String, multiple, Required):
  * **Name:** Facility Activity indicators
  * **Description:** Data elements used to decide whether a facility is active in a month. A facility is active in a month when at least one selected indicator has a non-missing value **≥ 0**.
  * **Choices:** `CONF`, `SUSP`, `TEST`, `PRES`.
  * **Default:** `CONF`, `PRES`.

* **`volume_activity_indicators`** (String, multiple, Required):
  * **Name:** Volume activity indicators
  * **Description:** Data elements summed to measure each facility's volume of activity, which defines the facility weights used by weighted reporting rates.
  * **Choices:** `CONF`, `SUSP`, `TEST`, `PRES`.
  * **Default:** `CONF`, `PRES`.

* **`dataelement_method_denominator`** (String, Required):
  * **Name:** Denominator method
  * **Description:** Defines the facilities expected to report (the denominator).
    * `ROUTINE_ACTIVE_FACILITIES`: facilities active at least once in the same **year**.
    * `PYRAMID_OPEN_FACILITIES`: facilities **open** in that month according to the pyramid (see Notes).
  * **Choices:** `ROUTINE_ACTIVE_FACILITIES`, `PYRAMID_OPEN_FACILITIES`.
  * **Default:** `ROUTINE_ACTIVE_FACILITIES`.

* **`use_weighted_reporting_rates`** (Boolean, Optional):
  * **Name:** Use weighted reporting rates
  * **Description:** When `true`, **`REPORTING_RATE`** is the volume-weighted rate for the selected denominator; when `false`, the unweighted rate.
  * **Default:** `false`.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`** and resolve **`COUNTRY_CODE`**.
2. **Routine source:** Map **`routine_data_choice`** to a dataset and filename, and check the file exists in that dataset. If it does not, the run **fails** (`FileNotFoundError`) before any computation.
3. **Parameters:** Save **`[COUNTRY_CODE]_parameters.json`** in **`data/dhis2/reporting_rate/`** with the notebook parameters (**`ROOT_PATH`**, **`ROUTINE_FILE`**, **`DATASET_ID`**, **`DATAELEMENT_METHOD_DENOMINATOR`**, **`ACTIVITY_INDICATORS`**, **`VOLUME_ACTIVITY_INDICATORS`**, **`USE_WEIGHTED_REPORTING_RATES`**).
4. **Computation:** Run **`snt_dhis2_reporting_rate_dataelement.ipynb`**:
   1. Load the routine file and **`[COUNTRY_CODE]_pyramid.parquet`**. Stop with an error if a volume indicator is missing from the routine data.
   2. Build the facility × month grid over the routine data's **`PERIOD`** range, using facilities at level **`ANALYTICS_ORG_UNITS_LEVEL`** of the pyramid.
   3. Flag each facility-month as active and open, flag facilities active at least once per year, and compute facility weights from the volume indicators.
   4. Aggregate to **ADM2 × month** and compute all reporting rate variants; keep the one selected by **`dataelement_method_denominator`** and **`use_weighted_reporting_rates`** as **`REPORTING_RATE`**.
   5. Write the final table and the intermediate tables (see Outputs).
5. **Output check:** Fail the run if the final parquet or CSV was not written during this run.
6. **Publish:** Upload the final parquet, CSV and parameters JSON to **`DHIS2_REPORTING_RATE`**.
7. **Reporting:** Run **`snt_dhis2_reporting_rate_dataelement_report.ipynb`**. With `run_report_only`, steps 2–6 are skipped and only this step runs.

## Inputs

* **Routine file** selected by **`routine_data_choice`** (see Parameters), on **`DHIS2_DATASET_FORMATTED`** or **`DHIS2_OUTLIERS_IMPUTATION`**. Required. Columns used: **`OU_ID`**, **`PERIOD`**, **`YEAR`**, **`MONTH`**, **`ADM2_ID`**, and the indicators **`CONF`**, **`PRES`**, **`SUSP`**, **`TEST`** that are present.
* **`[COUNTRY_CODE]_pyramid.parquet`** on **`DHIS2_DATASET_FORMATTED`**. Required. Columns used: the facility level **`LEVEL_[n]_ID`** / **`LEVEL_[n]_NAME`**, the admin columns named by **`DHIS2_ADMINISTRATION_1`** / **`DHIS2_ADMINISTRATION_2`** and their **`_ID`** counterparts, **`OPENING_DATE`**, **`CLOSED_DATE`**.
* **`configuration/SNT_config.json`** for **`COUNTRY_CODE`**, **`DHIS2_ADMINISTRATION_1`**, **`DHIS2_ADMINISTRATION_2`**, **`ANALYTICS_ORG_UNITS_LEVEL`**, and the dataset identifiers **`DHIS2_DATASET_FORMATTED`**, **`DHIS2_OUTLIERS_IMPUTATION`**, **`DHIS2_REPORTING_RATE`**.

## Outputs

**Workspace filesystem**

* **`data/dhis2/reporting_rate/[COUNTRY_CODE]_reporting_rate_dataelement.parquet`** and **`.csv`**
* **`data/dhis2/reporting_rate/[COUNTRY_CODE]_parameters.json`**
* Intermediate tables in **`pipelines/snt_dhis2_reporting_rate_dataelement/intermediate_results/`** (written, **not published**):
  * **`[COUNTRY_CODE]_reports_dataelement_fosa_monthly.parquet`**: facility × month flags and weights.
  * **`[COUNTRY_CODE]_reports_dataelement_monthly.parquet`** and **`.csv`**: ADM2 × month counts and every reporting rate variant.
  * **`[COUNTRY_CODE]_reports_dataelement_yearly.parquet`** and **`.csv`**: ADM2 × year counts and yearly rates. Read by the reporting notebook.
* Executed notebooks under **`pipelines/snt_dhis2_reporting_rate_dataelement/papermill_outputs/`** and report outputs under **`pipelines/snt_dhis2_reporting_rate_dataelement/reporting/outputs/`**.

**Published to `DHIS2_REPORTING_RATE`**

* **`[COUNTRY_CODE]_reporting_rate_dataelement.parquet`** and **`.csv`**
* **`[COUNTRY_CODE]_parameters.json`**

> **Notes for the Data Analyst:**
>
> - **Grain:** **ADM2 × month**. Columns: **`YEAR`**, **`MONTH`**, **`ADM2_ID`**, **`REPORTING_RATE`**.
> - **`REPORTING_RATE`**: proportion (0–1) of expected facilities that were active in the month.
>   - `ROUTINE_ACTIVE_FACILITIES`: facilities active this month / facilities active at least once in the same year.
>   - `PYRAMID_OPEN_FACILITIES`: facilities active this month / facilities open this month.
>   - Weighted (`use_weighted_reporting_rates` = `true`): numerator and denominator are sums of facility weights instead of facility counts.
> - **Active facility:** a non-missing value **≥ 0** on any activity indicator, so a reported zero counts as active and an empty value does not.
> - **Open facility:** has an **`OPENING_DATE`** on or before the month, no **`CLOSED_DATE`** on or before the month, and a name that does not match `CLOTUR` or `FERM` (case-insensitive). Facilities with no **`OPENING_DATE`** are never open, so incomplete pyramid dates lower the denominator.
> - **Facility weights:** a facility's mean monthly volume (sum of the volume indicators, averaged over months with volume > 0, across the whole period) divided by the ADM2 total of those means, times the number of facilities in the ADM2. Weights are constant across years. Facilities with no month of positive volume get no weight and drop out of weighted rates.
> - **Period range:** from the first to the last **`PERIOD`** in the routine data.
> - **Guarded execution:** a missing routine file stops the run before computation. A missing pyramid or volume indicator stops the notebook with an error. A missing activity indicator is logged as a warning first, then the computation fails at the activity step.
> - **Report-only runs** read the yearly intermediate CSV, so they need a previous full run in the same workspace.
