# SNT DHIS2 Reporting Rate Pipeline

This pipeline produces **monthly** DHIS2 reporting completeness at **ADM2** (district) resolution, using either a **dataset-based** method (pre-completed reporting counts from DHIS2) or a **data-element** method (facility-level presence of selected indicators). It saves results under **`data/dhis2/reporting_rate/`**, uploads them to **`DHIS2_REPORTING_RATE`**, and runs the reporting notebook for QA and narrative outputs.

## Parameters

* **`reporting_rate_method`** (String, Required):
  * **Name:** Reporting rate method
  * **Description:** Chooses whether reporting rates are derived from pre-aggregated DHIS2 reporting extracts (**`DATASET`**) or computed from routine indicator presence (**`DATAELEMENT`**).
  * **Choices/Default:** `DATASET`, `DATAELEMENT`. Default: `DATAELEMENT`.

* **`dataelement_method_numerator_conf`** (Boolean, Required):
  * **Name:** For method Data Element, calculate Numerator using CONF
  * **Description:** When **`reporting_rate_method`** is **`DATAELEMENT`**, if **`true`**, a facility counts toward the numerator when the **`CONF`** indicator is present or positive as defined in the notebook.
  * **Choices/Default:** Default: `true`.

* **`dataelement_method_numerator_susp`** (Boolean, Required):
  * **Name:** For method Data Element, calculate Numerator using SUSP
  * **Description:** Same as above for the **`SUSP`** indicator.
  * **Choices/Default:** Default: `true`.

* **`dataelement_method_numerator_test`** (Boolean, Required):
  * **Name:** For method Data Element, calculate Numerator using TEST
  * **Description:** Same as above for the **`TEST`** indicator.
  * **Choices/Default:** Default: `true`.

* **`dataelement_method_denominator`** (String, Required):
  * **Name:** For method Data Element: choice of Denominator
  * **Description:** Defines how the expected number of reporting facilities is computed for the data-element branch.
  * **Choices/Default:** `ROUTINE_ACTIVE_FACILITIES`, `PYRAMID_OPEN_FACILITIES`, `DHIS2_EXPECTED_REPORTS`. Default: `ROUTINE_ACTIVE_FACILITIES`. (**`DHIS2_EXPECTED_REPORTS`** requires notebook paths that load expected report counts.)

## Functionality Overview

1. Resolve workspace paths and ensure **`data/dhis2/reporting_rate/`** exists.
2. Load and validate **`configuration/SNT_config.json`** (country code, dataset identifiers, DHIS2 indicator definitions).
3. Execute **`pipelines/snt_dhis2_reporting_rate/code/snt_dhis2_reporting_rate.ipynb`** via Papermill with **`SNT_ROOT_PATH`** and the reporting-method parameters.
4. Persist the effective parameter set next to outputs for traceability.
5. Upload generated reporting-rate tables and the parameter file to the OpenHEXA dataset **`DHIS2_REPORTING_RATE`**.
6. Run **`snt_dhis2_reporting_rate_report.ipynb`** to refresh static report outputs.

## Inputs

* **`configuration/SNT_config.json`**: Country code, DHIS2 admin column names, dataset IDs, and indicator definitions.
* **`{COUNTRY_CODE}_routine.parquet`**: Monthly DHIS2 routine extract from the formatted routine dataset, with at least **`OU_ID`**, **`PERIOD`**, **`YEAR`**, **`MONTH`**, **`ADM1_ID`**, **`ADM2_ID`**, and configured indicator columns.
* **Conditional:** **`{COUNTRY_CODE}_reporting.parquet`** and/or **`{COUNTRY_CODE}_pyramid.parquet`** (or raw pyramid) when the selected method or denominator requires reporting extracts or facility open/close logic.

## Outputs

* **`{COUNTRY_CODE}_reporting_rate_dataset.parquet`** and **`.csv`**: Written when **`reporting_rate_method`** is **`DATASET`** (monthly **ADM2** reporting rate table).
* **`{COUNTRY_CODE}_reporting_rate_dataelement.parquet`** and **`.csv`**: Written when **`reporting_rate_method`** is **`DATAELEMENT`**.
* **Pipeline parameters JSON** in **`data/dhis2/reporting_rate/`**.
* **Papermill output notebooks** under **`papermill_outputs/`** and **reporting artefacts** under **`reporting/outputs/`**.

> **Notes for the Data Analyst:**
>
> - **`Temporal and spatial resolution`**: Outputs are **monthly** (**`PERIOD`** / **`YEAR`**–**`MONTH`**) at **ADM2**, consistent with the computation notebook.
> - **`reporting_rate_method`**: Only one of the dataset or data-element result files is produced per full run.
> - **`dataelement_method_denominator`**: Denominator choice changes which facility counts enter the denominator; some options require pyramid or expected-report inputs to exist in the linked datasets.
> - **`Upstream routine data`**: The notebook resolves the latest **`{COUNTRY_CODE}_routine.parquet`** from the configured formatted routine dataset; stale extracts propagate into reporting rates.
