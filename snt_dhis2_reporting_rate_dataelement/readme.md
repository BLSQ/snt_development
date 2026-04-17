# SNT DHIS2 Reporting Rate (Data Element) Pipeline

This pipeline estimates **monthly** facility **data completeness** at **ADM2** by building a full **`PERIOD` × `OU_ID`** grid from the DHIS2 pyramid, flagging reporting using configurable indicators, and applying either unweighted or volume-**weighted** reporting rates. Outputs land in **`data/dhis2/reporting_rate/`**, are pushed to **`DHIS2_REPORTING_RATE`**, and feed the companion reporting notebook.

## Parameters

* **`routine_data_choice`** (String, Required):
  * **Name:** Routine data source
  * **Description:** Selects which routine Parquet supplies monthly indicator values for merging with the pyramid grid.
  * **Choices/Default:** `raw` (**`{COUNTRY_CODE}_routine.parquet`** on **`DHIS2_DATASET_FORMATTED`**), `imputed` (**`{COUNTRY_CODE}_routine_outliers_imputed.parquet`** on **`DHIS2_OUTLIERS_IMPUTATION`**), `outliers_removed` (**`{COUNTRY_CODE}_routine_outliers_removed.parquet`** on **`DHIS2_OUTLIERS_IMPUTATION`**). Default: `imputed`.

* **`activity_indicators`** (Multi-select string, Required):
  * **Name:** Facility activity indicators
  * **Description:** Subset of indicators used to decide whether a facility is clinically active in a month (strictly positive value on any selected column counts as active).
  * **Choices/Default:** Any non-empty subset of **`CONF`**, **`SUSP`**, **`TEST`**, **`PRES`**. Default: **`CONF`**, **`PRES`**.

* **`volume_activity_indicators`** (Multi-select string, Required):
  * **Name:** Volume activity indicators
  * **Description:** Same choice set as **`activity_indicators`**; used to build **HF-level weights** for the weighted reporting rate branch.
  * **Choices/Default:** Subset of **`CONF`**, **`SUSP`**, **`TEST`**, **`PRES`**. Default: **`CONF`**, **`PRES`**.

* **`dataelement_method_denominator`** (String, Required):
  * **Name:** Denominator method
  * **Description:** Defines how the monthly expected facility universe is constructed for the data-element denominator.
  * **Choices/Default:** **`ROUTINE_ACTIVE_FACILITIES`**, **`PYRAMID_OPEN_FACILITIES`**. Default: **`ROUTINE_ACTIVE_FACILITIES`**.

* **`use_weighted_reporting_rates`** (Boolean, Optional):
  * **Name:** Use weighted reporting rates
  * **Description:** When **`true`**, the exported **`REPORTING_RATE`** column uses volume-weighted numerators and denominators; when **`false`**, unweighted rates populate that column.
  * **Choices/Default:** Default: **`false`**.

## Functionality Overview

1. Resolve paths, load **`SNT_config.json`**, and map **`routine_data_choice`** to a concrete routine filename.
2. Persist Papermill parameters (**`SNT_ROOT_PATH`**, **`ROUTINE_FILE`**, denominator, indicator lists, weighted flag) before executing the computation notebook.
3. Verify the routine file exists in the relevant dataset; stop early if the outliers pipeline has not produced imputed or removed files when those options are chosen.
4. Execute **`pipelines/snt_dhis2_reporting_rate_dataelement/code/snt_dhis2_reporting_rate_dataelement.ipynb`** to merge pyramid openings, monthly indicator activity, optional weights, and **ADM2** aggregates.
5. Upload **`{COUNTRY_CODE}_reporting_rate_dataelement`** outputs plus the parameter JSON to **`DHIS2_REPORTING_RATE`**.
6. Render **`snt_dhis2_reporting_rate_dataelement_report.ipynb`** outputs for analysts.

## Inputs

* **`configuration/SNT_config.json`**: Dataset IDs, **`COUNTRY_CODE`**, admin column names, and the list of DHIS2 indicators expected in routine data.
* **Routine Parquet** (see **`routine_data_choice`**): Monthly values per **`OU_ID`** / **`PERIOD`** with **`ADM1`** / **`ADM2`** identifiers.
* **`{COUNTRY_CODE}_pyramid.parquet`** (via dataset): Facility metadata including opening and closing dates for open-facility logic.

## Outputs

* **`{COUNTRY_CODE}_reporting_rate_dataelement.parquet`** and **`.csv`**: Columns include at least **`YEAR`**, **`MONTH`**, **`ADM2_ID`**, and **`REPORTING_RATE`** (weighted or unweighted per **`use_weighted_reporting_rates`**), plus intermediate count columns retained by the notebook.
* **Pipeline parameters JSON** under **`data/dhis2/reporting_rate/`**.
* **Papermill and reporting artefacts** under **`papermill_outputs/`** and **`reporting/outputs/`**.

> **Notes for the Data Analyst:**
>
> - **`Temporal and spatial resolution`**: Core logic operates on **monthly** **`PERIOD`** values and exports **ADM2-month** reporting rates.
> - **`activity_indicators`** and **`volume_activity_indicators`**: Activity drives the binary “reported this month” numerator; volume indicators drive **HF-level weights** used only when weighted rates are enabled.
> - **`dataelement_method_denominator`**: **`PYRAMID_OPEN_FACILITIES`** uses facility open/close dates; mis-specified dates in the pyramid will distort denominators.
> - **`Weighted reporting rates`**: Weights normalize mean monthly case volumes within **ADM2**; interpret weighted rates as completeness weighted by observed clinical volume proxies.
