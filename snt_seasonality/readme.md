# SNT Seasonality Pipeline

This umbrella pipeline evaluates **monthly** **ADM2** time series for **rainfall** (`MEAN` from ERA5-based extracts) and/or **confirmed cases** (`CONF` aggregated from routine DHIS2 data). It imputes sparse series where rules allow, classifies seasonality using sliding multi-month blocks, and writes paired wide tables and imputed series to `data/seasonality/` for ingestion into **SNT_SEASONALITY**.

## Parameters

- **`run_precipitation`** (bool): **Run precipitation seasonality.** When `true`, executes the shared notebook for the precipitation branch. Default: `true`.

- **`run_cases`** (bool): **Run cases seasonality.** When `true`, executes the notebook for the malaria cases branch. Default: `true`.

- **`minimum_periods`** (int, required): **Minimum number of periods.** Minimum distinct **year–month** observations required before analysis proceeds (notebook messaging references case-style thresholds; precipitation uses the same gate). Default: `48`.

- **`maximum_proportion_missings_overall`** (float, required): **Maximum proportion of missing datapoints overall.** Upper bound on share of missing values across the full space-time panel before aborting. Default: `0.1`.

- **`maximum_proportion_missings_per_district`** (float, required): **Maximum proportion of missing datapoints per district.** Per-`ADM2_ID` missing share threshold for imputation eligibility. Default: `0.2`.

- **`minimum_month_block_size`** (int, required): **Minimum month block size.** Shortest contiguous block length (months) tested for concentrated seasonality. Default: `3`.

- **`maximum_month_block_size`** (int, required): **Maximum month block size.** Longest block length (months) tested. Default: `5`.

- **`threshold_for_seasonality`** (float, required): **Threshold for seasonality.** Minimum share of annual rainfall or cases that must fall inside a candidate block to flag that start month. Default: `0.6`.

- **`threshold_proportion_seasonal_years`** (float, required): **Threshold proportion seasonal years.** Share of years that must be classified seasonal for an ADM2 to be seasonal overall. Default: `0.5`.

## Functionality Overview

1. Load **`SNT_config.json`**, build the parameter dictionary, and validate numeric ranges (non-negative proportions between 0 and 1, integer month counts).
2. For each enabled branch (**`precipitation`**, **`cases`**), run Papermill on the shared notebook with **`type_of_seasonality`** set, catching known “insufficient data” failures as warnings instead of hard failures.
3. Collect **`{COUNTRY_CODE}_precipitation_seasonality`**, **`{COUNTRY_CODE}_cases_seasonality`**, and matching **`_imputed`** tables when runs succeed.
4. Save consolidated parameters and upload all produced files to **`SNT_SEASONALITY`**.
5. Execute **`reporting/snt_seasonality_report.ipynb`** for documentation plots.

## Inputs

- **`configuration/SNT_config.json`**: `COUNTRY_CODE`, dataset IDs (ERA5 climate extract, DHIS2 formatted routine data), and paths to ADM2 geometries referenced inside the notebook.
- **`{COUNTRY_CODE}_total_precipitation_monthly.parquet`**: Monthly ERA5 rainfall totals per `ADM2_ID` (`YEAR`, `MONTH`, `MEAN`).
- **`{COUNTRY_CODE}_routine.parquet`**: Monthly DHIS2 routine file containing `CONF` and admin keys for case seasonality.
- **ADM2 spatial layer** (from configured dataset paths inside the notebook) used to align tabular outputs with administrative attributes.

## Outputs

- **`{COUNTRY_CODE}_precipitation_seasonality.parquet`** / **`.csv`** and **`{COUNTRY_CODE}_precipitation_imputed.parquet`** / **`.csv`**: When precipitation analysis runs successfully.
- **`{COUNTRY_CODE}_cases_seasonality.parquet`** / **`.csv`** and **`{COUNTRY_CODE}_cases_imputed.parquet`** / **`.csv`**: When cases analysis runs successfully.
- **Parameter JSON** saved beside the seasonality tables.
- **Papermill outputs** (`papermill_outputs/`) and **reporting outputs** (`reporting/outputs/`).

> **Notes for the Data Analyst:**

> - **`Temporal and spatial resolution`**: All series are **monthly** totals or counts aligned to **`ADM2_ID`** (`YEAR`, `MONTH`).

> - **`minimum_periods`**: Acts on the number of distinct **year–month** periods after expanding to the ADM2–month grid; short histories skip analysis with logged errors.

> - **`run_precipitation` / `run_cases`**: Branches are independent; disable either side when upstream extracts are unavailable to avoid wasted runs.

> - **`Imputation`**: SARIMA-based filling only proceeds when overall and per-district missingness rules pass; otherwise the notebook aborts that branch.

> - **`threshold_for_seasonality`**: Compares each candidate multi-month block’s share of the annual total (12-month forward window or calendar year depending on downstream helpers) against this proportion.
