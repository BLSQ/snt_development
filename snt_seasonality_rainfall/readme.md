# SNT Seasonality Rainfall Pipeline

This pipeline quantifies **rainfall seasonality** at **ADM2** using **monthly** ERA5-derived precipitation totals (`MEAN`), evaluates multi-month concentrated rainfall windows, and exports a single wide indicator table. Outputs are stored under `data/seasonality_rainfall/` and registered with **SNT_SEASONALITY_RAINFALL**, followed by the rainfall reporting notebook.

## Parameters

- **`get_minimum_month_block_size`** (int, required): **Minimum number of months per block.** Allowed values `3`, `4`, or `5`. Default: `3`. Passed to the notebook as `minimum_month_block_size`.

- **`get_maximum_month_block_size`** (int, required): **Maximum number of months per block.** Allowed values `3`, `4`, or `5`. Default: `5`. Passed as `maximum_month_block_size`; must be greater than or equal to the minimum.

- **`get_threshold_for_seasonality`** (float, required): **Minimal proportion of rainfall for seasonality.** Share of annual rainfall that must fall inside a candidate forward-looking block (e.g., `0.6` = 60%). Default: `0.6`.

- **`get_threshold_proportion_seasonal_years`** (float, optional): **Minimal proportion of seasonal years.** Share of years that must be seasonal for an ADM2 to be classified seasonal overall. Default: `0.5`.

- **`use_calendar_year_denominator`** (bool, optional): **Use calendar year as denominator.** `false` (default) uses a 12-month forward-looking sliding window (WHO-style annualization); `true` uses January–December calendar-year totals.

## Functionality Overview

1. Load **`SNT_config.json`**, derive **`COUNTRY_CODE`**, and map pipeline arguments to the notebook parameter names.
2. Validate numeric ranges (positive proportions ≤ 1, integer months, min ≤ max).
3. Execute the notebook to load **`{COUNTRY_CODE}_total_precipitation_monthly.parquet`**, impute eligible missing months, compute block-wise seasonality metrics, and merge **ADM2** attributes.
4. Save parameters JSON and upload **`{COUNTRY_CODE}_rainfall_seasonality`** outputs to **`SNT_SEASONALITY_RAINFALL`**.
5. Run **`reporting/snt_seasonality_rainfall_report.ipynb`** for analyst-facing visuals.

## Inputs

- **`configuration/SNT_config.json`**: Includes **`ERA5_DATASET_CLIMATE`** and country metadata consumed inside the notebook.
- **`{COUNTRY_CODE}_total_precipitation_monthly.parquet`**: Monthly rainfall totals per `ADM2_ID` with `YEAR`, `MONTH`, and `MEAN`.
- **ADM2 geometries** (loaded via configured dataset helpers in the notebook) for district names/IDs attached to outputs.

## Outputs

- **`{COUNTRY_CODE}_rainfall_seasonality.parquet`** and **`.csv`**: Wide ADM2 table with seasonality classification, block duration, onset month, and rainfall proportion columns produced by the notebook.
- **Pipeline parameters JSON** under `data/seasonality_rainfall/`.
- **Papermill output** in `papermill_outputs/` and **static report assets** in `reporting/outputs/`.

> **Notes for the Data Analyst:**

> - **`Temporal and spatial resolution`**: Inputs and outputs are **monthly** time series summarized at **ADM2**; ERA5 totals are already aggregated to that level in the extract file.

> - **`use_calendar_year_denominator`**: Changes how “annual rainfall” is defined when testing concentrated blocks; compare WHO sliding-year vs calendar-year results before switching production settings.

> - **`get_threshold_for_seasonality`**: Works on the forward-looking block sum divided by the annual denominator chosen above; raising the threshold tightens what counts as a concentrated rainy season.

> - **`get_threshold_proportion_seasonal_years`**: Controls how stable the seasonal signal must be across years before labeling an ADM2 seasonal.

> - **`Imputation guardrails`**: The notebook mirrors the legacy seasonality checks (overall and per-district missingness); districts with excessive gaps error out of imputation.
