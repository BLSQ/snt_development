# SNT Seasonality Cases Pipeline

This pipeline measures **malaria case seasonality** at **ADM2** using **monthly** **`CONF`** counts from the formatted DHIS2 routine extract. It mirrors the rainfall seasonality mathematics—forward-looking multi-month blocks, optional calendar-year denominators, and SARIMA imputation under missingness caps—and publishes `{COUNTRY_CODE}_cases_seasonality` artefacts to **SNT_SEASONALITY_CASES**.

## Parameters

- **`get_minimum_month_block_size`** (int, required): **Minimum number of months per block.** Choices `3`, `4`, or `5`. Default: `3`. Notebook parameter: `minimum_month_block_size`.

- **`get_maximum_month_block_size`** (int, required): **Maximum number of months per block.** Choices `3`, `4`, or `5`. Default: `5`. Notebook parameter: `maximum_month_block_size` (must be ≥ minimum).

- **`get_threshold_for_seasonality`** (float, required): **Minimal proportion of cases for seasonality.** Share of annual confirmed cases that must fall inside a candidate block (e.g., `0.6` = 60%). Default: `0.6`.

- **`get_threshold_proportion_seasonal_years`** (float, optional): **Minimal proportion of seasonal years.** Fraction of years that must be flagged seasonal to classify an ADM2 as seasonal overall. Default: `0.5`.

- **`use_calendar_year_denominator`** (bool, optional): **Use calendar year as denominator.** `false` (default) applies the WHO-style 12-month forward window for annualized case totals; `true` uses January–December sums.

## Functionality Overview

1. Load **`SNT_config.json`**, read **`COUNTRY_CODE`**, and translate pipeline kwargs to notebook-friendly names.
2. Validate thresholds and month-block integers (including min ≤ max).
3. Papermill the notebook to aggregate routine **`CONF`** to **`ADM2_ID` × `YEAR` × `MONTH`**, enforce minimum period counts (notebook default **`minimum_periods = 36`** internal guard), impute where allowed, and compute seasonality metrics (**`SEASONALITY_CASES`**, block duration, onset month, **`CASES_PROPORTION`**, etc.).
4. Persist parameters JSON and upload **`{COUNTRY_CODE}_cases_seasonality`** files to **`SNT_SEASONALITY_CASES`**.
5. Run **`reporting/snt_seasonality_cases_report.ipynb`**.

## Inputs

- **`configuration/SNT_config.json`**: Supplies dataset identifiers (DHIS2 formatted routine) and geometry references.
- **`{COUNTRY_CODE}_routine.parquet`**: Monthly DHIS2 extract containing `CONF`, `ADM2_ID`, `YEAR`, and `MONTH` (plus other routine columns ignored by this pipeline’s seasonality math).
- **ADM2 boundary attributes** pulled with the spatial helpers in the notebook for naming and QC columns.

## Outputs

- **`{COUNTRY_CODE}_cases_seasonality.parquet`** and **`.csv`**: Wide ADM2 table with case-based seasonality indicators.
- **Pipeline parameters JSON** under `data/seasonality_cases/`.
- **Papermill output** (`papermill_outputs/`) and **reporting outputs** (`reporting/outputs/`).

> **Notes for the Data Analyst:**

> - **`Temporal and spatial resolution`**: Inputs are **monthly confirmed cases** aggregated to **ADM2**; outputs remain one row per district with seasonality metadata.

> - **`CONF` column**: The notebook hard-codes confirmed malaria cases as the analytic signal; other indicators are not considered unless the notebook is changed.

> - **`use_calendar_year_denominator`**: Aligns the “annual case burden” definition with either sliding 12-month windows or strict calendar years—pick the option that matches national reporting guidance.

> - **`get_threshold_proportion_seasonal_years`**: Higher values demand more consistent inter-annual seasonality before marking a district seasonal.

> - **`Minimum history`**: The notebook enforces a floor on the number of year–month periods; short extracts stop with logged errors rather than producing partial classifications.
