# SNT DHIS2 Formatting Pipeline

The `snt_dhis2_formatting` pipeline, processes and formats DHIS2 data extracts for the SNT workflow.

## Description

The `snt_dhis2_formatting` pipeline orchestrates the formatting of various DHIS2 data extracts. It reads raw data, runs a series of Jupyter notebooks to transform and format the data, and then saves the output. The pipeline can also pull the latest versions of the processing scripts from a repository and generate a final report.

The pipeline performs the following steps:
1. **Script Update:** Optionally pulls the latest data processing scripts from a configured repository.
2. **Configuration:** Loads and validates the `SNT_config.json` configuration file.
3. **Data Formatting:** Executes a series of Jupyter notebooks to format different DHIS2 data extracts, including:
    - Analytics data (`snt_dhis2_formatting_routine.ipynb`)
    - Population data (`snt_dhis2_formatting_population.ipynb`)
    - Geospatial shapes data (`snt_dhis2_formatting_shapes.ipynb`)
    - Population pyramid data (`snt_dhis2_formatting_pyramid.ipynb`)
    - Reporting rates data (`snt_dhis2_formatting_reporting_rates.ipynb`)
4. **Output Storage:** Saves the formatted data into `.parquet` and `.csv` files.
5. **Dataset Update:** Adds the newly formatted files to a designated dataset.
6. **Reporting:** Runs a final Jupyter notebook (`snt_dhis2_formatting_report.ipynb`) to generate a report based on the formatted data.

## Parameters

The pipeline accepts the following parameters:

| Parameter | Type | Description | Default | Required |
|---|---|---|---|---|
| `run_report_only` | `bool` | If set to `True`, the pipeline will only execute the reporting notebook and skip all data formatting steps. | `False` | No |
| `pull_scripts` | `bool` | If set to `True`, the pipeline will pull the latest scripts from the repository before execution. | `False` | No |

## Outputs

The primary outputs of this pipeline are formatted data files and a report.

### Formatted Data

The pipeline generates the following files in the `workspace/data/dhis2/extracts_formatted/` directory, where `{country_code}` is derived from the configuration file:

- `{country_code}_routine.parquet`
- `{country_code}_routine.csv`
- `{country_code}_population.parquet`
- `{country_code}_population.csv`
- `{country_code}_shapes.geojson`
- `{country_code}_pyramid.parquet`
- `{country_code}_pyramid.csv`
- `{country_code}_reporting.parquet`
- `{country_code}_reporting.csv`

### Report

A report is generated from the `snt_dhis2_formatting_report.ipynb` notebook and the output is stored in the `workspace/pipelines/snt_dhis2_formatting/reporting/outputs/` directory.