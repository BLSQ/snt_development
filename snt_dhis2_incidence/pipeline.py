import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    validate_config,
)


# --- Quick & dirty: local copy to test new save_pipeline_parameters (JSON + CSV, list return)
# Remove this block and add save_pipeline_parameters to the import above once snt_utils is pushed ---
def save_pipeline_parameters(
    pipeline_name: str,
    parameters: dict[str, Any],
    output_path: Path,
    country_code: str,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Local copy for testing: saves parameters to JSON + CSV (EXECUTION_TIMESTAMP as column)."""
    output_path.mkdir(parents=True, exist_ok=True)
    execution_timestamp = datetime.now(timezone.utc).isoformat()

    parameters_log = {
        "pipeline_name": pipeline_name,
        "execution_timestamp": execution_timestamp,
        "country_code": country_code,
        "parameters": parameters,
    }
    if extra_metadata:
        parameters_log["metadata"] = extra_metadata

    json_path = output_path / f"{country_code}_parameters.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parameters_log, f, indent=2, default=str)

    row: dict[str, Any] = {
        "EXECUTION_TIMESTAMP": execution_timestamp,
        "pipeline_name": pipeline_name,
        "country_code": country_code,
        **{k: v for k, v in parameters.items()},
    }
    if extra_metadata:
        row.update(extra_metadata)

    csv_path = output_path / f"{country_code}_parameters.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    current_run.log_info(f"Pipeline parameters saved to {json_path.name} and {csv_path.name}")
    return [json_path, csv_path]


# -----------------------------------------------------------------------------------------------


@pipeline("snt_dhis2_incidence")
@parameter(
    "n1_method",
    name="Method for N1 calculations",
    help="Calculate N1 using `PRES` or `SUSP-TEST`",
    choices=["PRES", "SUSP-TEST"],
    type=str,
    required=True,
)
@parameter(
    "routine_data_choice",
    name="Routine data to use",
    help="Which routine data to use for the analysis. Options: 'raw' data is simply formatted and aligned;"
    "'raw_without_outliers' is the raw data after outliers removed (based on `outlier_detection_method`);"
    " 'imputed' contains imputed values after outliers removal",
    choices=["raw", "raw_without_outliers", "imputed"],
    type=str,
    required=True,
)
@parameter(
    "outlier_detection_method",
    name="Outlier detection method",
    help="Method used to detect outliers in the routine data",
    choices=["mean", "median", "iqr", "trend", "mg_partial", "mg_complete"],
    type=str,
    required=True,
)
@parameter(
    "use_csb_data",
    name="Use care seeking behaviour (CSB) data (source: DHS)",
    help="If True, the pipeline will use care seeking behaviour data (source: DHS) for the analysis,"
    " and calculate incidence adjusted for care seeking behaviour ('INCIDENCE_ADJ_CARESEEKING')",
    type=bool,
    default=False,
    required=True,
)
@parameter(
    "use_adjusted_population",
    name="Use adjusted population",
    help="If enabled, use adjusted population data for incidence calculations",
    type=bool,
    default=False,
    required=True,
)
@parameter(
    "disaggregation_selection",
    name="Disaggregation selection (NER only)",
    help="Select the disaggregation for indicence computation, available only for Niger.",
    multiple=False,
    choices=["pregnant", "under5"],
    type=str,
    default=None,
    required=False,
)
@parameter(
    "run_report_only",
    name="Run Report only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull notebooks from repository",
    help="Pull the latest notebooks from the GitHub repository."
    " Note: this will overwrite any local changes to the notebooks!",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_incidence(
    n1_method: str,
    routine_data_choice: str,
    outlier_detection_method: str,
    use_csb_data: bool,
    use_adjusted_population: bool,
    disaggregation_selection: str,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Pipeline entry point for running the SNT DHIS2 incidence notebook with specified parameters."""
    
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_incidence",
            report_scripts=["snt_dhis2_incidence_report.ipynb"],
            code_scripts=["snt_dhis2_incidence.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 Incidence pipeline...")
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_incidence"
        data_path = root_path / "data" / "dhis2" / "incidence"
        pipeline_path.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        # Helper to format the parameters for injection
        notebook_params = {
            "N1_METHOD": n1_method,
            "ROUTINE_DATA_CHOICE": routine_data_choice,
            "OUTLIER_DETECTION_METHOD": outlier_detection_method,
            "USE_CSB_DATA": use_csb_data,
            "USE_ADJUSTED_POPULATION": use_adjusted_population,
            "DISAGGREGATION_SELECTION": (
                disaggregation_selection.upper() if disaggregation_selection else None
            ),
            "ROOT_PATH": root_path.as_posix(),
        }

        if not run_report_only:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_incidence.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters=notebook_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            # Save pipeline parameters for provenance (JSON + CSV with EXECUTION_TIMESTAMP column)
            params_files = save_pipeline_parameters(
                pipeline_name="snt_dhis2_incidence",
                parameters={
                    "n1_method": n1_method,
                    "routine_data_choice": routine_data_choice,
                    "outlier_detection_method": outlier_detection_method,
                    "use_csb_data": use_csb_data,
                    "use_adjusted_population": use_adjusted_population,
                    "disaggregation_selection": disaggregation_selection,
                },
                output_path=data_path,
                country_code=country_code,
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_INCIDENCE"],
                country_code=country_code,
                file_paths=[
                    *params_files,
                    *[
                        p
                        for p in (
                            data_path.glob(
                                f"{country_code}_incidence_year_routine-data-*_rr-method-*.parquet"
                            )
                        )
                    ],
                    *[
                        p
                        for p in (
                            data_path.glob(f"{country_code}_incidence_year_routine-data-*_rr-method-*.csv")
                        )
                    ],
                ],
            )

        else:
            current_run.log_info("Skipping incidence calculations, running only the reporting.")

        # GP MODIFIED: parameters are now injected also into the report notebook nb
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_incidence_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            nb_parameters=notebook_params,
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
        )

        current_run.log_info("Pipeline finished!")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


if __name__ == "__main__":
    snt_dhis2_incidence()