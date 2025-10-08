from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    dataset_file_exists,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
)


@pipeline("snt_dhis2_population_transformation")
@parameter(
    "adjust_population",
    name="Adjust using WorldPop UN estimates",
    help="Use WorldPop UN adjusted estimates to scale the DHIS2 population totals.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull scripts",
    help="Pull the latest scripts from the repository (useful if you want to update the pipeline scripts).",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_population_transformation(adjust_population: bool, run_report_only: bool, pull_scripts: bool):
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations or
    expensive computations.
    """
    # set paths
    snt_root_path = Path(workspace.files_path)
    snt_pipeline_path = snt_root_path / "pipelines" / "snt_dhis2_population_transformation"
    snt_dhis2_pop_transform_path = snt_root_path / "data" / "dhis2" / "population_transformed"

    # create paths if they don't exist
    snt_pipeline_path.mkdir(parents=True, exist_ok=True)
    snt_dhis2_pop_transform_path.mkdir(parents=True, exist_ok=True)

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_population_transformation",
            report_scripts=["snt_dhis2_population_transformation_report.ipynb"],
            code_scripts=[
                "snt_dhis2_population_transformation.ipynb",
            ],
        )

    try:
        if not run_report_only:
            # Load configuration
            snt_config_dict = load_configuration_snt(
                config_path=snt_root_path / "configuration" / "SNT_config.json"
            )

            # Validate configuration
            validate_config(snt_config_dict)

            # get country identifier for naming
            country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
            if country_code is None:
                current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

            # Apply transformation to population data
            dhis2_population_transformation(
                snt_root_path=snt_root_path,
                pipeline_root_path=snt_pipeline_path,
                snt_config=snt_config_dict,
                adjust_population=adjust_population,
            )

            add_files_to_dataset(
                dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get(
                    "DHIS2_POPULATION_TRANSFORMATION", None
                ),
                country_code=country_code,
                file_paths=[
                    snt_dhis2_pop_transform_path / f"{country_code}_population.parquet",
                    snt_dhis2_pop_transform_path / f"{country_code}_population.csv",
                ],
            )

        run_report_notebook(
            nb_file=snt_pipeline_path / "reporting" / "snt_dhis2_population_transformation_report.ipynb",
            nb_output_path=snt_pipeline_path / "reporting" / "outputs",
            nb_parameters=None,
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
        )

    except Exception as e:
        current_run.log_error(f"Error in SNT DHIS2 formatting: {e}")
        raise


def dhis2_population_transformation(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
    adjust_population: bool,
) -> None:
    """Format DHIS2 analytics data for SNT."""
    current_run.log_info("Running DHIS2 population data transformations.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
        "ADJUST_WITH_WORLDPOP": adjust_population,
    }

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_population.parquet"):
        current_run.log_info(
            f"File {country_code} DHIS2 population formatted not found, skipping transformation."
        )
        return

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "snt_dhis2_population_transformation.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
        )
    except Exception as e:
        raise Exception(f"Error in formatting analytics data: {e}") from e


if __name__ == "__main__":
    snt_dhis2_population_transformation()
