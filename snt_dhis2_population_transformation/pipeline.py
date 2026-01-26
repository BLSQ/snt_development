from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    load_scripts_for_pipeline,
    get_repository,
    add_files_to_dataset,
    dataset_file_exists,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
)


# Quick & dirty: Override pull_scripts_from_repository with modified version
# This includes automatic snt_utils.r download (SNT25-279)
def pull_scripts_from_repository(
    pipeline_name: str,
    report_scripts: list[str],
    code_scripts: list[str],
    repo_path: Path = Path("/tmp"),
    repo_name: str = "snt_development",
    pipeline_parent_folder: Path = Path(workspace.files_path, "pipelines"),
) -> None:
    """Pull the latest pipeline scripts from the SNT repository and update the local workspace.

    Parameters
    ----------
    pipeline_name : str
        The name of the pipeline for which scripts are being updated.
    report_scripts : list[str]
        List of reporting script names to be updated.
    code_scripts : list[str]
        List of code script names to be updated.
    repo_path : Path, optional
        The path to the repository where the scripts are stored (default is "/tmp").
    repo_name : str, optional
        The name of the repository from which to pull the scripts (default is "snt_development").
        It also corresponds to the folder where the repo is stored.
    pipeline_parent_folder : Path, optional
        The path to the pipeline location (not the full path!) in the workspace where
        the scripts will be replaced (default is "pipelines" in the SNT workspaces files path).

    This function attempts to update reporting scripts and logs errors or warnings if the update fails.
    
    Also automatically pulls snt_utils.r from the snt_utils repository.
    """
    # Pull snt_utils.r from the snt_utils repository
    # The file is located at code/snt_utils.R in the repo and will be saved to code/snt_utils.r in the workspace
    try:
        current_run.log_info("Pulling snt_utils.r from repository (SNT25-279)")
        snt_root_path = Path(workspace.files_path)
        code_path = snt_root_path / "code"
        code_path.mkdir(parents=True, exist_ok=True)
        # Source: code/snt_utils.R in the snt_utils repository
        # Destination: code/snt_utils.r in the workspace
        script_paths = {Path("code/snt_utils.R"): code_path / "snt_utils.r"}
        load_scripts_for_pipeline(
            snt_script_paths=script_paths,
            repository_path=Path("/tmp"),
            repository_name="snt_utils",
        )
        current_run.log_info(f"Successfully updated {code_path / 'snt_utils.r'} from repository")
    except Exception as e:
        current_run.log_warning(
            f"Failed to update snt_utils.r from repository: {e}. "
            "Using existing version if available."
        )

    # Paths Repository -> Workspace
    repository_source = repo_path / repo_name / "pipelines" / pipeline_name
    pipeline_target = pipeline_parent_folder / pipeline_name

    # Create the mapping of script paths
    reporting_paths = {
        (repository_source / "reporting" / r): (pipeline_target / "reporting" / r)
        for r in report_scripts
    }
    code_paths = {
        (repository_source / "code" / c): (pipeline_target / "code" / c)
        for c in code_scripts
    }

    current_run.log_info(
        f"Updating scripts {', '.join(report_scripts + code_scripts)} from repository '{repo_name}'"
    )

    try:
        # Pull scripts from the SNT repository (replace local)
        load_scripts_for_pipeline(
            snt_script_paths=reporting_paths | code_paths,
            repository_path=repo_path,
            repository_name=repo_name,
        )
    except Exception as e:
        current_run.log_error(f"Error: {e}")
        current_run.log_warning("Continuing without scripts update.")


@pipeline("snt_dhis2_population_transformation")
@parameter(
    "adjust_population",
    name="Adjust using UN totals",
    help="Use UN totals to scale the DHIS2 population.",
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
        "ADJUST_WITH_UNTOTALS": adjust_population,
    }

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_population.parquet"):
        current_run.log_warning(
            f"File {country_code} DHIS2 population formatted not found, "
            "perhaps DHIS2 formatting pipeline has not yet been execute. Skipping transformation."
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
