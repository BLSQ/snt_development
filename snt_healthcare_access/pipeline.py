from pathlib import Path
from openhexa.sdk import current_run, pipeline, File, parameter, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    pull_scripts_from_repository,
)


@pipeline("snt_healthcare_access")
@parameter(
    "input_fosa_file",
    name="FOSA location file (.csv)",
    type=File,
    required=False,
    default=None,
    help="If not provided, the DHIS2 pyramid metadata file will be used.",
)
@parameter(
    "input_radius_meters",
    name="Radius around FOSA (meters)",
    type=int,
    default=5000,
    required=False,
)
@parameter(
    "input_pop_file",
    name="Population raster file (.tif)",
    type=File,
    required=False,
    default=None,
    help="If not provided, the default WorldPop raster will be used.",
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
def snt_healthcare_access(
    input_fosa_file: File,
    input_radius_meters: int,
    input_pop_file: File,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Pipeline to run computation and report notebooks for healthcare access.

    Determining the percentage of population which is within a given radius of at
    least one FOrmation SAnitaire.
    """
    # paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_healthcare_access"
    data_output_path = snt_root_path / "data" / "healthcare_access"
    # ensure directories exist
    pipeline_path.mkdir(parents=True, exist_ok=True)
    (pipeline_path / "reporting" / "outputs").mkdir(parents=True, exist_ok=True)
    data_output_path.mkdir(parents=True, exist_ok=True)

    num_km = input_radius_meters / 1000
    if input_fosa_file is not None:
        current_run.log_info(f"Coordinates file: {input_fosa_file.path}")
    current_run.log_info(f"Using radii of {num_km} km around each FOSA.")
    if input_pop_file is not None:
        current_run.log_info(f"Population raster: {input_pop_file.path}")

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_healthcare_access",
            report_scripts=["snt_healthcare_accessreport.ipynb"],
            code_scripts=["snt_healthcare_access.ipynb"],
        )

    try:
        # Load configuration
        snt_config_dict = load_configuration_snt(
            config_path=Path(snt_root_path, "configuration", "SNT_config.json")
        )

        # Validate configuration
        validate_config(snt_config_dict)

        # get country identifier for naming
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")

        if not run_report_only:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_healthcare_access.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters={
                    "FOSA_FILE": input_fosa_file.path if input_fosa_file is not None else None,
                    "RADIUS_METERS": input_radius_meters,
                    "POP_FILE": input_pop_file.path if input_pop_file is not None else None,
                },
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            # add files to a new dataset version
            add_files_to_dataset(
                dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("SNT_HEALTHCARE_ACCESS", None),
                country_code=country_code,
                file_paths=[
                    data_output_path / f"{country_code}_population_covered_health.parquet",
                    data_output_path / f"{country_code}_population_covered_health.csv",
                ],
            )

        else:
            current_run.log_info("Skipping incidence calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_healthcare_access_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
        )

        current_run.log_info("Pipeline finished!")
    except Exception as e:
        current_run.log_error(f"Error occurred while executing the pipeline: {e}")
        raise


if __name__ == "__main__":
    snt_healthcare_access()
