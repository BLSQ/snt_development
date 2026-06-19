#%% imports

from pathlib import Path
from openhexa.sdk import current_run, pipeline, File, parameter, workspace
from worldpopclient import WorldPopClient
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    pull_scripts_from_repository,
    save_pipeline_parameters
)

#%% pipeline definition

@pipeline("snt_healthcare_access")
@parameter(
    code="input_fosa_file",
    name="Optional FOSA location file (.csv)",
    type=File,
    required=False,
    default=None,
    help="If not provided, the DHIS2 pyramid metadata file will be used.",
)
@parameter(
    code="wpop_year",
    name="Year",
    help="Reference year for population data.",
    type=int,
    required=True,
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
    wpop_year: int,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Pipeline to run computation and report notebooks for healthcare access.

    Determining the percentage of population which is within 5 km of at
    least one FOrmation SAnitaire (FOSA).
    """
    # paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_healthcare_access"
    wpop_raster_path = snt_root_path / "data" / "worldpop" / "rasters"
    data_output_path = snt_root_path / "data" / "healthcare_access"
    data_intermediate_path = data_output_path / "intermediate_results"
    # ensure necessary directories exist
    pipeline_path.mkdir(parents=True, exist_ok=True)
    (pipeline_path / "reporting" / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    data_output_path.mkdir(parents=True, exist_ok=True)
    data_intermediate_path.mkdir(parents=True, exist_ok=True)

    # validate input parameter values
    if input_fosa_file is not None:
        current_run.log_info(f"FOSA coordinates file: {input_fosa_file.path}")
    else:
        current_run.log_info(f"Using default FOSA data.")

    if not (2015 <= wpop_year <= 2030):
        msg = f"Year {wpop_year} is out of range. WorldPop rasters are available for 2015–2030."
        current_run.log_warning(msg)
        raise ValueError(msg)
    else:
        current_run.log_info(f"Year for raster population data: {wpop_year}.")

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_healthcare_access",
            report_scripts=["snt_healthcare_access_report.ipynb"],
            code_scripts=["snt_healthcare_access.ipynb"],
        )

    try:
 
        # Load & validate configuration file
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")

        shapes = retrieve_shapes(snt_config=snt_config)
        if shapes is None:
            current_run.log_error("No valid shapes available. Processing stopped.")
            raise ValueError

        # if user chooses to run the computation notebook
        if not run_report_only:

            # save the input parameter values to file
            parameters_file = save_pipeline_parameters(
            pipeline_name="snt_healthcare_access",
            parameters={
                "input_fosa_file": input_fosa_file,
                "wpop_year": wpop_year,
                "run_report_only": run_report_only,
                "pull_scripts": pull_scripts,
            },
            output_path=data_output_path,
            country_code=country_code
            )

            # the params to use in the computation notebook
            input_params = {
                "FOSA_FILE": input_fosa_file.path if input_fosa_file is not None else None,
                "WORLDPOP_YEAR": wpop_year
                }

            # download worldpop data if it doesn't already exist in the folder
            pop_path = get_or_download_worldpop_raster(
                    country_code=country_code,
                    ref_year=str(wpop_year),
                    raster_dir=wpop_raster_path,
                )
            if pop_path is None:
                current_run.log_error(f"Could not retrieve population raster for {wpop_year}. Stopping.")
                raise ValueError(f"Population raster unavailable for {wpop_year}.")
            
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_healthcare_access.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            # add files to a new dataset version
            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_HEALTHCARE_ACCESS", None),
                country_code=country_code,
                file_paths=[
                    data_output_path / f"{country_code}_population_covered_health.parquet",
                    data_output_path / f"{country_code}_population_covered_health.csv",
                    parameters_file,
                ],
            )

        else:
            current_run.log_info("Skipping calculations, running only the reporting.")

        # in all cases, run the reporting notebook
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_healthcare_access_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        current_run.log_info("Pipeline executed successfully!")
    except Exception as e:
        current_run.log_error(f"Error occurred while executing the pipeline: {e}")
        raise


#%% functions used in pipeline

def get_or_download_worldpop_raster(country_code: str, ref_year: int, raster_dir: Path) -> Path | None:
    """Return the path to the population raster for the given country and year.
    Uses an existing file if found, otherwise downloads it from WorldPop.
    """

    try:
        raster_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        current_run.log_warning(f"Failed to create directory {raster_dir}: {e}")
        return None
    
    existing = list(raster_dir.glob(f"{country_code.lower()}_pop_{ref_year}_*.tif")) # the MAP pipeline saves extractions using lowercase for the country name => using the same here
    if existing:
        current_run.log_info(f"Population raster found: {existing[0]}.")
        return existing[0]

    current_run.log_info(f"No raster found for {ref_year}. Downloading from WorldPop.")

    try:
        wpop_client = WorldPopClient()
        wpop_output_raster_path = wpop_client.download_data_for_country(
            country_iso3=country_code.upper(),
            year=str(ref_year),
            output_dir=raster_dir,
            overwrite=False,
        )
        current_run.log_info(f"Raster downloaded: {wpop_output_raster_path}.")
        return wpop_output_raster_path
    except Exception as e:
        current_run.log_warning(f"WorldPop download failed for {country_code} - {ref_year}: {e}")
        return None


if __name__ == "__main__":
    snt_healthcare_access()
