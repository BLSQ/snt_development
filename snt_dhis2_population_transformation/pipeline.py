from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace, File
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    dataset_file_exists,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    save_pipeline_parameters,
    get_file_from_dataset,
)


@pipeline("snt_dhis2_population_transformation")
@parameter(
    "tot_pop_reference",
    name="Part 1: Population reference",
    help=(
        "Total population used to scale DHIS2 population data. When provided, "
        "DHIS2 values are adjusted proportionally to match this total."
    ),
    type=int,
    default=None,
    required=False,
)
@parameter(
    "pop_under_5",
    name="Part 2: Proportion population under 5",
    help=(
        "Proportion of the total population aged under 5 (e.g. 0.17 for 17%). "
        "Used to disaggregate population figures into the under-5 age group."
    ),
    type=float,
    default=None,
    required=False,
)
@parameter(
    "pop_pregnant_women",
    name="Part 2: Proportion population pregnant women",
    help=(
        "Proportion of the total population of pregnant women (e.g. 0.05 for 5%). "
        "Used to disaggregate population figures into the pregnant-women group."
    ),
    type=float,
    default=None,
    required=False,
)
@parameter(
    "pop_0_1_y",
    name="Part 2: Proportion population 0-1 years",
    help=(
        "Proportion of the total population aged 0-1 years (e.g. 0.04 for 4%). "
        "Used to disaggregate population figures into the 0-1 age group."
    ),
    type=float,
    default=None,
    required=False,
)
@parameter(
    "pop_1_2_y",
    name="Part 2: Proportion population 1-2 years",
    help=(
        "Proportion of the total population aged 1-2 years (e.g. 0.03 for 3%). "
        "Used to disaggregate population figures into the 1-2 age group."
    ),
    type=float,
    default=None,
    required=False,
)
@parameter(
    "pop_5_10_y",
    name="Part 2: Proportion population 5-10 years",
    help=(
        "Proportion of the total population aged 5-10 years (e.g. 0.06 for 6%). "
        "Used to disaggregate population figures into the 5-10 age group."
    ),
    type=float,
    default=None,
    required=False,
)
@parameter(
    "pop_5_36_m",
    name="Part 2: Proportion population 5-36 months",
    help=(
        "Proportion of the total population aged 5-36 months (e.g. 0.06 for 6%). "
        "Used to disaggregate population figures into the 5-36 months age group."
    ),
    type=float,
    default=None,
    required=False,
)
@parameter(
    "disaggregation_file",
    name="Part 2: Use disaggregation proportions (.csv)",
    type=File,
    required=False,
    default=None,
    help="Select user-uploaded file with population disaggregations proportions at ADM2 level.",
)
@parameter(
    "growth_factor",
    name="Part 3: Projection growth rate",
    help=(
        "Annual growth rate (e.g. 0.03 for 3%) used to project "
        "DHIS2 population figures into past and future years."
    ),
    type=float,
    default=None,
    required=True,
)
@parameter(
    "year_reference",
    name="Part 3: Projection reference year",
    help=(
        "Base year from which DHIS2 population figures are projected. "
        "This year must be available in the population data. "
        "Defaults to the latest year available"
    ),
    type=int,
    default=None,
    required=True,
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
def snt_dhis2_population_transformation(
    tot_pop_reference: int,
    growth_factor: float,
    year_reference: int,
    pop_under_5: float,
    pop_pregnant_women: float,
    pop_0_1_y: float,
    pop_1_2_y: float,
    pop_5_10_y: float,
    pop_5_36_m: float,
    disaggregation_file: File,
    run_report_only: bool,
    pull_scripts: bool,
):
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
        # Load configuration (needed for report and for main run)
        snt_config_dict = load_configuration_snt(
            config_path=snt_root_path / "configuration" / "SNT_config.json"
        )
        validate_config(snt_config_dict)
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
        if country_code is None:
            current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

        if not run_report_only:
            if disaggregation_file and not Path(disaggregation_file.path).exists():
                current_run.log_error(f"Disaggregation file not found: {disaggregation_file.path}")
                raise FileNotFoundError

            years_available = get_available_years_from_dhis2_population_data(snt_config_dict)
            if not years_available:
                current_run.log_error("No DHIS2 population data available.")
                raise ValueError

            if year_reference not in years_available:
                current_run.log_error(
                    f"Population reference year {year_reference} is not available in population data. "
                    f"Available years are: {[int(y) for y in years_available]}"
                )
                raise ValueError

            parameters = {
                "TOT_POP_REFERENCE": tot_pop_reference,
                "GROWTH_FACTOR": growth_factor,
                "YEAR_REFERENCE": year_reference,
                "POP_UNDER_5": pop_under_5,
                "POP_PREGNANT_WOMEN": pop_pregnant_women,
                "POP_0_1_Y": pop_0_1_y,
                "POP_1_2_Y": pop_1_2_y,
                "POP_5_10_Y": pop_5_10_y,
                "POP_5_36_M": pop_5_36_m,
                "DISAGGREGATION_FILE": disaggregation_file.path if disaggregation_file else None,
            }

            params_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_population_transformation",
                parameters=parameters,
                output_path=snt_dhis2_pop_transform_path,
                country_code=country_code,
            )
            current_run.log_info(f"Saved pipeline parameters to {params_file}")

            # Apply transformation to population data
            dhis2_population_transformation(
                snt_root_path=snt_root_path,
                pipeline_root_path=snt_pipeline_path,
                snt_config=snt_config_dict,
                nb_parameter=parameters,
            )

            add_files_to_dataset(
                dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get(
                    "DHIS2_POPULATION_TRANSFORMATION", None
                ),
                country_code=country_code,
                file_paths=[
                    snt_dhis2_pop_transform_path / f"{country_code}_population.parquet",
                    snt_dhis2_pop_transform_path / f"{country_code}_population.csv",
                    params_file,
                ],
            )

        run_report_notebook(
            nb_file=snt_pipeline_path / "reporting" / "snt_dhis2_population_transformation_report.ipynb",
            nb_output_path=snt_pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

    except Exception as e:
        current_run.log_error(f"Error in population transformation: {e}")
        raise


def dhis2_population_transformation(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
    nb_parameter: dict,
) -> None:
    """Format DHIS2 analytics data for SNT."""
    current_run.log_info("Running DHIS2 population data transformations.")

    # set parameters for notebook
    nb_parameter.update({"SNT_ROOT_PATH": str(snt_root_path)})

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_population.parquet"):
        current_run.log_warning(
            f"File {country_code} DHIS2 population formatted not found, "
            "perhaps DHIS2 formatting pipeline has not yet been executed. Skipping process."
        )
        return

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "snt_dhis2_population_transformation.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )
    except Exception as e:
        raise Exception(f"Error in executing population transformation notebook: {e}") from e


def get_available_years_from_dhis2_population_data(snt_config_dict: dict) -> list[int]:
    """Get the years available in the DHIS2 population data.

    Returns:
        A sorted list of years available in the population data, or an empty list.
    """
    country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
    pop_data = get_file_from_dataset(
        dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED"),
        filename=f"{country_code}_population.parquet",
    )

    if pop_data is not None and not pop_data.empty:
        return sorted(pop_data.YEAR.unique())
    return []


if __name__ == "__main__":
    snt_dhis2_population_transformation()
