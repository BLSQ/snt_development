import time
from pathlib import Path
import pandas as pd

from openhexa.sdk import current_run, parameter, pipeline, workspace, File
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    save_pipeline_parameters,
    get_file_from_dataset,
    check_outputs_generated,
)


@pipeline("snt_dhis2_population_transformation")
@parameter(
    "pop_source",
    name="Population source",
    help=(
        "Choose the source of the population data. DHIS2 (DHIS2_DATASET_FORMATTED dataset) "
        "or User-provided (SNT_POPULATION_USER_PROVIDED dataset)."
    ),
    type=str,
    default="DHIS2",
    required=True,
    choices=["DHIS2", "User-provided"],
)
@parameter(
    "tot_pop_reference",
    name="Part 1: Population reference",
    help=(
        "Total population used to scale population data. When provided, "
        "population values are adjusted proportionally to match this total. "
        "(e.g. 1000000 for a total population of 1 million people)."
    ),
    type=int,
    default=None,
    required=False,
)
@parameter(
    "tot_pop_reference_year",
    name="Part 1: Population year reference",
    help=(
        "Year corresponding to the total population reference. "
        "This year must be available in the population data. "
        "Defaults to the latest year available in the population data. "
        "(e.g. 2025)."
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
    "pop_50_plus",
    name="Part 2: Proportion population 50 plus years",
    help=(
        "Proportion of the total population aged 50 years and above (e.g. 0.06 for 6%). "
        "Used to disaggregate population figures into the 50 plus years age group."
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
        "Annual growth rate (e.g. 0.03 for 3%) used to project population figures into past and future years."
    ),
    type=float,
    default=None,
    required=False,
)
@parameter(
    "growth_reference_year",
    name="Part 3: Projection reference year",
    help=(
        "Base year from which population figures are projected. "
        "This year must be available in the population data. "
        "Defaults to the latest year available."
    ),
    type=int,
    default=None,
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
def snt_dhis2_population_transformation(
    pop_source: str,
    tot_pop_reference: int,
    tot_pop_reference_year: int,
    pop_under_5: float,
    pop_pregnant_women: float,
    pop_0_1_y: float,
    pop_1_2_y: float,
    pop_5_10_y: float,
    pop_5_36_m: float,
    pop_50_plus: float,
    disaggregation_file: File,
    growth_factor: float,
    growth_reference_year: int,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Transform the selected population data and publish it to the population transformation dataset.

    Loads the population of the selected source (DHIS2 or user-provided), validates the inputs and
    runs the transformation notebook: optional scaling to a reference total, disaggregations from
    proportion parameters and/or a CSV file, and growth projections. The outputs are published to
    DHIS2_POPULATION_TRANSFORMATION and the reporting notebook is executed.
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
        country_code = snt_config_dict["SNT_CONFIG"]["COUNTRY_CODE"]
    except Exception as e:
        current_run.log_error(f"Failed to load configuration: {e}")
        raise

    if not run_report_only:
        try:
            population_data = get_population_from_source(snt_config_dict, pop_source)
        except Exception as e:
            msg = f"Population not available in {pop_source}: {e}."
            current_run.log_error(msg)
            raise FileNotFoundError(msg) from e

        if disaggregation_file and not Path(disaggregation_file.path).exists():
            current_run.log_error(f"Disaggregation file not found: {disaggregation_file.path}")
            raise FileNotFoundError(f"Disaggregation file not found: {disaggregation_file.path}")

        if disaggregation_file:
            try:
                validate_disaggregation_file(Path(disaggregation_file.path), country_code)
            except Exception as e:
                msg = f"Disaggregation file validation failed: {e}"
                current_run.log_error(msg)
                raise ValueError(msg) from e

        try:
            years_available = sorted(population_data["YEAR"].unique())
        except Exception as e:
            current_run.log_error(f"Failed to determine years available in population data: {e}")
            raise ValueError(f"Failed to determine years available in population data: {e}") from e
        if not years_available:
            current_run.log_error("Years available in population data are empty.")
            raise ValueError

        tot_pop_reference_year_res = None
        if tot_pop_reference:
            tot_pop_reference_year_res = resolve_reference_year(
                years_available, tot_pop_reference_year, var_name="Total population"
            )

        growth_reference_year_res = None
        if growth_factor:
            growth_reference_year_res = resolve_reference_year(
                years_available, growth_reference_year, var_name="Growth projection"
            )

        parameters = {
            "DATA_SOURCE": pop_source,
            "TOT_POP_REFERENCE": tot_pop_reference,
            "TOT_POP_REFERENCE_YEAR": tot_pop_reference_year_res,
            "GROWTH_FACTOR": growth_factor,
            "GROWTH_REFERENCE_YEAR": growth_reference_year_res,
            "POP_UNDER_5": pop_under_5,
            "POP_PREGNANT_WOMEN": pop_pregnant_women,
            "POP_0_1_Y": pop_0_1_y,
            "POP_1_2_Y": pop_1_2_y,
            "POP_5_10_Y": pop_5_10_y,
            "POP_5_36_M": pop_5_36_m,
            "POP_50_PLUS": pop_50_plus,
            "DISAGGREGATION_FILE": disaggregation_file.path if disaggregation_file else None,
        }

        params_file = save_pipeline_parameters(
            pipeline_name="snt_dhis2_population_transformation",
            parameters=parameters,
            output_path=snt_dhis2_pop_transform_path,
            country_code=country_code,
        )
        current_run.log_info(f"Saved pipeline parameters to {params_file}")

        expected_outputs = [
            snt_dhis2_pop_transform_path / f"{country_code}_population.parquet",
            snt_dhis2_pop_transform_path / f"{country_code}_population.csv",
        ]

        run_start_ts = time.time()
        try:
            # Apply transformation to population data
            dhis2_population_transformation(
                snt_root_path=snt_root_path,
                pipeline_root_path=snt_pipeline_path,
                snt_config=snt_config_dict,
                nb_parameter=parameters,
            )
        except Exception as e:
            current_run.log_error(f"Failed to apply population transformation: {e}")
            raise

        check_outputs_generated(file_paths=expected_outputs, run_start_ts=run_start_ts)

        try:
            add_files_to_dataset(
                dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get(
                    "DHIS2_POPULATION_TRANSFORMATION", None
                ),
                country_code=country_code,
                file_paths=[*expected_outputs, params_file],
            )
        except Exception as e:
            current_run.log_error(f"Failed to add files to dataset: {e}")
            raise

    try:
        run_report_notebook(
            nb_file=snt_pipeline_path / "reporting" / "snt_dhis2_population_transformation_report.ipynb",
            nb_output_path=snt_pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )
    except Exception as e:
        current_run.log_error(f"Failed to run reporting notebook: {e}")
        raise


def dhis2_population_transformation(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
    nb_parameter: dict,
) -> None:
    """Run the population transformation notebook on the selected population source.

    Args:
        snt_root_path: Root path of the SNT workspace files.
        pipeline_root_path: Path of the pipeline folder containing the code notebook.
        snt_config: Dictionary containing SNT configuration and dataset identifiers.
        nb_parameter: Parameters injected into the notebook (updated with SNT_ROOT_PATH).

    Raises:
        Exception: If the notebook execution fails.
    """
    current_run.log_info("Running population data transformations.")

    # set parameters for notebook
    nb_parameter.update({"SNT_ROOT_PATH": str(snt_root_path)})

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "snt_dhis2_population_transformation.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=snt_config["SNT_CONFIG"]["COUNTRY_CODE"],
        )
    except Exception as e:
        raise Exception(f"Error in executing population transformation notebook: {e}") from e


def get_population_from_source(snt_config_dict: dict, pop_source: str) -> pd.DataFrame:
    """Load the population table of the selected source from its dataset.

    Args:
        snt_config_dict: Dictionary containing SNT configuration and dataset identifiers.
        pop_source: The source of the population data ("DHIS2" or "User-provided").

    Returns:
        A pandas DataFrame containing the population data.

    Raises:
        ValueError: If the population file cannot be loaded from the dataset, or is empty.
    """
    country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
    if pop_source == "DHIS2":
        dataset_source = "DHIS2_DATASET_FORMATTED"
        filename = f"{country_code}_population.parquet"
    else:
        dataset_source = "SNT_POPULATION_USER_PROVIDED"
        filename = f"{country_code}_population_user.parquet"

    population_data = get_file_from_dataset(
        dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"][dataset_source],
        filename=filename,
    )

    if population_data.empty:
        raise ValueError("Population data is empty.")

    return population_data


def resolve_reference_year(
    years_available: list[int], reference_year: int | None, var_name: str = "Total population"
) -> int:
    """Resolve the reference year to use for population scaling or growth projections.

    Args:
        years_available: A list of years available in the population data.
        reference_year: The user-provided reference year.
        var_name: The name of the variable for which the reference year is being resolved (used for logging).

    Returns:
        The resolved reference year to use for population transformations.
    """
    latest_year = years_available[-1]
    if reference_year is None:
        current_run.log_warning(
            f"No {var_name} reference year provided. Defaulting to latest available year {latest_year}."
        )
        return latest_year

    if reference_year not in years_available:
        current_run.log_warning(
            f"{var_name} reference year {reference_year} not available in population data. "
            f"Defaulting to latest available year {latest_year}."
        )
        return latest_year

    return reference_year


def validate_disaggregation_file(file_path: Path, country_code: str) -> None:
    """Validate the header of the user-uploaded disaggregation CSV before running the notebook.

    Accepts comma- or semicolon-separated files (same rule as `load_csv_file()` in R).

    Args:
        file_path: Path to the disaggregation CSV file.
        country_code: Country code, used to name the expected template in error messages.

    Raises:
        ValueError: If ADM2_ID is missing, or YEAR / POPULATION columns are present.
    """
    with Path(file_path).open(encoding="utf-8-sig", errors="replace", newline="") as f:
        header_line = f.readline()
    sep = ";" if header_line.count(";") > header_line.count(",") else ","
    columns = {col.strip().strip('"').upper() for col in header_line.split(sep)}
    expected_template = f"{country_code}_population_disaggregation_template.csv"

    if "ADM2_ID" not in columns:
        current_run.log_error(
            f"Disaggregation file {Path(file_path).name} has no ADM2_ID column. "
            "Check the file is comma- or semicolon-separated and based on "
            f"{expected_template} (created by the DHIS2 formatting pipeline under uploads/)."
        )
        raise ValueError("Invalid disaggregation file: missing ADM2_ID column.")

    unexpected = sorted(columns & {"YEAR", "POPULATION"})
    if unexpected:
        current_run.log_error(
            f"Disaggregation file {Path(file_path).name} contains column(s) {', '.join(unexpected)}. "
            "This looks like the population user template; "
            f"please use {expected_template} instead."
        )
        raise ValueError(f"Invalid disaggregation file: unexpected column(s) {', '.join(unexpected)}.")


if __name__ == "__main__":
    snt_dhis2_population_transformation()
