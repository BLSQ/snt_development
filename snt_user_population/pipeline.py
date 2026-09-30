from openhexa.sdk import current_run, pipeline, parameter, File, workspace
from pathlib import Path

import numpy as np
import pandas as pd
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    get_file_from_dataset,
    save_pipeline_parameters,
    validate_config,
)

# Ticket:
# https://bluesquare.atlassian.net/browse/SNT25-684

# Columns of the population user template ({COUNTRY_CODE}_population_user_template.csv)
ADM_COLS = ["ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID"]
REQUIRED_COLS = ["YEAR", *ADM_COLS, "POPULATION"]
DISAGGREGATION_COLS = [
    "POP_UNDER_5",
    "POP_PREGNANT_WOMEN",
    "POP_0_1_Y",
    "POP_1_2_Y",
    "POP_5_10_Y",
    "POP_5_36_M",
    "POP_50_PLUS",
]
EXPECTED_COLS = [*REQUIRED_COLS, *DISAGGREGATION_COLS]


@pipeline("snt_user_population")
@parameter(
    "user_file",
    name="Upload user population file (.csv)",
    type=File,
    required=True,
    default=None,
    help="Select user-uploaded file based on population template in CSV format.",
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook.",
    type=bool,
    default=False,
    required=False,
)
def snt_user_population(user_file: File, run_report_only: bool):
    """Orchestrate the SNT user population pipeline."""
    # set paths
    snt_root_path = Path(workspace.files_path)
    snt_pipeline_path = snt_root_path / "pipelines" / "snt_user_population"
    snt_user_population_data_path = snt_root_path / "data" / "user_population"

    # create paths if they don't exist
    snt_pipeline_path.mkdir(parents=True, exist_ok=True)
    snt_user_population_data_path.mkdir(parents=True, exist_ok=True)

    current_run.log_info("Pulling pipeline scripts from repository.")
    pull_scripts_from_repository(
        pipeline_name="snt_user_population",
        report_scripts=["snt_user_population_report.ipynb"],
    )

    try:
        # Load configuration (needed for report and for main run)
        snt_config_dict = load_configuration_snt(
            config_path=snt_root_path / "configuration" / "SNT_config.json"
        )
        validate_config(snt_config_dict)
    except Exception as e:
        current_run.log_error(f"Error occurred loading configuration: {e}")
        raise

    try:
        country_code = snt_config_dict["SNT_CONFIG"]["COUNTRY_CODE"]
    except Exception as e:
        current_run.log_error(f"Error occurred extracting country code 'COUNTRY_CODE': {e}")
        raise

    try:
        dataset_pop_user_id = snt_config_dict["SNT_DATASET_IDENTIFIERS"]["SNT_POPULATION_USER_PROVIDED"]
    except Exception as e:
        current_run.log_error(
            f"Error occurred extracting dataset population user ID 'SNT_POPULATION_USER_PROVIDED': {e}"
        )
        raise

    if not run_report_only:
        # Load pyramid data from the dataset
        pyramid_data = get_file_from_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED", None),
            filename=f"{country_code}_pyramid.parquet",
        )

        if pyramid_data is None or pyramid_data.empty:
            current_run.log_error(
                f"{country_code}_pyramid.parquet not found in DHIS2_DATASET_FORMATTED, "
                "perhaps DHIS2 formatting pipeline has not yet been executed."
            )
            raise FileNotFoundError(f"{country_code}_pyramid.parquet not found.")

        if user_file is None:
            current_run.log_error(
                "No user population file selected. Please select a file based on the template."
            )
            raise ValueError("Missing user population file.")

        if not Path(user_file.path).exists():
            current_run.log_error(f"User population file not found: {user_file.path}")
            raise FileNotFoundError(user_file.path)

        pyramid_adm = get_pyramid_adm_units(pyramid_data, snt_config_dict, country_code)

        # Load and validate the file from the user
        user_population = read_user_csv(Path(user_file.path))
        user_population = validate_user_population_file(user_population, pyramid_adm, country_code)

        # save the user-provided population file to the designated path
        user_population_path = snt_user_population_data_path / f"{country_code}_population_user.parquet"
        user_population_csv_path = snt_user_population_data_path / f"{country_code}_population_user.csv"
        user_population.to_parquet(user_population_path, index=False)
        user_population.to_csv(user_population_csv_path, index=False)
        current_run.log_info(f"User population data saved under: {user_population_path}")

        try:
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_user_population",
                parameters={
                    "user_file": user_file.path,
                },
                output_path=snt_user_population_data_path,
                country_code=country_code,
            )
        except Exception as e:
            current_run.log_error(f"Failed to save pipeline parameters: {e}")
            raise

        try:
            add_files_to_dataset(
                dataset_id=dataset_pop_user_id,
                country_code=country_code,
                file_paths=[
                    user_population_path,
                    user_population_csv_path,
                    parameters_file,
                ],
            )
        except Exception as e:
            current_run.log_error(f"Failed to add files to dataset: {e}")
            raise

    try:
        run_report_notebook(
            nb_file=snt_pipeline_path / "reporting" / "snt_user_population_report.ipynb",
            nb_output_path=snt_pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )
    except Exception as e:
        current_run.log_error(f"Error in running report notebook: {e}")
        raise

    current_run.log_info("User population pipeline completed successfully.")


def get_pyramid_adm_units(pyramid_data: pd.DataFrame, snt_config: dict, country_code: str) -> pd.DataFrame:
    """Extract the unique ADM1/ADM2 units from the formatted pyramid, with standard column names.

    The pyramid columns are taken from DHIS2_ADMINISTRATION_1 / DHIS2_ADMINISTRATION_2 in the
    config (uppercased, e.g. LEVEL_2_NAME); the ID column is the name column with _NAME -> _ID.

    Args:
        pyramid_data: Formatted pyramid ({COUNTRY_CODE}_pyramid.parquet).
        snt_config: SNT configuration dictionary.
        country_code: Country code, used in the error message.

    Returns:
        Unique rows with columns ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID.

    Raises:
        KeyError: If the configured admin columns are not in the pyramid.
    """
    adm_1_name = snt_config["SNT_CONFIG"]["DHIS2_ADMINISTRATION_1"].upper()
    adm_2_name = snt_config["SNT_CONFIG"]["DHIS2_ADMINISTRATION_2"].upper()
    pyramid_cols = [
        adm_1_name,
        adm_1_name.replace("_NAME", "_ID"),
        adm_2_name,
        adm_2_name.replace("_NAME", "_ID"),
    ]
    missing_pyramid_cols = [col for col in pyramid_cols if col not in pyramid_data.columns]
    if missing_pyramid_cols:
        current_run.log_error(
            f"Column(s) {', '.join(missing_pyramid_cols)} not found in {country_code}_pyramid.parquet. "
            "Check DHIS2_ADMINISTRATION_1 / DHIS2_ADMINISTRATION_2 in SNT_config.json."
        )
        raise KeyError(f"Missing pyramid column(s): {', '.join(missing_pyramid_cols)}")
    return pyramid_data[pyramid_cols].drop_duplicates().set_axis(ADM_COLS, axis=1)


def read_user_csv(file_path: Path) -> pd.DataFrame:
    """Read the user CSV, accepting comma- or semicolon-separated files (same rule as R `load_csv_file()`).

    Semicolon-separated files are read with a decimal comma. UTF-8 (with or without BOM) is tried
    first, then Latin-1 as fallback for files saved with a Windows encoding.

    Args:
        file_path: Path to the user-uploaded CSV file.

    Returns:
        The file contents as a DataFrame, with all values read as strings.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with file_path.open(encoding=encoding, newline="") as f:
                header_line = f.readline()
            sep, decimal = (";", ",") if header_line.count(";") > header_line.count(",") else (",", ".")
            data = pd.read_csv(file_path, sep=sep, dtype=str, encoding=encoding)
            break
        except UnicodeDecodeError:
            current_run.log_info(
                f"Failed to read {file_path} with encoding {encoding}. Trying next encoding."
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            current_run.log_error(f"Could not read user population file {file_path.name}: {e}")
            raise ValueError(f"Invalid user population file: {file_path.name}") from e
    else:
        current_run.log_error(f"Could not decode {file_path.name} with any supported encoding.")
        raise ValueError(f"Failed to read {file_path} with all attempted encodings.")

    data.columns = [col.strip().upper() for col in data.columns]
    current_run.log_info(f"User file loaded: {file_path.name} ({data.shape[0]} rows, separator '{sep}').")
    data.attrs["decimal"] = decimal
    return data


def to_numeric(values: pd.Series, decimal: str) -> pd.Series:
    """Convert a string column to numeric, returning NaN for empty or invalid values.

    Args:
        values: Column read as strings.
        decimal: Decimal mark used in the file ("." or ",").

    Returns:
        The column as float values.
    """
    cleaned = values.str.strip()
    if decimal == ",":
        cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def validate_user_population_file(
    user_population: pd.DataFrame, pyramid_adm: pd.DataFrame, country_code: str
) -> pd.DataFrame:
    """Validate the user population file against the template structure and the pyramid.

    Stops the run on structural problems (missing columns, no usable rows). Data issues
    (missing/invalid values, unknown or missing org units, name mismatches, duplicates) are
    logged as warnings and the file is kept as provided.

    Args:
        user_population: User file as read by `read_user_csv()` (string columns).
        pyramid_adm: Unique ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID rows from the formatted pyramid.
        country_code: Country code, used to name the expected template in error messages.

    Returns:
        The user population table restricted to the template columns, with YEAR and POPULATION
        as integer (POPULATION rounded up) and the disaggregation columns as numeric.
    """
    check_required_columns(user_population, country_code)
    user_population = select_expected_columns(user_population)

    # raw: stripped strings as typed by the user (for messages); data: parsed values
    raw = user_population.fillna("").apply(lambda col: col.str.strip())
    value_cols = get_value_columns(raw)
    data = parse_values(raw, value_cols, user_population.attrs.get("decimal", "."))

    check_year_and_population_filled(data)
    check_year_values(data, raw)
    check_value_columns(data, raw, value_cols)
    check_org_units_in_pyramid(data, pyramid_adm)
    check_names_against_pyramid(data, pyramid_adm)
    check_duplicates(data)
    data = round_up_population(data)

    current_run.log_info(
        f"User population file validated: {len(data)} rows, years: "
        f"{', '.join(str(y) for y in sorted(data['YEAR'].dropna().unique()))}."
    )
    return data


def select_expected_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the template columns (EXPECTED_COLS), in template order, and warn about any others.

    Disaggregation columns are optional: those absent from the file are not added.

    Args:
        data: User population table.

    Returns:
        A copy of the table restricted to the expected columns present in the file.
    """
    extra_cols = [col for col in data.columns if col not in EXPECTED_COLS]
    if extra_cols:
        current_run.log_warning(
            f"User population file has {len(extra_cols)} unexpected column(s), not published: "
            f"{', '.join(extra_cols)}."
        )
    selected = data[[col for col in EXPECTED_COLS if col in data.columns]].copy()
    selected.attrs = data.attrs  # keep the decimal mark set by read_user_csv()
    return selected


def get_value_columns(data: pd.DataFrame) -> list[str]:
    """List the numeric value columns: POPULATION and the POP_* disaggregations.

    Args:
        data: User population table.

    Returns:
        Names of the value columns, in file order.
    """
    return [col for col in data.columns if col == "POPULATION" or col.startswith("POP_")]


def parse_values(raw: pd.DataFrame, value_cols: list[str], decimal: str) -> pd.DataFrame:
    """Convert YEAR to integer and the value columns to numbers (missing when empty or invalid).

    Args:
        raw: User population table with stripped string values.
        value_cols: Value columns to convert (see `get_value_columns()`).
        decimal: Decimal mark used in the file ("." or ",").

    Returns:
        A copy of the table with YEAR as Int64 (non-whole numbers left empty), value columns as
        float, and empty text cells as NaN.
    """
    data = raw.mask(raw.eq(""))
    year = to_numeric(raw["YEAR"], decimal)
    data["YEAR"] = year.where(year % 1 == 0).astype("Int64")
    for col in value_cols:
        data[col] = to_numeric(raw[col], decimal)
    return data


def check_required_columns(data: pd.DataFrame, country_code: str) -> None:
    """Stop the run if a column of the user template is missing.

    Args:
        data: User population table.
        country_code: Country code, used to name the expected template in the error message.

    Raises:
        ValueError: If any of REQUIRED_COLS is missing.
    """
    missing_cols = [col for col in REQUIRED_COLS if col not in data.columns]
    if missing_cols:
        current_run.log_error(
            f"User population file is missing column(s): {', '.join(missing_cols)}. "
            f"Expected at least: {', '.join(REQUIRED_COLS)} "
            f"(see {country_code}_population_user_template.csv)."
        )
        raise ValueError(f"Invalid user population file: missing column(s) {', '.join(missing_cols)}.")


def check_year_and_population_filled(data: pd.DataFrame) -> None:
    """Stop the run if no row has both YEAR and POPULATION (e.g. an unfilled template).

    Args:
        data: Parsed user population table.

    Raises:
        ValueError: If YEAR and POPULATION are never both filled in.
    """
    if not data[["YEAR", "POPULATION"]].notna().all(axis=1).any():
        current_run.log_error("User population file has no row with both YEAR and POPULATION filled in.")
        raise ValueError("Invalid user population file: YEAR and POPULATION are empty.")


def check_year_values(data: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Warn about rows with a missing or non-whole-number YEAR.

    Args:
        data: Parsed user population table.
        raw: User population table with stripped string values.
    """
    invalid_year = data["YEAR"].isna()  # empty, non-numeric or not a whole number (see parse_values)
    if invalid_year.any():
        examples = raw["YEAR"][invalid_year].replace("", "<empty>").unique()[:5]
        current_run.log_warning(
            f"{invalid_year.sum()} row(s) with missing or invalid YEAR (e.g. {', '.join(examples)})."
        )


def check_value_columns(data: pd.DataFrame, raw: pd.DataFrame, value_cols: list[str]) -> None:
    """Warn about non-numeric or negative values, and missing POPULATION.

    Empty cells are allowed in the disaggregation columns; for POPULATION they are reported.

    Args:
        data: Parsed user population table.
        raw: User population table with stripped string values.
        value_cols: Value columns to check (see `get_value_columns()`).
    """
    for col in value_cols:
        invalid = data[col].isna() & raw[col].ne("")
        if invalid.any():
            examples = raw[col][invalid].unique()[:5]
            current_run.log_warning(
                f"{invalid.sum()} invalid (non-numeric) value(s) in {col} (e.g. {', '.join(examples)})."
            )
        negative = data[col] < 0
        if negative.any():
            current_run.log_warning(f"{negative.sum()} negative value(s) in {col}.")

    missing_pop = raw["POPULATION"].eq("").sum()
    if missing_pop:
        current_run.log_warning(f"{missing_pop} row(s) with missing POPULATION.")


def check_org_units_in_pyramid(data: pd.DataFrame, pyramid_adm: pd.DataFrame) -> None:
    """Warn about ADM2_IDs unknown to the pyramid, and pyramid ADM2 units missing for a YEAR.

    Coverage is checked per YEAR, so a unit filled in for one year but not another is reported.
    Rows with a missing or invalid YEAR are left out of the coverage check (reported separately).

    Args:
        data: Parsed user population table.
        pyramid_adm: Unique ADM rows from the formatted pyramid.
    """
    pyramid_ids = set(pyramid_adm["ADM2_ID"])

    unknown_ids = sorted(set(data["ADM2_ID"].dropna()) - pyramid_ids)
    if unknown_ids:
        current_run.log_warning(
            f"{len(unknown_ids)} ADM2_ID(s) not found in the pyramid: {', '.join(unknown_ids[:10])}"
            f"{' ...' if len(unknown_ids) > 10 else ''}"
        )

    for year, year_data in data.dropna(subset=["YEAR"]).groupby("YEAR"):
        missing_ids = pyramid_ids - set(year_data["ADM2_ID"].dropna())
        if missing_ids:
            current_run.log_warning(
                f"{len(missing_ids)} ADM2 unit(s) from the pyramid are missing for YEAR {year}."
            )


def check_names_against_pyramid(data: pd.DataFrame, pyramid_adm: pd.DataFrame) -> None:
    """Warn when ADM1 name/ID or ADM2 name differ from the pyramid for the same ADM2_ID.

    Args:
        data: User population table.
        pyramid_adm: Unique ADM rows from the formatted pyramid.
    """
    merged = (
        data[ADM_COLS]
        .drop_duplicates()
        .merge(pyramid_adm, on="ADM2_ID", how="inner", suffixes=("", "_PYRAMID"))
    )
    merged = merged.fillna("")  # so a value blank on both sides is not a mismatch
    for col in ["ADM1_NAME", "ADM1_ID", "ADM2_NAME"]:
        mismatch = merged[merged[col] != merged[f"{col}_PYRAMID"]]
        if not mismatch.empty:
            examples = [
                f"{r[col] or '<empty>'} (pyramid: {r[f'{col}_PYRAMID'] or '<empty>'})"
                for _, r in mismatch.head(5).iterrows()
            ]
            current_run.log_warning(
                f"{len(mismatch)} ADM2 unit(s) with {col} different from the pyramid, "
                f"e.g. {'; '.join(examples)}."
            )


def round_up_population(data: pd.DataFrame) -> pd.DataFrame:
    """Convert POPULATION to integer, rounding decimal values up (ceiling).

    Args:
        data: Parsed user population table.

    Returns:
        The table with POPULATION as Int64 (missing values kept empty).
    """
    has_decimals = data["POPULATION"].notna() & (data["POPULATION"] % 1 != 0)
    if has_decimals.any():
        current_run.log_info(f"{has_decimals.sum()} POPULATION value(s) with decimals rounded up to integer.")
    data["POPULATION"] = np.ceil(data["POPULATION"]).astype("Int64")
    return data


def check_duplicates(data: pd.DataFrame) -> None:
    """Warn about rows sharing the same YEAR and ADM2_ID.

    Args:
        data: Parsed user population table.
    """
    duplicated = data.duplicated(subset=["YEAR", "ADM2_ID"], keep=False) & data["YEAR"].notna()
    if duplicated.any():
        current_run.log_warning(f"{duplicated.sum()} row(s) share the same YEAR and ADM2_ID.")


if __name__ == "__main__":
    snt_user_population()
