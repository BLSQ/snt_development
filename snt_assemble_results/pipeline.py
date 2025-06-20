import json
from pathlib import Path

import pandas as pd
from openhexa.sdk import current_run, pipeline, workspace
from snt_pipeline_utils import (
    add_files_to_dataset,
    get_file_from_dataset,
    load_configuration_snt,
    validate_config,
)


@pipeline("snt_assemble_results")
def snt_assemble_results():
    """Assemble SNT results by loading configuration, validating it, and preparing paths for processing.

    Raises
    ------
    Exception
        If any error occurs during configuration loading or validation.
    """
    # paths
    root_path = Path(workspace.files_path)
    # pipeline_path = root_path / "pipelines" / "snt_assemble_results"
    results_path = root_path / "results"

    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        current_run.log_debug("config loaded")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")

        assemble_snt_results(snt_config=snt_config, output_path=results_path)

        build_metadata_table(output_path=results_path, country_code=country_code)

        add_files_to_dataset(
            dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_RESULTS"),
            country_code=country_code,
            file_paths=[
                results_path / f"{country_code}_results_dataset.csv",
                results_path / f"{country_code}_results_dataset.parquet",
                results_path / f"{country_code}_metadata.csv",
                results_path / f"{country_code}_metadata.parquet",
            ],
        )

    except Exception as e:
        current_run.log_error(f"Pipeline fail with error: {e}")
        raise


def assemble_snt_results(snt_config: dict, output_path: Path) -> None:
    """Assembles SNT results using the provided configuration dictionary."""
    results_table = build_results_table(snt_config)
    results_table = add_population_to(results_table, snt_config)
    results_table = add_incidence_to(results_table, snt_config)
    results_table = add_prevalence_to(results_table, snt_config)
    results_table = add_mortality_to(results_table, snt_config)
    results_table = add_map_access_to(results_table, snt_config)
    results_table = add_seasonality_to(results_table, snt_config)
    results_table = add_dhs_access_to(results_table, snt_config)

    # Save files
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    output_path.mkdir(parents=True, exist_ok=True)
    results_table.to_parquet(output_path / f"{country_code}_results_dataset.parquet", index=False)
    results_table.to_csv(output_path / f"{country_code}_results_dataset.csv", index=False)


def add_population_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add population data to the results table by merging with DHIS2 population information.

    Selection :
        matching: ADM2_ID
        values : ADM2_POP

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which population data will be added.
    snt_config : dict
        The SNT configuration dictionary containing dataset identifiers and country code.

    Returns
    -------
    pd.DataFrame
        The updated results table with population data merged.
    """
    current_run.log_info("Loading DHIS2 population data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
    try:
        dhis2_population = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_population.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"Error while loading population data: {e}")
        return table

    latest_period = dhis2_population["PERIOD"].max()
    update_metadata(variable="ADM2_POP", attribute="PERIOD", value=latest_period)

    dhis2_population_f = dhis2_population[
        (dhis2_population["ADM2_ID"].isin(table["ADM2_ID"])) & (dhis2_population["PERIOD"] == latest_period)
    ]

    merged_df = table.merge(dhis2_population_f[["ADM2_ID", "TOT_POPULATION"]], how="left", on="ADM2_ID")
    table["ADM2_POP"] = merged_df["TOT_POPULATION"]

    return table


def add_incidence_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    # INCIDENCE_CRUDE_MEDIAN, INCIDENCE_PRESUMED_MEDIAN, INCIDENCE_RR_MEDIAN, INCIDENCE_RR_TSR_MEDIAN
    return table


def add_prevalence_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    # PFPR_U5_MIS, PFPR_U5_MAP, PFPR_2TO10_MAP
    return table


def add_mortality_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    # U5_MORTALITY_MIS, U5_MORTALITY_IHME
    return table


def add_map_access_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    # ITN_ACCESS_MAP, ITN_USE_MAP, ITN_USE_RATE_MAP
    return table


def add_seasonality_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    # PROPORTION_SEASONAL_YEARS, PROPORTION_SEASONAL_RAINFALL_YEARS
    return table


def add_dhs_access_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    # ITN_ACCESS_SAMPLE_AVERAGE, ITN_ACCESS_CI_LOWER_BOUND, ITN_ACCESS_CI_UPPER_BOUND
    # DTP3_SAMPLE_AVERAGE, DTP3_CI_LOWER_BOUND, DTP3_CI_UPPER_BOUND
    return table


def build_results_table(snt_config: dict) -> pd.DataFrame:
    """Build and return a results table DataFrame with predefined column names and admin names.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with the specified columns for SNT results.
    """
    column_names = [
        "ADM1",
        "ADM1_ID",
        "ADM2",
        "ADM2_ID",
        "ADM2_POP",
        "INCIDENCE_CRUDE_MEDIAN",
        "INCIDENCE_PRESUMED_MEDIAN",
        "INCIDENCE_RR_MEDIAN",
        "INCIDENCE_RR_TSR_MEDIAN",
        "PFPR_U5_MIS",
        "PFPR_U5_MAP",
        "PFPR_2TO10_MAP",
        "U5_MORTALITY_MIS",
        "U5_MORTALITY_IHME",
        "MORBIDITY_RISK1",
        "MORBIDITY_MORTALITY_RISK2",
        "ITN_ACCESS_MAP",
        "ITN_USE_MAP",
        "ITN_USE_RATE_MAP",
        "PROPORTION_SEASONAL_YEARS",
        "PROPORTION_SEASONAL_RAINFALL_YEARS",
        "ITN_ACCESS_SAMPLE_AVERAGE",
        "ITN_ACCESS_CI_LOWER_BOUND",
        "ITN_ACCESS_CI_UPPER_BOUND",
        "DTP3_SAMPLE_AVERAGE",
        "DTP3_CI_LOWER_BOUND",
        "DTP3_CI_UPPER_BOUND",
    ]

    # Load names from pyramid
    current_run.log_debug("Loading DHIS2 pyramid")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
    dhis2_pyramid = get_file_from_dataset(
        dataset_id=dataset_id,
        filename=f"{country_code}_pyramid.parquet",
    )
    # Complete table with administrative names/ids
    results_table = pd.DataFrame(columns=column_names)
    admin1_name = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_1").upper()
    admin2_name = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_2").upper()
    admin1_id = admin1_name.replace("_NAME", "_ID")
    admin2_id = admin2_name.replace("_NAME", "_ID")
    results_table[["ADM1", "ADM1_ID", "ADM2", "ADM2_ID"]] = (
        dhis2_pyramid[[admin1_name, admin1_id, admin2_name, admin2_id]].drop_duplicates().to_numpy()
    )

    return results_table


def update_metadata(variable: str, attribute: str, value: any) -> None:
    """Update a specific attribute of a variable in the metadata JSON file.

    Parameters
    ----------
    variable : str
        The variable name in the metadata JSON to update.
    attribute : str
        The attribute of the variable to update.
    value : any
        The new value to set for the attribute.

    Raises
    ------
    KeyError
        If the variable or attribute does not exist in the metadata JSON.
    """
    # NOTE: hardcoded SNT path
    metadata_path = Path(workspace.files_path) / "configuration" / "metadata.json"

    try:
        with Path.open(metadata_path, "r") as file:
            metadata_json = json.load(file)
    except Exception as e:
        raise Exception(f"Error reading the metadata file {e}") from e

    if variable in metadata_json:
        if attribute in metadata_json.get(variable):
            metadata_json[variable][attribute] = value
        else:
            raise KeyError(f"Attribute '{attribute}' not found in variable '{variable}'.")
    else:
        raise KeyError(f"Variable '{variable}' not found in JSON file.")

    with Path.open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=4, ensure_ascii=True)

    current_run.log_info(f"SNT metadata variable {variable} updated: {metadata_path}")


def build_metadata_table(output_path: Path, country_code: str):
    """Build and save a metadata table from the metadata JSON file.

    Parameters
    ----------
    output_path : Path
        The directory where the metadata table files will be saved.
    country_code : str
        The country code used for naming the output files.

    Raises
    ------
    Exception
        If there is an error reading the metadata file.
    """
    # NOTE: hardcoded SNT path
    metadata_path = Path(workspace.files_path) / "configuration" / "metadata.json"

    try:
        with Path.open(metadata_path, "r") as file:
            metadata_json = json.load(file)
    except Exception as e:
        raise Exception(f"Error reading the metadata file {e}") from e

    # Convert to a pandas DataFrame
    metadata_table = pd.DataFrame.from_dict(metadata_json, orient="index")
    metadata_table = metadata_table.reset_index().rename(columns={"index": "VARIABLE"})
    # Save
    metadata_table.to_parquet(output_path / f"{country_code}_metadata.parquet", index=False)
    metadata_table.to_csv(output_path / f"{country_code}_metadata.csv", index=False)


if __name__ == "__main__":
    snt_assemble_results()
