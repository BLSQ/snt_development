import json
from pathlib import Path

import pandas as pd
from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_pipeline_utils import (
    add_files_to_dataset,
    copy_json_file,
    get_file_from_dataset,
    load_configuration_snt,
    validate_config,
)


@pipeline("snt_assemble_results")
@parameter(
    "incidence_selection",
    name="Incidence selection",
    type=str,
    multiple=False,
    choices=["any", "conf", "dummy_snis"],
    default="any",
    required=True,
)
@parameter(
    "map_selection",
    name="MAP indicators selection",
    type=str,
    multiple=True,
    choices=[
        "Pf_mortality_rate",
        "Pf_incidence_rate",
        "Pv_incidence_rate",
        "Pf_PR_rate",
        "Pv_PR_rate",
        "ITN_access_rate",
        "IRS_coverage_rate",
        "ITN_use_rate_rate",
        "Antimalarial_EFT_rate",
    ],
    default=[
        "Pf_mortality_rate",
        "Pf_incidence_rate",
        "Pv_incidence_rate",
        "Pf_PR_rate",
        "Pv_PR_rate",
        "ITN_access_rate",
        "IRS_coverage_rate",
        "ITN_use_rate_rate",
        "Antimalarial_EFT_rate",
    ],
    required=True,
)
def snt_assemble_results(incidence_selection: str, map_selection: list[str]):
    """Assemble SNT results by loading configuration, validating it, and preparing paths for processing.

    Raises
    ------
    Exception
        If any error occurs during configuration loading or validation.
    """
    # paths
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_assemble_results"
    results_path = root_path / "results"

    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        current_run.log_debug("config loaded")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")

        # Get metadata file
        copy_json_file(
            source_folder=root_path / "configuration",
            destination_folder=pipeline_path / "data",
            json_filename="SNT_metadata.json",
        )

        assemble_snt_results(
            snt_config=snt_config,
            output_path=results_path,
            incidence_selection=incidence_selection,
            map_selection=map_selection,
        )

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
        current_run.log_info("Pipeline completed successfully.")

    except Exception as e:
        current_run.log_error(f"Pipeline fail with error: {e}")
        raise


def assemble_snt_results(
    snt_config: dict, output_path: Path, incidence_selection: str, map_selection: list[str]
) -> None:
    """Assembles SNT results using the provided configuration dictionary."""
    # initialize table
    results_table = build_results_table(snt_config)

    # Add indicators based on source
    results_table = add_dhis2_indicators_to(results_table, snt_config, incidence_selection)
    results_table = add_map_indicators_to(results_table, snt_config, map_selection)
    results_table = add_seasonality_indicators_to(results_table, snt_config)
    # ADD THE REST OF THE INDICATORS

    # Save files
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    output_path.mkdir(parents=True, exist_ok=True)
    results_table.to_parquet(output_path / f"{country_code}_results_dataset.parquet", index=False)
    results_table.to_csv(output_path / f"{country_code}_results_dataset.csv", index=False)


def add_dhis2_indicators_to(table: pd.DataFrame, snt_config: dict, incidence_selection: str) -> pd.DataFrame:
    """Add DHIS2 indicators to the results table by sequentially applying indicator functions.

    Returns
    -------
    pd.DataFrame
        The updated results table with DHIS2 indicators added.
    """
    updated_table = add_population_to(table, snt_config)
    updated_table = add_incidence_indicators_to(updated_table, snt_config, incidence_selection)
    return updated_table


def add_population_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add population data to the results table by merging with DHIS2 population information.

    Selection :
        matching: ADM2_ID
        values : POPULATION

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

    latest_period = dhis2_population["YEAR"].max()
    update_metadata(variable="POPULATION", attribute="PERIOD", value=str(latest_period))

    table.update(
        table.merge(
            dhis2_population[dhis2_population["YEAR"] == latest_period][["ADM2_ID", "POPULATION"]],
            how="left",
            on="ADM2_ID",
            suffixes=("_old", ""),
        )["POPULATION"]
    )
    return table


def add_incidence_indicators_to(
    table: pd.DataFrame, snt_config: dict, incidence_selection: str
) -> pd.DataFrame:
    """Add incidence indicators to the results table using DHIS2 incidence data.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which incidence indicators will be added.
    snt_config : dict
        The SNT configuration dictionary containing dataset identifiers and country code.
    incidence_selection : list[str]
        List of incidence selection criteria.

    Returns
    -------
    pd.DataFrame
        The updated results table with incidence indicators added.
    """
    current_run.log_info("Loading incidence data")
    current_run.log_debug(f"Incidence selection: {incidence_selection}")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_INCIDENCE")
    try:
        dhis2_incidence = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_incidence_year_rr-method-{incidence_selection}.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"Error while loading incidence data: {e}")
        return table

    columns_selection = [
        "ADM2_ID",
        "CRUDE_INCIDENCE",
        "INCIDENCE_ADJ_TESTING",
        "INCIDENCE_ADJ_REPORTING",
        "INCIDENCE_ADJ_CARESEEKING",
    ]

    latest_period = dhis2_incidence["YEAR"].max()
    for col in columns_selection[1:]:
        update_metadata(variable=col, attribute="PERIOD", value=str(latest_period))
    dhis2_incidence.columns = dhis2_incidence.columns.str.upper()  # REMOVE THIS should already be formatted
    current_run.log_debug(f"Incidence columns: {dhis2_incidence.columns}")

    table.update(
        table.merge(dhis2_incidence[columns_selection], how="left", on="ADM2_ID", suffixes=("_old", ""))[
            columns_selection[1:]
        ]
    )
    return table


def add_map_indicators_to(table: pd.DataFrame, snt_config: dict, map_selection: list[str]) -> pd.DataFrame:
    """Add MAP indicators to the results table using the provided selection and configuration.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which MAP indicators will be added.
    snt_config : dict
        The SNT configuration dictionary containing dataset identifiers and country code.
    map_selection : list[str]
        List of MAP indicator names to select and add.

    Returns
    -------
    pd.DataFrame
        The updated results table with MAP indicators added.
    """
    current_run.log_info("Loading MAP data")
    current_run.log_debug(f"map selection: {map_selection}")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_MAP_EXTRACT")
    try:
        map_indicators = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_map_data.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"Error while loading MAP data: {e}")
        return table

    for metric in map_selection:
        try:
            indicator_data = map_indicators[map_indicators["METRIC_NAME"] == metric]
            latest_period = indicator_data["YEAR"].max()

            update_metadata(variable=metric.upper(), attribute="PERIOD", value=str(latest_period))

            current_run.log_debug(f"Latest period for {metric.upper()}: {latest_period}")

            pivot_df = indicator_data.pivot_table(index="ADM2_ID", columns="YEAR", values="VALUE")
            if isinstance(pivot_df.columns[0], str):
                latest_period = str(latest_period)
            else:
                latest_period = int(latest_period)
            latest_df = pivot_df[[latest_period]].rename(columns={latest_period: metric.upper()})
            latest_df = latest_df.reset_index()
            merged = table.merge(latest_df, how="left", on="ADM2_ID", suffixes=("_old", ""))
            table.update(merged[[metric.upper()]])
        except Exception as e:
            current_run.log_warning(f"Error while updating MAP data for metric {metric}: {e}")
            continue

    return table


def add_seasonality_indicators_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add seasonality indicators (precipitation and cases) to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which seasonality indicators will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with seasonality indicators added.
    """
    updated_table = add_precipitation_seasonality(table, snt_config)
    updated_table = add_cases_seasonality(updated_table, snt_config)
    return updated_table


def add_precipitation_seasonality(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add precipitation seasonality indicators to the results table using the provided configuration.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which precipitation seasonality indicators will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with precipitation seasonality indicators added.
    """
    current_run.log_info("Loading precipitation seasonality data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_SEASONALITY")
    try:
        seasonality_precipitation = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_precipitation_seasonality.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"No precipitation seasonality data available. Warning details: {e}")
        return table

    columns_selection = [
        "ADM2_ID",
        "SEASONALITY_PRECIPITATION",
        "SEASONAL_BLOCK_DURATION_PRECIPITATION",
    ]
    table.update(
        table.merge(
            seasonality_precipitation[columns_selection],
            how="left",
            on="ADM2_ID",
            suffixes=("_old", ""),
        )[columns_selection[1:]]
    )
    current_run.log_info("Precipitation seasonality data loaded successfully.")
    return table


def add_cases_seasonality(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add cases seasonality indicators to the results table using the provided configuration.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which cases seasonality indicators will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with cases seasonality indicators added.
    """
    current_run.log_info("Loading cases seasonality data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_SEASONALITY")
    try:
        seasonality_cases = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_cases_seasonality.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"No cases seasonality data available. Warning details: {e}")
        return table

    table.update(
        table.merge(
            seasonality_cases[["ADM2_ID", "SEASONALITY_CASES"]],
            how="left",
            on="ADM2_ID",
            suffixes=("_old", ""),
        )["SEASONALITY_CASES"]
    )
    current_run.log_info("Cases seasonality data loaded successfully.")
    return table


def build_results_table(snt_config: dict) -> pd.DataFrame:
    """Build and return a results table DataFrame with predefined column names and admin names.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with the specified columns for SNT results.
    """
    # NOTE: Should we consider taking the column names from the metadata file?
    column_names = [
        "ADM1",
        "ADM1_ID",
        "ADM2",
        "ADM2_ID",
        "POPULATION",
        "CRUDE_INCIDENCE",
        "INCIDENCE_ADJ_TESTING",
        "INCIDENCE_ADJ_REPORTING",
        "INCIDENCE_ADJ_CARESEEKING",
        "PF_PR_RATE",
        "PF_MORTALITY_RATE",
        "PF_INCIDENCE_RATE",
        "PV_INCIDENCE_RATE",
        "PV_PR_RATE",
        "ITN_ACCESS_RATE",
        "ITN_USE_RATE_RATE",
        "IRS_COVERAGE_RATE",
        "ANTIMALARIAL_EFT_RATE",
        "SEASONALITY_PRECIPITATION",
        "SEASONAL_BLOCK_DURATION_PRECIPITATION",
        "SEASONALITY_CASES",
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


def update_metadata(variable: str, attribute: str, value: any, filename: str = "SNT_metadata.json") -> None:
    """Update a specific attribute of a variable in the metadata JSON file.

    Parameters
    ----------
    variable : str
        The variable name in the metadata JSON to update.
    attribute : str
        The attribute of the variable to update.
    value : any
        The new value to set for the attribute.
    filename: str
        Metadata filename

    Raises
    ------
    KeyError
        If the variable or attribute does not exist in the metadata JSON.
    """
    # NOTE: hardcoded SNT paths
    root_path = Path(workspace.files_path)
    metadata_path = root_path / "pipelines" / "snt_assemble_results" / "data" / filename

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

    current_run.log_info(f"SNT metadata variable {variable} updated.")


def build_metadata_table(output_path: Path, country_code: str, filename: str = "SNT_metadata.json"):
    """Build and save a metadata table from the metadata JSON file.

    Parameters
    ----------
    output_path : Path
        The directory where the metadata table files will be saved.
    country_code : str
        The country code used for naming the output files.
    filename : str
        Metadata filename

    Raises
    ------
    Exception
        If there is an error reading the metadata file.
    """
    # NOTE: hardcoded SNT path
    root_path = Path(workspace.files_path)
    metadata_path = root_path / "pipelines" / "snt_assemble_results" / "data" / filename
    output_path.mkdir(parents=True, exist_ok=True)

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
