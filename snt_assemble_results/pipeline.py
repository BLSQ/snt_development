import json
from pathlib import Path
from typing import Any
import re

import pandas as pd
from openhexa.sdk import current_run, parameter, pipeline, workspace, File
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    copy_file,
    get_file_from_dataset,
    load_configuration_snt,
    validate_config,
    get_matching_filename_from_dataset_last_version,
)


@pipeline("snt_assemble_results")
@parameter(
    "incidence_metric",
    name="Incidence aggregation across years.",
    type=str,
    multiple=False,
    choices=["mean", "median"],
    default="mean",
    required=True,
)
@parameter(
    "incidence_years_to_include",
    name="Incidence calculation period (years back)",
    help=(
        "Select how many years to go back in the incidence mean calculation. "
        "(0 = all available years (default), -1 = most recent year, "
        "-2 or less = that many years including the most recent year)."
    ),
    type=int,
    default=0,
    required=True,
)
@parameter(
    "reporting_rate_metric",
    name="Reporting rate  aggregation across years.",
    type=str,
    multiple=False,
    choices=["mean", "median"],
    default="mean",
    required=True,
)
@parameter(
    "map_selection",
    name="MAP indicators selection",
    type=str,
    multiple=True,
    choices=[
        "Pf_Parasite_Rate",
        "Pf_Mortality_Rate",
        "Pf_Incidence_Rate",
        "Insecticide_Treated_Net_Access",
        "Insecticide_Treated_Net_Use_Rate",
        "IRS_Coverage",
        "Antimalarial_Effective_Treatment",
    ],
    default=[
        "Pf_Parasite_Rate",
        "Pf_Mortality_Rate",
        "Pf_Incidence_Rate",
        "Insecticide_Treated_Net_Access",
        "Insecticide_Treated_Net_Use_Rate",
        "IRS_Coverage",
        "Antimalarial_Effective_Treatment",
    ],
    required=True,
)
@parameter(
    "adm1_layers_file",
    name="Additional ADM 1 layers (.csv)",
    type=File,
    required=False,
    default=None,
    help="Select user-uploaded files with additional layers at level ADM 1.",
)
@parameter(
    "adm2_layers_file",
    name="Additional ADM 2 layers (.csv)",
    type=File,
    required=False,
    default=None,
    help="Select user-uploaded files with additional layers at level ADM 2.",
)
def snt_assemble_results(
    incidence_metric: str,
    incidence_years_to_include: int,
    reporting_rate_metric: str,
    map_selection: list[str],
    adm1_layers_file: File,
    adm2_layers_file: File,
) -> None:
    """Assemble SNT results by loading configuration, validating it, and preparing paths for processing.

    Raises
    ------
    Exception
        If any error occurs during configuration loading, processing or validation.
    """
    # paths
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_assemble_results"
    results_path = root_path / "results"

    if incidence_years_to_include > 0:
        message = "Number of incidence years must be 0 or negative (0 = all available years)."
        current_run.log_error(f"Invalid number of incidence years: {incidence_years_to_include}. {message}")
        raise ValueError(message)

    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")

        # Get metadata file
        copy_file(
            source_folder=root_path / "configuration",
            destination_folder=pipeline_path / "data",
            filename="SNT_metadata.json",
        )

        assemble_snt_results(
            snt_config=snt_config,
            output_path=results_path,
            incidence_metric=incidence_metric,
            incidence_years_to_include=incidence_years_to_include,
            reporting_rate_metric=reporting_rate_metric,
            map_selection=map_selection,
            additional_layer_files={
                "ADM1": adm1_layers_file.path if adm1_layers_file else None,
                "ADM2": adm2_layers_file.path if adm2_layers_file else None,
            },
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
    snt_config: dict,
    output_path: Path,
    incidence_metric: str,
    incidence_years_to_include: int,
    reporting_rate_metric: str,
    map_selection: list[str],
    additional_layer_files: dict[str, Any],
) -> None:
    """Assembles SNT results using the provided configuration dictionary."""
    # initialize table
    results_table = build_results_table(snt_config)

    # add indicators
    results_table = add_dhis2_indicators_to(
        results_table, snt_config, incidence_metric, incidence_years_to_include, reporting_rate_metric
    )
    results_table = add_map_indicators_to(results_table, snt_config, map_selection)
    results_table = add_seasonality_indicators_to(results_table, snt_config)
    results_table = add_dhs_indicators_to(results_table, snt_config)

    # add user uploaded indicators
    results_table = add_user_uploaded_indicators_to(results_table, additional_layer_files)

    # Save files
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    output_path.mkdir(parents=True, exist_ok=True)
    results_table.to_parquet(output_path / f"{country_code}_results_dataset.parquet", index=False)
    results_table.to_csv(output_path / f"{country_code}_results_dataset.csv", index=False)


def add_dhis2_indicators_to(
    table: pd.DataFrame,
    snt_config: dict,
    incidence_metric: str,
    incidence_years_to_include: int,
    reporting_rate_metric: str,
) -> pd.DataFrame:
    """Add DHIS2 indicators to the results table by sequentially applying indicator functions.

    Returns
    -------
    pd.DataFrame
        The updated results table with DHIS2 indicators added.
    """
    updated_table = add_population_to(table, snt_config)
    updated_table = add_reporting_rate_to(table, snt_config, reporting_rate_metric)
    updated_table = add_incidence_indicators_to(
        updated_table, snt_config, incidence_metric, incidence_years_to_include
    )
    return updated_table  # noqa: RET504


def any_columns_present(table: pd.DataFrame, required_columns: list[str]) -> bool:
    """Check if the table contains any of the required columns.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to check.
    required_columns : list[str]
        List of required column names.

    Returns
    -------
    bool
        True if any of the required columns are present, False otherwise.
    """
    present_columns = [col for col in required_columns if col in table.columns]
    return bool(present_columns)


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
    if not any_columns_present(table=table, required_columns=["POPULATION"]):
        current_run.log_info("No population columns present in the assembly table, skipping.")
        return table
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")  # fall back
    dataset_transform_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_POPULATION_TRANSFORMATION")
    transform_reference_year = (
        snt_config.get("DHIS2_DATA_DEFINITIONS", {}).get("POPULATION_DEFINITIONS", {}).get("REFERENCE_YEAR")
    )

    try:
        dhis2_population = get_file_from_dataset(
            dataset_id=dataset_transform_id,
            filename=f"{country_code}_population.parquet",
        )
        selected_year = transform_reference_year or select_reference_year_from(dhis2_population["YEAR"])
        current_run.log_info(
            "Population data loaded from transformed dataset (DHIS2_POPULATION_TRANSFORMATION)."
        )
    except Exception as e:
        current_run.log_warning(
            "No transformed population file found, falling back to dhis2 formatted population."
        )
        current_run.log_debug(f"Error loading population from DHIS2_POPULATION_TRANSFORMATION.{e}")
        try:
            dhis2_population = get_file_from_dataset(
                dataset_id=dataset_id,
                filename=f"{country_code}_population.parquet",
            )
            selected_year = dhis2_population["YEAR"].max()
            current_run.log_info(
                "Population data loaded from DHIS2 formatted dataset (DHIS2_DATASET_FORMATTED)."
            )
        except Exception as e:
            current_run.log_warning("Formatted population file not found, population layer skipped.")
            current_run.log_debug(f"Error loading population from DHIS2_DATASET_FORMATTED.{e}")
            return table

    if selected_year is None:
        return table

    current_run.log_info(f"Selected population reference year: {selected_year}")
    table.update(
        table.merge(
            dhis2_population[dhis2_population["YEAR"] == selected_year][["ADM2_ID", "POPULATION"]],
            how="left",
            on="ADM2_ID",
            suffixes=("_old", ""),
        )["POPULATION"]
    )

    update_metadata(variable="POPULATION", attribute="PERIOD", value=str(int(float(selected_year))))

    return table


def select_reference_year_from(
    years_series: pd.Series, years_to_past: int = 6, years_to_future: int = 6
) -> int:
    """Select the reference year from a series of years.

    Parameters
    ----------
    years_series : pd.Series
        Series containing year values.
    years_to_past : int, optional
        Number of years to look back from the reference year, by default 6.
    years_to_future : int, optional
        Number of years to look forward from the reference year, by default 6.

    Returns
    -------
    int
        The selected reference year.
    """
    current_run.log_info(
        "Automatic selection of reference year from year "
        f"series (-{years_to_past} to +{years_to_future} around reference year)."
    )

    unique_years = sorted(years_series.unique())
    if len(unique_years) < (years_to_past + years_to_future + 1):
        current_run.log_warning("Not enough years in the series to select a proper reference year.")
        return None

    return int(unique_years[years_to_past])


def add_reporting_rate_to(table: pd.DataFrame, snt_config: dict, reporting_rate_metric: str) -> pd.DataFrame:
    """Add reporting data to the results table by merging with DHIS2 rates information.

    Selection :
        matching: ADM2_ID
        values : REPORTING_RATE

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which population data will be added.
    snt_config : dict
        The SNT configuration dictionary containing dataset identifiers and country code.
    reporting_rate_metric : str
        Reporting rate metric selection.

    Returns
    -------
    pd.DataFrame
        The updated results table with population data merged.
    """
    current_run.log_info("Loading DHIS2 Reporting rates data")
    if not any_columns_present(table=table, required_columns=["REPORTING_RATE"]):
        current_run.log_info("No reporting rate columns present in the assembly table, skipping.")
        return table
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_REPORTING_RATE")

    # Determine which RR method was used based on Incidence filename (dataset)
    reporting_method = get_reporting_method_from_incidence_filename(snt_config)
    if not reporting_method:
        current_run.log_warning(
            "No reporting method found in incidence filename. Reporting rate data not added."
        )
        return table
    current_run.log_debug(f"Using reporting method: {reporting_method}")

    try:
        dhis2_reporting = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_reporting_rate_{reporting_method}.parquet",
        )
    except Exception as e:
        current_run.log_warning("Error while loading reporting rate data, data not added.")
        current_run.log_debug(f"Error while loading reporting rate data: {e}")
        return table

    latest_period = dhis2_reporting["YEAR"].max()
    update_metadata(variable="REPORTING_RATE", attribute="PERIOD", value=str(int(float(latest_period))))

    # Aggregate reporting rates per ADM2_ID across all periods (months/years)
    if reporting_rate_metric == "mean":
        dhis2_reporting_agg = dhis2_reporting.groupby("ADM2_ID")["REPORTING_RATE"].mean().reset_index()
    elif reporting_rate_metric == "median":
        dhis2_reporting_agg = dhis2_reporting.groupby("ADM2_ID")["REPORTING_RATE"].median().reset_index()
    else:
        current_run.log_warning(
            f"Reporting rate metric {reporting_rate_metric} not recognized. Using 'mean' as default."
        )
        dhis2_reporting_agg = dhis2_reporting.groupby("ADM2_ID")["REPORTING_RATE"].mean().reset_index()

    dhis2_reporting_agg = dhis2_reporting_agg.rename(columns={"REPORTING_RATE": "AGG_REPORTING_RATE"})
    table_updated = table.merge(dhis2_reporting_agg, on="ADM2_ID", how="left")
    table_updated["REPORTING_RATE"] = table_updated["AGG_REPORTING_RATE"].mul(100).round(1)
    return table_updated.drop(columns=["AGG_REPORTING_RATE"])


def add_incidence_indicators_to(
    table: pd.DataFrame, snt_config: dict, incidence_metric: str, incidence_years_to_include: int
) -> pd.DataFrame:
    """Add incidence indicators to the results table using DHIS2 incidence data.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which incidence indicators will be added.
    snt_config : dict
        The SNT configuration dictionary containing dataset identifiers and country code.
    incidence_metric : str
        Incidence metric selection.
    incidence_years_to_include : int
        Number of years to include for incidence calculation.

    Returns
    -------
    pd.DataFrame
        The updated results table with incidence indicators added.
    """
    current_run.log_info("Loading incidence data")
    columns_selection = [
        "INCIDENCE_CRUDE",
        "INCIDENCE_ADJ_TESTING",
        "INCIDENCE_ADJ_REPORTING",
        "INCIDENCE_ADJ_CARESEEKING",
    ]
    if not any_columns_present(table=table, required_columns=columns_selection):
        current_run.log_info("No incidence columns present in the assembly table, skipping.")
        return table
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_INCIDENCE")

    try:
        f_name = get_matching_filename_from_dataset_last_version(
            dataset_id=dataset_id,
            filename_pattern=f"{country_code}_incidence_year_routine-data-*_rr-method-*.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"Error while retrieving incidence filename: {e}")
        return table

    try:
        dhis2_incidence = get_file_from_dataset(dataset_id=dataset_id, filename=f_name)
        current_run.log_debug(f"Incidence file selection: {f_name}")
    except Exception as e:
        current_run.log_warning("Error while loading incidence data, data not added.")
        current_run.log_debug(f"Error while loading incidence data: {e}")
        return table

    dhis2_incidence.columns = dhis2_incidence.columns.str.upper()  # This should be already formatted
    min_year = int(float(dhis2_incidence["YEAR"].min()))
    period_end = int(float(dhis2_incidence["YEAR"].max()))

    # Select periods
    if incidence_years_to_include == 0:
        period_start = min_year
    else:
        period_start = period_end + int(incidence_years_to_include) + 1
        if period_start < min_year:
            period_start = min_year

    matched_columns = [col for col in columns_selection if col in dhis2_incidence.columns]
    if not matched_columns:
        current_run.log_warning("No matching incidence columns found for aggregation.")
        return table

    missing_columns = [col for col in columns_selection if col not in dhis2_incidence.columns]
    if missing_columns:
        current_run.log_warning(f"Missing columns in incidence data: {missing_columns}")

    # select only the columns present in the assembly table
    table_match_cols = [m for m in matched_columns if m in table.columns]
    if table_match_cols == []:
        current_run.log_warning("No matching incidence columns found in assembly table, skipping.")
        return table

    # Filter by periods
    current_run.log_info(f"Incidence years included: from {period_start} to {period_end}")
    dhis2_incidence = dhis2_incidence[
        (dhis2_incidence["YEAR"] >= period_start) & (dhis2_incidence["YEAR"] <= period_end)
    ]

    # Compute incidence metric
    dhis2_incidence_agg = dhis2_incidence.groupby("ADM2_ID", as_index=False).agg(
        {col: incidence_metric for col in table_match_cols}
    )

    # merge
    merged = table.merge(
        dhis2_incidence_agg[["ADM2_ID"] + table_match_cols],
        how="left",
        on="ADM2_ID",
        suffixes=("", "_new"),
    )

    if period_start == period_end:
        period_str = str(period_end)
    else:
        period_str = f"{period_start}-{period_end}"

    # Update each column if it is available in dhis2_incidence
    for col in table_match_cols:
        table[col] = pd.to_numeric(merged[f"{col}_new"], errors="coerce").round(2)
        update_metadata(variable=col, attribute="PERIOD", value=period_str)

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
    if not any_columns_present(
        table=table,
        required_columns=[
            "PF_PR_RATE",
            "PF_MORTALITY_RATE",
            "PF_INCIDENCE_RATE",
            "ITN_ACCESS_RATE",
            "ITN_USE_RATE_RATE",
            "IRS_COVERAGE_RATE",
            "ANTIMALARIAL_EFT_RATE",
        ],
    ):
        current_run.log_info("No MAP columns present in the assembly table, skipping.")
        return table
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

    col_mappings = {
        "Pf_Parasite_Rate": "PF_PR_RATE",
        "Pf_Mortality_Rate": "PF_MORTALITY_RATE",
        "Pf_Incidence_Rate": "PF_INCIDENCE_RATE",
        "Insecticide_Treated_Net_Access": "ITN_ACCESS_RATE",
        "Insecticide_Treated_Net_Use_Rate": "ITN_USE_RATE_RATE",
        "IRS_Coverage": "IRS_COVERAGE_RATE",
        "Antimalarial_Effective_Treatment": "ANTIMALARIAL_EFT_RATE",
    }

    convertions = {
        "PF_PR_RATE": 100,
        "PF_MORTALITY_RATE": 100000,
        "PF_INCIDENCE_RATE": 1000,
        "ITN_ACCESS_RATE": 100,
        "ITN_USE_RATE_RATE": 100,
        "IRS_COVERAGE_RATE": 100,
        "ANTIMALARIAL_EFT_RATE": 100,
    }
    columns_mapped = []
    for metric in map_selection:
        if col_mappings[metric] not in table.columns:
            current_run.log_info(f"Metric {metric} not found in assembly table, skipping.")
            continue

        indicator_data = map_indicators[map_indicators["METRIC_NAME"] == metric].copy()
        indicator_data = indicator_data[indicator_data["STATISTIC"] == "mean"].copy()
        if indicator_data.empty:
            current_run.log_warning(f"No metric {metric} data found in MAP dataset, skipping.")
            continue

        try:
            latest_period = indicator_data["YEAR"].max()
            indicator_data = indicator_data[indicator_data["YEAR"] == latest_period].copy()
            update_metadata(variable=col_mappings[metric], attribute="PERIOD", value=str(latest_period))
            current_run.log_debug(f"Latest period for {metric.upper()}: {latest_period}")
            indicator_df = indicator_data[["ADM2_ID", "VALUE"]].copy()
            indicator_df = indicator_df.rename(columns={"VALUE": col_mappings[metric]})
            merged = table.merge(indicator_df, how="left", on="ADM2_ID", suffixes=("_old", ""))
            table.update(merged[[col_mappings[metric]]])
            columns_mapped.append(col_mappings[metric])
        except Exception as e:
            current_run.log_warning(f"Error while updating MAP data for metric {metric}: {e}")
            continue

    # convertions
    for col in columns_mapped:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
            table[col] = table[col] * convertions[col]
            table[col] = table[col].round(2)
        else:
            current_run.log_warning(f"The column {col} was not found in results while updating MAP data.")

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
    return updated_table  # noqa: RET504


def add_precipitation_seasonality(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add precipitation seasonality indicators to the results table using the provided configuration.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which rainfall seasonality indicators will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with rainfall seasonality indicators added.
    """
    current_run.log_info("Loading rainfall seasonality data")
    columns_selection = [
        "SEASONALITY_RAINFALL",
        "SEASONAL_BLOCK_DURATION_RAINFALL",
    ]
    if not any_columns_present(table=table, required_columns=columns_selection):
        current_run.log_info("No rainfall seasonality columns present in the assembly table, skipping.")
        return table
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_SEASONALITY_RAINFALL")
    try:
        seasonality_precipitation = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_rainfall_seasonality.parquet",
        )
    except Exception as e:
        current_run.log_warning("No rainfall seasonality data available.")
        current_run.log_debug(f"Error while loading rainfall seasonality data: {e}")
        return table

    # skip if no columns present in the assembly table
    columns_present = [col for col in columns_selection if col in table.columns]
    if columns_present == []:
        current_run.log_warning("No rainfall seasonality columns found in assembly table, skipping.")
        return table

    table.update(
        table.merge(
            seasonality_precipitation[["ADM2_ID"] + columns_present],
            how="left",
            on="ADM2_ID",
            suffixes=("_old", ""),
        )[["ADM2_ID"] + columns_present]
    )

    if "SEASONALITY_RAINFALL" in table.columns:
        table["SEASONALITY_RAINFALL"] = pd.to_numeric(table["SEASONALITY_RAINFALL"], errors="coerce")
        table["SEASONALITY_RAINFALL"] = table["SEASONALITY_RAINFALL"].replace(
            {0: "not-seasonal", 1: "seasonal"}
        )
    if "SEASONAL_BLOCK_DURATION_RAINFALL" in table.columns:
        table["SEASONAL_BLOCK_DURATION_RAINFALL"] = pd.to_numeric(
            table["SEASONAL_BLOCK_DURATION_RAINFALL"], errors="coerce"
        )
    current_run.log_info("Rainfall seasonality data loaded successfully.")
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
    columns_selection = ["SEASONALITY_CASES", "SEASONAL_BLOCK_DURATION_CASES"]
    if not any_columns_present(table=table, required_columns=columns_selection):
        current_run.log_info("No cases seasonality columns present in the assembly table, skipping.")
        return table
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_SEASONALITY_CASES")
    try:
        seasonality_cases = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_cases_seasonality.parquet",
        )
    except Exception as e:
        current_run.log_warning("No cases seasonality data available.")
        current_run.log_debug(f"Error while loading seasonality cases data: {e}")
        return table

    # skip if no columns present in the assembly table
    columns_present = [col for col in columns_selection if col in table.columns]
    if columns_present == []:
        current_run.log_warning("No cases seasonality columns found in assembly table, skipping.")
        return table

    table.update(
        table.merge(
            seasonality_cases[["ADM2_ID"] + columns_present],
            how="left",
            on="ADM2_ID",
            suffixes=("_old", ""),
        )[["ADM2_ID"] + columns_present]
    )

    if "SEASONALITY_CASES" in table.columns:
        table["SEASONALITY_CASES"] = pd.to_numeric(table["SEASONALITY_CASES"], errors="coerce")
        table["SEASONALITY_CASES"] = table["SEASONALITY_CASES"].replace({0: "not-seasonal", 1: "seasonal"})
    if "SEASONAL_BLOCK_DURATION_CASES" in table.columns:
        table["SEASONAL_BLOCK_DURATION_CASES"] = pd.to_numeric(
            table["SEASONAL_BLOCK_DURATION_CASES"], errors="coerce"
        )
    current_run.log_info("Cases seasonality data loaded successfully.")
    return table


def add_dhs_indicators_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add various DHS indicators to the results table using the provided configuration.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which DHS indicators will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with DHS indicators added.
    """
    updated_table = add_careseeking_to(table, snt_config)
    updated_table = add_dropout_dtp_to(updated_table, snt_config)
    updated_table = add_proportion_dpt_to(updated_table, snt_config)
    updated_table = add_under5_mortality_to(updated_table, snt_config)
    updated_table = add_under5_prevalence_to(updated_table, snt_config)
    updated_table = add_itn_access_sample_to(updated_table, snt_config)
    updated_table = add_itn_use_to(updated_table, snt_config)
    return updated_table  # noqa: RET504


def add_careseeking_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add care seeking data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which care seeking data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with care seeking data added.
    """
    current_run.log_info("Loading DHS care seeking data")
    columns_selection = [
        "PCT_PUBLIC_CARE",
        "PCT_PRIVATE_CARE",
        "PCT_NO_CARE",
    ]
    if not any_columns_present(table=table, required_columns=columns_selection):
        current_run.log_info("DHS care seeking columns not present in the assembly table, skipping.")
        return table
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    try:
        # NOTE : FIX this
        dhs_careseeking = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_DHS_ADM1_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet",
        )
        current_run.log_debug(f"Columns: {dhs_careseeking.columns}")
    except Exception as e:
        current_run.log_warning("Error while loading dhs careseeking data, data not added")
        current_run.log_debug(f"Error while loading dhs careseeking data: {e}")
        return table

    matched_columns = [col for col in columns_selection if col in dhs_careseeking.columns]
    if matched_columns == []:
        current_run.log_warning("No care seeking columns found in DHS dataset, skipping.")
        return table
    missing_columns = [col for col in columns_selection if col not in dhs_careseeking.columns]
    if missing_columns:
        current_run.log_warning(f"Missing columns in care seeking dataset: {missing_columns}")
    present_columns = [col for col in matched_columns if col in table.columns]
    if present_columns == []:
        current_run.log_info("No care seeking columns found in assembly table, skipping.")
        return table

    merged = table.merge(
        dhs_careseeking[["ADM1_ID"] + present_columns],  # NOTE: only ADM1_ID level
        how="left",
        on="ADM1_ID",
        suffixes=("", "_new"),
    )

    # Update each column if it is available in dhs_careseeking
    for col in present_columns:
        table[col] = pd.to_numeric(merged[f"{col}_new"], errors="coerce").round(2)
        # NOTE: Theres no period available in the file (!)
        # update_metadata(variable=col, attribute="PERIOD", value=str(latest_period))
        current_run.log_info(f"DHS care seeking data {col} updated.")  # log each column

    return table


def add_dropout_dtp_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add dropout DTP data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which dropout DTP data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with dropout DTP data added.
    """
    current_run.log_info("Loading DHS dropout dtp data")
    columns_selection = [
        "PCT_DROPOUT_DTP_1_2",
        "PCT_DROPOUT_DTP_2_3",
        "PCT_DROPOUT_DTP_1_3",
    ]
    if not any_columns_present(table=table, required_columns=columns_selection):
        current_run.log_info("DHS dropout columns not present in the assembly table, skipping.")
        return table
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    try:
        # NOTE : FIX this
        dhs_dropout = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=f"{country_code}_DHS_ADM1_PCT_DROPOUT_DTP.parquet",
        )
        current_run.log_debug(f"Columns: {dhs_dropout.columns}")
    except Exception as e:
        current_run.log_warning("Error while loading dropout Ddtp data. Data not added.")
        current_run.log_debug(f"Error while loading dropout Ddtp data: {e}")
        return table

    matched_columns = [col for col in columns_selection if col in dhs_dropout.columns]
    missing_columns = [col for col in columns_selection if col not in dhs_dropout.columns]
    if missing_columns:
        current_run.log_warning(f"Missing columns in dropout data: {missing_columns}")
    present_columns = [col for col in matched_columns if col in table.columns]
    if present_columns == []:
        current_run.log_warning("No dropout data columns found in assembly table, skipping.")
        return table

    merged = table.merge(
        dhs_dropout[["ADM1_ID"] + present_columns],  # NOTE: only ADM1_ID level
        how="left",
        on="ADM1_ID",
        suffixes=("", "_new"),
    )

    # Update each column if it is available in dhs_dropout
    for col in present_columns:
        table[col] = pd.to_numeric(merged[f"{col}_new"], errors="coerce").round(2)
        # NOTE: Theres no period available in the file (!)
        # update_metadata(variable=col, attribute="PERIOD", value=str(latest_period))
        current_run.log_info(f"DHS dropout dtp data {col} updated.")  # log each column

    return table


def add_proportion_dpt_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add proportion DTP indicators from DHS to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which proportion DTP indicators will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with proportion DTP indicators added.
    """
    updated_table = add_proportion_dtp_1_to(table, snt_config)
    updated_table = add_proportion_dtp_2_to(updated_table, snt_config)
    updated_table = add_proportion_dtp_3_to(updated_table, snt_config)
    return updated_table  # noqa: RET504


def add_proportion_dtp_1_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add proportion DTP1 data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which proportion DTP1 data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with proportion DTP1 data added.
    """
    current_run.log_info("Loading DHS proportion dtp 1 data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    if "PCT_DTP1_SAMPLE_AVERAGE" not in table.columns:
        current_run.log_info("DHS proportion dtp 1 column not present in assembly table, skipping.")
        return table
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_PCT_DTP1.parquet",
        column_id="ADM1_ID",
        column_data="PCT_DTP1_SAMPLE_AVERAGE",
        msg_text="DHS proportion dtp 1",
    )


def add_proportion_dtp_2_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add proportion DTP2 data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which proportion DTP2 data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with proportion DTP2 data added.
    """
    current_run.log_info("Loading DHS proportion dtp 2 data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    if "PCT_DTP2_SAMPLE_AVERAGE" not in table.columns:
        current_run.log_info("DHS proportion dtp 2 column not present in assembly table, skipping.")
        return table
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_PCT_DTP2.parquet",
        column_id="ADM1_ID",
        column_data="PCT_DTP2_SAMPLE_AVERAGE",
        msg_text="DHS proportion dtp 2",
    )


def add_proportion_dtp_3_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add proportion DTP3 data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which proportion DTP3 data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with proportion DTP3 data added.
    """
    current_run.log_info("Loading DHS proportion dtp 3 data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    if "PCT_DTP3_SAMPLE_AVERAGE" not in table.columns:
        current_run.log_info("DHS proportion dtp 3 column not present in assembly table, skipping.")
        return table
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_PCT_DTP3.parquet",
        column_id="ADM1_ID",
        column_data="PCT_DTP3_SAMPLE_AVERAGE",
        msg_text="DHS proportion dtp 3",
    )


def add_under5_mortality_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add under 5 mortality data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which under 5 mortality data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with under 5 mortality data added.
    """
    indicator_msg = "DHS under 5 mortality"
    current_run.log_info(f"Loading {indicator_msg} data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    if "U5MR_PERMIL_SAMPLE_AVERAGE" not in table.columns:
        current_run.log_info("DHS under 5 mortality column not present in assembly table, skipping.")
        return table
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_U5MR_PERMIL.parquet",
        column_id="ADM1_ID",
        column_data="U5MR_PERMIL_SAMPLE_AVERAGE",
        msg_text=indicator_msg,
    )


def add_under5_prevalence_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add under 5 prevalence data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which under 5 prevalence data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with under 5 prevalence data added.
    """
    indicator_msg = "DHS under 5 prevalence"
    current_run.log_info(f"Loading {indicator_msg} data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    if "PCT_U5_PREV_RDT_SAMPLE_AVERAGE" not in table.columns:
        current_run.log_info("DHS under 5 prevalence column not present in assembly table, skipping.")
        return table
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_PCT_U5_PREV_RDT.parquet",
        column_id="ADM1_ID",
        column_data="PCT_U5_PREV_RDT_SAMPLE_AVERAGE",
        msg_text=indicator_msg,
    )


def add_itn_access_sample_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add ITN access data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which ITN access data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with ITN access data added.
    """
    current_run.log_info("Loading DHS ITN access data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    if "PCT_ITN_ACCESS_SAMPLE_AVERAGE" not in table.columns:
        current_run.log_info("DHS ITN access column not present in assembly table, skipping.")
        return table
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_PCT_ITN_ACCESS.parquet",
        column_id="ADM1_ID",
        column_data="PCT_ITN_ACCESS_SAMPLE_AVERAGE",
        msg_text="DHS ITN access",
    )


def add_itn_use_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
    """Add ITN use data from DHS indicators to the results table.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which ITN use data will be added.
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    pd.DataFrame
        The updated results table with ITN use data added.
    """
    current_run.log_info("Loading DHS ITN use data")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS")
    if "PCT_ITN_USE_SAMPLE_AVERAGE" not in table.columns:
        current_run.log_info("DHS ITN use column not present in assembly table, skipping.")
        return table
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_PCT_ITN_USE.parquet",
        column_id="ADM1_ID",
        column_data="PCT_ITN_USE_SAMPLE_AVERAGE",
        msg_text="DHS ITN use",
    )


def add_user_uploaded_indicators_to(table: pd.DataFrame, additional_layer_files: dict) -> pd.DataFrame:
    """Add user-uploaded indicators to the results table using the provided configuration.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which user-uploaded indicators will be added.
    additional_layer_files : dict
        A dictionary containing user-uploaded layer files.

    Returns
    -------
    pd.DataFrame
        The updated results table with user-uploaded indicators added.
    """
    adm1_path = additional_layer_files.get("ADM1")
    if adm1_path:
        table = _update_table_from_file(
            table=table,
            file_path=Path(adm1_path),
            level="ADM1",
            id_col="ADM1_ID",
            cols_to_drop=["ADM1_NAME"],
            invalid_cols={"ADM2_ID", "ADM2_NAME"},
        )

    adm2_path = additional_layer_files.get("ADM2")
    if adm2_path:
        table = _update_table_from_file(
            table=table,
            file_path=Path(adm2_path),
            level="ADM2",
            id_col="ADM2_ID",
            cols_to_drop=["ADM1_ID", "ADM1_NAME", "ADM2_NAME"],  # avoid replacing values in these columns
            invalid_cols=set(),
        )

    return table


def _update_table_from_file(
    table: pd.DataFrame,
    file_path: Path,
    level: str,
    id_col: str,
    cols_to_drop: list[str],
    invalid_cols: set[str],
) -> pd.DataFrame:
    """Update the given table DataFrame by merging it with a user-uploaded file on a specified column.

    Parameters
    ----------
    table : pd.DataFrame
        The DataFrame to update.
    file_path : Path
        The path to the user-uploaded file.
    level : str
        The administrative level (e.g., "ADM1", "ADM2").
    id_col : str
        The column name to join on.
    cols_to_drop : str
        The names of columns to drop from the user-uploaded data.
    invalid_cols : set
        A set of invalid columns that should not be present in the user-uploaded data.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after merging and updating the specified column.
    """
    if not file_path.exists():
        current_run.log_info(f"User provided file at level {level} not found: {file_path}.")
        return table

    current_run.log_info(f"Adding {level} user uploaded data: {file_path.name}")
    user_data = pd.read_csv(file_path)
    user_data.columns = user_data.columns.str.upper()
    user_data = user_data.drop(columns=cols_to_drop, errors="ignore")
    present_invalid_cols = invalid_cols & set(user_data.columns)
    if present_invalid_cols:
        current_run.log_warning(
            f"{level} user uploaded file: {file_path.name} "
            f" contains {', '.join(present_invalid_cols)} columns. This file will be ignored."
        )
        return table

    table = table.set_index(id_col)
    user_data = user_data.set_index(id_col)
    for col in user_data.columns:
        if col in table.columns:
            # update only matching column
            table[col].update(user_data[col])
        else:
            current_run.log_warning(
                f"Column name: {col} from file: {file_path.name} does not "
                "match the names defined in metadata."
            )

    return table.reset_index()


def update_table_with(
    table_df: pd.DataFrame,
    dataset_id: str,
    filename: str,
    column_id: str,
    column_data: str,
    msg_text: str = "population",
    rounding: int = 2,
) -> pd.DataFrame:
    """Update the given table DataFrame by merging it with a dataset file on a specified column.

    Parameters
    ----------
    table_df : pd.DataFrame
        The DataFrame to update.
    dataset_id : str
        The dataset identifier to fetch the file from.
    filename : str
        The filename to load from the dataset.
    column_id : str
        The column name to join on.
    column_data : str
        The column name containing the data to update.
    msg_text : str, optional
        Message text for logging purposes (default is "population").
    rounding : int, optional
        Number of decimal places to round the updated column (default is 2).

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after merging and updating the specified column.
    """
    try:
        dataset_file = get_file_from_dataset(
            dataset_id=dataset_id,
            filename=filename,
        )
    except Exception as e:
        current_run.log_warning(f"Error while loading {msg_text} data, data not added")
        current_run.log_debug(f"Error {msg_text} details: {e}")
        return table_df

    dataset_file.columns = [col.upper() for col in dataset_file.columns]
    missing_columns = [col for col in [column_id, column_data] if col not in dataset_file.columns]
    if missing_columns:
        current_run.log_warning(f"Missing columns in {msg_text} file: {missing_columns}")
        return table_df

    merged = table_df.merge(
        dataset_file[[column_id, column_data]], how="left", on=column_id, suffixes=("", "_new")
    )
    # table_df[column_data] = merged[f"{column_data}_new"] ## if round:
    table_df[column_data] = pd.to_numeric(merged[f"{column_data}_new"], errors="coerce").round(rounding)
    current_run.log_info(f"{msg_text} column {column_data} updated.")

    return table_df


def build_results_table(snt_config: dict) -> pd.DataFrame:
    """Build and return a results table DataFrame with predefined column names and admin names.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with the specified columns for SNT results.
    """
    # Read the json metadata
    # NOTE: Hardcoded path
    metadata_json = read_json(
        Path(workspace.files_path) / "pipelines" / "snt_assemble_results" / "data" / "SNT_metadata.json"
    )

    # Load names from pyramid
    current_run.log_debug("Loading DHIS2 pyramid")
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
    dhis2_pyramid = get_file_from_dataset(
        dataset_id=dataset_id,
        filename=f"{country_code}_pyramid.parquet",
    )

    # Complete table with administrative names/ids
    column_names = [
        "ADM1_NAME",
        "ADM1_ID",
        "ADM2_NAME",
        "ADM2_ID",
    ] + list(metadata_json.keys())
    # TODO: define a schema for the results table using metadata_json file
    # table_schema = {
    #         "ADM1_NAME": pd.Series(dtype="string"),
    #         "ADM1_ID": pd.Series(dtype="string"),
    #         "ADM2_NAME": pd.Series(dtype="string"),
    #         "ADM2_ID": pd.Series(dtype="string"),
    #         **{key: pd.Series(dtype="float") for key in metadata_json.keys()}
    #     }
    results_table = pd.DataFrame(columns=column_names)
    # results_table = results_table.astype(table_schema)
    admin1_name = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_1").upper()
    admin2_name = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_2").upper()
    admin1_id = admin1_name.replace("_NAME", "_ID")
    admin2_id = admin2_name.replace("_NAME", "_ID")
    results_table[["ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID"]] = (
        dhis2_pyramid[[admin1_name, admin1_id, admin2_name, admin2_id]].drop_duplicates().to_numpy()
    )

    return results_table


def read_json(json_path: Path) -> json:
    """Read and return the contents of a JSON file.

    Parameters
    ----------
    json_path : Path
        The path to the JSON file.

    Returns
    -------
    dict
        The contents of the JSON file as a dictionary.

    Raises
    ------
    Exception
        If there is an error reading the metadata file.
    """
    try:
        with Path.open(json_path, "r") as file:
            return json.load(file)
    except Exception as e:
        raise Exception(f"Error reading the metadata file {e}") from e


def update_metadata(variable: str, attribute: str, value: Any, filename: str = "SNT_metadata.json") -> None:  # noqa: ANN401
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
    # NOTE: hardcoded SNT metadata path
    metadata_path = Path(workspace.files_path) / "pipelines" / "snt_assemble_results" / "data" / filename
    metadata_json = read_json(metadata_path)

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
    # NOTE: hardcoded SNT metadata path
    metadata_path = Path(workspace.files_path) / "pipelines" / "snt_assemble_results" / "data" / filename
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_json = read_json(metadata_path)
    # Ensure PERIOD values are strings
    for node in metadata_json.values():
        period = node.get("PERIOD")
        if period is not None:
            node["PERIOD"] = str(period)
    # Convert to a pandas DataFrame
    metadata_table = pd.DataFrame.from_dict(metadata_json, orient="index")
    metadata_table = metadata_table.reset_index().rename(columns={"index": "VARIABLE"})
    # Save
    metadata_table.to_parquet(output_path / f"{country_code}_metadata.parquet", index=False)
    metadata_table.to_csv(output_path / f"{country_code}_metadata.csv", index=False)


def get_reporting_method_from_incidence_filename(snt_config: dict) -> str:
    """Get the reporting method from the incidence filename in the incidence dataset.

    Parameters
    ----------
    snt_config : dict
        The SNT configuration dictionary.

    Returns
    -------
    str
        The reporting method extracted from the incidence filename.
    """
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    try:
        filename = get_matching_filename_from_dataset_last_version(
            dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_INCIDENCE"),
            filename_pattern=f"{country_code}_incidence_year_routine-*_rr-method-*.parquet",
        )
    except Exception:
        return None

    match = re.search(r"rr-method-([^.]+)\.parquet", filename)
    if match:
        current_run.log_debug(f"Reporting method found in incidence filename: {match.group(1)}")
        return match.group(1)
    return None


if __name__ == "__main__":
    snt_assemble_results()
