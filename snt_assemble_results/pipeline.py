import json
from pathlib import Path
from typing import Any
import re

import pandas as pd
from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    copy_json_file,
    get_file_from_dataset,
    load_configuration_snt,
    validate_config,
    get_matching_filename_from_dataset_last_version,
)


@pipeline("snt_assemble_results")
@parameter(
    "incidence_metric",
    name="Select the metric to aggregate incidence data across years.",
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
def snt_assemble_results(incidence_metric: str, map_selection: list[str]):
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
            incidence_metric=incidence_metric,
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
    snt_config: dict,
    output_path: Path,
    incidence_metric: str,
    map_selection: list[str],
) -> None:
    """Assembles SNT results using the provided configuration dictionary."""
    # initialize table
    results_table = build_results_table(snt_config)

    # Add indicators based on source
    # DHIS2 indicators:
    #   -POPULATION
    #   -REPORTING_RATE
    #   -INCIDENCE_CRUDE
    results_table = add_dhis2_indicators_to(results_table, snt_config, incidence_metric)
    results_table = add_map_indicators_to(results_table, snt_config, map_selection)
    results_table = add_seasonality_indicators_to(results_table, snt_config)
    results_table = add_dhs_indicators_to(results_table, snt_config)

    # Save files
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    output_path.mkdir(parents=True, exist_ok=True)
    results_table.to_parquet(output_path / f"{country_code}_results_dataset.parquet", index=False)
    results_table.to_csv(output_path / f"{country_code}_results_dataset.csv", index=False)


def add_dhis2_indicators_to(table: pd.DataFrame, snt_config: dict, incidence_metric: str) -> pd.DataFrame:
    """Add DHIS2 indicators to the results table by sequentially applying indicator functions.

    Returns
    -------
    pd.DataFrame
        The updated results table with DHIS2 indicators added.
    """
    updated_table = add_population_to(table, snt_config)
    updated_table = add_reporting_rate_to(table, snt_config)
    updated_table = add_incidence_indicators_to(updated_table, snt_config, incidence_metric)
    return updated_table  # noqa: RET504


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
    table.update(
        table.merge(
            dhis2_population[dhis2_population["YEAR"] == latest_period][["ADM2_ID", "POPULATION"]],
            how="left",
            on="ADM2_ID",
            suffixes=("_old", ""),
        )["POPULATION"]
    )

    update_metadata(variable="POPULATION", attribute="PERIOD", value=str(int(float(latest_period))))

    return table


def add_reporting_rate_to(table: pd.DataFrame, snt_config: dict) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        The updated results table with population data merged.
    """
    current_run.log_info("Loading DHIS2 population data")
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
            filename=f"{country_code}_reporting_rate_{reporting_method}_month.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"Error while loading population data: {e}")
        return table

    latest_period = dhis2_reporting["YEAR"].max()
    update_metadata(variable="REPORTING_RATE", attribute="PERIOD", value=str(int(float(latest_period))))
    # Average of reporting dates per ADM2_ID across all months and years
    dhis2_reporting_agg = dhis2_reporting.groupby("ADM2_ID")["REPORTING_RATE"].mean().reset_index()
    dhis2_reporting_agg = dhis2_reporting_agg.rename(columns={"REPORTING_RATE": "AVG_REPORTING_RATE"})

    table_updated = table.merge(dhis2_reporting_agg, on="ADM2_ID", how="left")
    table_updated["REPORTING_RATE"] = table_updated["AVG_REPORTING_RATE"].round(2)
    return table_updated.drop(columns=["AVG_REPORTING_RATE"])


def add_incidence_indicators_to(table: pd.DataFrame, snt_config: dict, incidence_metric: str) -> pd.DataFrame:
    """Add incidence indicators to the results table using DHIS2 incidence data.

    Parameters
    ----------
    table : pd.DataFrame
        The results table to which incidence indicators will be added.
    snt_config : dict
        The SNT configuration dictionary containing dataset identifiers and country code.
    incidence_metric : str
        Incidence metric selection.

    Returns
    -------
    pd.DataFrame
        The updated results table with incidence indicators added.
    """
    current_run.log_info("Loading incidence data")

    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_INCIDENCE")

    try:
        f_name = get_matching_filename_from_dataset_last_version(
            dataset_id=dataset_id,
            filename_pattern=f"{country_code}_incidence_year_routine-data-*_rr-method-*.parquet",
        )
    except Exception as e:
        current_run.log_warning(f"Error while getting incidence filename: {e}")
        return table

    try:
        dhis2_incidence = get_file_from_dataset(dataset_id=dataset_id, filename=f_name)
        current_run.log_debug(f"Incidence file selection: {f_name}")
    except Exception as e:
        current_run.log_warning(f"Error while loading incidence data: {e}")
        current_run.log_warning(
            f"Please make sure the incidence file {f_name} exists or re-run the pipeline."
        )
        return table

    columns_selection = [
        "INCIDENCE_CRUDE",
        "INCIDENCE_ADJ_TESTING",
        "INCIDENCE_ADJ_REPORTING",
        "INCIDENCE_ADJ_CARESEEKING",
    ]

    period_start = int(float(dhis2_incidence["YEAR"].min()))
    period_end = int(float(dhis2_incidence["YEAR"].max()))
    dhis2_incidence.columns = dhis2_incidence.columns.str.upper()  # This should be already formatted
    matched_columns = [col for col in columns_selection if col in dhis2_incidence.columns]
    current_run.log_debug(f"Found incidence cols: {matched_columns}")
    if not matched_columns:
        current_run.log_warning("No matching incidence columns found for aggregation.")
        return table

    missing_columns = [col for col in columns_selection if col not in dhis2_incidence.columns]
    if missing_columns:
        current_run.log_warning(f"Missing columns in incidence data: {missing_columns}")

    # Compute incidence metric
    dhis2_incidence_agg = dhis2_incidence.groupby("ADM2_ID", as_index=False).agg(
        {col: incidence_metric for col in matched_columns}
    )

    # merge
    merged = table.merge(
        dhis2_incidence_agg[["ADM2_ID"] + matched_columns],
        how="left",
        on="ADM2_ID",
        suffixes=("", "_new"),
    )

    # Update each column if it is available in dhis2_incidence
    for col in matched_columns:
        table[col] = pd.to_numeric(merged[f"{col}_new"], errors="coerce").round(2)
        update_metadata(variable=col, attribute="PERIOD", value=f"{period_start}-{period_end}")

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
    for metric in map_selection:
        try:
            indicator_data = map_indicators[map_indicators["METRIC_NAME"] == metric]
            latest_period = indicator_data["YEAR"].max()

            update_metadata(variable=col_mappings[metric], attribute="PERIOD", value=str(latest_period))

            current_run.log_debug(f"Latest period for {metric.upper()}: {latest_period}")

            pivot_df = indicator_data.pivot_table(index="ADM2_ID", columns="YEAR", values="VALUE")
            if isinstance(pivot_df.columns[0], str):
                latest_period = str(latest_period)
            else:
                latest_period = int(latest_period)
            latest_df = pivot_df[[latest_period]].rename(columns={latest_period: col_mappings[metric]})
            latest_df = latest_df.reset_index()
            merged = table.merge(latest_df, how="left", on="ADM2_ID", suffixes=("_old", ""))
            table.update(merged[[col_mappings[metric]]])
        except Exception as e:
            current_run.log_warning(f"Error while updating MAP data for metric {metric}: {e}")
            continue

    # convertions
    for col in col_mappings.values():
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

    table["SEASONALITY_PRECIPITATION"] = pd.to_numeric(table["SEASONALITY_PRECIPITATION"], errors="coerce")
    table["SEASONALITY_PRECIPITATION"] = table["SEASONALITY_PRECIPITATION"].replace(
        {0: "not-seasonal", 1: "seasonal"}
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

    table["SEASONALITY_PRECIPITATION"] = pd.to_numeric(table["SEASONALITY_PRECIPITATION"], errors="coerce")
    table["SEASONALITY_CASES"] = table["SEASONALITY_CASES"].replace({0: "not-seasonal", 1: "seasonal"})
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
        current_run.log_warning(f"Error while loading dhs careseeking data: {e}")
        return table

    columns_selection = [
        "PCT_PUBLIC_CARE",
        "PCT_PRIVATE_CARE",
        "PCT_NO_CARE",
    ]

    matched_columns = [col for col in columns_selection if col in dhs_careseeking.columns]
    current_run.log_debug(f"Found care seeking cols: {matched_columns}")
    missing_columns = [col for col in columns_selection if col not in dhs_careseeking.columns]
    if missing_columns:
        current_run.log_warning(f"Missing columns in care seeking data: {missing_columns}")

    merged = table.merge(
        dhs_careseeking[["ADM1_ID"] + matched_columns],  # NOTE: only ADM1_ID level
        how="left",
        on="ADM1_ID",
        suffixes=("", "_new"),
    )

    # Update each column if it is available in dhs_careseeking
    for col in matched_columns:
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
        current_run.log_warning(f"Error while loading dropout Ddtp data: {e}")
        return table

    columns_selection = [
        "PCT_DROPOUT_DTP_1_2",
        "PCT_DROPOUT_DTP_2_3",
        "PCT_DROPOUT_DTP_1_3",
    ]
    matched_columns = [col for col in columns_selection if col in dhs_dropout.columns]
    current_run.log_debug(f"Found care seeking cols: {matched_columns}")
    missing_columns = [col for col in columns_selection if col not in dhs_dropout.columns]
    if missing_columns:
        current_run.log_warning(f"Missing columns in care seeking data: {missing_columns}")
    merged = table.merge(
        dhs_dropout[["ADM1_ID"] + matched_columns],  # NOTE: only ADM1_ID level
        how="left",
        on="ADM1_ID",
        suffixes=("", "_new"),
    )

    # Update each column if it is available in dhs_dropout
    for col in matched_columns:
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
    return update_table_with(
        table_df=table,
        dataset_id=dataset_id,
        filename=f"{country_code}_DHS_ADM1_PCT_ITN_USE.parquet",
        column_id="ADM1_ID",
        column_data="PCT_ITN_USE_SAMPLE_AVERAGE",
        msg_text="DHS ITN use",
    )


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
        current_run.log_warning(f"Error while loading {msg_text} data: {e}")
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
        return match.group(1)
    return None


if __name__ == "__main__":
    snt_assemble_results()
