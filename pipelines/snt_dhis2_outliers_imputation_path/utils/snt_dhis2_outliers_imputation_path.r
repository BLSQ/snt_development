# Helpers for PATH outliers imputation notebook.

#' Initialize runtime context for the PATH outliers pipeline.
#'
#' Creates standard project paths, loads shared dependencies and utilities,
#' initializes OpenHEXA SDK access, and returns a context object used by
#' notebooks.
#'
#' @param root_path Project root folder (workspace).
#' @param required_packages Character vector of R packages to install/load.
#' @param load_openhexa Logical; import OpenHEXA SDK when TRUE.
#' @return Named list with paths, OpenHEXA handle, and parsed config.
bootstrap_path_context <- function(
    root_path = "~/workspace",
    required_packages = c("arrow", "tidyverse", "jsonlite", "DBI", "RPostgres", "reticulate", "glue"),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")
    output_dir <- file.path(data_path, "dhis2", "outliers_imputation")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }
    assign("openhexa", openhexa, envir = .GlobalEnv)

    config_json <- tryCatch(
        {
            jsonlite::fromJSON(file.path(config_path, "SNT_config.json"))
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading configuration {conditionMessage(e)}")
            log_msg(msg)
            stop(msg)
        }
    )

    return(list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        OUTPUT_DIR = output_dir,
        openhexa = openhexa,
        config_json = config_json
    ))
}

#' Load DHIS2 routine input data with validation and logging.
#'
#' Reads the latest routine parquet file from OpenHEXA, logs dataset details,
#' optionally casts YEAR and MONTH to integers, and validates indicator columns.
#' Stops execution with a clear error when required fields are missing.
#'
#' @param dataset_name OpenHEXA dataset identifier/name.
#' @param country_code Country code used in routine filename prefix.
#' @param required_indicators Optional character vector of required indicators.
#' @param cast_year_month Logical; cast YEAR/MONTH columns to integer.
#' @return Data frame containing validated routine data.
load_routine_data <- function(dataset_name, country_code, required_indicators = NULL, cast_year_month = TRUE) {
    dhis2_routine <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, paste0(country_code, "_routine.parquet"))
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading DHIS2 routine data file for {country_code} : {conditionMessage(e)}")
            log_msg(msg)
            stop(msg)
        }
    )

    log_msg(glue::glue("DHIS2 routine data loaded from dataset : {dataset_name}"))
    log_msg(glue::glue("DHIS2 routine data loaded has dimensions: {nrow(dhis2_routine)} rows, {ncol(dhis2_routine)} columns."))

    if (cast_year_month && all(c("YEAR", "MONTH") %in% colnames(dhis2_routine))) {
        dhis2_routine[c("YEAR", "MONTH")] <- lapply(dhis2_routine[c("YEAR", "MONTH")], as.integer)
    }

    if (!is.null(required_indicators)) {
        missing_indicators <- setdiff(required_indicators, colnames(dhis2_routine))
        if (length(missing_indicators) > 0) {
            msg <- paste("[ERROR] Missing indicator column(s) in routine data:", paste(missing_indicators, collapse = ", "))
            log_msg(msg)
            stop(msg)
        }
    }

    dhis2_routine
}

#' Build PATH-ready long routine table.
#'
#' Selects required administrative/time columns, pivots indicator values to
#' long format, and completes missing PERIOD/INDICATOR combinations for each
#' location.
#'
#' @param dhis2_routine Routine data in wide format.
#' @param DHIS2_INDICATORS Indicator column names to pivot.
#' @return Long-format routine data frame used by PATH detection.
build_path_routine_long <- function(dhis2_routine, DHIS2_INDICATORS) {
    dhis2_routine %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM1_NAME", "ADM2_ID", "ADM2_NAME", "OU_ID", "OU_NAME", "PERIOD", DHIS2_INDICATORS))) %>%
        tidyr::pivot_longer(cols = dplyr::all_of(DHIS2_INDICATORS), names_to = "INDICATOR", values_to = "VALUE") %>%
        tidyr::complete(tidyr::nesting(ADM1_ID, ADM1_NAME, ADM2_ID, ADM2_NAME, OU_ID, OU_NAME), PERIOD, INDICATOR) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "INDICATOR", "VALUE")))
}

#' Remove duplicate observations in PATH long routine data.
#'
#' Detects duplicate keys at ADM/OU/PERIOD/INDICATOR level, logs duplicate
#' counts, and keeps distinct rows only when duplicates exist.
#'
#' @param dhis2_routine_long Long-format routine data frame.
#' @return List with cleaned `data` and `duplicated` summary table.
remove_path_duplicates <- function(dhis2_routine_long) {
    duplicated <- dhis2_routine_long %>%
        dplyr::group_by(ADM1_ID, ADM2_ID, OU_ID, PERIOD, INDICATOR) %>%
        dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
        dplyr::filter(n > 1L)

    if (nrow(duplicated) > 0) {
        log_msg(glue::glue("Removing {nrow(duplicated)} duplicated values."))
        dhis2_routine_long <- dhis2_routine_long %>%
            dplyr::distinct(ADM1_ID, ADM2_ID, OU_ID, PERIOD, INDICATOR, .keep_all = TRUE)
    }

    list(data = dhis2_routine_long, duplicated = duplicated)
}

#' Detect potential stock-out exceptions in PATH logic.
#'
#' Flags periods where PRES is marked outlier while TEST is unusually low and
#' PRES remains within a reasonable upper range, indicating likely stock-out
#' behavior rather than true anomaly.
#'
#' @param dhis2_routine_outliers Routine table with OUTLIER_TREND and stats.
#' @param MEAN_DEVIATION Deviation multiplier used in PATH thresholds.
#' @return Data frame of flagged stock-out exception keys.
detect_possible_stockout <- function(dhis2_routine_outliers, MEAN_DEVIATION) {
    low_testing_periods <- dhis2_routine_outliers %>%
        dplyr::filter(INDICATOR == "TEST") %>%
        dplyr::mutate(
            low_testing = dplyr::case_when(VALUE < MEAN_80 ~ TRUE, TRUE ~ FALSE),
            upper_limit_tested = MEAN_80 + MEAN_DEVIATION * SD_80
        ) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "low_testing", "upper_limit_tested")))

    dhis2_routine_outliers %>%
        dplyr::filter(OUTLIER_TREND == TRUE) %>%
        dplyr::left_join(low_testing_periods, by = c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD")) %>%
        dplyr::mutate(POSSIBLE_STKOUT = dplyr::case_when(low_testing == TRUE & INDICATOR == "PRES" & VALUE < upper_limit_tested ~ TRUE, TRUE ~ FALSE)) %>%
        dplyr::filter(POSSIBLE_STKOUT == TRUE) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "POSSIBLE_STKOUT")))
}

#' Detect potential epidemic exceptions in PATH logic.
#'
#' Identifies periods where CONF is outlier and TEST also supports epidemic
#' behavior (test outlier or TEST >= CONF), so values should not be suppressed
#' as reporting anomalies.
#'
#' @param dhis2_routine_outliers Routine table with OUTLIER_TREND and stats.
#' @param MEAN_DEVIATION Deviation multiplier used in PATH thresholds.
#' @return Data frame of flagged epidemic exception keys.
detect_possible_epidemic <- function(dhis2_routine_outliers, MEAN_DEVIATION) {
    dhis2_routine_outliers %>%
        dplyr::filter(INDICATOR == "TEST" | INDICATOR == "CONF") %>%
        dplyr::rename(total = VALUE) %>%
        dplyr::mutate(max_value = MEAN_80 + MEAN_DEVIATION * SD_80) %>%
        dplyr::select(-c("MEAN_80", "SD_80")) %>%
        tidyr::pivot_wider(names_from = INDICATOR, values_from = c(total, max_value, OUTLIER_TREND)) %>%
        tidyr::unnest(cols = dplyr::everything()) %>%
        dplyr::mutate(POSSIBLE_EPID = dplyr::case_when(
            OUTLIER_TREND_CONF == TRUE & (OUTLIER_TREND_TEST == TRUE | total_TEST >= total_CONF) ~ TRUE,
            TRUE ~ FALSE
        )) %>%
        dplyr::filter(POSSIBLE_EPID == TRUE) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "POSSIBLE_EPID")))
}

#' Apply PATH exception logic and build cleaned outlier table.
#'
#' Joins stock-out and epidemic exception flags, updates OUTLIER_TREND after
#' exception rules, and standardizes key output columns including YEAR/MONTH.
#'
#' @param dhis2_routine_outliers Base PATH outlier table.
#' @param possible_stockout Output from `detect_possible_stockout`.
#' @param possible_epidemic Output from `detect_possible_epidemic`.
#' @return Cleaned long-format outlier table for imputation/export.
build_path_clean_outliers <- function(dhis2_routine_outliers, possible_stockout, possible_epidemic) {
    dhis2_routine_outliers %>%
        dplyr::left_join(possible_stockout, by = c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD")) %>%
        dplyr::mutate(OUTLIER_TREND_01 = dplyr::case_when(OUTLIER_TREND == TRUE & INDICATOR == "PRES" & POSSIBLE_STKOUT == TRUE ~ FALSE, TRUE ~ OUTLIER_TREND)) %>%
        dplyr::left_join(possible_epidemic, by = c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD")) %>%
        dplyr::mutate(OUTLIER_TREND_02 = dplyr::case_when(OUTLIER_TREND_01 == TRUE & INDICATOR %in% c("CONF", "TEST") & POSSIBLE_EPID == TRUE ~ TRUE, TRUE ~ OUTLIER_TREND_01)) %>%
        dplyr::select(-OUTLIER_TREND) %>%
        dplyr::rename(OUTLIER_TREND = OUTLIER_TREND_02) %>%
        dplyr::mutate(
            YEAR = as.integer(substr(PERIOD, 1, 4)),
            MONTH = as.integer(substr(PERIOD, 5, 6))
        ) %>%
        dplyr::select(dplyr::all_of(c(
            "PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID",
            "INDICATOR", "VALUE", "MEAN_80", "SD_80",
            "OUTLIER_TREND", "POSSIBLE_STKOUT", "POSSIBLE_EPID"
        )))
}

#' Impute PATH outliers and enforce TEST/CONF consistency.
#'
#' Replaces flagged values using MEAN_80, reshapes data to evaluate TEST vs CONF
#' consistency, and reverts impossible imputations when they create TEST < CONF
#' while original values were logically consistent.
#'
#' @param routine_data_outliers_clean Clean outlier table from PATH logic.
#' @return Long-format routine table with VALUE_OLD, VALUE_IMPUTED and flags.
impute_path_outliers <- function(routine_data_outliers_clean) {
    routine_data_outliers_clean %>%
        dplyr::rename(VALUE_OLD = VALUE) %>%
        dplyr::mutate(VALUE_IMPUTED = ifelse(OUTLIER_TREND == TRUE, MEAN_80, VALUE_OLD)) %>%
        dplyr::select(dplyr::all_of(c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID", "INDICATOR", "VALUE_OLD", "VALUE_IMPUTED", "OUTLIER_TREND"))) %>%
        tidyr::pivot_wider(names_from = INDICATOR, values_from = c(VALUE_OLD, VALUE_IMPUTED, OUTLIER_TREND)) %>%
        dplyr::mutate(reverse_val = dplyr::case_when(
            !is.na(VALUE_IMPUTED_TEST) & !is.na(VALUE_IMPUTED_CONF) &
                VALUE_IMPUTED_TEST < VALUE_IMPUTED_CONF &
                VALUE_OLD_TEST > VALUE_OLD_CONF ~ TRUE,
            TRUE ~ FALSE
        )) %>%
        dplyr::mutate(
            VALUE_IMPUTED_TEST = ifelse(reverse_val == TRUE, VALUE_OLD_TEST, VALUE_IMPUTED_TEST),
            OUTLIER_TREND_TEST = ifelse(reverse_val == TRUE, FALSE, OUTLIER_TREND_TEST)
        ) %>%
        dplyr::mutate(
            VALUE_IMPUTED_CONF = ifelse(reverse_val == TRUE, VALUE_OLD_CONF, VALUE_IMPUTED_CONF),
            OUTLIER_TREND_CONF = ifelse(reverse_val == TRUE, FALSE, OUTLIER_TREND_CONF)
        ) %>%
        dplyr::select(-reverse_val) %>%
        tidyr::pivot_longer(
            cols = dplyr::starts_with("VALUE_OLD_") | dplyr::starts_with("VALUE_IMPUTED_") | dplyr::starts_with("OUTLIER_TREND_"),
            names_to = c(".value", "INDICATOR"),
            names_pattern = "(.*)_(.*)$"
        ) %>%
        dplyr::arrange(ADM1_ID, ADM2_ID, OU_ID, PERIOD, INDICATOR) %>%
        dplyr::select(dplyr::all_of(c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID", "INDICATOR", "VALUE_OLD", "VALUE_IMPUTED", "OUTLIER_TREND")))
}
