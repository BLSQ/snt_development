# ================================================
# Title: Main helpers for the PATH outliers imputation pipeline
# Description: Loading, deduplication, exception (stock-out, epidemic) and imputation helpers
#   sourced by the pipeline notebook.
# Dependencies: dplyr, tidyr, glue
# ================================================

# Load base snt utils 
source(file.path("~/workspace", "code", "snt_utils.r"))


#' Load DHIS2 Routine Input Data with Validation and Logging
#'
#' Reads the latest routine parquet file from OpenHEXA, logs dataset details,
#' optionally casts YEAR and MONTH to integers, and validates indicator columns.
#' Stops execution with a clear error when required fields are missing.
#'
#' @param dataset_name Character. OpenHEXA dataset identifier/name.
#' @param country_code Character. Country code used in the routine filename prefix.
#' @param required_indicators Character vector. Optional indicator columns that must
#'   be present in the loaded data.
#' @param cast_year_month Logical. Cast the YEAR/MONTH columns to integer when TRUE.
#' @return Data frame containing the validated routine data.
#'
#' @export
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


#' Remove Duplicate Observations in PATH Long Routine Data
#'
#' Detects duplicate keys at ADM/OU/PERIOD/INDICATOR level, logs duplicate
#' counts, and keeps distinct rows only when duplicates exist.
#'
#' @param dhis2_routine_long Data frame. Long-format routine data.
#' @return List with the cleaned `data` and the `duplicated` summary table.
#'
#' @export
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


#' Detect Potential Stock-Out Exceptions in PATH Logic
#'
#' Flags periods where PRES is marked outlier while TEST is unusually low and
#' PRES remains within a reasonable upper range, indicating likely stock-out
#' behavior rather than true anomaly.
#'
#' @param dhis2_routine_outliers Data frame. Routine table with OUTLIER_TREND and
#'   the MEAN_80 / SD_80 statistics.
#' @param DEVIATION_MEAN Numeric. Deviation multiplier used in PATH thresholds.
#' @return Data frame of flagged stock-out exception keys.
#'
#' @export
detect_possible_stockout <- function(dhis2_routine_outliers, DEVIATION_MEAN) {
    low_testing_periods <- dhis2_routine_outliers %>%
        dplyr::filter(INDICATOR == "TEST") %>%
        dplyr::mutate(
            low_testing = dplyr::case_when(VALUE < MEAN_80 ~ TRUE, TRUE ~ FALSE),
            upper_limit_tested = MEAN_80 + DEVIATION_MEAN * SD_80
        ) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "low_testing", "upper_limit_tested")))

    dhis2_routine_outliers %>%
        dplyr::filter(OUTLIER_TREND == TRUE) %>%
        dplyr::left_join(low_testing_periods, by = c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD")) %>%
        dplyr::mutate(POSSIBLE_STKOUT = dplyr::case_when(low_testing == TRUE & INDICATOR == "PRES" & VALUE < upper_limit_tested ~ TRUE, TRUE ~ FALSE)) %>%
        dplyr::filter(POSSIBLE_STKOUT == TRUE) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "POSSIBLE_STKOUT")))
}


#' Detect Potential Epidemic Exceptions in PATH Logic
#'
#' Identifies periods where CONF is outlier and TEST also supports epidemic
#' behavior (test outlier or TEST >= CONF), so values should not be suppressed
#' as reporting anomalies.
#'
#' @param dhis2_routine_outliers Data frame. Routine table with OUTLIER_TREND and
#'   the MEAN_80 / SD_80 statistics.
#' @param DEVIATION_MEAN Numeric. Deviation multiplier used in PATH thresholds.
#' @return Data frame of flagged epidemic exception keys.
#'
#' @export
detect_possible_epidemic <- function(dhis2_routine_outliers, DEVIATION_MEAN) {
    dhis2_routine_outliers %>%
        dplyr::filter(INDICATOR == "TEST" | INDICATOR == "CONF") %>%
        dplyr::rename(total = VALUE) %>%
        dplyr::mutate(max_value = MEAN_80 + DEVIATION_MEAN * SD_80) %>%
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


#' Apply PATH Exception Logic and Build Cleaned Outlier Table
#'
#' Joins stock-out and epidemic exception flags, updates OUTLIER_TREND after
#' exception rules, and standardizes key output columns including YEAR/MONTH.
#'
#' @param dhis2_routine_outliers Data frame. Base PATH outlier table.
#' @param possible_stockout Data frame. Output from `detect_possible_stockout()`.
#' @param possible_epidemic Data frame. Output from `detect_possible_epidemic()`.
#' @return Cleaned long-format outlier table for imputation/export.
#'
#' @export
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


#' Impute PATH Outliers and Enforce TEST/CONF Consistency
#'
#' Replaces flagged values using MEAN_80, reshapes data to evaluate TEST vs CONF
#' consistency, and reverts impossible imputations when they create TEST < CONF
#' while original values were logically consistent.
#'
#' @param routine_data_outliers_clean Data frame. Clean outlier table from
#'   `build_path_clean_outliers()`.
#' @return Long-format routine table with VALUE_OLD, VALUE_IMPUTED and
#'   OUTLIER_TREND columns.
#'
#' @export
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
