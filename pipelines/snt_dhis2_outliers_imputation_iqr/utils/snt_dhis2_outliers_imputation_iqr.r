# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   


#' Load DHIS2 Routine Input Data with Validation and Logging
#'
#' Reads the latest routine parquet file from OpenHEXA, logs dataset details,
#' optionally casts YEAR and MONTH to integers, and validates indicator columns.
#' Stops execution with a clear error when required fields are missing.
#'
#' @param dataset_name Character. OpenHEXA dataset identifier/name.
#' @param country_code Character. Country code used in the routine filename prefix.
#' @param required_indicators Character vector. Indicator columns that must be present;
#'   stops execution if any are missing. Default: NULL (no validation).
#' @param cast_year_month Logical. If TRUE, casts the YEAR/MONTH columns to integer
#'   when both are present. Default: TRUE.
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

  
#' Impute Flagged Outliers Using a Centered Moving Average
#'
#' For each ADM/OU/indicator time series, values marked as outliers are
#' replaced by a centered moving average (ceiling), preserving non-outlier
#' observations.
#'
#' @param dt Data frame or data.table. Routine data in long format.
#' @param outlier_col Character. Name of the logical outlier flag column.
#' @param n Integer. Size of the centered rolling window, in periods. Default: 3.
#' @return Data frame with VALUE_IMPUTED and MOVING_AVG columns added, and the
#'   TO_IMPUTE helper column removed.
#'
#' @export
impute_outliers_dt <- function(dt, outlier_col, n = 3) {
    dt <- data.table::as.data.table(dt)
    data.table::setorder(dt, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, PERIOD, YEAR, MONTH)
    dt[, TO_IMPUTE := data.table::fifelse(get(outlier_col) == TRUE, NA_real_, VALUE)]
    dt[, MOVING_AVG := data.table::frollapply(
        TO_IMPUTE,
        N = n,
        FUN = function(x) {
            m <- mean(x, na.rm = TRUE)
            if (is.nan(m)) NA_real_ else ceiling(m)
        },
        align = "center"
    ), by = .(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)]
    dt[, VALUE_IMPUTED := data.table::fifelse(is.na(TO_IMPUTE), MOVING_AVG, TO_IMPUTE)]
    dt[, c("TO_IMPUTE") := NULL]
    return(as.data.frame(data.table::copy(dt)))
}


#' Build Final Routine Output Tables (Imputed or Removed)
#'
#' Reshapes long-format routine values back to wide indicator columns, joins
#' location names, and standardizes output columns expected by downstream
#' datasets and reporting.
#'
#' @param df Data frame. Long-format routine data including VALUE_IMPUTED.
#' @param outlier_column Character. Outlier flag column used to filter removed records.
#' @param dhis2_indicators Character vector. Indicator columns to keep in the final table.
#' @param fixed_cols Character vector. Fixed identifier/date columns in long format.
#' @param pyramid_names Data frame. Mapping table with ADM/OU names.
#' @param remove Logical. When TRUE, returns outlier-removed data instead of imputed data. Default FALSE.
#' @return Wide routine data frame ready for export.
#'
#' @export
format_routine_data_selection <- function(
    df,
    outlier_column,
    dhis2_indicators,
    fixed_cols,
    pyramid_names,
    remove = FALSE
) {
    if (remove) {
        df <- df %>% dplyr::filter(!.data[[outlier_column]])
    }

    target_cols <- c(
        "PERIOD", "YEAR", "MONTH", "ADM1_NAME", "ADM1_ID",
        "ADM2_NAME", "ADM2_ID", "OU_ID", "OU_NAME", dhis2_indicators
    )

    output <- df %>%
        dplyr::select(-VALUE) %>%
        dplyr::rename(VALUE = VALUE_IMPUTED) %>%
        dplyr::select(dplyr::all_of(fixed_cols), INDICATOR, VALUE) %>%
        dplyr::mutate(VALUE = ifelse(is.nan(VALUE), NA_real_, VALUE)) %>%
        tidyr::pivot_wider(names_from = "INDICATOR", values_from = "VALUE") %>%
        dplyr::left_join(pyramid_names, by = c("ADM1_ID", "ADM2_ID", "OU_ID"))

    return(output %>% dplyr::select(dplyr::all_of(intersect(target_cols, names(output)))))
}
