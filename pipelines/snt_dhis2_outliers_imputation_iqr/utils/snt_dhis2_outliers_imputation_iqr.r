# ================================================
# Title: Main helpers for the IQR outliers imputation pipeline
# Description: Routine data loading helper sourced by the pipeline notebook (imputation and output
#   formatters: code/snt_utils.r).
# Dependencies: glue
# ================================================

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
