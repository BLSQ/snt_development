# ================================================
# Title: Main helpers for the median outliers imputation pipeline
# Description: Loading and output-formatting helpers sourced by the pipeline notebook (imputation:
#   impute_outliers() in code/snt_utils.r).
# Dependencies: dplyr, tidyr, glue
# ================================================

# Load base snt utils
source(file.path("~/workspace", "code", "snt_utils.r"))


#' Load DHIS2 Routine Input Data
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
load_routine_data <- function(
    dataset_name,
    country_code,
    required_indicators = NULL,
    cast_year_month = TRUE
) {
    dhis2_routine <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, paste0(country_code, "_routine.parquet"))
        },
        error = function(e) {
            msg <- glue::glue(
                "[ERROR] Error while loading DHIS2 routine data file for {country_code} : ",
                "{conditionMessage(e)}"
            )
            log_msg(msg)
            stop(msg)
        }
    )

    log_msg(glue::glue("DHIS2 routine data loaded from dataset : {dataset_name}"))
    log_msg(glue::glue(
        "DHIS2 routine data loaded has dimensions: {nrow(dhis2_routine)} rows, {ncol(dhis2_routine)} columns."
    ))

    if (cast_year_month && all(c("YEAR", "MONTH") %in% colnames(dhis2_routine))) {
        dhis2_routine[c("YEAR", "MONTH")] <- lapply(dhis2_routine[c("YEAR", "MONTH")], as.integer)
    }

    if (!is.null(required_indicators)) {
        missing_indicators <- setdiff(required_indicators, colnames(dhis2_routine))
        if (length(missing_indicators) > 0) {
            msg <- paste(
                "[ERROR] Missing indicator column(s) in routine data:",
                paste(missing_indicators, collapse = ", ")
            )
            log_msg(msg)
            stop(msg)
        }
    }

    dhis2_routine
}


#' Build the Final Routine Output Table (Imputed or Removed)
#'
#' Reshapes long-format routine values back to wide indicator columns, joins
#' location names, and standardizes the output columns expected by downstream
#' datasets and reporting.
#'
#' @param df Data frame. Long-format routine data including VALUE_IMPUTED.
#' @param outlier_column Character. Outlier flag column used to filter removed records.
#' @param DHIS2_INDICATORS Character vector. Indicator columns to keep in the final table.
#' @param fixed_cols Character vector. Fixed identifier/date columns in long format.
#' @param pyramid_names Data frame. Mapping table carrying the ADM/OU names.
#' @param remove Logical. Return outlier-removed data when TRUE, imputed data otherwise.
#' @return Wide routine data frame ready for export.
#'
#' @export
format_routine_data_selection <- function(
    df,
    outlier_column,
    DHIS2_INDICATORS,
    fixed_cols,
    pyramid_names,
    remove = FALSE
) {
    if (remove) {
        df <- df %>% dplyr::filter(!.data[[outlier_column]])
    }
    target_cols <- c(
        "PERIOD", "YEAR", "MONTH", "ADM1_NAME", "ADM1_ID",
        "ADM2_NAME", "ADM2_ID", "OU_ID", "OU_NAME", DHIS2_INDICATORS
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
