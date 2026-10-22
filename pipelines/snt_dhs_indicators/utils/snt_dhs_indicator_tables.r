#' Compute and export a DHS indicator table with confidence intervals.
#'
#' Runs survey-weighted estimation by admin unit, formats confidence interval
#' and sample-average columns, converts proportions to percentages, merges with
#' admin reference table, and exports CSV/Parquet outputs.
#'
#' @param design_obj Survey design object.
#' @param indicator_name Indicator variable name in design data.
#' @param output_indicator_name Indicator label used in output columns/files.
#' @param admin_name_col Admin grouping column name.
#' @param admin_data Admin reference table for full joins.
#' @param output_data_path Directory to write output files.
#' @param filename_without_extension Output file stem.
#' @return Formatted indicator table.
compute_and_export_indicator_table <- function(
    design_obj,
    indicator_name,
    output_indicator_name = indicator_name,
    admin_name_col,
    admin_data,    
    output_data_path,
    filename_without_extension
) {
    table_content <- survey::svyby(
        formula = as.formula(paste("~", indicator_name)),
        by = reformulate(admin_name_col),
        FUN = survey::svymean,
        design = design_obj,
        level = 0.95,
        vartype = "ci",
        na.rm = TRUE,
        influence = TRUE
    )

    data.table::setDT(table_content)
    lower_bound_col <- glue::glue("{toupper(output_indicator_name)}_CI_LOWER_BOUND")
    upper_bound_col <- glue::glue("{toupper(output_indicator_name)}_CI_UPPER_BOUND")
    sample_avg_col <- glue::glue("{toupper(output_indicator_name)}_SAMPLE_AVERAGE")

    names(table_content)[names(table_content) == "ci_l"] <- lower_bound_col
    names(table_content)[names(table_content) == "ci_u"] <- upper_bound_col
    names(table_content)[names(table_content) == indicator_name] <- sample_avg_col

    table_content[get(lower_bound_col) < 0, (lower_bound_col) := 0]
    table_content[get(upper_bound_col) > 1, (upper_bound_col) := 1]
    table_content[, (lower_bound_col) := get(lower_bound_col) * 100]
    table_content[, (upper_bound_col) := get(upper_bound_col) * 100]
    table_content[, (sample_avg_col) := get(sample_avg_col) * 100]

    table_content <- data.table::merge.data.table(admin_data, table_content, by = admin_name_col, all.x = TRUE)
    utils::write.csv(table_content, file = file.path(output_data_path, paste0(filename_without_extension, ".csv")), row.names = FALSE)
    arrow::write_parquet(table_content, file.path(output_data_path, paste0(filename_without_extension, ".parquet")))
    table_content
}


#' Compute DTP dose indicators and prepare dropout base table.
#'
#' Iterates over configured DTP doses, computes/export per-dose indicators, and
#' aggregates them into a single table used later to derive dropout metrics.
#'
#' @param dtp_design Survey design object for vaccination analysis.
#' @param vaccination_doses Integer vector of dose numbers (e.g., 1:3).
#' @param indicator_access Prefix used for access indicator naming.
#' @param admin_name_col Admin grouping column name.
#' @param admin_cols Admin key columns used for table merges.
#' @param admin_data Admin reference table.
#' @param output_data_path Directory to write output files.
#' @param country_code Country code used in filenames.
#' @param data_source DHS source code used in filenames.
#' @param admin_level Admin level label used in filenames.
#' @return List with `dtp_dropout` table and `dose_tables`.
compute_dtp_indicator_tables <- function(
    dtp_design,
    vaccination_doses,
    indicator_access,
    admin_name_col,
    admin_cols,
    admin_data,
    output_data_path,
    country_code,
    data_source,
    admin_level
) {
    dtp_dropout <- data.table::copy(admin_data)
    dose_tables <- list()

    for (dose_number in vaccination_doses) {
        vaccine_colname <- glue::glue("DTP{dose_number}")
        table_name <- glue::glue("{toupper(indicator_access)}{dose_number}")
        filename_without_extension <- glue::glue("{country_code}_{data_source}_{admin_level}_{table_name}")

        df <- compute_and_export_indicator_table(
            design_obj = dtp_design,
            indicator_name = vaccine_colname,
            output_indicator_name = table_name,
            admin_name_col = admin_name_col,
            admin_data = admin_data,
            output_data_path = output_data_path,
            filename_without_extension = filename_without_extension
        )

        dose_tables[[table_name]] <- df
        dtp_dropout <- data.table::merge.data.table(dtp_dropout, df, by = admin_cols)
    }

    list(dtp_dropout = dtp_dropout, dose_tables = dose_tables)
}


#' Compute and export DTP dropout indicators.
#'
#' Derives dropout percentages between dose pairs from dose-level sample
#' averages, removes intermediate CI/sample columns, and exports final table.
#'
#' @param dtp_dropout Base table containing per-dose sample averages.
#' @param vaccination_doses Integer vector of dose numbers.
#' @param indicator_access Prefix used for dose access columns.
#' @param indicator_attrition Prefix used for dropout columns.
#' @param output_data_path Directory to write output files.
#' @param country_code Country code used in filenames.
#' @param data_source DHS source code used in filenames.
#' @param admin_level Admin level label used in filenames.
#' @return Dropout table with derived attrition columns.
compute_and_export_dtp_dropout <- function(
    dtp_dropout,
    vaccination_doses,
    indicator_access,
    indicator_attrition,
    output_data_path,
    country_code,
    data_source,
    admin_level
) {
    dtp_dropout[, grep("BOUND", names(dtp_dropout), value = TRUE) := NULL]

    for (current_dose in vaccination_doses) {
        for (reference_dose in 1:(current_dose - 1)) {
            if ((reference_dose >= 1) & (reference_dose < current_dose)) {
                attrition_col <- glue::glue("{toupper(indicator_attrition)}_{reference_dose}_{current_dose}")
                numerator_colname <- glue::glue("{toupper(indicator_access)}{current_dose}_SAMPLE_AVERAGE")
                denominator_colname <- glue::glue("{toupper(indicator_access)}{reference_dose}_SAMPLE_AVERAGE")
                dtp_dropout[, (attrition_col) := (1 - get(numerator_colname) / get(denominator_colname)) * 100]
            }
        }
    }

    dtp_dropout[, grep("SAMPLE_AVERAGE", names(dtp_dropout), value = TRUE) := NULL]
    filename <- glue::glue("{country_code}_{data_source}_{admin_level}_{indicator_attrition}")
    data.table::fwrite(dtp_dropout, file = file.path(output_data_path, paste0(filename, ".csv")))
    arrow::write_parquet(dtp_dropout, file.path(output_data_path, paste0(filename, ".parquet")))
    dtp_dropout
}


#' Get the Survey Year from a DHS Year Column
#'
#' Reads a year column from DHS recode data, either calendar years or century
#' month codes (CMC), and returns the latest year, the earliest year, or a
#' year-range label. Two-digit years from older surveys are read as 19xx.
#' If the column is missing, a warning is raised and NA is returned.
#'
#' @param df Data frame or data.table. DHS recode data (e.g. HR, PR, KR).
#' @param col Character. Name of the year column (e.g. "HV007", or "HV008" with `is_cmc = TRUE`).
#' @param type Character. "latest" (default), "earliest", or "label" (e.g. "2013-2014", or a single year).
#' @param is_cmc Logical. TRUE if `col` holds century month codes to convert to years. Default: FALSE.
#' @return Integer year for "latest" / "earliest", character label for "label"; NA if the column is
#'   missing or holds no values.
#'
#' @export
get_survey_year <- function(df, col, type = c("latest", "earliest", "label"), is_cmc = FALSE) {
    type <- match.arg(type)
    empty <- if (type == "label") NA_character_ else NA_integer_

    if (!col %in% names(df)) {
        warning(sprintf("Column '%s' not found in the data", col))
        return(empty)
    }

    years <- as.numeric(df[[col]])
    if (is_cmc) {
        years <- floor((years - 1) / 12) + 1900
        current_year <- as.integer(format(Sys.Date(), "%Y"))
        if (any(years > current_year, na.rm = TRUE)) {
            msg <- "Converted years are in the future. Is the column really a CMC? Try is_cmc = FALSE."
            warning(msg)
            log_msg(msg, "warning")
        }
    }

    years <- years[!is.na(years)]
    if (length(years) == 0) return(empty)

    # Handle two-digit years from older surveys (e.g. 92 -> 1992)
    years <- ifelse(years < 100, years + 1900, years)
    yrs <- sort(unique(as.integer(years)))

    switch(type,
        latest   = max(yrs),
        earliest = min(yrs),
        label    = if (length(yrs) > 1) paste(range(yrs), collapse = "-") else as.character(yrs)
    )
}
