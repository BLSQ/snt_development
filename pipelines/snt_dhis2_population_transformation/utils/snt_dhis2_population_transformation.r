# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   


# -----------------------------------------------------------------------------------------
# Population transformation util functions ------------------------------------------------
# -----------------------------------------------------------------------------------------


#' Validate and Resolve Reference Year
#'
#' Checks if a provided reference year exists among the available years.
#' If the year is NULL or missing from the data, it defaults to the maximum
#' available year and logs a warning.
#'
#' @param available_years Numeric or character vector of years present in the population data.
#' @param reference_year The year to validate (numeric or string). Can be NULL.
#'
#' @return A numeric or string representing the resolved reference year.
#'
#' @export
resolve_reference_year <- function(available_years, reference_year = NULL) {
        
    latest_year <- max(available_years, na.rm = TRUE)
    
    # No year provided — default to latest
    if (is.null(reference_year)) {
        log_msg(glue("No reference year provided, defaulting to: {latest_year}."))
        return(latest_year)
    }
    
    # Year provided but not found in data — fallback to latest
    if (!reference_year %in% available_years) {
        log_msg(glue("Reference year {reference_year} not found in population data, falling back to: {latest_year}."), "warning")
        return(latest_year)
    }
    
    # Year found — use it
    return(reference_year)
}


#' Project Specific Population Columns Backward
#'
#' Projects target population columns backward in time from a base year, by
#' repeatedly dividing by (1 + growth_factor) for each year moving away from
#' the base year.
#'
#' @param ref_data Dataframe of the base year.
#' @param years Vector of years to project.
#' @param growth_factor Numeric growth rate.
#' @param target_columns Character vector of column names to project.
#'
#' @return Data frame with one row per input year (stacked via rbind), with target_columns
#'   scaled down for each year, or NULL if years is empty.
#'
#' @export
project_backward <- function(ref_data, years, growth_factor, target_columns) {
    if (length(years) == 0) return(NULL)
    
    # Validation
    missing_cols <- setdiff(target_columns, colnames(ref_data))
    if (length(missing_cols) > 0) {
        stop(glue::glue("The following target columns were not found in ref_data: {paste(missing_cols, collapse = ', ')}"))
    }
    
    results <- list()
    current_data <- ref_data
    ordered_years <- sort(years, decreasing = TRUE)
    
    for (yr in ordered_years) {
        current_data[["YEAR"]] <- yr
        current_data[target_columns] <- lapply(current_data[target_columns], function(x) {
          round(x / (1 + growth_factor))
        })    
        results[[as.character(yr)]] <- current_data
    }
    
    return(do.call(rbind, results))
}


#' Project Specific Population Columns Forward
#'
#' Projects target population columns forward in time from a base year, by
#' repeatedly multiplying by (1 + growth_factor) for each year moving away from
#' the base year.
#'
#' @param ref_data Dataframe of the base year.
#' @param years Vector of years to project.
#' @param growth_factor Numeric growth rate.
#' @param target_columns Character vector of column names to project (e.g., c("TOTAL_POP", "FEMALE_POP")).
#'
#' @return Data frame with one row per input year (stacked via rbind), with target_columns
#'   scaled up for each year, or NULL if years is empty.
#'
#' @export
project_forward <- function(ref_data, years, growth_factor, target_columns) {
    if (length(years) == 0) return(NULL)
    
    # Validation: Ensure all target columns exist in the data
    missing_cols <- setdiff(target_columns, colnames(ref_data))
    if (length(missing_cols) > 0) {
        stop(glue::glue("The following target columns were not found in ref_data: {paste(missing_cols, collapse = ', ')}"))
    }
    
    results <- list()
    current_data <- ref_data
    ordered_years <- sort(years)
    
    for (yr in ordered_years) {
        current_data[["YEAR"]] <- yr
        current_data[target_columns] <- lapply(current_data[target_columns], function(x) {
          round(x * (1 + growth_factor))
        })
        results[[as.character(yr)]] <- current_data
    }
    
    return(do.call(rbind, results))
}


#' Validate Disaggregation Proportion Columns
#'
#' Checks that each given column has at least one value and that every non-NA value is a
#' proportion between 0 and 1. Empty columns (all NA) are skipped, and columns with any value
#' outside that range are logged as an error. Both are excluded from the returned column list,
#' so the corresponding disaggregation is not computed.
#'
#' @param disaggregation_table A data frame containing the proportion columns (already numeric).
#' @param columns Character vector of column names to validate.
#'
#' @return Character vector with the subset of columns that are non-empty and whose values are
#'   all within [0, 1].
#'
#' @export
validate_proportion_columns <- function(disaggregation_table, columns) {
    valid_columns <- c()
    for (col in columns) {
        values <- disaggregation_table[[col]]
        if (all(is.na(values))) {
            log_msg(glue::glue("Disaggregation column '{col}' is empty and will be ignored."))
            next
        }
        out_of_range <- !is.na(values) & (values < 0 | values > 1)
        if (any(out_of_range)) {
            invalid_ids <- head(disaggregation_table$ADM2_ID[out_of_range], 5)
            log_msg(glue::glue(
                "Disaggregation column '{col}' has {sum(out_of_range)} value(s) outside the [0, 1] range ",
                "(e.g. ADM2_ID: {paste(invalid_ids, collapse=', ')}). Values must be proportions (e.g. 0.17 for 17%). ",
                "The '{col}' disaggregation will be ignored."
            ), "error")
        } else {
            valid_columns <- c(valid_columns, col)
        }
    }
    return(valid_columns)
}


#' Create Disaggregated Population
#'
#' Disaggregates the total population into specific demographic groups (e.g. age,
#' pregnant women) using ADM2-level proportions from a secondary table. Proportions
#' are joined on ADM2_ID only, so the same proportion is applied to every year of the
#' population table. Metadata columns (YEAR, ADM1/ADM2 names and ids) and the
#' population column itself are never treated as disaggregations.
#'
#' @param population_table A data frame containing at least 'ADM2_ID' and the population column.
#' @param disaggregation_table A data frame containing 'ADM2_ID' and demographic proportion columns.
#' @param population_col Name of the total population column used as the base for the
#'   disaggregations. Defaults to "POPULATION".
#'
#' @return population_table with one column added per valid disaggregation_table column (non-empty,
#'   all values within [0, 1], see validate_proportion_columns()), computed as population_col times
#'   that proportion (any pre-existing column of the same name is overwritten). If ADM2_ID is
#'   duplicated in disaggregation_table, only its first row is used. Returned unchanged if no
#'   column is valid.
#'
#' @export
add_population_disaggregations <- function(
    population_table, 
    disaggregation_table,
    population_col="POPULATION"
) {

    # Standard checks
    if (!population_col %in% colnames(population_table)) stop(glue::glue("[ERROR] Missing {population_col} column in population table"))
    if (!"ADM2_ID" %in% colnames(population_table)) stop("[ERROR] Missing ADM2_ID column in population table")
    if (!"ADM2_ID" %in% colnames(disaggregation_table)) stop("[ERROR] Missing ADM2_ID column in disaggregation_table")
    
    # Identify target columns and convert to numeric
    meta_cols <- c("YEAR", "ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID")
    disagg_cols <- setdiff(colnames(disaggregation_table), c(meta_cols, population_col))

    population_table[[population_col]] <- as.numeric(population_table[[population_col]])
    disaggregation_table[disagg_cols] <- suppressWarnings(lapply(disaggregation_table[disagg_cols], as.numeric))

    # Keep only non-empty columns whose values are proportions (within [0, 1])
    valid_cols <- validate_proportion_columns(disaggregation_table, disagg_cols)

    for (col in valid_cols) {
        action <- "Creating"
        if (col %in% colnames(population_table)) {
            action <- "Overwriting"
            log_msg(glue::glue("Column '{col}' already exists in the population table; it will be overwritten by values from the disaggregation file."), "warning")
        }
        log_msg(glue::glue("{action} population disaggregation ({population_col} * disaggregation): {col}"))
    }

    # Early exit if no valid columns exist
    if (length(valid_cols) == 0) return(population_table)

    # Ensure one row per ADM2_ID, otherwise the join duplicates population rows
    duplicated_ids <- unique(disaggregation_table$ADM2_ID[duplicated(disaggregation_table$ADM2_ID)])
    if (length(duplicated_ids) > 0) {
        log_msg(glue::glue(
            "Disaggregation file has duplicated ADM2_ID(s): {paste(head(duplicated_ids, 10), collapse=', ')}. ",
            "Only the first row of each duplicated ADM2_ID will be used."
        ), "error")
        disaggregation_table <- disaggregation_table %>% distinct(ADM2_ID, .keep_all = TRUE)
    }

    result <- population_table %>%
        select(-any_of(valid_cols)) %>% 
        left_join(disaggregation_table[c("ADM2_ID", valid_cols)], by = "ADM2_ID") %>%
        mutate(across(all_of(valid_cols), ~ round(.data[[population_col]] * .x)))
    
    return(result) 
}