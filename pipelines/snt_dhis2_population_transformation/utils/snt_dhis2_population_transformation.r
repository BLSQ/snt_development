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


#' Create Disaggregated Population
#'
#' This function takes a base population table and disaggregates the total 
#' population into specific demographic groups (e.g., age, gender) based on 
#' proportions provided in a secondary table.
#'
#' @param population_table A data frame containing at least 'ADM2_ID' and 'POPULATION'.
#' @param disaggregation_table A data frame containing 'ADM2_ID' and demographic proportion columns.
#' @return population_table with one column added per disaggregation_table column that has at
#'   least one non-NA proportion, computed as POPULATION times that proportion (any pre-existing
#'   column of the same name is overwritten). Returned unchanged if no column in
#'   disaggregation_table has any non-NA values.
#'
#' @export
add_population_disaggregations <- function(
    population_table, 
    disaggregation_table
) {

    # Standard checks
    if (!"POPULATION" %in% colnames(population_table)) stop("[ERROR] Missing POPULATION column in population table")
    if (!"ADM2_ID" %in% colnames(population_table)) stop("[ERROR] Missing ADM2_ID column in population table")
    if (!"ADM2_ID" %in% colnames(disaggregation_table)) stop("[ERROR] Missing ADM2_ID column in disaggregation_table")
    
    # Identify target columns and convert to numeric
    meta_cols <- c("ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID")
    disagg_cols <- setdiff(colnames(disaggregation_table), meta_cols)

    population_table[["POPULATION"]] <- as.numeric(population_table[["POPULATION"]])    
    disaggregation_table[disagg_cols] <- suppressWarnings(lapply(disaggregation_table[disagg_cols], as.numeric))    
    
    # create a list of Valid columns
    valid_cols <- c()
    for (col in disagg_cols) {
        action <- "Creating"
        if (any(!is.na(disaggregation_table[[col]]))) {  # Has at least some non-NA values
            if (col %in% colnames(population_table)) {
                action <- "Overwriting"
                log_msg(glue::glue("Column '{col}' already exists in the population table; it will be overwritten by values from the disaggregation file."), "warning")
            }            
            log_msg(glue::glue("{action} population disagregation: {col}"))
            valid_cols <- c(valid_cols, col)
        }         
    }
     
    # Early exit if no valid columns exist
    if (length(valid_cols) == 0) return(population_table)    
    
    result <- population_table %>% 
        select(-any_of(valid_cols)) %>% 
        left_join(disaggregation_table[c("ADM2_ID", valid_cols)], by = "ADM2_ID") %>%
        mutate(across(all_of(valid_cols), ~ round(POPULATION * .x)))
    
    return(result) 
}