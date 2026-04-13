# Shared helpers for snt_dhis2_formatting code notebooks.

# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with SNT paths.
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH='~/workspace', 
    packages=c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue")
) {
        
    # List required pcks
    required_packages <- unique(c(packages, "reticulate"))
    install_and_load(required_packages)

    # Set environment to load openhexa.sdk from the right environment
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    
    # Attempt to import the SDK
    tryCatch({
        sdk <- reticulate::import("openhexa.sdk")
        assign("openhexa", sdk, envir = .GlobalEnv)
    }, error = function(e) {
        log_msg("Could not import openhexa.sdk. Ensure it is installed in /opt/conda/bin/python", "warning")
    })    

    # Set paths (add paths here)
    paths_to_check = list(
        CONFIG_PATH = file.path(SNT_ROOT_PATH, "configuration"),        
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads"),
        DATA_PATH = file.path(SNT_ROOT_PATH, "data")
    )

    # create if they do not exist
    lapply(paths_to_check, dir.create, recursive = TRUE, showWarnings = FALSE)
    
    return(paths_to_check)
}


#' Load SNT Configuration File
#' Reads and parses a JSON configuration file.
#' @param snt_config_path Character. Path to the configuration JSON file.
#' @return List containing parsed configuration.
#'
#' @export
load_snt_config <- function(snt_config_path) {

    # config file path 
    config_json <- tryCatch({ fromJSON(snt_config_path) },
                error = function(e) {
                    stop(glue::glue("[ERROR] Error while loading configuration: {snt_config_path}"))
                })
    
    log_msg(paste0("SNT configuration loaded from  : ", snt_config_path))
    return(config_json)    
}


#' Load Dataset File from OpenHEXA
#' Retrieves the latest version of a file from an OpenHEXA dataset.
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @param verbose Bool. Log messages
#' @return Dataframe containing the loaded data.
#'
#' @export
load_dataset_file <- function (dataset_id, filename, verbose=TRUE) {
    data <- tryCatch({ 
            get_latest_dataset_file_in_memory(dataset_id, filename) 
        }, error = function(e) {
            stop(glue("[ERROR] Error while loading {filename} file from dataset: {dataset_id}"))
    })

    if (verbose) {        
        log_msg(glue("{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse=', ')}]"))
    }    
    return(data)
}


#' Load a CSV File with Error Handling
#'
#' @description 
#' Attempts to read a CSV file from the specified path. If the file cannot be loaded, 
#' it logs a high-level error message and stops execution.
#'
#' @param csv_file_path String representing the file path to the CSV.
#' @return A dataframe containing the contents of the CSV file.
#' 
#' @export
load_csv_file <- function(csv_file_path) {
    csv_data <- tryCatch({ read.csv(csv_file_path) },
        error = function(e) {
            stop(glue::glue("[ERROR] Error while loading the file: {csv_file_path}"))
        }
    )
    log_msg(glue::glue("File loaded: {csv_file_path}"))
    return(csv_data)
}


# -----------------------------------------------------------------------------------------
# Population transformation util functions ------------------------------------------------
# -----------------------------------------------------------------------------------------


#' Compute Yearly Population Scaling Factors
#'
#' @description Calculates the total population per year from the input data and 
#' derives a scaling factor by dividing a reference value by those annual totals. 
#'
#' @param data A dataframe or tibble containing at least \code{YEAR} and \code{POPULATION} columns.
#' @param reference A numeric value representing the target population (numerator) 
#' used to calculate the scaling factor.
#'
#' @return A tibble with three columns: \code{YEAR}, \code{total_year_pop}, and \code{scaling_factor}.
#' @importFrom dplyr group_by summarise mutate
#' @export
compute_yearly_scaling <- function(data, reference, pop_column="POPULATION") {
    
    # Compute totals per DHIS2 year
    totals <- data %>%
            group_by(YEAR) %>%
            summarise(total_year_pop = sum(.data[[pop_column]], na.rm = TRUE))
    
    # Compute scaling factor per year
    result <- totals %>%
      mutate(scaling_factor = reference / total_year_pop)
    
    return(result)
}


#' Compute Adjusted Population
#'
#' @description Merges yearly scaling factors into a dataset, calculates rounded 
#' adjusted population values, and logs verification totals for each year.
#'
#' @param data A dataframe containing \code{YEAR} and \code{POPULATION} columns.
#' @param year_scales A dataframe containing \code{YEAR}, \code{total_year_pop}, 
#' and \code{scaling_factor}.
#' @param source_column Character string. The name of the existing population column.
#' @param target_column Character string. The name for the new adjusted column.
#'
#' @return A dataframe with an added \code{POPULATION_SCALED} column and 
#' intermediate scaling columns removed.
#' 
#' @export
compute_adjusted_population <- function(data, year_scales, source_column="POPULATION", target_column="POPULATION_SCALED") {
    
    cols_in_data <- colnames(data)
    pop_column_sel <- cols_in_data[toupper(cols_in_data) == toupper(source_column)]
    
    if (length(pop_column_sel) == 0) {
        stop(glue::glue("[ERROR] Column: {source_column} not found in population data."))
    }
    
    # Compute adjusted population
    data_merged <- data %>%
      left_join(year_scales, by = "YEAR") %>%
      mutate(!!target_column := round(.data[[pop_column_sel]] * scaling_factor))

    if (any(is.na(data_merged$scaling_factor))) {
        log_msg("Some years in the data do not have scaling factors!", "warning")
    }
    
    # util logs
    for (i in seq_len(nrow(year_scales))) {
        row <- year_scales[i, ]       
        dhis2_total <- sum(data_merged[[pop_column_sel]][data_merged$YEAR == row$YEAR], na.rm = TRUE)
        dhis2_total_scd <- sum(data_merged[[target_column]][data_merged$YEAR == row$YEAR], na.rm = TRUE)        
        log_msg(glue("DHIS2 population year {row$YEAR} ({dhis2_total}) scaled: {dhis2_total_scd} (scaling_factor={round(row$scaling_factor, 3)})."))
    }
    
    # Cleanup: Drop the helper columns
    data_merged <- data_merged %>% select(-any_of(c("total_year_pop", "scaling_factor")))
    
    return(data_merged)
} 


#' Validate and Resolve Reference Year
#'
#' @description 
#' Checks if a provided reference year exists within a population dataset. 
#' If the year is NULL or missing from the data, it defaults to the maximum 
#' available year and logs a warning.
#'
#' @param dhis2_population A dataframe containing at least a \code{YEAR} column.
#' @param reference_year The year to validate (numeric or string). Can be NULL.
#'
#' @return A numeric or string representing the resolved reference year.
#' 
#' @export
resolve_reference_year <- function(dhis2_population, reference_year = NULL) {
    
    # 1. Check if the provided year exists in the dataset
    year_exists <- !is.null(reference_year) && (reference_year %in% dhis2_population$YEAR)
    
    if (!year_exists) {
        not_found <- reference_year
        # Default to the most recent year available
        resolved_year <- max(dhis2_population$YEAR, na.rm = TRUE)
        
        # 2. Log warning if a non-NULL year was provided but not found
        if (!is.null(not_found)) {
            log_msg(
                glue::glue("Reference year {not_found} is not present in the population data, using last year: {resolved_year}."), 
                "warning"
            )
        } else {
            log_msg(glue::glue("No reference year provided, defaulting to: {resolved_year}."))
        }
        
        return(resolved_year)
    }
    
    return(reference_year)
}


#' Project Specific Population Columns Backward
#' @param ref_data Dataframe of the base year.
#' @param years Vector of years to project.
#' @param growth_factor Numeric growth rate.
#' @param target_columns Character vector of column names to project.
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
#' @param ref_data Dataframe of the base year.
#' @param years Vector of years to project.
#' @param growth_factor Numeric growth rate.
#' @param target_columns Character vector of column names to project (e.g., c("TOTAL_POP", "FEMALE_POP")).
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
#' @return A combined data frame with new columns for each valid disaggregated population group.
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
    disagg_cols <- setdiff(colnames(disaggregation_data), meta_cols)

    population_table[["POPULATION"]] <- as.numeric(population_table[["POPULATION"]])    
    disaggregation_table[disagg_cols] <- suppressWarnings(lapply(disaggregation_table[disagg_cols], as.numeric))    
    
    # create a list of Valid columns
    valid_cols <- c()
    for (col in disagg_cols) {
        if (any(!is.na(disaggregation_table[[col]]))) {  # Has at least some non-NA values
            if (col %in% colnames(population_table)) {                
                log_msg(glue::glue("The column '{col}' already exists in the population table; it will be overwritten by values from the disaggregation file."), "warning")
            }            
            log_msg(glue::glue("Creating population disagregation: {col}"))
            valid_cols <- c(valid_cols, col)
        }         
    }
    
    # Early exit if no valid columns exist
    if (length(valid_cols) == 0) return(population_table)    
    
    result <- population_table %>% 
        left_join(disaggregation_table[c("ADM2_ID", valid_cols)], by = "ADM2_ID") %>%
        mutate(across(all_of(valid_cols), ~ round(POPULATION * .x)))
    
    return(result) 
}