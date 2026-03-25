
# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with CONFIG_PATH, POPULATION_DATA_PATH.
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH='~/workspace', 
    packages=c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate")
) {
    
    # Set project folders
    setup_variable <- list(
        CONFIG_PATH = file.path(SNT_ROOT_PATH, "configuration"),  # Missing comma fixed
        POPULATION_DATA_PATH = file.path(SNT_ROOT_PATH, "data", "dhis2", "population_transformed")
    )
    
    # List required pcks
    required_packages <- packages
    install_and_load(required_packages)
    
    # Set environment to load openhexa.sdk from the right environment
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    # Import OpenHEXA SDK and make it available in the environment 
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)
    
    return(setup_variable)
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
                    msg <- glue::glue("[ERROR] Error while loading configuration: {snt_config_path}")
                    cat(msg)
                    stop(msg)
                })
    
    log_msg(paste0("SNT configuration loaded from  : ", snt_config_path))
    return(config_json)    
}


#' Load Dataset File from OpenHEXA
#' Retrieves the latest version of a file from an OpenHEXA dataset.
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @return Dataframe containing the loaded data.
#'
#' @export
load_dataset_file <- function (dataset_id, filename) {
    data <- tryCatch({ 
            get_latest_dataset_file_in_memory(dataset_id, filename) 
        }, error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading {filename} file for: {conditionMessage(e)}")
            log_msg(msg, "error")
            stop(msg)
    })
    
    msg <- glue("{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse=', ')}]")
    log_msg(msg)
    
    return(data)
}


#' Load a CSV file from the given path.
#' Logs a message on success, or throws an error if the file cannot be read.
#' @param csv_file_path - path to the CSV file
#' @return data.frame with the file contents
load_csv_file <- function(csv_file_path) {
    csv_data <- tryCatch(
        { read.csv(csv_file_path) },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading the file: {csv_file_path}")
            cat(msg)
            stop(msg)
        }
    )
    log_msg(paste0("File loaded: ", csv_file_path))
    return(csv_data)
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