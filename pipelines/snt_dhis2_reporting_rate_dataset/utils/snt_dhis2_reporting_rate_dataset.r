# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with CONFIG_PATH, UPLOADS_PATH, DATA_PATH.
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate")
) {
    setup_variable <- list(
        CONFIG_PATH  = file.path(SNT_ROOT_PATH, "configuration"),
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads"),
        DATA_PATH    = file.path(SNT_ROOT_PATH, "data")
    )

    install_and_load(packages)

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
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
    config_json <- tryCatch(
        { jsonlite::fromJSON(snt_config_path) },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading configuration: {snt_config_path}")
            cat(msg)
            stop(msg)
        }
    )
    log_msg(paste0("SNT configuration loaded from: ", snt_config_path))
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
load_dataset_file <- function(dataset_id, filename) {
    data <- tryCatch(
        { get_latest_dataset_file_in_memory(dataset_id, filename) },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading {filename} file: {conditionMessage(e)}")
            log_msg(msg, "error")
            stop(msg)
        }
    )
    msg <- glue::glue("{filename} data loaded from dataset: {dataset_id} dataframe dimensions: [{paste(dim(data), collapse = ', ')}]")
    log_msg(msg)
    return(data)
}
