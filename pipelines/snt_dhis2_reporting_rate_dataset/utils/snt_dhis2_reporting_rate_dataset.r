# Load base utils
# Keep helpers small and reusable; pipeline-specific assignments stay in notebook code.
source(file.path("~/workspace/code", "snt_utils.r"))


# JSON reader for this pipeline only (`snt_utils.r` unchanged).
read_workspace_json_file <- function(json_path, resource_label = "JSON file") {
    json_path <- as.character(json_path)[[1L]]
    tryCatch(
        jsonlite::fromJSON(json_path),
        error = function(e) {
            stop(paste0(
                "[ERROR] Error while loading ",
                resource_label,
                " from `",
                json_path,
                "`: ",
                conditionMessage(e)
            ))
        }
    )
}


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with `paths_to_check` (CONFIG_PATH, UPLOADS_PATH, DATA_PATH) and the
#'   same three paths at the top level for backward compatibility (`setup$CONFIG_PATH`, …).
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c(
        "arrow", "dplyr", "tidyr", "ggplot2",
        "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate"
    )
) {
    paths_to_check <- list(
        CONFIG_PATH  = file.path(SNT_ROOT_PATH, "configuration"),
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads"),
        DATA_PATH    = file.path(SNT_ROOT_PATH, "data")
    )
    setup_variable <- c(
        list(paths_to_check = paths_to_check),
        paths_to_check
    )

    install_and_load(packages)

    if (Sys.getenv("PROJ_LIB", "") == "") {
        Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    }
    if (Sys.getenv("GDAL_DATA", "") == "") {
        Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    }
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)

    return(setup_variable)
}


#' Load SNT Configuration File
#' @param snt_config_path Character. Full path to `SNT_config.json`.
#' @export
load_snt_config <- function(snt_config_path) {
    config_json <- read_workspace_json_file(snt_config_path, "configuration")
    log_msg(paste0("SNT configuration loaded from: ", snt_config_path))
    return(config_json)
}


#' Fail if Papermill did not inject `ROUTINE_FILE` and `DATASET_ID`.
#' @export
stop_if_dataset_reporting_papermill_params_missing <- function() {
    required <- c("ROUTINE_FILE", "DATASET_ID")
    missing <- required[!vapply(required, exists, logical(1), inherits = TRUE)]
    if (length(missing) > 0) {
        stop(
            "[ERROR] Missing pipeline parameters (Papermill): ",
            paste(missing, collapse = ", "),
            ". Expected only ROUTINE_FILE and DATASET_ID from `snt_dhis2_reporting_rate_dataset`."
        )
    }
}


#' Load Dataset File from OpenHEXA
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @param verbose Logical. If TRUE, log dataframe dimensions after a successful load.
#' @export
load_dataset_file <- function(dataset_id, filename, verbose = TRUE) {
    data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_id, filename)
        },
        error = function(e) {
            stop(glue::glue("[ERROR] Error while loading {filename} file from dataset: {dataset_id}"))
        }
    )
    if (verbose) {
        log_msg(glue::glue(
            "{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse = ', ')}]"
        ))
    }
    return(data)
}


