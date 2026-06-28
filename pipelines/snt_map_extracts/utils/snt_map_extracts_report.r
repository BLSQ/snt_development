
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


#' Print dataframe dimensions with a readable label.
#'
#' @param df Data frame-like object.
#' @param name Optional display name (defaults to variable name).
#' @return Invisibly prints dimensions to console.
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
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
            stop(glue::glue("[ERROR] Error while loading {filename} file from dataset: {dataset_id}"))
    })

    if (verbose) {        
        log_msg(glue::glue("{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse=', ')}]"))
    }    
    return(data)
}



#' Build one choropleth plot per MAP metric.
#'
#' Generates a list of `ggplot` map objects by filtering input rows for each
#' metric and applying a shared visual style.
#'
#' @param map_data_joined Spatial table containing `METRIC_NAME` and `VALUE`.
#' @param metrics Character vector of metric names to plot.
#' @param year reference year of the data/indicator
#' @param na_color color for missing values
#' @param plot_caption map caption (defaults to NULL)
#' 
#' @return List of `ggplot` objects, one per metric.
build_metric_plots <- function(
    map_data_joined,
    metrics,
    year,
    na_color = "#D3D3D3",
    plot_caption = NULL
) {
    purrr::map(metrics, function(metric) {
        ggplot2::ggplot(map_data_joined %>% dplyr::filter(METRIC_NAME == metric)) +
            ggplot2::geom_sf(ggplot2::aes(fill = VALUE), color = "white") +
            ggplot2::scale_fill_viridis_c(option = "C", na.value = na_color) +
            
            # titles
            ggplot2::labs(
            title = metric,
            subtitle = year,
            caption = plot_caption,
            fill = "Valeur"
            ) +
        
            # map theme
            ggplot2::theme_void() +
            ggplot2::theme(
                plot.title = ggplot2::element_text(
                    face = "bold", size = 10,
                    margin = ggplot2::margin(b = 4)
                ),
                plot.subtitle = ggplot2::element_text(
                    size = 8, colour = "grey40",
                    margin = ggplot2::margin(b = 8)
                ),
                plot.caption = ggplot2::element_text(
                    size = 8,
                    colour = "grey55",
                    hjust = 1,
                    margin = ggplot2::margin(t = 8)
                ),
                legend.position = "right",
                legend.text = ggplot2::element_text(size = 8),
                plot.margin = ggplot2::margin(10, 10, 10, 10)
            )
    })
}
