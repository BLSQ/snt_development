#' Print dataframe dimensions with a readable label.
#'
#' @param df Data frame-like object.
#' @param name Optional display name (defaults to variable name).
#' @return Invisibly prints dimensions to console.
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

#' Load a MAP report input file from dataset with logging/error handling.
#'
#' @param dataset_name Dataset identifier/name.
#' @param filename File name to download from latest dataset version.
#' @param label Human-readable label for logs/errors.
#' @return Loaded dataset object (data frame / sf depending on source file).
load_map_report_input <- function(dataset_name, filename, label = "dataset file") {
    data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, filename)
        },
        error = function(e) {
            msg <- paste("Error while loading", label, "for file:", filename, conditionMessage(e))
            cat(msg)
            stop(msg)
        }
    )
    log_msg(paste0(label, " loaded from dataset: ", dataset_name, " dataframe dimensions: ", paste(dim(data), collapse = ", ")))
    data
}


#' Build one choropleth plot per MAP metric.
#'
#' Generates a list of `ggplot` map objects by filtering input rows for each
#' metric and applying a shared visual style.
#'
#' @param map_data_joined Spatial table containing `METRIC_NAME` and `VALUE`.
#' @param metrics Character vector of metric names to plot.
#' @return List of `ggplot` objects, one per metric.
build_metric_plots <- function(map_data_joined, metrics) {
    purrr::map(metrics, function(metric) {
        ggplot2::ggplot(map_data_joined %>% dplyr::filter(METRIC_NAME == metric)) +
            ggplot2::geom_sf(ggplot2::aes(fill = VALUE), color = "white") +
            ggplot2::scale_fill_viridis_c(option = "C", na.value = "lightgrey") +
            ggplot2::labs(
                title = paste0(metric),
                fill = "Valeur"
            ) +
            ggplot2::theme_minimal(base_size = 16) +
            ggplot2::theme(
                plot.title = ggplot2::element_text(size = 20, face = "bold"),
                legend.title = ggplot2::element_text(size = 16),
                legend.text = ggplot2::element_text(size = 14)
            )
    })
}
