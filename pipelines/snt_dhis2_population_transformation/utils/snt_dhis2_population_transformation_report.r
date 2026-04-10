# Shared helpers for snt_dhis2_population_transformation reporting notebook.

#' Print dataframe dimensions with a readable label.
#'
#' @param df Data frame-like object.
#' @param name Optional display name (defaults to variable name).
#' @return Invisibly prints dimensions to console.
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

#' Build dynamic legend labels from numeric breakpoints.
#'
#' Labels are formatted in thousands with apostrophe separators, producing
#' lower-than, interval, and upper-than categories.
#'
#' @param breaks Numeric vector of threshold values.
#' @return Character vector of category labels.
create_dynamic_labels <- function(breaks) {
    fmt <- function(x) {
        format(x / 1000, big.mark = "'", scientific = FALSE, trim = TRUE)
    }

    labels <- c(
        paste0("< ", fmt(breaks[1]), "k"),
        paste0(fmt(breaks[-length(breaks)]), " - ", fmt(breaks[-1]), "k"),
        paste0("> ", fmt(breaks[length(breaks)]), "k")
    )
    return(labels)
}


#' Parse metadata scale definition into numeric vector.
#'
#' Handles both JSON-encoded strings and list/vector representations.
#'
#' @param scale_value Raw metadata scale field.
#' @return Numeric vector of break values.
parse_metadata_scale <- function(scale_value) {
    if (is.character(scale_value) && length(scale_value) == 1) {
        return(jsonlite::fromJSON(scale_value))
    }
    as.numeric(unlist(scale_value, use.names = FALSE))
}


#' Build faceted choropleth for transformed population data.
#'
#' Joins population and shapes data, bins target population values into
#' configured categories, and returns a yearly faceted `ggplot` map.
#'
#' @param population_data_filtered Population table for plotting.
#' @param shapes_data Geometry table (`sf`) for ADM boundaries.
#' @param population_column Column name containing numeric population values.
#' @param breaks_values Numeric breakpoints used for binning.
#' @param labels Category labels matching breaks.
#' @param legend_title Legend title text.
#' @param plot_title Main title text.
#' @param palette_values Color vector for categories.
#' @return `ggplot` object ready to print/save.
build_population_choropleth <- function(
    population_data_filtered,
    shapes_data,
    population_column,
    breaks_values,
    labels,
    legend_title,
    plot_title,
    palette_values
) {
    names(palette_values) <- labels

    population_data_filtered %>%
        dplyr::mutate(
            CATEGORY_POPULATION = cut(
                .data[[population_column]],
                breaks = c(0, breaks_values, Inf),
                labels = labels,
                right = TRUE,
                include.lowest = TRUE
            )
        ) %>%
        dplyr::left_join(shapes_data, by = dplyr::join_by(ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID)) %>%
        ggplot2::ggplot() +
        ggplot2::geom_sf(
            ggplot2::aes(geometry = geometry, fill = CATEGORY_POPULATION),
            color = "black",
            linewidth = 0.25,
            show.legend = TRUE
        ) +
        ggplot2::labs(
            title = plot_title,
            subtitle = "Source: NMDR / DHIS2",
            fill = legend_title
        ) +
        ggplot2::scale_fill_manual(values = palette_values, limits = labels, drop = FALSE) +
        ggplot2::facet_wrap(~YEAR, ncol = 3) +
        ggplot2::theme_void() +
        ggplot2::theme(
            plot.title = ggplot2::element_text(face = "bold"),
            plot.subtitle = ggplot2::element_text(margin = ggplot2::margin(5, 0, 20, 0)),
            legend.position = "bottom",
            legend.title = ggplot2::element_text(face = "bold"),
            legend.title.position = "top",
            strip.text = ggplot2::element_text(face = "bold"),
            legend.key.height = grid::unit(0.5, "line"),
            legend.margin = ggplot2::margin(20, 0, 0, 0)
        )
}
