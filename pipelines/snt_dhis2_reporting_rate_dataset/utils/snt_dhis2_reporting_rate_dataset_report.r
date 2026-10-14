# ================================================
# Title: Report helpers for the Reporting Rate (Dataset) pipeline
# Description: Reporting-rate categories, yearly aggregation, plot builders and plot export,
#   sourced by the reporting notebook.
# Dependencies: tidyverse (dplyr, ggplot2, forcats), sf, jsonlite, glue
# Requires: code/snt_utils.r and code/snt_palettes.r, loaded by the reporting notebook before
#   this file.
# ================================================


#' Get the Reporting Rate Scale Breaks
#'
#' Reads the reporting-rate break values from the SNT metadata
#' (`REPORTING_RATE$SCALE`), accepting either a JSON-encoded string or a
#' list/vector. Falls back to the default breaks, with a `[WARNING]`, when
#' none are defined.
#'
#' @param metadata_json List. Parsed SNT metadata (`SNT_metadata.json`).
#' @param default_breaks Numeric vector. Breaks used when the metadata defines
#'   none. Default: c(0.5, 0.8, 0.9, 0.95, 1.00).
#' @return Numeric vector of break values.
#'
#' @export
get_reporting_rate_breaks <- function(metadata_json, default_breaks = c(0.5, 0.8, 0.9, 0.95, 1.00)) {
    scale_raw <- metadata_json$REPORTING_RATE$SCALE
    if (is.character(scale_raw) && length(scale_raw) == 1) {
        break_vals <- jsonlite::fromJSON(scale_raw)
    } else {
        break_vals <- unlist(scale_raw, use.names = FALSE)
    }
    break_vals <- as.numeric(break_vals)

    if (length(break_vals) == 0) {
        log_msg("[WARNING] No break values found in SNT_metadata.json for REPORTING_RATE$SCALE. Using default values.", "warning")
        break_vals <- default_breaks
    }

    log_msg(paste0("Reporting Rate scale break values: ", paste(break_vals, collapse = ", ")))
    break_vals
}


#' Build the Reporting Rate Categories
#'
#' Builds the cut points (from 0 to Inf), the category labels and the matching
#' colour palette from the break values. The palette is named with the labels
#' in reverse order, so that the lowest category gets the "worst" colour.
#'
#' @param break_vals Numeric vector. Break values (see `get_reporting_rate_breaks()`);
#'   assumes the data starts at 0.
#' @return Named list with `full_breaks` (numeric), `labels` (character) and
#'   `palette` (named character vector of colours).
#'
#' @export
build_reporting_rate_categories <- function(break_vals) {
    full_breaks <- c(0, break_vals, Inf)

    labels <- c(
        paste0("< ", break_vals[1]),                                      # First label
        paste0(break_vals[-length(break_vals)], " - ", break_vals[-1]),   # Middle labels
        paste0("> ", break_vals[length(break_vals)])                      # Last label
    )

    palette <- get_range_from_count(length(labels))
    names(palette) <- rev(labels)

    list(full_breaks = full_breaks, labels = labels, palette = palette)
}


#' Add the Reporting Rate Category Column
#'
#' Bins `REPORTING_RATE` into the categories, intervals closed on the right so
#' that 1.00 falls in the top bounded category (e.g. "0.95 - 1").
#'
#' @param df Data frame. Table with a `REPORTING_RATE` column.
#' @param categories List. Output of `build_reporting_rate_categories()`.
#' @return Data frame with an added `REPORTING_RATE_CATEGORY` factor column.
#'
#' @export
add_reporting_rate_category <- function(df, categories) {
    df %>%
        mutate(
            REPORTING_RATE_CATEGORY = cut(
                REPORTING_RATE,
                breaks = categories$full_breaks,
                labels = categories$labels,
                right = TRUE,
                include.lowest = TRUE
            )
        )
}


#' Aggregate Reporting Rates to Yearly Means
#'
#' Averages the monthly reporting rates per district and year (ignoring NA
#' values), keeping the geometry for mapping, and recomputes the category on
#' the yearly mean.
#'
#' @param data_to_plot Data frame. Monthly reporting rates joined to the shapes
#'   (`geometry`, `ADM2_ID`, `ADM2_NAME`, `ADM1_NAME`, `YEAR`, `REPORTING_RATE`).
#' @param categories List. Output of `build_reporting_rate_categories()`.
#' @return Data frame with one row per district and year, with `REPORTING_RATE`
#'   and `REPORTING_RATE_CATEGORY`.
#'
#' @export
aggregate_reporting_rate_yearly <- function(data_to_plot, categories) {
    data_to_plot %>%
        group_by(geometry, ADM2_ID, ADM2_NAME, ADM1_NAME, YEAR) %>%
        summarise(
            REPORTING_RATE = mean(REPORTING_RATE, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        add_reporting_rate_category(categories)
}


#' Plot Monthly Reporting Rates as Lines and Points
#'
#' One line and point series per district, by month, faceted by year and
#' coloured by category. The y axis extends past 1 when such values exist.
#'
#' @param data_to_plot Data frame. Monthly reporting rates with
#'   `REPORTING_RATE_CATEGORY`.
#' @param palette Named character vector. Category colours.
#' @param break_vals Numeric vector. Break values, used as y-axis breaks.
#' @param subtitle Character. Plot subtitle.
#' @return ggplot object.
#'
#' @export
plot_reporting_rate_linepoint <- function(data_to_plot, palette, break_vals, subtitle) {
    ggplot(data = data_to_plot) +
        geom_line(
            aes(x = MONTH, y = REPORTING_RATE, group = ADM2_ID, color = REPORTING_RATE_CATEGORY),
            alpha = 0.3,
            show.legend = FALSE
        ) +
        geom_point(aes(x = MONTH, y = REPORTING_RATE, group = ADM2_ID, color = REPORTING_RATE_CATEGORY)) +
        facet_grid(~YEAR) +
        scale_color_manual(
            values = palette,
            na.value = "white",
            name = "Reporting Rate Categories"
        ) +
        scale_x_continuous(breaks = seq(1, 12, 1)) +
        scale_y_continuous(
            breaks = c(0, break_vals),
            # Dynamically set max value to fit actual data (do show values >1 if present)
            limits = c(0, max(data_to_plot$REPORTING_RATE, na.rm = TRUE) + 0.1)
        ) +
        labs(
            title = "Reporting Rate (Dataset)",
            subtitle = subtitle,
            x = "Month",
            y = "Reporting Rate\n(Dataset)"
        ) +
        theme_minimal() +
        theme(
            plot.subtitle = element_text(margin = margin(0, 0, 20, 0)),
            legend.position = "none",
            legend.title = element_blank(),
            axis.title.y = element_blank(),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank(),
            strip.placement = "outside",
            strip.text = element_text(face = "bold", size = 10)
        )
}


#' Plot Monthly Reporting Rates as a Heatmap
#'
#' One tile per district and month, filled by category, faceted by year
#' (columns) and ADM1 (rows).
#'
#' @param data_to_plot Data frame. Monthly reporting rates with
#'   `REPORTING_RATE_CATEGORY`, `ADM2_NAME` and `ADM1_NAME`.
#' @param palette Named character vector. Category colours.
#' @param subtitle Character. Plot subtitle.
#' @return ggplot object.
#'
#' @export
plot_reporting_rate_heatmap <- function(data_to_plot, palette, subtitle) {
    ggplot(data = data_to_plot) +
        geom_tile(
            aes(x = MONTH, y = fct_rev(ADM2_NAME), fill = REPORTING_RATE_CATEGORY),
            color = "white",
            show.legend = TRUE
        ) +
        scale_fill_manual(
            values = palette,
            na.value = "white",
            name = "Reporting Rate: "
        ) +
        scale_x_continuous(breaks = seq(1, 12, 1)) +
        labs(
            title = "Reporting Rate (Dataset)",
            subtitle = subtitle,
            x = "Month"
        ) +
        facet_grid(
            rows = vars(ADM1_NAME), cols = vars(YEAR),
            scales = "free_y", space = "free_y",
            switch = "y"
        ) +
        theme_minimal() +
        theme(
            plot.subtitle = element_text(margin = margin(0, 0, 20, 0)),
            legend.position = "bottom",
            legend.key.height = unit(0.25, "cm"),
            axis.text.x = element_text(size = 7),
            axis.title.y = element_blank(),
            panel.grid.minor = element_blank(),
            panel.grid.major = element_blank(),
            strip.placement = "outside",
            strip.text = element_text(face = "bold", size = 10)
        ) +
        guides(fill = guide_legend(nrow = 1))
}


#' Map Monthly Reporting Rates by District
#'
#' Choropleth of the category per district, faceted by year (rows) and month
#' (columns).
#'
#' @param data_to_plot Data frame. Monthly reporting rates with
#'   `REPORTING_RATE_CATEGORY` and a `geometry` column.
#' @param palette Named character vector. Category colours.
#' @param subtitle Character. Plot subtitle.
#' @return ggplot object.
#'
#' @export
plot_reporting_rate_monthly_map <- function(data_to_plot, palette, subtitle) {
    ggplot(data = data_to_plot) +
        geom_sf(
            aes(fill = REPORTING_RATE_CATEGORY, geometry = geometry),
            color = "white",
            size = 0.01
        ) +
        scale_fill_manual(
            values = palette,
            na.value = "white"
        ) +
        theme_void() +
        theme(
            plot.subtitle = element_text(margin = margin(5, 0, 20, 0)),
            legend.position = "bottom",
            legend.title = element_blank(),
            legend.key.height = unit(0.25, "cm")
        ) +
        labs(
            title = "Reporting Rate (Dataset)",
            subtitle = subtitle
        ) +
        facet_grid(
            rows = vars(YEAR),
            cols = vars(MONTH),
            switch = "both"
        ) +
        guides(fill = guide_legend(nrow = 1))
}


#' Map Yearly Mean Reporting Rates by District
#'
#' Choropleth of the yearly-mean category per district, faceted by year.
#'
#' @param data_to_plot_year Data frame. Output of `aggregate_reporting_rate_yearly()`.
#' @param palette Named character vector. Category colours.
#' @param subtitle Character. Plot subtitle.
#' @return ggplot object.
#'
#' @export
plot_reporting_rate_yearly_map <- function(data_to_plot_year, palette, subtitle) {
    ggplot(data = data_to_plot_year) +
        geom_sf(
            aes(fill = REPORTING_RATE_CATEGORY, geometry = geometry),
            color = "white",
            size = 0.01
        ) +
        scale_fill_manual(
            values = palette,
            na.value = "white"
        ) +
        theme_void() +
        theme(
            plot.subtitle = element_text(margin = margin(5, 0, 20, 0)),
            legend.position = "bottom"
        ) +
        labs(
            title = "Reporting Rate (Dataset) - mean per Year",
            subtitle = subtitle,
            fill = "Reporting Rate: "
        ) +
        facet_grid(cols = vars(YEAR)) +
        guides(fill = guide_legend(nrow = 1))
}


#' Save a Report Plot as PNG
#'
#' Saves the plot at 200 dpi (dimensions in cm), creating the folder if needed,
#' and logs the saved path.
#'
#' @param plot ggplot object. Plot to save.
#' @param filename Character. PNG file name.
#' @param figures_path Character. Folder where the PNG is written.
#' @param width Numeric. Width in cm.
#' @param height Numeric. Height in cm.
#' @param bg Character. Background colour; NULL uses the plot theme's
#'   background. Default: NULL.
#' @return Invisibly, the path of the saved file.
#'
#' @export
save_reporting_rate_plot <- function(plot, filename, figures_path, width, height, bg = NULL) {
    ggsave(
        filename = filename,
        plot = plot,
        path = figures_path,
        create.dir = TRUE,
        width = width,
        height = height,
        units = "cm",
        bg = bg,
        dpi = 200
    )
    out_file <- file.path(figures_path, filename)
    log_msg(glue::glue("📊 Plot saved to: {out_file}"))
    invisible(out_file)
}
