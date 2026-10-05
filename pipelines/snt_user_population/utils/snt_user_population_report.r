# Helpers for the snt_user_population reporting notebook.

ADM_COLUMNS <- c("ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID")


#' Print dataframe dimensions
#'
#' Prints the number of rows and columns of a data frame with a readable label.
#'
#' @param df Data frame-like object.
#' @param name Character. Display name (defaults to the variable name).
#' @return NULL, invisibly. Called for its side effect of printing.
#'
#' @export
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
    invisible(NULL)
}


#' Check the population file admin units against the shapes
#'
#' Every ADM1_NAME / ADM1_ID / ADM2_NAME / ADM2_ID combination in the user population
#' file must exist in the shapes, otherwise those units cannot be drawn. Stops with an
#' `[ERROR]` listing the mismatched units (up to 10) and what the shapes contain for the
#' same ADM2_ID. Shapes units absent from the file are only logged: they are drawn grey.
#'
#' @param population_data Data frame. User population table.
#' @param shapes_data sf / data frame. Shapes with the ADM columns.
#' @param country_code Character. Country code, used to name the files in the message.
#' @param adm_cols Character vector. Admin columns to match on.
#' @return NULL, invisibly. Stops if any file unit does not match the shapes.
#'
#' @export
check_population_matches_shapes <- function(
    population_data,
    shapes_data,
    country_code,
    adm_cols = ADM_COLUMNS
) {
    shapes_units <- as.data.frame(shapes_data) %>%
        dplyr::distinct(dplyr::across(dplyr::all_of(adm_cols)))
    file_units <- population_data %>%
        dplyr::distinct(dplyr::across(dplyr::all_of(adm_cols)))

    unmatched <- dplyr::anti_join(file_units, shapes_units, by = adm_cols)
    if (nrow(unmatched) > 0) {
        details <- unmatched %>%
            dplyr::left_join(shapes_units, by = "ADM2_ID", suffix = c("", "_SHAPES")) %>%
            dplyr::mutate(detail = dplyr::if_else(
                is.na(ADM2_NAME_SHAPES),
                glue::glue("{ADM1_NAME} / {ADM2_NAME} (ADM2_ID {ADM2_ID}): ADM2_ID not found in shapes"),
                glue::glue(
                    "{ADM1_NAME} ({ADM1_ID}) / {ADM2_NAME} (ADM2_ID {ADM2_ID}): ",
                    "shapes has {ADM1_NAME_SHAPES} ({ADM1_ID_SHAPES}) / {ADM2_NAME_SHAPES}"
                )
            )) %>%
            dplyr::pull(detail)
        stop(
            glue::glue(
                "[ERROR] {nrow(unmatched)} admin unit(s) in {country_code}_population.parquet ",
                "do not match {country_code}_shapes.geojson on {paste(adm_cols, collapse = ', ')}. ",
                "Correct the uploaded file so the names and IDs match the shapes, then re-run the pipeline.\n",
                "{paste(utils::head(details, 10), collapse = '\n')}",
                "{if (length(details) > 10) '\n...' else ''}"
            ),
            call. = FALSE
        )
    }

    n_missing <- nrow(dplyr::anti_join(shapes_units, file_units, by = adm_cols))
    if (n_missing > 0) {
        log_msg(glue::glue("{n_missing} shapes ADM2 unit(s) have no row in the population file (drawn grey)."))
    }
    log_msg(glue::glue("All {nrow(file_units)} admin units of the population file match the shapes."))
    invisible(NULL)
}


#' Compute choropleth class thresholds
#'
#' Classifies the values with `classInt::classIntervals` and returns the interior
#' thresholds (min and max excluded), rounded to 2 significant digits for readable
#' legends. Missing values are ignored; duplicate thresholds created by rounding are
#' dropped, so fewer than `n_classes - 1` thresholds may be returned.
#'
#' @param values Numeric vector of values to classify.
#' @param n_classes Integer. Target number of classes.
#' @param style Character. classInt classification style (e.g. "kmeans", "jenks").
#' @return Numeric vector of sorted, unique thresholds (length 0 when values are constant).
#'
#' @export
compute_value_intervals <- function(values, n_classes, style = "kmeans") {
    values <- values[!is.na(values)]
    n_classes <- min(n_classes, length(unique(values)))
    if (n_classes < 2) {
        return(numeric(0))
    }
    breaks <- suppressWarnings(classInt::classIntervals(values, n = n_classes, style = style)$brks)
    thresholds <- signif(breaks[-c(1, length(breaks))], 2)
    return(sort(unique(thresholds)))
}


#' Build legend labels from class thresholds
#'
#' Produces "< t1", "t1 - t2", ..., "> tn" labels, with short-scale number formatting
#' (e.g. 950, 12K, 1.2M).
#'
#' @param thresholds Numeric vector of sorted thresholds (see `compute_value_intervals()`).
#' @return Character vector of `length(thresholds) + 1` labels ("All values" when empty).
#'
#' @export
create_dynamic_labels <- function(thresholds) {
    if (length(thresholds) == 0) {
        return("All values")
    }
    fmt <- scales::label_number(scale_cut = scales::cut_short_scale(), accuracy = 0.1, drop0trailing = TRUE)
    labels <- c(
        paste0("< ", fmt(thresholds[1])),
        if (length(thresholds) > 1) paste0(fmt(thresholds[-length(thresholds)]), " - ", fmt(thresholds[-1])),
        paste0("> ", fmt(thresholds[length(thresholds)]))
    )
    return(labels)
}


#' Build a yearly choropleth of a population indicator
#'
#' Bins the indicator into the given classes, joins it to the shapes and returns a map
#' faceted by YEAR. All shapes are drawn grey underneath, so units without data for a
#' year stay visible.
#'
#' @param population_data Data frame. Population table with YEAR, the ADM columns and the indicator.
#' @param shapes_data sf. Shapes with the ADM columns and geometry.
#' @param population_column Character. Name of the indicator column to map.
#' @param thresholds Numeric vector. Class thresholds (see `compute_value_intervals()`).
#' @param labels Character vector. Class labels (see `create_dynamic_labels()`).
#' @param legend_title Character. Legend title.
#' @param plot_title Character. Plot title.
#' @param palette_values Character vector. Colours; spread evenly over the classes when longer.
#' @param na_fill Character. Fill colour for units without data.
#' @return ggplot object.
#'
#' @export
build_population_choropleth <- function(
    population_data,
    shapes_data,
    population_column,
    thresholds,
    labels,
    legend_title,
    plot_title,
    palette_values,
    na_fill = "#D3D3D3"
) {
    palette_values <- unname(palette_values)
    palette_values <- palette_values[round(seq(1, length(palette_values), length.out = length(labels)))]
    names(palette_values) <- labels

    plot_data <- population_data %>%
        dplyr::mutate(
            CATEGORY = cut(
                .data[[population_column]],
                breaks = c(-Inf, thresholds, Inf),
                labels = labels,
                right = TRUE
            )
        ) %>%
        dplyr::inner_join(
            shapes_data %>% dplyr::select(dplyr::all_of(ADM_COLUMNS)),
            by = ADM_COLUMNS
        ) %>%
        sf::st_as_sf()

    ggplot2::ggplot() +
        ggplot2::geom_sf(data = shapes_data, fill = na_fill, color = "black", linewidth = 0.25) +
        ggplot2::geom_sf(
            data = plot_data,
            ggplot2::aes(fill = CATEGORY),
            color = "black",
            linewidth = 0.25,
            show.legend = TRUE
        ) +
        ggplot2::labs(
            title = plot_title,
            subtitle = "Source : fichier de population fourni par l'utilisateur",
            fill = legend_title
        ) +
        ggplot2::scale_fill_manual(values = palette_values, limits = labels, drop = FALSE, na.value = na_fill) +
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


#' Save a plot as PNG and display it in the notebook
#'
#' Writes the plot to disk and embeds the PNG in the notebook output, instead of
#' rendering the (heavier) plot object itself.
#'
#' @param plot ggplot object.
#' @param output_file Character. Destination PNG path.
#' @param width Numeric. Width in cm.
#' @param height Numeric. Height in cm.
#' @param dpi Numeric. Resolution.
#' @return Character. The saved file path, invisibly.
#'
#' @export
save_and_display_plot <- function(plot, output_file, width = 21, height = 15, dpi = 150) {
    ggplot2::ggsave(
        filename = output_file,
        plot = plot,
        create.dir = TRUE,
        units = "cm",
        width = width,
        height = height,
        dpi = dpi,
        bg = "white"
    )
    log_msg(glue::glue("Figure saved: {output_file}"))
    IRdisplay::display_png(file = output_file)
    invisible(output_file)
}
