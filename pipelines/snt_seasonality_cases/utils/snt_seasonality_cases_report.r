# Helpers for seasonality-cases reporting notebook.


cases_month_labels_en <- function() {
    c(
        "1" = "January", "2" = "February", "3" = "March", "4" = "April",
        "5" = "May", "6" = "June", "7" = "July", "8" = "August",
        "9" = "September", "10" = "October", "11" = "November", "12" = "December"
    )
}


cases_month_palette <- function() {
    c(
        "January" = "#9E0142",
        "February" = "#D53E4F",
        "March" = "#F46D43",
        "April" = "#FDAE61",
        "May" = "#FEE08B",
        "June" = "#E6F598",
        "July" = "#ABDDA4",
        "August" = "#66C2A5",
        "September" = "#3288BD",
        "October" = "#5E4FA2",
        "November" = "#C51B7D",
        "December" = "#8E0152"
    )
}


make_cases_start_month_plot <- function(
    plot_data,
    season_start_month_col,
    subtitle_text,
    data_source
) {
    if (!season_start_month_col %in% names(plot_data)) {
        return(NULL)
    }

    month_labels <- cases_month_labels_en()
    month_colors <- cases_month_palette()
    plot_data$START_MONTH_FACTOR <- factor(
        as.character(plot_data[[season_start_month_col]]),
        levels = as.character(1:12),
        labels = month_labels
    )

    ggplot2::ggplot(plot_data) +
        ggplot2::geom_sf(ggplot2::aes(fill = .data$START_MONTH_FACTOR), color = "black", size = 0.1) +
        ggplot2::scale_fill_manual(
            values = month_colors,
            na.value = "white",
            drop = FALSE,
            guide = ggplot2::guide_legend(nrow = 1)
        ) +
        ggplot2::theme_void() +
        ggplot2::labs(
            title = "Starting month of seasonal block",
            subtitle = subtitle_text,
            caption = paste("Data source:", data_source),
            fill = NULL
        ) +
        ggplot2::theme(
            plot.title = ggplot2::element_text(face = "bold", size = 13),
            plot.subtitle = ggplot2::element_text(size = 7),
            legend.position = "bottom",
            legend.text = ggplot2::element_text(size = 9)
        )
}


make_cases_proportion_plot <- function(
    plot_data,
    subtitle_text,
    data_source,
    proportion_col = "CASES_PROPORTION"
) {
    if (!proportion_col %in% names(plot_data)) {
        return(NULL)
    }

    proportion_values <- suppressWarnings(as.numeric(plot_data[[proportion_col]]))
    plot_data$PROPORTION_CAT <- cut(
        proportion_values,
        breaks = c(-Inf, 0, 0.2, 0.4, 0.6, 0.8, 1.0, Inf),
        labels = c("< 0%", "0 - 20%", "20 - 40%", "40 - 60%", "60 - 80%", "80 - 100%", "> 100%"),
        include.lowest = TRUE
    )

    proportion_palette <- c(
        "< 0%" = "#455A64",
        "0 - 20%" = "#43A047",
        "20 - 40%" = "#8BC34A",
        "40 - 60%" = "#FDD835",
        "60 - 80%" = "#FF9800",
        "80 - 100%" = "#E65100",
        "> 100%" = "#B71C1C"
    )

    ggplot2::ggplot(plot_data) +
        ggplot2::geom_sf(ggplot2::aes(fill = .data$PROPORTION_CAT), color = "black", size = 0.1) +
        ggplot2::scale_fill_manual(
            values = proportion_palette,
            limits = names(proportion_palette),
            na.value = "white",
            drop = FALSE
        ) +
        ggplot2::theme_void() +
        ggplot2::labs(
            title = "Percentage of annual cases falling within the seasonal block",
            subtitle = subtitle_text,
            caption = paste("Data source:", data_source),
            fill = NULL
        ) +
        ggplot2::theme(
            legend.position = "bottom",
            plot.title = ggplot2::element_text(size = 13, face = "bold"),
            plot.subtitle = ggplot2::element_text(size = 7),
            legend.text = ggplot2::element_text(size = 10)
        ) +
        ggplot2::guides(fill = ggplot2::guide_legend(nrow = 1))
}
