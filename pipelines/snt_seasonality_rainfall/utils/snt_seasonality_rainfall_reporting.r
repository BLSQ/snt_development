# Helpers for rainfall seasonality reporting notebook.

#' @description make English language labels for month numbers
rainfall_month_labels_en <- function() {
    c(
        "1" = "January", "2" = "February", "3" = "March", "4" = "April",
        "5" = "May", "6" = "June", "7" = "July", "8" = "August",
        "9" = "September", "10" = "October", "11" = "November", "12" = "December"
    )
}

#' @description make French language labels for month numbers
rainfall_month_labels_fr <- function() {
    c(
        "1" = "Janvier", "2" = "Février", "3" = "Mars", "4" = "Avril",
        "5" = "Mai", "6" = "Juin", "7" = "Juillet", "8" = "Août",
        "9" = "Septembre", "10" = "Octobre", "11" = "Novembre", "12" = "Décembre"
    )
}

#' @description make a palette for month names in French
rainfall_month_palette_fr <- function() {
    c(
        "Janvier" = "#1E90FF",
        "Février" = "#2060E8",
        "Mars" = "#5030D0",
        "Avril" = "#8020B8",
        "Mai" = "#B01890",
        "Juin" = "#D41E55",
        "Juillet" = "#E83020",
        "Août" = "#F56800",
        "Septembre" = "#F5A800",
        "Octobre" = "#C8C800",
        "Novembre" = "#60C820",
        "Décembre" = "#20B860"
    )
}

#' @description make a palette for month names in English
rainfall_month_palette_en <- function() {
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

#' @description
#' plot the start month of the rainy season with custom color palette
#'
#' @param plot_data a data frame with spatial data and a column for the start month of the rainy season
#' @param season_start_month_col column that gives the start month values
#' @param subtitle_text text for the plot subtitle
#' @param data_source source of the data for the plot caption
#'
#' @return a ggplot object or NULL if season_start_month_col is not found in plot_data
make_rainfall_start_month_plot <- function(
    plot_data,
    season_start_month_col,
    subtitle_text,
    data_source
) {
    if (!season_start_month_col %in% names(plot_data)) {
        return(NULL)
    }

    month_labels <- rainfall_month_labels_fr()
    month_colors <- rainfall_month_palette_fr()
    plot_data$START_MONTH_FACTOR <- factor(
        as.character(plot_data[[season_start_month_col]]),
        levels = as.character(1:12),
        labels = month_labels
    )

    ggplot2::ggplot(plot_data) +
        ggplot2::geom_sf(ggplot2::aes(fill = .data$START_MONTH_FACTOR), color = "black", size = 0.1) +
        ggplot2::scale_fill_manual(
            values = month_colors,
            na.value="#D3D3D3",
            drop = FALSE,
            guide = ggplot2::guide_legend(nrow = 1)
        ) +
        ggplot2::theme_void() +
        ggplot2::labs(
            title = "Début de la saison pluvieuse",
            subtitle = subtitle_text,
            caption = paste("Données:", data_source),
            fill = NULL
        ) +
        ggplot2::theme(
            plot.title = ggplot2::element_text(face = "bold", size = 10),
            plot.subtitle = ggplot2::element_text(size = 6),
            legend.position = "bottom",
            legend.text = ggplot2::element_text(size = 8)
        ) +
        guides(fill=guide_legend(nrow = 2))
}


#' @description
#' rainfall proportion plot with custom color palette
#' 
#' @param plot_data a data frame containingwith spatial data and a column for rainfall proportion
#' @param subtitle_text text for the plot subtitle
#' @param data_source source of the data for the plot caption
#' @param proportion_col column of the rainfall proportion values; default is RAIN_PROPORTION
#'
#' @return a ggplot object or NULL if proportion_col is not found in plot_data
make_rainfall_proportion_plot <- function(
    plot_data,
    subtitle_text,
    data_source,
    proportion_col = "RAIN_PROPORTION"
) {
    if (!proportion_col %in% names(plot_data)) {
        return(NULL)
    }

    proportion_values <- suppressWarnings(as.numeric(plot_data[[proportion_col]]))
    plot_data$PROPORTION_CAT <- cut(
        proportion_values,
        breaks = c(-Inf, 0.2, 0.4, 0.6, 0.8, Inf),
        labels = c("<20%", "20 - 40%", "40 - 60%", "60 - 80%", ">80%"),
        include.lowest = TRUE
    )

    proportion_palette <- c(
        "<20%" = "#C8DDD9",
        "20 - 40%" = "#9DBFBB",
        "40 - 60%" = "#5E9490",
        "60 - 80%" = "#2E6460",
        ">80%" = "#264A48"
    )
  
    ggplot2::ggplot(plot_data) +
        ggplot2::geom_sf(ggplot2::aes(fill = .data$PROPORTION_CAT), color = "black", size = 0.1) +
        ggplot2::scale_fill_manual(
            values = proportion_palette,
            na.value="#D3D3D3",
            limits = names(proportion_palette),
            drop = FALSE
        ) +
        ggplot2::theme_void() +
        ggplot2::labs(
            title = "Précipitations durant la saison pluvieuse (%)",
            subtitle = subtitle_text,
            caption = paste("Données:", data_source),
            fill = NULL
        ) +
        ggplot2::theme(
            legend.position = "bottom",
            plot.title = ggplot2::element_text(size = 10, face = "bold"),
            plot.subtitle = ggplot2::element_text(size = 6),
            legend.text = ggplot2::element_text(size = 8)
        ) +
        ggplot2::guides(fill = ggplot2::guide_legend(ncol = 3))
}
