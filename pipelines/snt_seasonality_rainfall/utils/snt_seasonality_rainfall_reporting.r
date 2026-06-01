# Helpers for rainfall seasonality reporting notebook.

#' @description
#' plot the start month of the rainy season with custom color palette
#'
#' @param plot_data a data frame with spatial data and a column for the start month of the rainy season
#' @param season_start_month_col column that gives the start month values
#' @param color_vector vector of colors to be used for January-December
#' @param plot_title text for the plot main title
#' @param plot_subtitle text for the plot subtitle
#' @param plot_caption text for the plot caption
#' @param missing_label label for the missing values (non seasonal areas)
#'
#' @return a ggplot object or NULL if season_start_month_col is not found in plot_data
make_rainfall_start_month_plot <- function(
    plot_data,
    season_start_month_col,
    color_vector,
    color_labels,
    plot_title,
    plot_subtitle,
    plot_caption,
    missing_label = "Non saisonnier"
) {
  
  # validate inputs
  stopifnot(
    "seasonality start plot_data must be an sf object" = inherits(plot_data, "sf"),
    "seasonality start month column must be a single string" = is.character(season_start_month_col) && length(season_start_month_col) == 1,
    "seasonality start month column not found in plot_data" = season_start_month_col %in% names(plot_data),
    "seasonality start color_vector must have exactly 12 elements" = length(color_vector) == 12,
    "seasonality start color_labels must have exactly 12 elements" = length(color_labels) == 12
  )
 
  month_vals <- plot_data[[season_start_month_col]]
  valid_vals <- month_vals[!is.na(month_vals)]
  if (!all(valid_vals %in% 1:12)) {
    stop("Column '", season_start_month_col,
        "' contains values outside 1–12: ",
        paste(sort(unique(valid_vals[!valid_vals %in% 1:12])), collapse = ", "))
  }
 
  # make a factor column with ordered levels 1–12 for the months
  # NA stays NA so it gets the na.value color in scale_fill_manual
  plot_data <- plot_data |>
    dplyr::mutate(
      .month_factor = factor(.data[[season_start_month_col]], levels = 1:12)
    )
 
  # make color / label scales including only the months that are in the data
  present_months <- sort(unique(as.integer(
    levels(droplevels(plot_data$.month_factor))
  )))
  present_months <- present_months[present_months %in% 1:12]
 
  scale_values <- stats::setNames(color_vector[present_months],
                                  as.character(present_months))
  scale_breaks <- as.character(present_months)
  scale_labels <- color_labels[present_months]
 
  # make the plot
  seasonality_start_plot <- ggplot2::ggplot(data = plot_data) +
 
    # geo layer
    ggplot2::geom_sf(
      ggplot2::aes(fill = .month_factor),
      colour = "white",
      linewidth = 0.15
    ) +
 
    # discrete color scale with NA handling
    ggplot2::scale_fill_manual(
      values = scale_values,
      breaks = scale_breaks,
      labels = scale_labels,
      na.value = "#D3D3D3",
      name = NULL,
      guide = ggplot2::guide_legend(
        title = NULL,
        override.aes = list(colour = "white", linewidth = 0.3),
        label.position = "right",
        keywidth = ggplot2::unit(0.9, "lines"),
        keyheight = ggplot2::unit(0.9, "lines")
      )
    ) +
 
    # titles
    ggplot2::labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
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
 
  # append NA note to caption if any missing values exist in the data, so they also appear in the legend
  if (anyNA(month_vals)) {
    seasonality_start_plot <- seasonality_start_plot + ggplot2::labs(
      caption = paste0(plot_caption,
        if (nchar(plot_caption) > 0) "\n" else "",
        "\u25A0 : ", missing_label)
      )
  }
 
  return(seasonality_start_plot)
}


#' @description
#' rainfall proportion plot with custom color palette
#' 
#' @param plot_data a data frame containingwith spatial data and a column for rainfall proportion
#' @param subtitle_text text for the plot subtitle
#' @param data_source source of the data for the plot caption
#' @param proportion_colname column of the rainfall proportion values; default is RAIN_PROPORTION
#'
#' @return a ggplot object or NULL if proportion_col is not found in plot_data
make_rainfall_proportion_plot <- function(
    plot_data,
    proportion_colname,
    plot_title,
    plot_subtitle,
    plot_caption,
    color_vector = c(
        "#C8DDD9",
        "#9DBFBB",
        "#5E9490",
        "#2E6460",
        "#264A48"
    )
) {
 
  #validate inputs
  if (!inherits(plot_data, "sf")) {
    stop("plot_data must be an sf object.")
  }
 
  if (!proportion_colname %in% names(plot_data)) {
    warning("The rainfall proportion column not found in plot_data. Returning NULL.")
    return(NULL)
  }
 
  prop_vals <- plot_data[[proportion_colname]]
 
  if (!is.numeric(prop_vals)) {
    stop("The rainfall proportion column must be numeric.")
  }
 
  # bin into five ordered categories
  bin_breaks <- c(-Inf, 0.20, 0.40, 0.60, 0.80, Inf)
  bin_labels <- c("<20%", "20 - 40%", "40 - 60%", "60 - 80%", ">80%")
 
  plot_data <- plot_data |>
    dplyr::mutate(
      .prop_rescaled = prop_vals,
      .rain_category = cut(
        .prop_rescaled,
        breaks         = bin_breaks,
        labels         = bin_labels,
        include.lowest = TRUE,
        right          = FALSE
      )
    )
 
  # make plot
  rain_proportion_plot <- ggplot2::ggplot(data = plot_data) +
 
    # geo layer
    ggplot2::geom_sf(
      ggplot2::aes(fill = .rain_category),
      colour    = "white",
      linewidth = 0.15
    ) +
 
    # discrete colors and handle missings
    ggplot2::scale_fill_manual(
      values   = color_vector,
      breaks   = bin_labels,
      na.value = "#D3D3D3",
      name     = NULL,
      drop     = TRUE,
      guide    = ggplot2::guide_legend(
        title          = NULL,
        override.aes   = list(colour = "white", linewidth = 0.3),
        label.position = "right",
        keywidth       = ggplot2::unit(0.9, "lines"),
        keyheight      = ggplot2::unit(0.9, "lines")
      )
    ) +
 
    # titles
    ggplot2::labs(
      title    = plot_title,
      subtitle = plot_subtitle,
      caption  = plot_caption
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
 
  # add NA note to caption if any missing values in data
  if (anyNA(prop_vals)) {
    rain_proportion_plot <- rain_proportion_plot + ggplot2::labs(
      caption = paste0(data_source,
        if (nchar(data_source) > 0) "\n" else "",
        "\u25A0 Non saisonnier")
    )
  }
 
  return(rain_proportion_plot)
}
