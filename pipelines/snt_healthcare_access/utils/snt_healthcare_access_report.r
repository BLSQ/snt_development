# Helpers for the access to healthcare pipeline report

#' @description
#' map overlaying a) healthcare unit locations, b) administrative boundaries, and c) buffer zones around healthcare units, projected to a common CRS
#'
#' @param admin_unit_vect: sf vector of administrative boundaries
#' @param points_sf_vect: sf vector with coordinate columns of healthcare units
#' @param buffer_vect: sf vector of buffer zones around healthcare units
#' @param epsg_value_degrees: EPSG code (in degrees) for CRS
#' @param plot_title: title of plot
#'
#' @return ggplot object showing the spatial overlay
make_overlaid_sf_plot <- function(  
  admin_unit_vect,
  points_sf_vect,
  buffer_vect,
  epsg_value_degrees,
  plot_title
){

  # get all 3 data objects to the same projection

  # a) ensure the healthcare locations have the proper projection
  points_sf_vect <- reproject_epsg(points_sf_vect, epsg_value_degrees)

  # b) ensure the admin geo data has the proper projection
  admin_unit_vect <- reproject_epsg(admin_unit_vect, epsg_value_degrees)

  # c) ensure the buffer data has the proper projection
  buffer_vect <- reproject_epsg(buffer_vect, epsg_value_degrees)

  plot <- ggplot() +
    geom_sf(data = admin_unit_vect, fill = "gray95", color = "black") +
    geom_sf(data = buffer_vect, fill = "dodgerblue", alpha = 0.3) +
    geom_sf(data = points_sf_vect, color = "dodgerblue4", size = 0.5) +
    theme_minimal() +
    ggtitle(plot_title)


  return(plot)
}

#' @description
#' choropleth map of population
#'
#' @param input_data   An sf object containing the geometries and data to plot.
#' @param target_colname  A string with the name of the numeric column to map.
#' @param low_color    Color used for the low end of the gradient (default: "#f7fbff").
#' @param high_color   Color used for the high end of the gradient (default: "#08306b").
#' @param plot_title   Title string displayed above the map.
#' @param plot_subtitle  Subtitle string displayed below the title.
#' @param plot_caption  Caption string displayed at the bottom of the map.
#'
#' @return ggplot object
make_population_choropleth_map <- function(
    input_data,
    target_colname,
    low_color    = "#f7fbff",
    high_color   = "#08306b",
    plot_title   = NULL,
    plot_subtitle = NULL,
    plot_caption = NULL
) {
  # validate inputs
  if (!inherits(input_data, "sf")) {
    stop("The data to plot must be an sf object")
  }
  if (!target_colname %in% names(input_data)) {
    stop("Population column is not part of the data")
  }
  if (!is.numeric(input_data[[target_colname]])) {
    stop("Population column must be numeric")
  }
 
  # plot
  ggplot(data = input_data) +
    geom_sf(
      aes(fill = .data[[target_colname]]),
      color = "white",
      linewidth = 0.2
    ) +
    
    scale_fill_gradient(
      low  = low_color,
      high = high_color,
      name = NULL,
      na.value = "#D3D3D3"
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
}