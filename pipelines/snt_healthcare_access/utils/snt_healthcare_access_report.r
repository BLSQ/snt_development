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
    geom_sf(data = points_sf_vect, color = "dodgerblue4", size = 0.7) +
    theme_minimal() +
    ggtitle(plot_title)


  return(plot)
}

#' @description
#' choropleth map of population
#'
#' @param input_data sf object with the geometries and data to plot
#' @param target_colname name of column to map
#' @param low_color color for the low end of the gradient (defaults to "#f7fbff")
#' @param high_color color for the high end of the gradient (defaults to"#08306b")
#' @param plot_title title of the plot
#' @param plot_subtitle subtitle of the plot
#' @param plot_caption caption of the plot
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

#' @description
#' choropleth map of the number of FOSAs per administrative unit
#'
#' @param spatial_data sf polygon object with the administrative boundaries
#' @param fosa_data sf point object with the FOSA locations
#' @param epsg_value_degrees CRS for the map
#' @param spatial_data_id_colname column which is the unique identifier of the polygons
#' @param low_color color for the low end of the gradient (defaults to "#5db4ff")
#' @param high_color color for the high end of the gradient (defaults to"#ffd852")
#' @param plot_title title of the plot
#' @param plot_subtitle subtitle of the plot
#' @param plot_caption caption of the plot
#'
#' @return ggplot object
make_fosa_choropleth_map <- function(
  spatial_data,
  fosa_data,
  epsg_value_degrees,
  spatial_data_id_colname,
  low_color = "#5db4ff",
  high_color = "#ffd852",
  plot_title = NULL,
  plot_subtitle = NULL,
  plot_caption = NULL
) {
  
  # make both sf objects have the same CRS
  fosa_data <- reproject_epsg(fosa_data, epsg_value_degrees)
  spatial_data <- reproject_epsg(spatial_data, epsg_value_degrees)
  
  # spatial join: attach polygon attributes to each point
  joined <- sf::st_join(fosa_data, spatial_data, join = sf::st_within)
  
  # compute the point counts per polygon
  point_counts <- joined |>
    sf::st_drop_geometry() |>
    dplyr::group_by(.data[[spatial_data_id_colname]]) |>
    dplyr::summarise(fosa_count = dplyr::n(), .groups = "drop")
  
  # join counts back to the polygon sf object
  spatial_data <- spatial_data |>
    dplyr::left_join(point_counts, by = spatial_data_id_colname) |>
    dplyr::mutate(fosa_count = tidyr::replace_na(fosa_count, 0))
  
  # make plot
  ggplot2::ggplot(data = spatial_data) +
    ggplot2::aes(fill = fosa_count) +
    ggplot2::geom_sf(colour = "white", linewidth = 0.3) +
    ggplot2::scale_fill_gradient(
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