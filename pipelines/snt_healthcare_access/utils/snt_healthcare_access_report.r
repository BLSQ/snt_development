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
#' choropleth map of raster data with polygon borders
#'
#' @param input_raster spatraster with object to plot
#' @param input_vector sf polygon to overlay as boundaries
#' @param epsg_value_degrees CRS for the map (defaults to 4326)
#' @param low_color color for the low end of the gradient (defaults to "#ead7f8")
#' @param high_color color for the high end of the gradient (defaults to "#2e0044")
#' @param na_color color for missing values
#' @param plot_title title of the map (defaults to NULL)
#' @param plot_subtitle subtitle of the map (defaults to NULL)
#' @param plot_caption plot caption (defaults to NULL)
#'
#' @return ggplot plot
plot_raster_with_boundaries <- function(
    input_raster,
    input_vector,
    epsg_value_degrees = 4326,
    low_color = "#ead7f8",
    high_color = "#2e0044",
    na_color = "#D3D3D3",
    plot_title = NULL,
    plot_subtitle = NULL,
    plot_caption = NULL
) {

  # validate inputs
  if (!inherits(input_raster, "SpatRaster")) {
    stop("The input_raster must be a terra SpatRaster object.")
  }
  if (!inherits(input_vector, "sf")) {
    stop("The input_vector must be an sf object.")
  }
  if (!is.numeric(epsg_value_degrees) || length(epsg_value_degrees) != 1) {
    stop("The CRS value (epsg_value_degrees) must be a single numeric EPSG code (e.g. 4326).")
  }

  target_crs <- paste0("EPSG:", epsg_value_degrees)

  # reproject the raster if needed
  if (!terra::same.crs(input_raster, target_crs)) {
    message("Reprojecting raster to EPSG:", epsg_value_degrees)
    input_raster <- terra::project(input_raster, target_crs)
  }

  # reproject the sf object if needed
  if (sf::st_crs(input_vector)$epsg != epsg_value_degrees) {
    message("Reprojecting vector to EPSG:", epsg_value_degrees)
    input_vector <- sf::st_transform(input_vector, epsg_value_degrees)
  }

  # get the raster layer name
  layer_name <- names(input_raster)[1]

  # plot the map
  raster_plot <- ggplot2::ggplot() +
    tidyterra::geom_spatraster(data = input_raster) +
    ggplot2::geom_sf(
      data = input_vector,
      fill = NA,
      color = "white",
      linewidth = 0.2
    ) +
    ggplot2::scale_fill_gradient(
      low = low_color,
      high = high_color,
      name = NULL,
      na.value = na_color
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

  return(raster_plot)
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
#' @param na_color color for missing values
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
  na_color = "#D3D3D3",
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
      low = low_color,
      high = high_color,
      name = NULL,
      na.value = na_color
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

