#' Prepare spatial/admin objects used in healthcare-access computation.
#'
#' Reprojects shapes, removes invalid geometries, derives non-spatial admin
#' table, and computes country union geometry for clipping/intersection steps.
#'
#' @param spatial_units_data Input spatial units (`sf`).
#' @param country_epsg_degrees Target geographic EPSG code.
#' @return List with cleaned `spatial_units_data`, `admin_data`, and `all_country`.
prepare_spatial_admin_objects <- function(spatial_units_data, country_epsg_degrees) {
    spatial_units_data <- reproject_epsg(spatial_units_data, country_epsg_degrees)

    n_before <- nrow(spatial_units_data)
    spatial_units_data <- spatial_units_data %>%
        dplyr::filter(!is.na(sf::st_is_valid(.)), sf::st_is_valid(.), !sf::st_is_empty(.))
    if (nrow(spatial_units_data) < n_before) {
        log_msg(glue::glue("Dropped {n_before - nrow(spatial_units_data)} spatial unit(s) with null/empty/invalid geometry."))
    }

    admin_data <- data.table::setDT(sf::st_drop_geometry(spatial_units_data))
    all_country <- sf::st_union(spatial_units_data)

    list(
        spatial_units_data = spatial_units_data,
        admin_data = admin_data,
        all_country = all_country
    )
}


#' make circles of a given radius around each point (longitude/latitude) in the sf vector input data
#'
#' @param input_vect: sf vector of spatial points (in any CRS)
#' @param coordinate_colnames: names of the longitude and latitude columns
#' @param epsg_value_degrees: EPSG code for the geographic (degree-based) CRS (eg, for Burkina 4326)
#' @param epsg_value_meters: EPSG code for the projected (meter-based) CRS (eg, for Burkina 3857)
#' @param radius_meters: radius (in meters) of the  coverage area to create around each point, defaults to 5000 (5 km or approx. 60' walk)
#'
#' @return: sf vector of the circle coverages in the degree CRS
#'
#' @details 
#' 1. check that input is in the correct degree CRS (reproject if needed)
#' 2. project it to a meter CRS for distance calculations
#' 3. create circular buffers (coverage radii) around each point
#' 4. reproject the buffer geometries back to the original degree CRS
make_coverage_radii_sf <- function(
  input_vect,
  coordinate_colnames,
  epsg_value_degrees,
  epsg_value_meters,
  radius_meters = 5000
){

  # check CRS and reproject to degree CRS if necessary
  input_vect <- reproject_epsg(input_vect, epsg_value_degrees)
  
  # reproject to a meter CRS
  vect_meters <- st_transform(input_vect, epsg_value_meters)
  
  # create the circles/buffers around each point
  coverage_radii_meters <- st_buffer(vect_meters, dist = radius_meters)
  
  # reproject back to degree CRS for mapping
  coverage_radii_degrees <- st_transform(coverage_radii_meters, epsg_value_degrees)
  
  return(coverage_radii_degrees)
}


#' make a new raster layer aligned with the original raster, where each cell is a specific value if it intersects any buffer in the vector data and another specific value if not
#'
#' @param buffer_vect: vector with the buffer geometries to rasterize
#' @param raster_data: raster to use as the template for resolution and extent
#' @param epsg_value_degrees: EPSG of the target CRS in degrees
#' @param value_inside: value to assign to raster cells that intersect any buffer
#' @param value_outside: value to assign to raster cells that do not intersect any buffer
#'
#' @return raster with cells assigned values based on intersection with the buffer vector
make_rasterized_inclusion_data <- function(
  buffer_vect, 
  raster_data,
  epsg_value_degrees,
  value_inside = 1,
  value_outside = 0
){

  # reproject raster to the correct CRS (degrees)
  raster_data <- project(raster_data, glue("epsg:{epsg_value_degrees}"))
  
  # if buffer CRS differs, reproject buffer to raster CRS
  buffer_vect <- reproject_epsg(buffer_vect, epsg_value_degrees)
  
  # convert sf to terra SpatVector for rasterization
  buffer_vect_terra <- terra::vect(buffer_vect)
  
  # rasterize the buffer: cells inside = value_inside, outside = value_outside
  inclusion_data <- terra::rasterize(
    buffer_vect_terra,
    raster_data,
    field = value_inside,
    background = value_outside
  )
  
  return(inclusion_data)
}


#' Compute total and covered population by administrative unit.
#'
#' Aggregates raster-based total and covered populations over admin polygons,
#' computes percent coverage, and joins results back to admin attributes.
#'
#' @param pop_total_raster Raster of total population.
#' @param pop_covered_raster Raster of covered population.
#' @param adm_raster Rasterized admin-id grid used for zonal aggregation.
#' @param admin_col Admin ID column name used for joins.
#' @param admin_data Admin attributes table.
#' @return Data table with `POP_TOTAL`, `POP_COVERED`, and `PCT_HEALTH_ACCESS`.
compute_population_by_admin <- function(pop_total_raster, pop_covered_raster, adm_raster, admin_col, admin_data) {
    pop_total_by_adm2 <- terra::zonal(
        pop_total_raster,
        adm_raster,
        fun = "sum",
        na.rm = TRUE
    )
    log_msg("Aggregated the total population by spatial units.")

    pop_cov_by_adm2 <- terra::zonal(
        pop_covered_raster,
        adm_raster,
        fun = "sum",
        na.rm = TRUE
    )
    log_msg("Aggregated the covered population by spatial units.")

    adm2_pop_total <- data.table::setDT(as.data.frame(pop_total_by_adm2))
    adm2_pop_covered <- data.table::setDT(as.data.frame(pop_cov_by_adm2))
    output_df <- data.table::merge.data.table(adm2_pop_total, adm2_pop_covered, by = admin_col, all = TRUE)

    if (nrow(output_df) != nrow(adm2_pop_total)) {
        stop("Error: There was an error when computing covered population.")
    }

    output_df$PCT_HEALTH_ACCESS <- output_df$POP_COVERED * 100 / output_df$POP_TOTAL
    data.table::merge.data.table(admin_data, output_df, by = admin_col, all.x = TRUE)
}
