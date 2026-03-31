load_spatial_units_data <- function(shapes_file, dhis2_dataset, country_code) {
    if (!is.null(shapes_file) && !is.na(shapes_file) && trimws(shapes_file) != "") {
        custom_shapes_path <- path.expand(shapes_file)
        if (!file.exists(custom_shapes_path)) {
            stop(glue::glue("[ERROR] Custom shapes file was provided but does not exist: {custom_shapes_path}"))
        }

        spatial_units_data <- tryCatch(
            {
                sf::st_read(custom_shapes_path, quiet = TRUE)
            },
            error = function(e) {
                stop(glue::glue(
                    "[ERROR] Error while loading custom shapes file: {custom_shapes_path} [ERROR DETAILS] {conditionMessage(e)}"
                ))
            }
        )

        log_msg(glue::glue("Custom shapes file loaded successfully: {custom_shapes_path}"))
        log_msg(
            "[WARNING] Using a custom shapefile: hierarchy may not align with the extracted DHIS2 pyramid. During data assembly, this mismatch can result in missing values for some organizational units (especially at ADM2 level) if IDs do not match or do not exist in both files.",
            level = "warning"
        )
        return(spatial_units_data)
    }

    spatial_units_data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dhis2_dataset, paste0(country_code, "_shapes.geojson"))
        },
        error = function(e) {
            stop(glue::glue(
                "[ERROR] Error while loading DHIS2 Shapes data for: {paste0(country_code, '_shapes.geojson')} [ERROR DETAILS] {conditionMessage(e)}"
            ))
        }
    )
    log_msg(glue::glue("Default HMIS/NMDR shapes file downloaded successfully from dataset: {dhis2_dataset}"))
    spatial_units_data
}


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
