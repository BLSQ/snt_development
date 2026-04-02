find_country_parquet_file <- function(dataset_last_version, country_code) {
    parquet_file <- NULL
    files_list <- reticulate::iterate(dataset_last_version$files)
    for (file in files_list) {
        if (endsWith(file$filename, ".parquet")) {
            parquet_file <- paste0(country_code, "_", substring(file$filename, 5))
            print(paste0("Parquet file found: ", parquet_file))
        }
    }
    parquet_file
}

load_worldpop_report_input <- function(dataset_name, filename, label = "dataset file") {
    data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, filename)
        },
        error = function(e) {
            msg <- paste("Error while loading", label, filename, conditionMessage(e))
            cat(msg)
            stop(msg)
        }
    )
    log_msg(paste0(label, " loaded from dataset: ", dataset_name, " dataframe dimensions: ", paste(dim(data), collapse = ", ")))
    data
}


get_comparison_years <- function(worldpop_population, dhis2_population) {
    worldpop_year <- min(worldpop_population$YEAR)
    if (worldpop_year %in% unique(dhis2_population$YEAR)) {
        dhis2_year <- worldpop_year
    } else if (worldpop_year < min(dhis2_population$YEAR)) {
        dhis2_year <- min(dhis2_population$YEAR)
    } else {
        dhis2_year <- max(dhis2_population$YEAR)
    }
    list(worldpop_year = worldpop_year, dhis2_year = dhis2_year)
}


build_adm2_comparison <- function(shapes_data, dhis2_population, worldpop_population, dhis2_year, worldpop_year) {
    dhis2_pop_renamed <- dhis2_population %>%
        dplyr::filter(YEAR == dhis2_year) %>%
        dplyr::select(ADM2_ID, dhis2_value = POPULATION)

    worldpop_pop_renamed <- worldpop_population %>%
        dplyr::filter(YEAR == worldpop_year) %>%
        dplyr::select(ADM2_ID, worldpop_value = POPULATION)

    comparison_df <- dplyr::left_join(shapes_data, dhis2_pop_renamed[, c("ADM2_ID", "dhis2_value")], by = "ADM2_ID")
    dplyr::left_join(comparison_df, worldpop_pop_renamed[, c("ADM2_ID", "worldpop_value")], by = "ADM2_ID")
}


build_adm1_comparison <- function(shapes_data, dhis2_population, worldpop_population) {
    dhis2_shapes_provinces <- shapes_data %>%
        dplyr::group_by(ADM1_ID) %>%
        dplyr::summarise(geometry = sf::st_union(geometry), .groups = "drop")

    dhis2_pop_prov <- dhis2_population %>%
        dplyr::group_by(ADM1_NAME, ADM1_ID) %>%
        dplyr::summarise(dhis2_value = sum(POPULATION, na.rm = TRUE), .groups = "drop")

    worldpop_pop_prov <- worldpop_population %>%
        dplyr::group_by(ADM1_NAME, ADM1_ID) %>%
        dplyr::summarise(worldpop_value = sum(POPULATION, na.rm = TRUE), .groups = "drop")

    comparison_df_prov <- dplyr::left_join(dhis2_shapes_provinces, dhis2_pop_prov, by = c("ADM1_ID"))
    dplyr::left_join(comparison_df_prov, worldpop_pop_prov, by = c("ADM1_ID"))
}
