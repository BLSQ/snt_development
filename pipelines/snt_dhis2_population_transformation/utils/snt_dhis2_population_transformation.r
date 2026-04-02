bootstrap_population_transformation_context <- function(
    root_path = "~/workspace",
    required_packages = c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate", "rlang"),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    population_data_path <- file.path(root_path, "data", "dhis2", "population_transformed")
    dir.create(population_data_path, recursive = TRUE, showWarnings = FALSE)

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }
    assign("openhexa", openhexa, envir = .GlobalEnv)

    list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        POPULATION_DATA_PATH = population_data_path,
        openhexa = openhexa
    )
}

get_total_population_reference <- function(config_json, adjust_with_untotals = FALSE) {
    if (!adjust_with_untotals) {
        return(NULL)
    }

    total_population_reference <- config_json$DHIS2_DATA_DEFINITIONS$POPULATION_DEFINITIONS[["TOTAL_POPULATION_REF"]]
    if (is.null(total_population_reference)) {
        log_msg("No total population reference found in 'snt_config'. Adjustment will be skipped.", "warning")
        return(NULL)
    }

    log_msg(glue::glue("Using total population reference from SNT configuration file: {total_population_reference}"))
    total_population_reference
}


apply_total_population_scaling <- function(dhis2_population, total_population_reference) {
    if (is.null(total_population_reference)) {
        return(dhis2_population)
    }

    year_totals <- dhis2_population %>%
        dplyr::group_by(YEAR) %>%
        dplyr::summarise(total_year_pop = sum(POPULATION, na.rm = TRUE), .groups = "drop") %>%
        dplyr::mutate(scaling_factor = total_population_reference / total_year_pop)

    dhis2_population_scaled <- dhis2_population %>%
        dplyr::left_join(year_totals, by = "YEAR") %>%
        dplyr::mutate(POPULATION_SCALED = round(POPULATION * scaling_factor)) %>%
        dplyr::select(-total_year_pop, -scaling_factor)

    for (i in seq_len(nrow(year_totals))) {
        row <- year_totals[i, ]
        dhis2_total <- sum(dhis2_population_scaled[dhis2_population_scaled$YEAR == row$YEAR, "POPULATION"], na.rm = TRUE)
        dhis2_total_scd <- sum(dhis2_population_scaled[dhis2_population_scaled$YEAR == row$YEAR, "POPULATION_SCALED"], na.rm = TRUE)
        log_msg(glue::glue("DHIS2 population year {row$YEAR} ({dhis2_total}) scaled: {dhis2_total_scd} (scaling_factor={round(row$scaling_factor, 3)})."))
    }

    dhis2_population_scaled
}


project_population_with_growth <- function(dhis2_population, growth_factor, reference_year, n_years_past = 6, n_years_future = 6) {
    population_column <- ifelse(("POPULATION_SCALED" %in% colnames(dhis2_population)), "POPULATION_SCALED", "POPULATION")
    columns_selection <- c("YEAR", "ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID", population_column)

    if (is.null(growth_factor)) {
        pop_result <- dhis2_population[order(dhis2_population$YEAR), columns_selection]
        pop_result <- pop_result %>% dplyr::rename(POPULATION = !!rlang::sym(population_column))
        return(pop_result)
    }

    if (is.null(reference_year) || !(reference_year %in% dhis2_population$YEAR)) {
        not_found <- reference_year
        reference_year <- max(dhis2_population$YEAR)

        if (!is.null(not_found)) {
            log_msg(
                glue::glue("Reference year {not_found} is not present in the population data, using last year: {reference_year}."),
                "warning"
            )
        }
    }

    log_msg(glue::glue("Applying growth factor {growth_factor} to project {tolower(population_column)} from reference year {reference_year}."))
    projection_years_backward <- seq(reference_year - 1, reference_year - n_years_past, by = -1)
    projection_years_forward <- seq(reference_year + 1, reference_year + n_years_future)

    dhis2_population_reference <- dhis2_population[dhis2_population$YEAR == reference_year, columns_selection]
    pop_result <- dhis2_population_reference
    population_forward <- dhis2_population_reference
    population_backward <- dhis2_population_reference

    for (year in projection_years_forward) {
        population_forward[["YEAR"]] <- year
        population_forward[[population_column]] <- round(population_forward[[population_column]] * (1 + growth_factor))
        pop_result <- rbind(pop_result, population_forward)
    }

    for (year in projection_years_backward) {
        population_backward[["YEAR"]] <- year
        population_backward[[population_column]] <- round(population_backward[[population_column]] / (1 + growth_factor))
        pop_result <- rbind(pop_result, population_backward)
    }

    pop_result <- pop_result[order(pop_result$YEAR), ]
    pop_result %>% dplyr::rename(POPULATION = !!rlang::sym(population_column))
}


add_population_disaggregations <- function(pop_result, pop_disagg) {
    if (is.null(pop_disagg) || length(pop_disagg) == 0) {
        message("No population disaggregations defined.")
        return(pop_result)
    }

    for (name in names(pop_disagg)) {
        value <- pop_disagg[[name]]
        log_msg(glue::glue("Adding disaggregation: {name}, Factor: {value}"))
        pop_result[[toupper(name)]] <- round(pop_result[["POPULATION"]] * value)
    }

    pop_result
}
