# Shared helpers for snt_dhis2_formatting code notebooks.

# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with CONFIG_PATH, POPULATION_DATA_PATH.
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH='~/workspace', 
    packages=c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate")
) {
        
    # List required pcks
    required_packages <- packages
    install_and_load(required_packages)
    
    # Set environment to load openhexa.sdk from the right environment
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    
    # Import OpenHEXA SDK and make it available in the environment 
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)
    
    return(
        list(
            CONFIG_PATH = file.path(SNT_ROOT_PATH, "configuration"),  
            FORMATTED_DATA_PATH = file.path(SNT_ROOT_PATH, "data", "dhis2", "extracts_formatted")
        )
    )
}


#' Load SNT Configuration File
#' Reads and parses a JSON configuration file.
#' @param snt_config_path Character. Path to the configuration JSON file.
#' @return List containing parsed configuration.
#'
#' @export
load_snt_config <- function(snt_config_path) {

    # config file path 
    config_json <- tryCatch({ fromJSON(snt_config_path) },
                error = function(e) {
                    msg <- glue::glue("[ERROR] Error while loading configuration: {snt_config_path}")
                    cat(msg)
                    stop(msg)
                })
    
    log_msg(paste0("SNT configuration loaded from  : ", snt_config_path))
    return(config_json)    
}


#' Load Dataset File from OpenHEXA
#' Retrieves the latest version of a file from an OpenHEXA dataset.
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @param verbose Bool. Log messages
#' @return Dataframe containing the loaded data.
#'
#' @export
load_dataset_file <- function (dataset_id, filename, verbose=TRUE) {
    data <- tryCatch({ 
            get_latest_dataset_file_in_memory(dataset_id, filename) 
        }, error = function(e) {
            if (verbose) log_msg(glue("[ERROR] Error while loading {filename} file for: {conditionMessage(e)}"), "error")
            stop(msg)
    })

    if (verbose) {
        msg <- glue("{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse=', ')}]")
        log_msg(msg)
    }    
    return(data)
}



build_routine_indicators <- function(routine_data_ind, dhis_indicator_definitions_clean) {
    empty_data_indicators <- c()

    for (indicator in names(dhis_indicator_definitions_clean)) {
        data_element_uids <- dhis_indicator_definitions_clean[[indicator]]
        col_names <- c()

        if (length(data_element_uids) > 0) {
            for (dx in data_element_uids) {
                dx_co <- gsub("\\.", "_", dx)
                if (grepl("_", dx_co)) {
                    col_names <- c(col_names, dx_co)
                } else {
                    if (!any(grepl(dx, colnames(routine_data_ind)))) {
                        msg <- paste0("Data element : ", dx, " of indicator ", indicator, " is missing in the DHIS2 routine data.")
                        log_msg(msg, level = "warning")
                    } else {
                        col_names <- c(col_names, colnames(routine_data_ind)[grepl(dx, colnames(routine_data_ind))])
                    }
                }
            }

            if (length(col_names) == 0) {
                msg <- paste0("No data elements available to build indicator : ", indicator, ", skipped.")
                log_msg(msg, level = "warning")
                empty_data_indicators <- c(empty_data_indicators, indicator)
                next
            }

            msg <- paste0("Building indicator : ", indicator, " -> column selection : ", paste(col_names, collapse = ", "))
            log_msg(msg)

            if (length(col_names) > 1) {
                sums <- rowSums(routine_data_ind[, col_names], na.rm = TRUE)
                all_na <- rowSums(!is.na(routine_data_ind[, col_names])) == 0
                sums[all_na] <- NA
                routine_data_ind[[indicator]] <- sums
            } else {
                routine_data_ind[indicator] <- routine_data_ind[, col_names]
            }
        } else {
            routine_data_ind[indicator] <- NA
            msg <- paste0("Building indicator : ", indicator, " -> column selection : NULL")
            log_msg(msg)
        }
    }

    list(
        routine_data_ind = routine_data_ind,
        empty_data_indicators = empty_data_indicators
    )
}

#
# Claude: Helpers for DHIS2 formatting pipeline.
#

extract_geo_coords <- function(geom_json) {
  if (is.na(geom_json) || !nzchar(geom_json)) {
    return(list(lon = NA_real_, lat = NA_real_, parse_ok = FALSE))
  }

  parsed <- tryCatch(jsonlite::fromJSON(geom_json), error = function(e) NULL)
  coords <- parsed$coordinates

  if (is.null(coords) || length(coords) < 2) {
    return(list(lon = NA_real_, lat = NA_real_, parse_ok = FALSE))
  }

  list(
    lon = suppressWarnings(as.numeric(coords[[1]])),
    lat = suppressWarnings(as.numeric(coords[[2]])),
    parse_ok = TRUE
  )
}

shift_decimal_left_to_right <- function(value, k) {
  if (is.na(value)) return(NA_real_)

  sign_val <- ifelse(value < 0, -1, 1)
  digits_only <- gsub("[^0-9]", "", as.character(abs(value)))
  if (!nzchar(digits_only) || nchar(digits_only) <= k) return(NA_real_)

  shifted <- paste0(substr(digits_only, 1, k), ".", substr(digits_only, k + 1, nchar(digits_only)))
  sign_val * suppressWarnings(as.numeric(shifted))
}

build_coordinate_candidates <- function(lon, lat, max_shift = 2) {
  candidates <- list(
    ORIGINAL = c(lon, lat),
    SWAP = c(lat, lon),
    FLIP_LAT = c(lon, -lat),
    FLIP_LON = c(-lon, lat),
    FLIP_BOTH = c(-lon, -lat),
    SWAP_FLIP_LAT = c(lat, -lon),
    SWAP_FLIP_LON = c(-lat, lon),
    SWAP_FLIP_BOTH = c(-lat, -lon)
  )

  for (k in seq_len(max_shift)) {
    lon_lr <- shift_decimal_left_to_right(lon, k)
    lat_lr <- shift_decimal_left_to_right(lat, k)

    candidates[[paste0("DECIMAL_LR_LON_", k)]] <- c(lon_lr, lat)
    candidates[[paste0("DECIMAL_LR_LAT_", k)]] <- c(lon, lat_lr)
    candidates[[paste0("DECIMAL_LR_BOTH_", k)]] <- c(lon_lr, lat_lr)
    candidates[[paste0("SWAP_DECIMAL_LR_LON_", k)]] <- c(lat_lr, lon)
    candidates[[paste0("SWAP_DECIMAL_LR_LAT_", k)]] <- c(lat, lon_lr)
    candidates[[paste0("SWAP_DECIMAL_LR_BOTH_", k)]] <- c(lat_lr, lon_lr)
  }

  candidates
}

prepare_country_boundary <- function(country_shapes_sf) {
  if (!inherits(country_shapes_sf, "sf")) {
    stop("Country shapes must be an sf object.")
  }

  if (is.na(sf::st_crs(country_shapes_sf))) {
    sf::st_crs(country_shapes_sf) <- 4326
  }

  country_shapes_sf <- sf::st_transform(country_shapes_sf, 4326)
  country_boundary <- sf::st_union(sf::st_geometry(country_shapes_sf))
  sf::st_sf(GEOMETRY = country_boundary)
}

points_within_country_batch <- function(lon_vec, lat_vec, boundary_sf) {
  out <- rep(FALSE, length(lon_vec))
  ok_idx <- which(
    !is.na(lon_vec) &
      !is.na(lat_vec) &
      (abs(lon_vec) <= 180) &
      (abs(lat_vec) <= 90)
  )

  if (length(ok_idx) == 0) return(out)

  pts <- sf::st_as_sf(
    data.frame(LONGITUDE = lon_vec[ok_idx], LATITUDE = lat_vec[ok_idx]),
    coords = c("LONGITUDE", "LATITUDE"),
    crs = 4326
  )

  out[ok_idx] <- as.logical(sf::st_within(pts, boundary_sf, sparse = FALSE)[, 1])
  out
}

fix_coordinate_pair_in_country <- function(lon, lat, boundary_sf, max_shift = 2) {
  if (is.na(lon) || is.na(lat)) {
    return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "MISSING_COORDINATES", VALID = FALSE))
  }

  candidates <- build_coordinate_candidates(lon, lat, max_shift = max_shift)
  candidate_names <- names(candidates)

  m <- matrix(NA_real_, nrow = length(candidate_names), ncol = 2)
  for (j in seq_along(candidate_names)) {
    cand <- candidates[[candidate_names[j]]]
    m[j, 1] <- as.numeric(cand[1])
    m[j, 2] <- as.numeric(cand[2])
  }

  earth_ok <- abs(m[, 1]) <= 180 & abs(m[, 2]) <= 90
  if (!any(earth_ok)) {
    return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "INVALID_NO_MATCH", VALID = FALSE))
  }

  pts <- sf::st_as_sf(
    data.frame(LONGITUDE = m[earth_ok, 1], LATITUDE = m[earth_ok, 2]),
    coords = c("LONGITUDE", "LATITUDE"),
    crs = 4326
  )

  inside <- rep(FALSE, nrow(m))
  inside[earth_ok] <- as.logical(sf::st_within(pts, boundary_sf, sparse = FALSE)[, 1])
  ok_idx <- which(earth_ok & inside)

  if (length(ok_idx) == 0) {
    return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "INVALID_NO_MATCH", VALID = FALSE))
  }

  j <- min(ok_idx)
  list(LONGITUDE = m[j, 1], LATITUDE = m[j, 2], METHOD = candidate_names[j], VALID = TRUE)
}
