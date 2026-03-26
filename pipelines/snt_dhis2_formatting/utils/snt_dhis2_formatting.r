# Helpers for DHIS2 formatting pipeline.

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
