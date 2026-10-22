# Store code to be sourced in the notebook in this same directory, so that the main notebook
# only shows the code relevant to the analysis, and not the boring routine setup, import and export.
# Each piece of code is wrapped in a function to keep the notebook clean.

message("This step sets up the environment for the DHIS2 incidence pipeline, including paths, config, and utility functions.
It basically handles all the boring stuff so that you can focus on the code that matters :)")

# Load base functions
source(file.path("~/workspace/code", "snt_utils.r"))


#' Resolve the routine data filename suffix
#'
#' Maps the global `ROUTINE_DATA_CHOICE` to the filename suffix of the routine file:
#' "raw" -> `_routine.parquet`, "raw_without_outliers" -> `_routine_outliers_removed.parquet`,
#' any other value -> `_routine_outliers_imputed.parquet`.
#'
#' @return Character. Filename suffix, without the country code prefix.
#' @export
resolve_routine_filename <- function() {
    if (ROUTINE_DATA_CHOICE == "raw") {
        return("_routine.parquet")
    }
    removed_status <- if (ROUTINE_DATA_CHOICE == "raw_without_outliers") "removed" else "imputed"
    glue::glue("_routine_outliers_{removed_status}.parquet")
}


#' Select the routine dataset and filename
#'
#' Uses the global `ROUTINE_DATA_CHOICE`: "raw" reads from `DHIS2_DATASET_FORMATTED`, any other
#' choice from `DHIS2_OUTLIERS_IMPUTATION`. The filename is the global `COUNTRY_CODE` followed by
#' the suffix from `resolve_routine_filename()`.
#'
#' @param config_json List. Parsed SNT configuration, with `SNT_DATASET_IDENTIFIERS`.
#' @return List with `routine_dataset_name` and `routine_filename` (both character).
#' @export
select_routine_dataset_and_filename <- function(config_json) {
    routine_dataset_name <- if (ROUTINE_DATA_CHOICE == "raw") {
        config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED
    } else {
        config_json$SNT_DATASET_IDENTIFIERS$DHIS2_OUTLIERS_IMPUTATION
    }
    routine_filename <- paste0(COUNTRY_CODE, resolve_routine_filename())

    log_msg(glue::glue("Selected routine dataset: {routine_dataset_name}, filename: {routine_filename}"))

    list(routine_dataset_name = routine_dataset_name, routine_filename = routine_filename)
}


#' Check that the fixed columns are present in the routine data
#'
#' Compares the columns of `dhis2_routine` with the global `fixed_cols` and logs a warning
#' listing any missing ones. Does not stop the run.
#'
#' @param dhis2_routine Data frame. Routine data as loaded from the dataset.
#' @return Invisible NULL. Called for its side effects.
#' @export
check_fixed_cols_in_routine <- function(dhis2_routine) {
    actual_cols <- colnames(dhis2_routine)
    missing_cols <- setdiff(fixed_cols, actual_cols)
    if (length(missing_cols) == 0) {
        log_msg(glue::glue("All expected 'fixed' columns present."))
    } else {
        log_msg(glue::glue("🚨 Missing Columns: {paste(missing_cols, collapse = ', ')}"), "warning")
    }
}

#' Check that the DHIS2 indicator columns are present in the routine data
#'
#' Compares the columns of `dhis2_routine` with the global `DHIS2_INDICATORS` and logs a warning
#' listing any missing ones. Does not stop the run.
#'
#' @param dhis2_routine Data frame. Routine data as loaded from the dataset.
#' @return Invisible NULL. Called for its side effects.
#' @export
check_dhis2_indicators_cols_in_routine <- function(dhis2_routine) {
    actual_cols <- colnames(dhis2_routine)
    missing_cols <- setdiff(DHIS2_INDICATORS, actual_cols)
    if (length(missing_cols) == 0) {
        log_msg(glue::glue("All DHIS2 indicators present in 'dhis2_routine'."))
    } else {
        log_msg(glue::glue("🚨 Missing DHIS2 INDICATORS: {paste(missing_cols, collapse = ', ')}"), "warning")
    }
}

#' Check that the PRES column can be used for N1
#'
#' Runs only when `N1_METHOD == "PRES"`: stops the run if `dhis2_routine` has no `PRES` column,
#' or if `PRES` contains only missing values. Expects `dhis2_routine` after
#' `prepare_disaggregated_indicators()`, so with a disaggregation `PRES` holds the values of
#' `PRES_{DISAGGREGATION_SELECTION}`, and the error message names that column. Otherwise prints a
#' note (notebook only, not logged).
#'
#' @param dhis2_routine Data frame. Routine data, after the disaggregation mapping.
#' @param N1_METHOD Character. "SUSP-TEST" or "PRES".
#' @param DISAGGREGATION_SELECTION Character or NULL. Population group, e.g. "UNDER_5".
#' @return Invisible NULL. Called for its side effects.
#' @export
check_pres_col <- function(dhis2_routine, N1_METHOD, DISAGGREGATION_SELECTION) {
    if (N1_METHOD == "PRES") {
        pres_col <- if (is.null(DISAGGREGATION_SELECTION)) "PRES" else glue::glue("PRES_{DISAGGREGATION_SELECTION}")
        if (!"PRES" %in% names(dhis2_routine)) {
            stop(glue::glue("[ERROR] Column `{pres_col}` missing from routine data. Cannot calculate N1 with 'PRES'."))
        }
        if (all(is.na(dhis2_routine$PRES))) {
            stop(glue::glue("[ERROR] Column `{pres_col}` contains only missing values. Cannot calculate N1 with 'PRES'."))
        }
        log_msg(glue::glue("Column `{pres_col}` is present and contains values. Proceeding."))
    } else {
        # This is just for the nb, no need to log in pipeline run
        print("N1_METHOD is not set to 'PRES'. No need to check for `PRES` column.")
    }
    invisible(NULL)
}

#' Load population data from the selected dataset
#'
#' Loads {country_code}_population.parquet from the latest version of the dataset resolved
#' by pipeline.py from the population selection (DHIS2, User-provided or Population-transformed).
#'
#' @param country_code Character. Country code used as the filename prefix.
#' @param population_dataset_id Character. Dataset ID of the selected population source.
#' @return Data frame of ADM2 x yearly population.
#' @export
load_population_data <- function(country_code, population_dataset_id) {
    dhis2_population_adm2 <- load_dataset_file(
        dataset_id=population_dataset_id, 
        filename=paste0(country_code, "_population.parquet")
    )
    log_msg(glue::glue("Population data loaded from {population_dataset_id}."))
    return(dhis2_population_adm2)
}

# --- DISAGGREGATION LOGIC --- --- --- --- --- --- ---

#' Map disaggregated indicator columns to the generic names
#'
#' When `DISAGGREGATION_SELECTION` is set, copies `{SUSP|PRES}_{SELECTION}`, `TEST_{SELECTION}`
#' and `CONF_{SELECTION}` into `SUSP` (or `PRES`), `TEST` and `CONF`, so the rest of the notebook
#' uses the generic names. The method prefix is `SUSP` for "SUSP-TEST" and `PRES` for "PRES".
#' Stops the run if any of the disaggregated columns is missing. When `DISAGGREGATION_SELECTION`
#' is NULL, the data is returned unchanged.
#'
#' Sets the globals `DISAGGREGATED_INDICATORS_FOUND` (TRUE when the mapping was applied, else
#' FALSE) and `target_colnames` (the disaggregated column names looked for).
#'
#' @param dhis2_routine Data frame. Routine data as loaded from the dataset.
#' @param DISAGGREGATION_SELECTION Character or NULL. Population group, e.g. "UNDER_5".
#' @param N1_METHOD Character. "SUSP-TEST" or "PRES".
#' @return Data frame. `dhis2_routine`, with the generic indicator columns overwritten when the
#'   mapping is applied.
#' @export
prepare_disaggregated_indicators <- function(dhis2_routine, DISAGGREGATION_SELECTION, N1_METHOD) {

  # Initial log msg
  log_msg(glue::glue("Starting preparation of disaggregated indicators for selection '{DISAGGREGATION_SELECTION}' and N1_METHOD '{N1_METHOD}'.", .null = "none"))

  # Initialize the flag locally
  DISAGGREGATED_INDICATORS_FOUND <<- FALSE # AVOID (!)

  if (!is.null(DISAGGREGATION_SELECTION) && N1_METHOD %in% c("SUSP-TEST", "PRES")) {
    # Determine the expected column names based on the disaggregation selection and method
    prefix_method <- ifelse(N1_METHOD == "SUSP-TEST", "SUSP", "PRES")
    prefix_fixed <- c("TEST", "CONF") 
    prefix_all    <- c(prefix_method, prefix_fixed) 
    target_colnames <<- glue::glue("{prefix_all}_{DISAGGREGATION_SELECTION}")
    log_msg(glue::glue("Looking for disaggregated indicators columns in routine data: {paste(target_colnames, collapse=', ')}"))
    
    if (all(target_colnames %in% colnames(dhis2_routine))) {
      # Map the disaggregated columns (e.g., SUSP_UNDER_5) to generic names (e.g., SUSP) so that 
      # the rest of the pipeline can use them without needing to know about the disaggregation
      dhis2_routine[prefix_all] <- dhis2_routine[target_colnames]
      
      for (col in target_colnames) {
        log_msg(glue::glue("Indicator Disaggregation: Successfully mapped indicator {col}"))
      }    
      
      # Signal success for the next code block
      DISAGGREGATED_INDICATORS_FOUND <<- TRUE 
    } else {
      missing_cols <- setdiff(target_colnames, colnames(dhis2_routine))
      log_msg(glue::glue("Indicator Disaggregation: Disaggregation on '{DISAGGREGATION_SELECTION}' failed."), "warning")
      log_msg(glue::glue("Indicator Disaggregation: Missing columns in routine dataset: {paste(missing_cols, collapse = ', ')}"), "warning")
      msg <- glue::glue("[ERROR] 🛑 Indicator Disaggregation: Required columns for disaggregation '{DISAGGREGATION_SELECTION}' are missing.")  
      stop(msg)
    }
  } else {
    # Print just in nb (not in pipeline logs)
    print("Indicator Disaggregation: No disaggregation applied based on the current configuration.")
  }
    
  return(dhis2_routine)
}

#' Select the population column matching the disaggregation
#'
#' When `DISAGGREGATED_INDICATORS_FOUND` is TRUE, copies `POP_{DISAGGREGATION_SELECTION}` into
#' `POPULATION` and sets the global `POPULATION_SELECTION` to that column name. Stops the run if
#' the column is not in the population data. When FALSE, the data is returned unchanged and its
#' existing `POPULATION` column is used.
#'
#' @param dhis2_population_adm2 Data frame. ADM2 x yearly population.
#' @param DISAGGREGATED_INDICATORS_FOUND Logical. Whether disaggregated indicators were mapped.
#' @param DISAGGREGATION_SELECTION Character or NULL. Population group, e.g. "UNDER_5".
#' @return Data frame. `dhis2_population_adm2`, with `POPULATION` set from the selected column.
#' @export
select_population_column <- function(dhis2_population_adm2, DISAGGREGATED_INDICATORS_FOUND, DISAGGREGATION_SELECTION) {
  if (DISAGGREGATED_INDICATORS_FOUND) { 
      POPULATION_SELECTION <<- paste0("POP_", DISAGGREGATION_SELECTION) # GP: Make this a global var so it can be used downstream
      if (POPULATION_SELECTION %in% colnames(dhis2_population_adm2)) {   
        # The selected column is assigned to POPULATION col so that later code can use it generically
        dhis2_population_adm2$POPULATION <- dhis2_population_adm2[[POPULATION_SELECTION]]
        log_msg(glue::glue(
            "Population Disaggregation: Column '{POPULATION_SELECTION}' selected as population values, 
            and renamed to 'POPULATION'."))
      } else {
          log_msg(glue::glue("Population Disaggregation: Column '{POPULATION_SELECTION}' not found in Population dataset!"), "error")
          stop(glue::glue("Population Disaggregation: Column '{POPULATION_SELECTION}' not found in Population dataset!"))
      }
  }
  # IMPORTANT: Return the modified dataframe back to the global environment!
  return(dhis2_population_adm2)
}


# --- --- --- --- --- --- --- --- ---

#' Load DHS careseeking data
#'
#' Loads `{country_code}_DHS_ADM2_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet` from the latest version
#' of the `DHS_INDICATORS` dataset, falling back to `{country_code}_DHS_ADM1_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet`
#' when the ADM2 file is not found. Keeps only `ADM1_ID` and `PCT_PUBLIC_CARE` and removes the
#' duplicate rows (the ADM1 survey value is repeated for each ADM2), so the single DHS value per
#' ADM1 applies to all ADM2 units and routine years (a `YEAR` column, if present, is dropped).
#' Logs a warning if an ADM1_ID still has more than one value. On success, sets the global
#' `careseeking_file_name` (used in the final log summary). On failure, logs a warning and returns
#' NULL; the run continues.
#'
#' @param country_code Character. Country code used as the filename prefix. Default: `COUNTRY_CODE`.
#' @param config List. Parsed SNT configuration, with `SNT_DATASET_IDENTIFIERS`.
#'   Default: `config_json`.
#' @return Data frame with `ADM1_ID` and `PCT_PUBLIC_CARE`, or NULL if no file could be loaded.
#' @export
load_dhs_careseeking_data <- function(country_code = COUNTRY_CODE, config = config_json) {
    # 1. Prepare identifiers (ADM2 file first, ADM1 file as fallback)
    dataset_name <- config$SNT_DATASET_IDENTIFIERS$DHS_INDICATORS
    file_names   <- glue::glue("{country_code}_DHS_{c('ADM2', 'ADM1')}_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet")

    # 2. Attempt to load, stopping at the first file found
    data <- NULL
    for (file_name in file_names) {
        data <- tryCatch({
            get_latest_dataset_file_in_memory(dataset_name, file_name)
        }, error = function(e) {
            # log_msg(paste("🛑 Error loading DHS data:", conditionMessage(e)), "error")
            return(NULL)
        })
        if (!is.null(data)) break
    }

    # 3. Validation & Logging
    if (!is.null(data)) {
        # DHS values come from a single survey and apply to all routine years:
        # keep only the columns used downstream (a YEAR column, if present, is not used)
        data <- dplyr::select(data, dplyr::any_of(c("ADM1_ID", "PCT_PUBLIC_CARE")))
        # The ADM1 value is repeated for each ADM2: keep one row per ADM1
        data <- dplyr::distinct(data)
        if (anyDuplicated(data$ADM1_ID) > 0) {
            log_msg("[WARNING] DHS careseeking data has more than one value per ADM1_ID; the first one is used.", "warning")
        }
        log_msg(glue::glue("✅ Care Seeking data: {file_name} loaded from {dataset_name}"))
        log_msg(glue::glue("Dimensions: {nrow(data)} rows, {ncol(data)} columns."))
        # Keep this global becuase it's needed for the final log summary (at the end of the script)
        careseeking_file_name <<- file_name # AVOID SETTING GLOBALS IN FUNCTIONS (!)
        return(data)
    } else {
        log_msg("Loading of careseeking data from DHS failed. Returning NULL.", "warning")
        return(NULL)
    }
}


#' Validate and format careseeking data
#'
#' Expects `ADM1_ID` and a percentage column: `CARESEEKING_PCT` for a user-provided file,
#' `PCT_PUBLIC_CARE` for DHS data. Converts the percentage column to numeric, scales it to 0-100
#' when its maximum is <= 1, and renames it to `CARESEEKING_PCT`. Logs an error and returns NULL
#' when a required column is missing or contains non-numeric values.
#'
#' @param data Data frame or NULL. Careseeking data, from a user file or from DHS.
#' @param from_file Logical. TRUE for a user-provided file, FALSE for DHS data.
#' @return Data frame with a `CARESEEKING_PCT` column (0-100), or NULL if `data` is NULL or
#'   validation fails.
#' @export
validate_and_format_careseeking_data <- function(data, from_file) {
    if (is.null(data)) {
        print("No careseeking data to validate.")
        return(NULL)
    }
    log_msg("Validating careseeking data...")
    
    # 1. Determine which column we expect based on the source
    # This keeps the logic modular and avoids repeating column-specific checks
    target_col <- if (from_file) "CARESEEKING_PCT" else "PCT_PUBLIC_CARE"
        
    # 2. Check for ADM1_ID and the source-specific percentage column
    required_cols <- c("ADM1_ID", target_col)
    missing_cols  <- setdiff(required_cols, colnames(data))
    if (length(missing_cols) > 0) {
        log_msg(paste("Required columns missing:", paste(missing_cols, collapse = ", ")), "error")
        return(NULL)
    }
    
    # 3. Ensure the percentage column is numeric
    if (!is.numeric(data[[target_col]])) {
        log_msg(paste("Column", target_col, "is not numeric. Attempting conversion..."), "warning")
        # Try to convert, but keep a backup to check for conversion failures
        converted_vals <- as.numeric(data[[target_col]])
        if (any(is.na(converted_vals)) && any(!is.na(data[[target_col]]))) {
            log_msg(paste("Non-numeric values found in", target_col, ". Validation failed."), "error")
            return(NULL)
        }
        data[[target_col]] <- converted_vals
    }
    
    # 4. Handle Range Adjustment (0-1 to 0-100)
    # We check if the maximum is <= 1 to see if it's likely a proportion
    max_val <- max(data[[target_col]], na.rm = TRUE)
    if (!is.na(max_val) && max_val <= 1) {
        log_msg(paste("Values in", target_col, "appear to be 0-1. Scaling to 0-100."), "warning")
        data[[target_col]] <- data[[target_col]] * 100
    }
    
    # 5. STANDARDIZATION STEP: Rename to CARESEEKING_PCT
    if (target_col != "CARESEEKING_PCT") {
        log_msg(paste("Renaming source column", target_col, "to standardized name 'CARESEEKING_PCT'."))
        colnames(data)[colnames(data) == target_col] <- "CARESEEKING_PCT"
    }
    log_msg("✅ Careseeking data validation and formatting completed.")
    return(data)
}


#' Join careseeking data to the monthly cases table
#'
#' Joins on ADM2_ID if present in the careseeking data, otherwise on ADM1_ID. If the careseeking
#' data also has a YEAR column, the join is done per year as well, so each year uses its own value,
#' and a warning lists the routine years with no careseeking value. Keeps one row per join key,
#' and sets missing or non-numeric `CARESEEKING_PCT` values to 100.
#'
#' @param main_df Data frame. Monthly cases table, with ADM1_ID, ADM2_ID and YEAR columns.
#' @param careseeking_data Data frame. Validated careseeking data, with CARESEEKING_PCT and
#'   ADM1_ID or ADM2_ID (optionally YEAR).
#' @return Data frame. `main_df` with a CARESEEKING_PCT column added.
#' @export
join_careseeking_data <- function(main_df, careseeking_data) {
  # --- 1. Determine the join key based on available columns ---
  if ("ADM2_ID" %in% colnames(careseeking_data)) {
    join_key <- "ADM2_ID"
  } else if ("ADM1_ID" %in% colnames(careseeking_data)) {
    join_key <- "ADM1_ID"
  } else {
    stop("[ERROR] Input data must contain 'ADM1_ID' or 'ADM2_ID'.")
  }
  # If YEAR is provided, join per year (YEAR is numeric in routine data, see load step)
  if ("YEAR" %in% colnames(careseeking_data)) {
    join_key <- c(join_key, "YEAR")
    careseeking_data <- careseeking_data |> mutate(YEAR = as.numeric(YEAR))
    log_msg(glue::glue("Careseeking data contains a YEAR column: joining by {paste(join_key, collapse = ', ')}."))
  }
  # --- 2. Robust Safeguard: Collapse the user data to the join_key level ---
  # This handles 'broadcasted' duplicates by ensuring only 1 row exists per ID (and YEAR)
  careseeking_data_clean <- careseeking_data |>
    select(all_of(join_key), CARESEEKING_PCT) |>
    # Ensure there's only one value per ID (and YEAR) to prevent row duplication
    distinct(across(all_of(join_key)), .keep_all = TRUE)
  # Warn about years in the routine data with no careseeking value: N3 will be missing for them
  if ("YEAR" %in% join_key) {
    missing_years <- setdiff(unique(main_df$YEAR), unique(careseeking_data_clean$YEAR))
    if (length(missing_years) > 0) {
      log_msg(glue::glue(
        "[WARNING] Careseeking data has no values for YEAR(s): {paste(sort(missing_years), collapse = ', ')}. ",
        "No careseeking adjustment (N3) is available for these years."
      ), "warning")
    }
  }
  # --- 3. Ensure CARESEEKING_PCT is numeric and handles any non-numeric issues ---
  careseeking_data_clean <- careseeking_data_clean |>
    mutate(CARESEEKING_PCT = as.numeric(CARESEEKING_PCT)) 
    if (any(is.na(careseeking_data_clean$CARESEEKING_PCT))) {
      careseeking_data_clean <- careseeking_data_clean |>
        mutate(CARESEEKING_PCT = ifelse(is.na(CARESEEKING_PCT), 100, CARESEEKING_PCT))
      log_msg("Warning: Non-numeric values found in CARESEEKING_PCT. These have been converted to 100%.", "warning")
    }
  # --- 4. Perform the join ---
  # If the key is ADM1, main_df (at ADM2 level) will correctly inherit the value for all its children.
  result <- main_df |>
    left_join(careseeking_data_clean, by = join_key)
  # ---
  return(result)
}


#' Load monthly reporting rate data
#'
#' Loads `{COUNTRY_CODE}_reporting_rate_dataelement.parquet` from the latest version of the
#' `DHIS2_REPORTING_RATE` dataset, or `{COUNTRY_CODE}_reporting_rate_dataset.parquet` if the first
#' cannot be loaded. Stops the run if neither can be loaded.
#'
#' Sets the globals `reporting_rate_month` (the loaded data), `REPORTING_RATE_METHOD`
#' ("dataelement" or "dataset"), `rr_dataset_name`, `file_name_de` and `file_name_ds`.
#'
#' @return Data frame. The first 3 rows of `reporting_rate_month`, for display in the notebook.
#' @export
load_reporting_rate_data <- function() {
    rr_dataset_name <<- config_json$SNT_DATASET_IDENTIFIERS$DHIS2_REPORTING_RATE
    file_name_de <<- paste0(COUNTRY_CODE, "_reporting_rate_dataelement.parquet")
    file_name_ds <<- paste0(COUNTRY_CODE, "_reporting_rate_dataset.parquet")
    reporting_rate_month <<- tryCatch({
        df_loaded <- get_latest_dataset_file_in_memory(rr_dataset_name, file_name_de)
        log_msg(glue("Reporting Rate data: `{file_name_de}` loaded from dataset: `{rr_dataset_name}`. Dataframe dimensions: {paste(dim(df_loaded), collapse=', ')}"))
        REPORTING_RATE_METHOD <<- "dataelement"
        df_loaded
    }, 
        error = function(e) {    
            cat(glue("[ERROR] Error while loading Reporting Rate 'dataelement' version for: {COUNTRY_CODE} {conditionMessage(e)}"))
            return(NULL)
    })
    if (is.null(reporting_rate_month)) {
        reporting_rate_month <<- tryCatch({
            df_loaded <- get_latest_dataset_file_in_memory(rr_dataset_name, file_name_ds) 
            log_msg(glue("Reporting Rate data: `{file_name_ds}` loaded from dataset: `{rr_dataset_name}`. Dataframe dimensions: {paste(dim(df_loaded), collapse=', ')}"))
            REPORTING_RATE_METHOD <<- "dataset"
            df_loaded
        }, 
        error = function(e) {    
            stop(glue("[ERROR] Error while loading Reporting Rate 'dataset' version for: {COUNTRY_CODE} {conditionMessage(e)}")) # raise error
        })
    }
    rm(df_loaded)
    log_msg(glue("Final Reporting Rate ({REPORTING_RATE_METHOD}) data frame dimensions: {paste(dim(reporting_rate_month), collapse=', ')}"))
    head(reporting_rate_month, 3)
}


#' Check the reporting rate data for missing values
#'
#' Counts NA values in `REPORTING_RATE` of the global `reporting_rate_month`, stores the count in
#' the global `na_count` and logs a warning when it is above 0. Logs an error when
#' `reporting_rate_month` is NULL. Does not stop the run.
#'
#' @return Invisible NULL. Called for its side effects.
#' @export
check_reporting_rate_data <- function() {
    if (!is.null(reporting_rate_month)) {
        na_count <<- sum(is.na(reporting_rate_month$REPORTING_RATE))     
        if (na_count > 0) {
            log_msg(glue::glue("⚠️ Warning: Reporting Rate data contains {na_count} missing values (NA) in 'REPORTING_RATE' column."), "warning")
        } else {
            log_msg(glue::glue("✅ Reporting Rate data contains no missing values (NA) in 'REPORTING_RATE' column."))
        }
    } else {
        log_msg(glue::glue("🚨 Reporting Rate data frame is NULL. Cannot check for missing values."), "error")
    }
}


#' Convert routine and reporting rate columns to numeric
#'
#' Creates the global `routine_data` from the global `dhis2_routine`, with YEAR, MONTH, CONF, TEST,
#' SUSP and PRES (where present) as numeric. Creates the global `reporting_rate_data` from the
#' global `reporting_rate_month`, with YEAR, MONTH and REPORTING_RATE as numeric, or logs a warning
#' when `reporting_rate_month` is NULL.
#'
#' @return Invisible NULL. Called for its side effects.
#' @export
enforce_numeric_cols <- function() {
    routine_data <<- dhis2_routine |>
        mutate(across(any_of(c("YEAR", "MONTH", "CONF", "TEST", "SUSP", "PRES")), as.numeric))
    log_msg(glue::glue("Created 'routine_data' dataframe. Ensured correct data types for DHIS2 routine data numerical columns: YEAR, MONTH, CONF, TEST, SUSP, PRES."))
    if (!is.null(reporting_rate_month)) {
        reporting_rate_data <<- reporting_rate_month |>
            mutate(across(c(YEAR, MONTH, REPORTING_RATE), as.numeric))
        log_msg("Created 'reporting_rate_data' dataframe. Ensured correct data types for Reporting Rate data numerical columns: YEAR, MONTH, REPORTING_RATE.")
    } else {
        log_msg("Reporting Rate data frame is NULL. Skipping data type enforcement for Reporting Rate.", "warning")
    }
}


#' Report months with a zero reporting rate
#'
#' Counts the rows of the global `reporting_rate_data` with `REPORTING_RATE == 0` and the number of
#' ADM2 affected, stores both in the global `zero_reporting` and logs them. N2 is NA for these rows
#' (set where N2 is calculated, not here). Logs an error when `reporting_rate_data` is NULL.
#'
#' @return Invisible NULL. Called for its side effects.
#' @export
handle_zeros_in_reporting_rate <- function() {
    if (!is.null(reporting_rate_data)) {
        zero_reporting <<- reporting_rate_data %>%
            filter(REPORTING_RATE == 0) %>%
            summarise(
                n_months_zero_reporting = n(),
                affected_zones = n_distinct(ADM2_ID)
            )
        if (zero_reporting$n_months_zero_reporting > 0) {    
            log_msg(glue::glue("🚨 Note: {zero_reporting$n_months_zero_reporting} rows had `REPORTING_RATE == 0` across ",
                         "{zero_reporting$affected_zones} ADM2. These N2 values were set to NA."))
        } else {
            log_msg(glue::glue("✅ Note: no ADM2 has `REPORTING_RATE == 0`. All N2 values were preserved."))
        }
    } else {
        log_msg("🚨 Reporting Rate data frame is NULL. Cannot check for zero reporting rates.", "error")
    }
}


#' Export the monthly cases table
#'
#' Writes `monthly_cases` to `{INTERMEDIATE_DATA_PATH}/{COUNTRY_CODE}_monthly_cases.parquet`, read
#' by the reporting notebook. The file is not added to any dataset.
#'
#' @param monthly_cases Data frame. Monthly cases table (ADM2 x MONTH).
#' @return Invisible NULL. Called for its side effects.
#' @export
export_monthly_cases <- function(monthly_cases) {
    file_path <- file.path(INTERMEDIATE_DATA_PATH, paste0(COUNTRY_CODE, "_monthly_cases.parquet"))
    arrow::write_parquet(monthly_cases, file_path)
    log_msg(glue::glue("Monthly cases data saved to: {file_path}"))
}


#' Check for months with zero presumed cases
#'
#' Runs only when `N1_METHOD == "PRES"`. Logs a warning with the number of ADM2 x MONTH rows
#' with `PRES == 0`, where N1 equals CONF, when there are any.
#'
#' @param monthly_cases Data frame. Monthly cases table (ADM2 x MONTH), with a PRES column.
#' @param N1_METHOD Character. "SUSP-TEST" or "PRES".
#' @return Invisible NULL. Called for its side effects.
#' @export
coherence_check_PRES <- function(monthly_cases, N1_METHOD) {
    # Run this check only if N1_METHOD == "PRES" (else, problem doesn't exist)
    if (N1_METHOD == "PRES") {
        nr_of_pres_0_adm2_month <- monthly_cases %>% filter(PRES == 0) %>% nrow()
        if (nr_of_pres_0_adm2_month > 0) {
            log_msg(glue::glue("🚨 Note: using `PRES` for incidence adjustement, but `PRES == 0` for {nr_of_pres_0_adm2_month} rows (ADM2 x MONTH)."), "warning")
        }        
    } else {
        log_msg(glue::glue("N1_METHOD is not set to 'PRES'. No need to check for coherence of `PRES` column."))
    }
}

#' Check for more tested than suspected cases
#'
#' Runs only when `N1_METHOD == "SUSP-TEST"`. Counts the ADM2 x MONTH rows where
#' `TEST > SUSP` and logs a warning when there are any. Expects `monthly_cases` before `TEST` is
#' capped to `SUSP`.
#'
#' @param monthly_cases Data frame. Monthly cases table (ADM2 x MONTH), with SUSP and TEST.
#' @param N1_METHOD Character. "SUSP-TEST" or "PRES".
#' @return Invisible NULL. Called for its side effects.
#' @export
coherence_check_SUSP_TEST <- function(monthly_cases, N1_METHOD) {
    if (N1_METHOD == "SUSP-TEST") {
        nr_of_negative <- monthly_cases %>% filter(TEST > SUSP) %>% nrow()
        if (nr_of_negative > 0) {
            log_msg(
            glue::glue("🚨 Note: using formula `SUSP - TEST` for incidence adjustement, but higher tested than suspected cases (`TEST > SUSP`) detected in {nr_of_negative} rows (ADM2 x MONTH). `TEST` is capped to `SUSP` in these rows for the N1 calculation."),
            "warning"
            )
        } else {
            log_msg(glue::glue("✅ Note: using `SUSP - TEST` for incidence adjustment, no cases where `TEST > SUSP` detected."))
        }
    } else {
        log_msg(glue::glue("N1_METHOD is not set to 'SUSP-TEST'. No need to check for coherence of `SUSP` and `TEST` columns."))
    }
    invisible(NULL)
}


#' Check for more confirmed than tested cases
#'
#' Counts the ADM2 x MONTH rows where `CONF / TEST > 1`, which includes rows with `TEST == 0` and
#' `CONF > 0`, and logs a warning when there are any.
#'
#' @param monthly_cases Data frame. Monthly cases table (ADM2 x MONTH), with CONF and TEST.
#' @return Invisible NULL. Called for its side effects.
#' @export
coherence_check_CONF_TEST <- function(monthly_cases) {
    more_confirmed_than_tested <- monthly_cases %>% 
        mutate(CONF_divby_TEST = CONF / TEST) %>% 
        filter(CONF_divby_TEST > 1) %>% 
        nrow() 
    if (more_confirmed_than_tested > 0) {
        log_msg(glue::glue("🚨 Note: higher confirmed than tested cases (`CONF/TEST`) detected in {more_confirmed_than_tested} rows (ADM2 x MONTH)."), "warning")
    } else {
        log_msg(glue::glue("✅ Note: no cases where `CONF > TEST` detected."))
    }
}


#' Check that an adjusted incidence is not below the incidence it adjusts
#'
#' Counts the ADM2 x YEAR rows where `incidence_col_2 < incidence_col_1`, stores the count in the
#' global `nr_of_impossible_values` and logs a warning when it is above 0. Also logs a warning when
#' every value of `incidence_col_2` is NA.
#'
#' @param yearly_incidence Data frame. Yearly incidence table (ADM2 x YEAR).
#' @param incidence_col_1 Character. Name of the less adjusted incidence column.
#' @param incidence_col_2 Character. Name of the more adjusted incidence column.
#' @return Invisible NULL. Called for its side effects.
#' @export
coherence_check_yearly_incidence <- function(yearly_incidence, incidence_col_1, incidence_col_2) {
    nr_of_impossible_values <<- yearly_incidence |>
      mutate(IMPOSSIBLE_VALUE = if_else(!!sym(incidence_col_2) < !!sym(incidence_col_1), TRUE, FALSE)) |>
      pull(IMPOSSIBLE_VALUE) |>
      sum(na.rm = TRUE) 
    if (nr_of_impossible_values > 0) {
      log_msg(glue::glue("🚨 Warning: found {nr_of_impossible_values} rows where {incidence_col_2} < {incidence_col_1}!"), "warning")
    } else log_msg(glue::glue("✅ For all YEAR and ADM2, `{incidence_col_1}` is smaller than `{incidence_col_2}` (as expected)."))
    # Check if all values in the column are NA, which indicates that the adjustment method did not work for any ADM2 and month, which is a problem.
    if (all(is.na(yearly_incidence[[incidence_col_2]]))) {
      log_msg(glue::glue("🚨 Warning: all values of `{incidence_col_2}` are `NA`s"), "warning")
    } 
}


#' Save the yearly incidence table
#'
#' Writes `yearly_incidence` to `{data_path}/{COUNTRY_CODE}_incidence{file_extension}` with the
#' given write function, creating the directory if needed.
#'
#' @param yearly_incidence Data frame. Yearly incidence table (ADM2 x YEAR).
#' @param data_path Character. Output directory.
#' @param file_extension Character. File extension, including the dot, e.g. ".parquet".
#' @param write_function Function. Called as `write_function(data, path)`, e.g. `write_csv` or
#'   `arrow::write_parquet`.
#' @return Invisible NULL. Called for its side effects.
#' @export
save_yearly_incidence <- function(yearly_incidence, data_path, file_extension, write_function) {
  base_name_parts <- c(COUNTRY_CODE, "_incidence")
  # --- Concatenate all parts to form the final filename ---
  file_name <- paste0(c(base_name_parts, file_extension), collapse = "")
  file_path <- file.path(data_path, file_name)
  output_dir <- dirname(file_path)
  # --- Check if the output directory exists, else create it ---
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  # --- Flexibility to use function as provided in argument: "write_csv" or "arrow::write_parquet" ... ---
  write_function(yearly_incidence, file_path)
  log_msg(glue::glue("Exporting : {file_path}"))
}