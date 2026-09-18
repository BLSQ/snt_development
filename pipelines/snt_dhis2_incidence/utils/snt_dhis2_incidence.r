# Store code to be sourced in the notebook in this same directory, so that the main notebook
# only shows the code relevant to the analysis, and not the boring routine setup, import and export.
# Each piece of code is wrapped in a function to keep the notebook clean.

message("This step sets up the environment for the DHIS2 incidence pipeline, including paths, config, and utility functions.
It basically handles all the boring stuff so that you can focus on the code that matters :)")

# Load base functions
source(file.path("~/workspace/code", "snt_utils.r"))


resolve_routine_filename <- function() {
    if (ROUTINE_DATA_CHOICE == "raw") {
        return("_routine.parquet")
    }
    removed_status <- if (ROUTINE_DATA_CHOICE == "raw_without_outliers") "removed" else "imputed"
    glue::glue("_routine_outliers_{removed_status}.parquet")
}


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

 
check_fixed_cols_in_routine <- function(dhis2_routine) {
    actual_cols <- colnames(dhis2_routine)
    missing_cols <- setdiff(fixed_cols, actual_cols)
    if (length(missing_cols) == 0) {
        log_msg(glue::glue("All expected 'fixed' columns present."))
    } else {
        log_msg(glue::glue("🚨 Missing Columns: {paste(missing_cols, collapse = ', ')}"), "warning")
    }
}

check_dhis2_indicators_cols_in_routine <- function(dhis2_routine) {
    actual_cols <- colnames(dhis2_routine)
    missing_cols <- setdiff(DHIS2_INDICATORS, actual_cols)
    if (length(missing_cols) == 0) {
        log_msg(glue::glue("All DHIS2 indicators present in 'dhis2_routine'."))
    } else {
        log_msg(glue::glue("🚨 Missing DHIS2 INDICATORS: {paste(missing_cols, collapse = ', ')}"), "warning")
    }
}

check_pres_col <- function(dhis2_routine) {
    if (exists("N1_METHOD") && N1_METHOD == "PRES") {
        pres_in_routine <- any(names(dhis2_routine) == "PRES")        
    if (!pres_in_routine) {
        log_msg(glue::glue("🛑 Column `PRES` missing from routine data!"), "error")
        stop()
    }
        log_msg(glue::glue("Column `PRES` is present. Proceeding."))
    } else {
        # This is just for the nb, no need to long in pipeline run
        print("N1_METHOD is not set to 'PRES'. No need to check for `PRES` column.")
    }
}

load_population_data <- function(config_json) {
    dhis2_pop_dataset <- if (USE_TRANSFORMED_POPULATION) {
        config_json$SNT_DATASET_IDENTIFIERS$DHIS2_POPULATION_TRANSFORMATION
    } else {
        config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED        
    }
        
    dhis2_population_adm2 <- load_dataset_file(dataset_id=dhis2_pop_dataset, filename= paste0(COUNTRY_CODE, "_population.parquet"))
    log_msg(glue::glue("DHIS2 population data loaded from {dhis2_pop_dataset}."))
    return(dhis2_population_adm2)
}

# --- DISAGGREGATION LOGIC --- --- --- --- --- --- ---

prepare_disaggregated_indicators <- function(dhis2_routine, DISAGGREGATION_SELECTION, N1_METHOD) {

  # Initial log msg
    log_msg(glue::glue("Starting preparation of disaggregated indicators for selection '{DISAGGREGATION_SELECTION}' and N1_METHOD '{N1_METHOD}'."))

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

# Rewrote select_population_column() to be mapping the DISAGGREGATION_SELECTION 
# only if it exists in the population dataset, otherwise log an error and break 
# (instead of silently using the non-disaggregated population)
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

load_dhs_careseeking_data <- function(country_code = COUNTRY_CODE, config = config_json) {
    # 1. Prepare identifiers
    dataset_name <- config$SNT_DATASET_IDENTIFIERS$DHS_INDICATORS
    file_name    <- glue::glue("{country_code}_DHS_ADM1_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet")
    
    # 2. Attempt to load
    data <- tryCatch({
        get_latest_dataset_file_in_memory(dataset_name, file_name)
    }, error = function(e) {
        # log_msg(paste("🛑 Error loading DHS data:", conditionMessage(e)), "error")
        return(NULL)
    })
    
    # 3. Validation & Logging
    if (!is.null(data)) {
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


join_careseeking_data <- function(main_df, careseeking_data) {
  # --- 1. Determine the join key based on available columns ---
  if ("ADM2_ID" %in% colnames(careseeking_data)) {
    join_key <- "ADM2_ID"
  } else if ("ADM1_ID" %in% colnames(careseeking_data)) {
    join_key <- "ADM1_ID"
  } else {
    stop("[ERROR] Input data must contain 'ADM1_ID' or 'ADM2_ID'.")
  }
  # --- 2. Robust Safeguard: Collapse the user data to the join_key level ---
  # This handles 'broadcasted' duplicates by ensuring only 1 row exists per ID   
  careseeking_data_clean <- careseeking_data |>
    select(all_of(join_key), CARESEEKING_PCT) |>
    # Ensure there's only one value per ID to prevent row duplication
    distinct(!!sym(join_key), .keep_all = TRUE)
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


export_monthly_cases <- function(monthly_cases) {
    file_path <- file.path(INTERMEDIATE_DATA_PATH, paste0(COUNTRY_CODE, "_monthly_cases.parquet"))
    arrow::write_parquet(monthly_cases, file_path)
    log_msg(glue::glue("Monthly cases data saved to: {file_path}"))
}


coherence_check_PRES <- function(monthly_cases) {
    # Run this check only if N1_METHOD == "PRES" (else, problem doesn't exist)
    if (N1_METHOD == "PRES") {
        nr_of_pres_0_adm2_month <<- monthly_cases |> filter(PRES == 0) |> nrow()
        log_msg(glue::glue("🚨 Note: using `PRES` for incidence adjustement, but `PRES == 0` for {nr_of_pres_0_adm2_month} rows (ADM2 x MONTH)."), "warning")
    } else {
        log_msg(glue::glue("N1_METHOD is not set to 'PRES'. No need to check for coherence of `PRES` column."))
    }
}

coherence_check_SUSP_TEST <- function(monthly_cases) {
  # Logically, there should not be more tested cases than suspected cases. 
    if (N1_METHOD == "SUSP-TEST") {
        nr_of_negative <<- monthly_cases |> mutate(SUSP_minus_TEST = SUSP - TEST) |> filter(SUSP_minus_TEST < 0) |> nrow() 
        if (nr_of_negative > 0) {
            log_msg(
            glue::glue("🚨 Note: using formula `SUSP - TEST` for incidence adjustement, but higher tested than suspected cases (`SUSP < TEST`) detected in {nr_of_negative} rows (ADM2 x MONTH)."),
            "warning"
            )
        } else {
            log_msg(glue::glue("✅ Note: using `SUSP - TEST` for incidence adjustment, no cases where `TEST > SUSP` detected."))
        }
    } else {
        log_msg(glue::glue("N1_METHOD is not set to 'SUSP-TEST'. No need to check for coherence of `SUSP` and `TEST` columns."))
    }
}


coherence_check_CONF_TEST <- function(monthly_cases) {
    more_confirmed_than_tested <<- monthly_cases |> mutate(CONF_divby_TEST = CONF / TEST) |> filter(CONF_divby_TEST > 1) |> nrow() 
    if (more_confirmed_than_tested > 0) {
        log_msg(glue::glue("🚨 Note: higher confirmed than tested cases (`CONF/TEST`) detected in {more_confirmed_than_tested} rows (ADM2 x MONTH)."), "warning")
    } else {
        log_msg(glue::glue("✅ Note: no cases where `CONF > TEST` detected."))
    }
}


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


# Reusable function to generate filename and save data ---------------------------------------------
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