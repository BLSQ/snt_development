# Store code to be sourced in the notebook in this same directory, so that the main notebook
# only shows the code relevant to the analysis, and not the boring routine setup, import and export.
# Each piece of code is wrapped in a function to keep the notebook clean.

message("This step sets up the environment for the DHIS2 incidence pipeline, including paths, config, and utility functions.
It basically handles all the boring stuff so that you can focus on the code that matters :)
")

setup_paths <- function() {
SNT_ROOT_PATH <<- "/home/hexa/workspace"
CODE_PATH <<- file.path(SNT_ROOT_PATH, 'code')
CONFIG_PATH <<- file.path(SNT_ROOT_PATH, 'configuration')
DATA_PATH <<- file.path(SNT_ROOT_PATH, 'data', 'dhis2', 'incidence')
INTERMEDIATE_DATA_PATH <<- file.path(DATA_PATH, "intermediate_results")
message("Paths set up:")
message("CODE_PATH: ", CODE_PATH)
message("CONFIG_PATH: ", CONFIG_PATH)
message("DATA_PATH: ", DATA_PATH)
message("INTERMEDIATE_DATA_PATH: ", INTERMEDIATE_DATA_PATH)
}

create_intermediate_data_dir <- function() {
if (!dir.exists(INTERMEDIATE_DATA_PATH)) {
dir.create(INTERMEDIATE_DATA_PATH, recursive = TRUE)
log_msg(glue("Created directory for intermediate results: {INTERMEDIATE_DATA_PATH}"))
}
}

load_utils <- function() {
utils_path <- "/home/hexa/workspace/code/snt_utils.r"
palettes_path <- "/home/hexa/workspace/code/snt_palettes.r"
source("/home/hexa/workspace/code/snt_utils.r")
message("Utils loaded from: ", utils_path)
source("/home/hexa/workspace/code/snt_palettes.r")
message("Palettes loaded from: ", palettes_path)
}

set_env_openhexa <- function() {
Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
message("Python config:")
print(reticulate::py_config())
openhexa <<- import("openhexa.sdk")
message("Openhexa SDK imported successfully.")
}

import_config_json <- function() {
config_path <- file.path(CONFIG_PATH, "SNT_config.json")
config_json <<- tryCatch({ fromJSON(config_path) },
error = function(e) {
msg <- paste0("[ERROR] Error while loading configuration: ", conditionMessage(e))
cat(msg)
stop(msg)
})
log_msg(glue("SNT configuration loaded from: {config_path}"))
}

config_generic <- function() {
COUNTRY_CODE <<- config_json$SNT_CONFIG$COUNTRY_CODE
ADMIN_1 <<- toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_1)
ADMIN_2 <<- toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_2)
log_msg(glue("Configuration values set: COUNTRY_CODE={COUNTRY_CODE}, ADMIN_1={ADMIN_1}, ADMIN_2={ADMIN_2}"))
}

config_incidence <- function() {
DHIS2_INDICATORS <<- names(config_json$DHIS2_DATA_DEFINITIONS$DHIS2_INDICATOR_DEFINITIONS)
log_msg(glue("DHIS2 indicators set: {paste(DHIS2_INDICATORS, collapse=', ')}"))
}

set_fixed_cols <- function() {
fixed_cols <<- c('OU_ID','PERIOD', 'YEAR', 'MONTH', 'ADM1_ID', 'ADM2_ID')
log_msg(glue("Fixed routine data ('dhis2_routine') columns set: {paste(fixed_cols, collapse=', ')}"))
}

resolve_routine_filename <- function() {
if (ROUTINE_DATA_CHOICE == "raw") return("_routine.parquet")
is_removed <<- FALSE
if (ROUTINE_DATA_CHOICE == "raw_without_outliers") is_removed <<- TRUE
removed_status <<- if (is_removed) "removed" else "imputed"
return(glue::glue("_routine_outliers_{removed_status}.parquet"))
}

select_routine_dataset_and_filename <- function() {
if (ROUTINE_DATA_CHOICE == "raw") {
routine_dataset_name <<- config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED
routine_name <<- resolve_routine_filename()
routine_filename <<- paste0(COUNTRY_CODE, routine_name)
} else {
routine_dataset_name <<- config_json$SNT_DATASET_IDENTIFIERS$DHIS2_OUTLIERS_IMPUTATION
routine_name <<- resolve_routine_filename()
routine_filename <<- paste0(COUNTRY_CODE, routine_name)
}
log_msg(glue("Selected routine dataset: {routine_dataset_name}, filename: {routine_filename}"))
}

load_dhis2_routine_data <- function() {
dhis2_routine <<- tryCatch({ get_latest_dataset_file_in_memory(routine_dataset_name, routine_filename) },
error = function(e) {
if (grepl("does not exist", conditionMessage(e), ignore.case = TRUE)) {
msg <- paste0("[ERROR] File not found! 🛑 The file ", routine_filename, " does not exist in ",  routine_dataset_name, ". To generate it, execute the pipeline DHIS2 Outliers Removal and Imputation.")
} else {
msg <- paste0("[ERROR] 🛑 Error while loading DHIS2 routine data file : ", routine_filename, ". [ERROR DETAILS] " , conditionMessage(e))
}
stop(msg)
})
log_msg(paste0("DHIS2 routine data : ", routine_filename, " loaded. Dims: ", paste(dim(dhis2_routine), collapse=", ")))
return(head(dhis2_routine, 3))
}

check_fixed_cols_in_routine <- function() {
actual_cols <- colnames(dhis2_routine)
missing_cols <- setdiff(fixed_cols, actual_cols)
if (length(missing_cols) == 0) {
log_msg(paste0("All expected 'fixed' columns present."))
} else {
log_msg(paste0("🚨 Missing Columns: ", paste(missing_cols, collapse = ", ")), "warning")
}
}

check_dhis2_indicators_cols_in_routine <- function() {
actual_cols <- colnames(dhis2_routine)
missing_cols <- setdiff(DHIS2_INDICATORS, actual_cols)
if (length(missing_cols) == 0) {
log_msg("All DHIS2 indicators present in 'dhis2_routine'.")
} else {
log_msg(paste0("🚨 Missing DHIS2 INDICATORS: ", paste(missing_cols, collapse = ", ")), "warning")
}
}

check_PRES_col <- function() {
if (exists("N1_METHOD") && N1_METHOD == "PRES") {
pres_in_routine <- any(names(dhis2_routine) == "PRES")
pres_in_config <- any(DHIS2_INDICATORS == "PRES")
    if (!pres_in_routine) {
        log_msg("🛑 Column `PRES` missing from routine data!", "error")
        stop()
    }
    log_msg("Column `PRES` is present. Proceeding.")
} else {
        # This is just for the nb, no need to long in pipeline run
        print("N1_METHOD is not set to 'PRES'. No need to check for `PRES` column.")
    }
}

load_population_data <- function() {
dhis2_pop_dataset <- if (USE_ADJUSTED_POPULATION) config_json$SNT_DATASET_IDENTIFIERS$DHIS2_POPULATION_TRANSFORMATION else config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED

dhis2_population_adm2 <<- get_latest_dataset_file_in_memory(dhis2_pop_dataset, paste0(COUNTRY_CODE, "_population.parquet"))
log_msg(glue("DHIS2 population data loaded from {dhis2_pop_dataset}."))
return(head(dhis2_population_adm2, 3))


}

# --- DISAGGREGATION LOGIC --- --- --- --- --- --- ---

# DISAGGREGATED_INDICATORS_FOUND <- FALSE # 👈 

# if (!is.null(DISAGGREGATION_SELECTION) && N1_METHOD %in% c("SUSP-TEST", "PRES")) {
#   # Determine the expected column names based on the disaggregation selection and method
#   prefix_method <- ifelse(N1_METHOD == "SUSP-TEST", "SUSP", "PRES")
#   prefix_fixed <- c("TEST", "CONF") 
#   prefix_all    <- c(prefix_method, prefix_fixed) 
#   target_colnames <- glue("{prefix_all}_{DISAGGREGATION_SELECTION}")
  
#   if (all(target_colnames %in% colnames(dhis2_routine))) {
#     # Map the disaggregated columns (e.g., SUSP_UNDER_5) to generic names (e.g., SUSP) so that 
#     # the rest of the pipeline can use them without needing to know about the disaggregation
#     dhis2_routine[prefix_all] <- dhis2_routine[target_colnames]
#     for (col in target_colnames) {
#       log_msg(glue("Population Disaggregation: Successfully mapped indicator: {col}"))
#     }    
#     # Signal success for the next code block
#     DISAGGREGATED_INDICATORS_FOUND <- TRUE # 👈
#   } else {
#     missing_cols <- setdiff(target_colnames, colnames(dhis2_routine))
#     log_msg(glue("Population Disaggregation: Disaggregation on '{DISAGGREGATION_SELECTION}' failed."), "warning")
#     log_msg(glue("Population Disaggregation: Missing columns in routine dataset: {paste(missing_cols, collapse = ', ')}"), "warning")
#     msg <- glue("[ERROR] 🛑 Population Disaggregation: Required columns for disaggregation '{DISAGGREGATION_SELECTION}' are missing.")        
#     stop(msg)
#   }
# } else {
#   # Print just in nb (not in pipeline logs)
#   print("Population Disaggregation: No disaggregation applied based on the current configuration.")
# }

prepare_disaggregated_indicators <- function(dhis2_routine, DISAGGREGATION_SELECTION, N1_METHOD) {
  # Initialize the flag locally
  DISAGGREGATED_INDICATORS_FOUND <<- FALSE 

  if (!is.null(DISAGGREGATION_SELECTION) && N1_METHOD %in% c("SUSP-TEST", "PRES")) {
    # Determine the expected column names based on the disaggregation selection and method
    prefix_method <- ifelse(N1_METHOD == "SUSP-TEST", "SUSP", "PRES")
    prefix_fixed <- c("TEST", "CONF") 
    prefix_all    <- c(prefix_method, prefix_fixed) 
    target_colnames <<- glue::glue("{prefix_all}_{DISAGGREGATION_SELECTION}")
    
    if (all(target_colnames %in% colnames(dhis2_routine))) {
      # Map the disaggregated columns (e.g., SUSP_UNDER_5) to generic names (e.g., SUSP) so that 
      # the rest of the pipeline can use them without needing to know about the disaggregation
      dhis2_routine[prefix_all] <- dhis2_routine[target_colnames]
      
      for (col in target_colnames) {
        log_msg(glue::glue("Population Disaggregation: Successfully mapped indicator: {col}"))
      }    
      
      # Signal success for the next code block
      DISAGGREGATED_INDICATORS_FOUND <<- TRUE 
    } else {
      missing_cols <- setdiff(target_colnames, colnames(dhis2_routine))
      log_msg(glue::glue("Population Disaggregation: Disaggregation on '{DISAGGREGATION_SELECTION}' failed."), "warning")
      log_msg(glue::glue("Population Disaggregation: Missing columns in routine dataset: {paste(missing_cols, collapse = ', ')}"), "warning")
      msg <- glue::glue("[ERROR] 🛑 Population Disaggregation: Required columns for disaggregation '{DISAGGREGATION_SELECTION}' are missing.")  
      stop(msg)
    }
  } else {
    # Print just in nb (not in pipeline logs)
    print("Population Disaggregation: No disaggregation applied based on the current configuration.")
  }
  # return(dhis2_routine)
    dhis2_routine <<- dhis2_routine
}

# if (DISAGGREGATED_INDICATORS_FOUND) { 
#     POPULATION_SELECTION <- paste0("POP_", DISAGGREGATION_SELECTION)    
#     if (!(POPULATION_SELECTION %in% colnames(dhis2_population_adm2))) {
#         log_msg(glue("Population Disaggregation: Column '{POPULATION_SELECTION}' not found in Population dataset."), "warning")
#         POPULATION_SELECTION <- "POPULATION"
#     }
#     # The selected column is assigned to POPULATION col so that later code can use it generically
#     dhis2_population_adm2$POPULATION <- dhis2_population_adm2[[POPULATION_SELECTION]]
#     log_msg(glue("Population Disaggregation: Column '{POPULATION_SELECTION}' selected as population values."))
# }

select_population_column <- function(dhis2_population_adm2, DISAGGREGATED_INDICATORS_FOUND, DISAGGREGATION_SELECTION) {
  # Default value for the selection if the condition isn't met or if it fails
  POPULATION_SELECTION <<- "POPULATION"
  if (DISAGGREGATED_INDICATORS_FOUND) { 
      POPULATION_SELECTION <<- paste0("POP_", DISAGGREGATION_SELECTION)    
      if (!(POPULATION_SELECTION %in% colnames(dhis2_population_adm2))) {
          log_msg(glue::glue("Population Disaggregation: Column '{POPULATION_SELECTION}' not found in Population dataset."), "warning")
          POPULATION_SELECTION <<- "POPULATION"
      }
      # The selected column is assigned to POPULATION col so that later code can use it generically
      dhis2_population_adm2$POPULATION <- dhis2_population_adm2[[POPULATION_SELECTION]]
      log_msg(glue::glue("Population Disaggregation: Column '{POPULATION_SELECTION}' selected as population values."))
  } else {
    # Print just in nb (not in pipeline logs)
    print("Population Disaggregation: No disaggregation applied based on the current configuration.")
  }
    dhis2_population_adm2 <<- dhis2_population_adm2
}

# --- --- --- --- --- --- --- --- ---


load_careseeking_data <- function() {
    if (USE_CSB_DATA == TRUE) {
        dataset_name <<- config_json$SNT_DATASET_IDENTIFIERS$DHS_INDICATORS
        file_name <<- glue::glue("{COUNTRY_CODE}_DHS_ADM1_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet")
        careseeking_data <<- tryCatch({ get_latest_dataset_file_in_memory(dataset_name, file_name) },          
                      error = function(e) {
                          msg <- paste("🛑 Error while loading DHS Care Seeking data file from `", dataset_name, file_name ,"`.", conditionMessage(e))  # log error message
                          log_msg(msg, "error")
                          return(NULL) # make object NULL on error
                      })
        if (!is.null(careseeking_data)) {
            log_msg(paste0("Care Seeking data : ", file_name, " loaded from dataset : ", dataset_name))
            log_msg(paste0("Care Seeking data frame dimensions: ", nrow(careseeking_data), " rows, ", ncol(careseeking_data), " columns."))
            head(careseeking_data)
        } else {
            log_msg(paste0("🚨 Care-seeking data not loaded due to an error, `careseeking_data` is set to `NULL`!"), "warning")
        }
    } else {
        careseeking_data <<- NULL
        print("USE_CSB_DATA is set to FALSE. Care-seeking data will be ignored and `careseeking_data` is set to `NULL`.")
    }
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
            log_msg(glue("⚠️ Warning: Reporting Rate data contains {na_count} missing values (NA) in 'REPORTING_RATE' column."), "warning")
        } else {
            log_msg("✅ Reporting Rate data contains no missing values (NA) in 'REPORTING_RATE' column.")
        }
    } else {
        log_msg("🚨 Reporting Rate data frame is NULL. Cannot check for missing values.", "error")
    }
}


enforce_numeric_cols <- function() {
    routine_data <<- dhis2_routine |>
        mutate(across(any_of(c("YEAR", "MONTH", "CONF", "TEST", "SUSP", "PRES")), as.numeric))
    log_msg("Created 'routine_data' dataframe. Ensured correct data types for DHIS2 routine data numerical columns: YEAR, MONTH, CONF, TEST, SUSP, PRES.")
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
            log_msg(glue("🚨 Note: {zero_reporting$n_months_zero_reporting} rows had `REPORTING_RATE == 0` across ",
                         "{zero_reporting$affected_zones} ADM2. These N2 values were set to NA."))
        } else {
            log_msg("✅ Note: no ADM2 has `REPORTING_RATE == 0`. All N2 values were preserved.")
        }
    } else {
        log_msg("🚨 Reporting Rate data frame is NULL. Cannot check for zero reporting rates.", "error")
    }
}


build_monthly_cases <- function(
    routine_data,
    reporting_rate_data,
    N1_METHOD,
    care_seeking_data_f = NULL,
    careseeking_data = NULL
) {
    monthly_cases <- routine_data |>
        dplyr::group_by(ADM1_ID, ADM2_ID, YEAR, MONTH) |>
        dplyr::summarise(
            CONF = sum(CONF, na.rm = TRUE),
            TEST = sum(TEST, na.rm = TRUE),
            SUSP = sum(SUSP, na.rm = TRUE),
            dplyr::across(dplyr::any_of("PRES"), ~sum(., na.rm = TRUE), .names = "PRES"),
            .groups = "drop"
        ) |>
        dplyr::mutate(TEST = ifelse(N1_METHOD == "SUSP-TEST" & !is.na(SUSP) & (TEST > SUSP), SUSP, TEST)) |>
        dplyr::left_join(reporting_rate_data, by = c("ADM2_ID", "YEAR", "MONTH")) |>
        dplyr::mutate(TPR = ifelse(!is.na(CONF) & !is.na(TEST) & (TEST != 0), CONF / TEST, 1))

    if (N1_METHOD == "SUSP-TEST") {
        monthly_cases <- monthly_cases %>%
            dplyr::mutate(N1 = CONF + ((SUSP - TEST) * TPR))
        log_msg("Calculating N1 as `N1 = CONF + ((SUSP - TEST) * TPR)`")
    } else if (N1_METHOD == "PRES") {
        if ("PRES" %in% names(monthly_cases) && !all(is.na(monthly_cases$PRES))) {
            monthly_cases <- monthly_cases %>%
                dplyr::mutate(N1 = CONF + (PRES * TPR))
            log_msg("ℹ️ Calculating N1 as `N1 = CONF + (PRES * TPR)`")
        } else {
            log_msg("🚨 Warning: 'PRES' not found in routine data or contains all `NA` values! 🚨 Calculating N1 using 'SUSP-TEST' method instead.")
            monthly_cases <- monthly_cases %>%
                dplyr::mutate(N1 = CONF + ((SUSP - TEST) * TPR))
        }
    } else {
        log_msg("Invalid N1_METHOD. Please use 'PRES' or 'SUSP-TEST'.")
    }

    monthly_cases <- monthly_cases %>%
        dplyr::mutate(N2 = ifelse(REPORTING_RATE == 0, NA_real_, N1 / REPORTING_RATE))

    if (!is.null(care_seeking_data_f)) {
        monthly_cases <- monthly_cases %>%
            dplyr::left_join(care_seeking_data_f %>% dplyr::select(ADM1_ID, PCT), by = c("ADM1_ID")) %>%
            dplyr::mutate(N3 = N2 / PCT) %>%
            dplyr::select(-PCT)
        log_msg("N2 adjusted by care seeking data (NER Specific).")
    }

    if (!is.null(careseeking_data)) {
        monthly_cases <- monthly_cases |>
            dplyr::mutate(YEAR = as.numeric(YEAR)) |>
            dplyr::left_join(careseeking_data, by = c("ADM1_ID")) |>
            dplyr::mutate(
                N3 = N2 + (N2 * PCT_PRIVATE_CARE / PCT_PUBLIC_CARE) + (N2 * PCT_NO_CARE / PCT_PUBLIC_CARE)
            )
    } else {
        print("🦘 Careseeking data not available, skipping calculation of N3.")
    }

    monthly_cases
}


build_yearly_incidence <- function(monthly_cases, dhis2_population_adm2, care_seeking_data_f = NULL, careseeking_data = NULL) {
    monthly_cases <- monthly_cases %>%
        dplyr::mutate(dplyr::across(where(is.numeric), as.numeric))

    population_data <- dhis2_population_adm2 %>%
        dplyr::mutate(dplyr::across(c(YEAR, POPULATION), as.numeric))

    yearly_incidence <- monthly_cases %>%
        dplyr::group_by(ADM2_ID, YEAR) %>%
        dplyr::summarise(
            dplyr::across(c(CONF, N1, N2), ~sum(.)),
            .groups = "drop"
        ) %>%
        dplyr::left_join(
            population_data,
            by = c("ADM2_ID", "YEAR")
        ) %>%
        dplyr::mutate(
            INCIDENCE_CRUDE = CONF / POPULATION * 1000,
            INCIDENCE_ADJ_TESTING = N1 / POPULATION * 1000,
            INCIDENCE_ADJ_REPORTING = N2 / POPULATION * 1000
        ) |>
        dplyr::ungroup()

    if (!is.null(care_seeking_data_f) && "N3" %in% names(monthly_cases)) {
        n3_data <- monthly_cases %>%
            dplyr::group_by(ADM2_ID, YEAR) %>%
            dplyr::summarise(N3 = sum(N3, na.rm = TRUE), .groups = "drop") |>
            dplyr::ungroup()

        yearly_incidence <- yearly_incidence %>%
            dplyr::left_join(n3_data, by = c("ADM2_ID", "YEAR")) %>%
            dplyr::mutate(INCIDENCE_ADJ_CARESEEKING = N3 / POPULATION * 1000)
    } else if (!is.null(careseeking_data) && "N3" %in% names(monthly_cases)) {
        n3_data <- monthly_cases %>%
            dplyr::group_by(ADM2_ID, YEAR) %>%
            dplyr::summarise(N3 = sum(N3, na.rm = TRUE), .groups = "drop") |>
            dplyr::ungroup()

        yearly_incidence <- yearly_incidence %>%
            dplyr::left_join(n3_data, by = c("ADM2_ID", "YEAR")) %>%
            dplyr::mutate(INCIDENCE_ADJ_CARESEEKING = N3 / POPULATION * 1000)
    } else {
        yearly_incidence <- yearly_incidence |>
            dplyr::mutate(INCIDENCE_ADJ_CARESEEKING = NA)
    }

    yearly_incidence
}


export_monthly_cases <- function(monthly_cases) {
    file_path <- file.path(INTERMEDIATE_DATA_PATH, paste0(COUNTRY_CODE, "_monthly_cases.parquet"))
    arrow::write_parquet(monthly_cases, file_path)
    log_msg(glue("Monthly cases data saved to: {file_path}"))
}


coherence_check_PRES <- function(monthly_cases) {
    # Run this check only if N1_METHOD == "PRES" (else, problem doesn't exist)
    if (N1_METHOD == "PRES") {
        nr_of_pres_0_adm2_month <<- monthly_cases |> filter(PRES == 0) |> nrow()
        log_msg(glue("🚨 Note: using `PRES` for incidence adjustement, but `PRES == 0` for {nr_of_pres_0_adm2_month} rows (ADM2 x MONTH)."), "warning")
    } else {
        log_msg("N1_METHOD is not set to 'PRES'. No need to check for coherence of `PRES` column.")
    }
}

coherence_check_SUSP_TEST <- function(monthly_cases) {
  # Logically, there should not be more tested cases than suspected cases. 
    if (N1_METHOD == "SUSP-TEST") {
        nr_of_negative <<- monthly_cases |> mutate(SUSP_minus_TEST = SUSP - TEST) |> filter(SUSP_minus_TEST < 0) |> nrow() 
        if (nr_of_negative > 0) {
            log_msg(
            glue("🚨 Note: using formula `SUSP - TEST` for incidence adjustement, but higher tested than suspected cases (`SUSP < TEST`) detected in {nr_of_negative} rows (ADM2 x MONTH)."),
            "warning"
            )
        } else {
            log_msg("✅ Note: using `SUSP - TEST` for incidence adjustment, no cases where `TEST > SUSP` detected.")
        }
    } else {
        log_msg("N1_METHOD is not set to 'SUSP-TEST'. No need to check for coherence of `SUSP` and `TEST` columns.")
    }
}


coherence_check_CONF_TEST <- function(monthly_cases) {
    more_confirmed_than_tested <<- monthly_cases |> mutate(CONF_divby_TEST = CONF / TEST) |> filter(CONF_divby_TEST > 1) |> nrow() 
    if (more_confirmed_than_tested > 0) {
        log_msg(glue("🚨 Note: higher confirmed than tested cases (`CONF/TEST`) detected in {more_confirmed_than_tested} rows (ADM2 x MONTH)."), "warning")
    } else {
        log_msg("✅ Note: no cases where `CONF > TEST` detected.")
    }
}


coherence_checkes_yearly_incidence <- function(yearly_incidence, incidence_col_1, incidence_col_2) {
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
  log_msg(paste0("Exporting : ", file_path))
}