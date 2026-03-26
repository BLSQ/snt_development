# Report helpers for magic glasses outliers imputation pipeline.
.this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
.this_dir <- if (exists("PIPELINE_PATH", inherits = TRUE)) {
    file.path(get("PIPELINE_PATH", inherits = TRUE), "utils")
} else if (!is.na(.this_file)) {
    dirname(.this_file)
} else {
    getwd()
}
source(file.path(.this_dir, "snt_dhis2_outliers_imputation_magic_glasses.r"))

