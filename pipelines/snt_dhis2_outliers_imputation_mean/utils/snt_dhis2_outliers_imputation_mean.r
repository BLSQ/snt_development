# Main helpers for mean outliers imputation pipeline.
.this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
.this_dir <- if (!is.na(.this_file)) dirname(.this_file) else getwd()
source(file.path(.this_dir, "bootstrap.R"))
source(file.path(.this_dir, "imputation_utils.R"))

