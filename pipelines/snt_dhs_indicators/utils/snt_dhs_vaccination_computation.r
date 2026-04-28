# Utils entrypoint for snt_dhs_vaccination_computation.ipynb
if (!exists("ROOT_PATH", inherits = TRUE)) ROOT_PATH <- "~/workspace"
if (!exists("PIPELINE_PATH", inherits = TRUE)) PIPELINE_PATH <- file.path(ROOT_PATH, "pipelines", "snt_dhs_indicators")
source(file.path(PIPELINE_PATH, "utils", "snt_dhs_indicator_tables.r"))
