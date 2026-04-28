# Utils entrypoint for snt_dhs_vaccination_report.ipynb
if (!exists("ROOT_PATH", inherits = TRUE)) ROOT_PATH <- "~/workspace"
if (!exists("PIPELINE_PATH", inherits = TRUE)) PIPELINE_PATH <- file.path(ROOT_PATH, "pipelines", "snt_dhs_indicators")
if (!exists("CODE_PATH", inherits = TRUE)) CODE_PATH <- file.path(ROOT_PATH, "code")
if (file.exists(file.path(CODE_PATH, "snt_utils.r"))) source(file.path(CODE_PATH, "snt_utils.r"))
source(file.path(PIPELINE_PATH, "utils", "snt_dhs_indicator_tables.r"))
