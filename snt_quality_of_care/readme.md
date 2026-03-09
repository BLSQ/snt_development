SNT Quality of Care Pipeline

Description

This pipeline computes district-year quality-of-care indicators from DHIS2 outliers-imputed routine data and generates yearly ADM2 maps.

Parameters

  outlier_imputation_method (String, required)
    Name: Outlier imputation method
    Description: Select which imputed routine file to load from DHIS2_OUTLIERS_IMPUTATION.
    Choices/Default: mean, median, iqr, trend, mg-partial, mg-complete. Default: mean.

  run_report_only (Boolean, optional)
    Name: Run reporting only
    Description: Skip computations and run only reporting notebook.
    Choices/Default: TRUE/FALSE. Default: FALSE.

  pull_scripts (Boolean, optional)
    Name: Pull scripts
    Description: Pull latest scripts from repository before run.
    Choices/Default: TRUE/FALSE. Default: FALSE.
