# **SNT ERA5 Rainfall Seasonality Pipeline**

This pipeline estimates **the existence and duration of rainfall seasonality** using ERA5 rainfall data and DHIS2 spatial and health pyramid data. It uses the algorithm recommended by [the World Health Organization](https://iris.who.int/server/api/core/bitstreams/78b4fed5-a372-4f87-8a04-edec47e092d5/content) to establish which administrative units exhibit rainfall patterns that warrant seasonal malaria chemoprevention strategies.

Rainfall is often used as a proxy for malaria cases because malaria transmission depends heavily on environmental conditions, especially on water availability. This is because malaria is transmitted by female _Anopheles_ mosquitoes, which lay their eggs in standing water. In this context, increased rainfall, which creates more water bodies, expands breeding sites for mosquitoes.

The primary reason for using rainfall to proxy malarial cases, is because rainfall data are widely available whereas in certain areas, case surveillance data may be incomplete.

The pipeline does as follows, for each administrative unit:
1. Audit and preprocess rainfall data
2. Create month-blocks (windows) of several, pre-determined durations
3. Compute the amount of rainfall in each block/window, proportional to a year's amount
3. Classify each administrative unit as seasonal or non-seasonal, based on the result obtained at step 3.
4. Establish the (minimal) duration of the seasonal block/window


## **Parameters**

* **`get_minimum_month_block_size` (Integer, required)**
    * **Name**: Minimum number of months per block
    * **Description**: The shortest duration (in number of months) during which the existence of seasonality will be analyzed
    * **Choices**: 3, 4, 5
    * **Default**: 4
* **`get_maximum_month_block_size` (Integer, required)**
    * **Name**: Maximum number of months per block
    * **Description**: The longest duration (in number of months) during which the existence of seasonality will be analyzed
    * **Choices**: 3, 4, 5
    * **Default**: 4
* **`get_threshold_for_seasonality`(Float, required)**
    * **Name**: Minimal proportion of rainfall for seasonality
    * **Description**: The minimal amount of rainfall during a given block/window of time, relative to the rainfall during a whole year, to warrant considering that rainfall followed a seasonal pattern in the given administrative unit and year
    * **Default**: 0.6
* **`get_threshold_proportion_seasonal_years`(Float, optional)**
    * **Name**: Minimal proportion of seasonal years
    * **Description**: The minimal amount of years where the seasonal pattern occurs, relative to the total number of years in the analysis, for the administrative unit to be flagged as seasonal from a rainfall perspective
    * **Default**: 0.5
* **`run_report_only` (Boolean, optional)**
    * **Name**: Run reporting only
    * **Description**:
    * **Default**: False
* **`pull_scripts` (Boolean, optional)**
    * **Name**: Pull scripts
    * **Description**: Pull the latest scripts from the repository
    * **Default**: False

## **Functionality Overview**

The pipeline performs the following main operations:

1. **Data Loading**: Loads the DHIS2 health pyramid data and the ERA5 rainfall data
2. **Audit of data quality**: Evaluates if the input data is of sufficient quality to perform the analyses; in the negative, the pipeline stops  
3. **Preprocessing**: Identifies if missing data is present; in the affirmative, a Seasonal Autoregressive Integrated Moving Average (SARIMA) model is employed, to impute the missing values for each administrative unit; then, it formats the data for analysis
4. **Rainfall seasonal patterns per observation**: For each combination of:
    - administrative unit
    - year
    - duration (number of months per block/window)
    - starting month of the block/window
