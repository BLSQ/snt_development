# **SNT ERA5 Rainfall Seasonality Pipeline**

This pipeline estimates **the existence and duration of rainfall seasonality** using ERA5 rainfall data and DHIS2 spatial data. It uses the algorithm recommended by [the World Health Organization](https://iris.who.int/server/api/core/bitstreams/78b4fed5-a372-4f87-8a04-edec47e092d5/content) to establish which administrative units exhibit rainfall patterns that warrant seasonal malaria chemoprevention strategies.

Rainfall is often used as a proxy for malaria cases because malaria transmission depends heavily on environmental conditions, especially on water availability. This is because malaria is transmitted by female _Anopheles_ mosquitoes, which lay their eggs in standing water. In this context, increased rainfall, which creates more water bodies, expands breeding sites for mosquitoes.

The primary reason for using rainfall to proxy malarial cases, is because rainfall data are widely available whereas in certain areas, case surveillance data may be incomplete.

The pipeline does as follows, for each administrative unit:
1. Audit and preprocess rainfall data
2. Create month-blocks (windows) of several, pre-determined durations ({n})
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
    * **Description**: The longest duration (in number of months) during which the existence of seasonality will be analyzed; this value needs to be at least as large as `get_minimum_month_block_size`, or the pipeline will stop
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

1. **Data Loading**: Loads the DHIS2 spatial data and the ERA5 rainfall data
2. **Audit of data quality**: Evaluates if the input data is of sufficient quality to perform the analyses; in the negative, the pipeline stops  
3. **Preprocessing**: Identifies if missing data is present; in the affirmative, a Seasonal Autoregressive Integrated Moving Average (SARIMA) model is employed, to impute the missing values for each administrative unit; then, it formats the data for analysis
4. **Rainfall seasonal patterns per observation**: For each combination of:
    - administrative unit
    - year
    - duration (the {n} number of months per block/window)
    - starting month of the block/window:

    1. Computes the proportion of rainfall, relative to the whole year, using forward-facing sliding windows for both the numerator and the denominator
    2. Comparing the proportions computed at step 3.1. against the value of the `get_threshold_for_seasonality` parameter, the pipeline checks whether a given month of the given year, marks the beginning of a concentrated rainfall period for the administrative unit in question
    3. Flags the month in question as a valid start month for the respective administrative unit. This means that most of the year's rainfall falls within the {n}-month window beginning in that month and year
5. **Classify administrative units as seasonal or non-seasonal**: for each administrative unit, the pipeline evaluates whether the it consistently exhibits this rainfall concentration pattern. For each month and block duration ({n}), the pipeline calculates the proportion of years in which that month was flagged as a valid start (from Step 4.3.). If this proportion exceeds the `proportion_seasonal_years_threshold`, the district is classified as seasonal for that block duration. This ensures that the classification reflects a recurring climatic pattern, rather than an isolated event.
 
5. **Determine season duration**: Some districts may qualify as seasonal for multiple block lengths (such as both 4-month and 5-month windows). For each administrative unit, the pipeline compares all qualifying block durations {n} and selects the smallest one. This represents the shortest continuous period in which the required proportion of annual rainfall is concentrated.

6. **Visualization**: The results of the classification and of the season duration (steps 5. and 6.) are plotted as choropleth maps.

## **Inputs**

The pipeline requires the following inputs :
 
* **DHIS2 spatial data**: A file named "\[COUNTRY\_CODE\]\_shapes.geojson" from the "SNT\_DHIS2\_FORMATTED" dataset, containing the geometries of the administrative units.  
* **ERA5 rainfall data**: A file named "\[COUNTRY\_CODE\]\_total_precipitation_monthly.parquet" from the "SNT\_ERA5\_CLIMATE" dataset, containing the monthly amount of rainfall for each administrative unit. 
* **Configuration**: The "SNT\_config.json" file containing country codes and dataset identifiers.

## **Outputs**

* **Output tables**: these files contain the final output tables with both the classification and duration of the season for each administrative unit (`ADM2` level). They are saved to the SNT\_SEASONALITY\_RAINFALL dataset:
    - **\[COUNTRY\_CODE\]\_rainfall_seasonality.parquet**
    - **\[COUNTRY\_CODE\]\_rainfall_seasonality.csv**

* **Output report**: this file contains the visualization (plots and maps) of the results generated by the pipeline. It is saved as .html file in the workspace, under `pipelines/snt_seasonality_rainfall/reporting/outputs/`, and is also added to the pipeline run as output file:
    - **snt_seasonality\_rainfall\_report\_\[COUNTRY\_CODE\]\_OUTPUT_\[timestamp].html**

---------

> **Notes for the Data Analyst**:

> - **`SEASONALITY_RAINFALL`**: In the output table, this is the column which indicates final decision on whether or not an administrative unit has been classified as seasonal (1) or not (0).
> - **`SEASONAL_BLOCK_DURATION_RAINFALL`**: In the output table, this is the shortest duration ({n} number of months) for which the administrative unit qualifies as seasonal from a rainfall perspective. Missing values for this column are in direct link with the value 0 for the column `SEASONALITY_RAINFALL` and indicate that the administrative unit is not seasonal.
> - In the computation process, both the numerator and the denominator in the algorithm follow the World Health Organization approach, of creating forward-facing sliding windows.  
> - **`ADM2_ID`**: The unique identifier for the administrative level 2 area.  
> - **`YEAR`** & **`MONTH`**: Together, they form the time period.  
