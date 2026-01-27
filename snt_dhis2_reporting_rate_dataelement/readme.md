# **SNT DHIS2 Reporting Rate (Data Element) Pipeline**

This pipeline estimates **routine health facility reporting rates** using HMIS data and facility metadata. It calculates reporting rates by analyzing facility activity (reporting of specific malaria-related indicators) against operational status or annual activity, with options for outlier handling and volume-based weighting.

First, facility **activity** is assessed **monthly** using a small set of key **Facility Activity Indicators**. A facility is considered active in a month if at least one of these indicators has a non-missing value (zero values are counted as valid reports). These monthly activity signals are then used to determine whether a facility is active during the year, meaning it reported at least once in that year.

Separately, facility operational status is derived from the facility master (pyramid) dataset using opening and closing dates and explicit closure markers in facility names.
Reporting rates are computed by comparing monthly activity (**numerator** \= number of facilities active in the month) against one of **two denominators**:

* **Operational facilities**: facilities that are declared open in the month (based on opening/closing dates and excluding facilities marked as closed)
* **Annual active facilities**: facilities that reported at least once during the year

In addition to unweighted reporting rates, the pipeline also computes **weighted reporting rates**, where each facility is weighted by its average reported malaria workload based on the selected **Volume Activity Indicators**, so that high-volume facilities contribute more to the overall reporting rate.

## **Parameters**

* **`outliers_method`** (String, required):  
  * **Name:** Outliers detection method  
  * **Description:** Specifies which method was used to detect outliers in the input routine data. Select "Routine data (Raw)" to use raw data without outlier processing.  
  * **Choices:** Routine data (Raw), Mean (Classic), Median (Classic), IQR (Classic), Trend (PATH), MG Partial (MagicGlasses2), MG Complete (MagicGlasses2).  
  * **Default:** None  
* **`use_removed_outliers`** (Boolean, optional):  
  * **Name:** Use routine data with outliers removed  
  * **Description:** If enabled, the pipeline uses routine data where detected outliers have been removed (set to null). If disabled (default), it uses data where outliers have been imputed (replaced), or raw data if "Routine data (Raw)" was selected.  
  * **Default:** False  
* **`activity_indicators`** (List of Strings, required):  
  * **Name:** Facility Activity indicators  
  * **Description:** Defines the set of data elements used to determine if a facility is "active". A facility is considered active in a given period if at least one of these indicators has a non-missing value greater than or equal to zero.  
  * **Choices:** CONF, SUSP, TEST, PRES.  
  * **Default:** \['CONF', 'PRES'\]  
* **`volume_activity_indicators`** (List of Strings, required):  
  * **Name:** Volume activity indicators  
  * **Description:** Defines the set of data elements used to determine the volume of activity (workload). These indicators are used to calculate weights for the "Weighted Reporting Rates" calculation.  
  * **Choices:** CONF, SUSP, TEST, PRES.  
  * **Default:** \['CONF', 'PRES'\]  
* **`dataelement_method_denominator`** (String, required):  
  * **Name:** Denominator method  
  * **Description:** Determines how the total number of expected facilities (denominator) is calculated.  
    * ROUTINE\_ACTIVE\_FACILITIES: Denominator is the number of facilities active (reported at least once) during the entire current year.  
    * PYRAMID\_OPEN\_FACILITIES: Denominator is the number of facilities considered structurally "Open" (operational) during the specific period.  
  * **Default:** None  
* **`use_weighted_reporting_rates`** (Boolean, optional):  
  * **Name:** Use weighted reporting rates  
  * **Description:** If enabled, reporting rates are weighted based on the facility's volume of activity (derived from volume\_activity\_indicators). High-volume facilities will have a greater impact on the final aggregated rate.  
  * **Default:** False

## **Functionality Overview**

The pipeline performs the following key operations:

1. **Data Loading:** Loads the appropriate DHIS2 routine data file based on the selected outliers\_method (raw, imputed, or removed) and the facility metadata (pyramid).  
2. **Facility Activity Assessment:** Evaluates facility activity monthly. A facility is flagged as ACTIVE\_THIS\_PERIOD if it reports valid data (≥0) for any of the selected activity\_indicators.  
3. **Operational Status Check:** Identifies OPEN facilities by filtering out those with explicit closure keywords in their names (e.g., "CLOTUR", "FERME") and checking opening/closing dates against the reporting period.  
4. **Annual Activity Check:** Determines if a facility was ACTIVE\_THIS\_YEAR by checking if it reported data at least once during the calendar year.  
5. **Weight Calculation (Optional):** If weighting is enabled, calculates a weight for each facility based on the average monthly volume of the volume\_activity\_indicators.  
6. **Reporting Rate Calculation:** Computes the reporting rate aggregated to **Administrative Level 2 (ADM2)** and **Monthly resolution** using the selected denominator method:  
   * **Method 1 (Routine Active):** Active Facilities in Period / Active Facilities in Year.  
   * **Method 2 (Pyramid Open):** Active Facilities in Period / Open Facilities in Period.  
7. **Export:** Selects the appropriate reporting rate column (weighted or unweighted) and exports the data.

## **Inputs**

The pipeline requires the following inputs from the data lake:

* **DHIS2 Routine Data:** A parquet file containing routine health data. The specific file depends on the `outliers_method` parameter:  
  * Raw: \[COUNTRY\_CODE\]\_routine.parquet  
  * Processed: \[COUNTRY\_CODE\]\_routine\_outliers-\[method\]\_\[imputed|removed\].parquet  
* **DHIS2 Pyramid Data:** A file named "\[COUNTRY\_CODE\]\_pyramid.parquet" from the "SNT\_DHIS2\_FORMATTED" dataset, containing facility metadata (opening dates, hierarchy).  
* **Configuration:** The "SNT\_config.json" file containing country codes and dataset identifiers.

## **Outputs**

* **\[COUNTRY\_CODE\]\_reporting\_rate\_dataelement.parquet** and **\[COUNTRY\_CODE\]\_reporting\_rate\_dataelement.csv**: These files contain the calculated reporting rates aggregated at the **`ADM2` level** and **`MONTH`ly resolution**. Both files are saved to the DHIS2\_REPORTING\_RATE dataset.


> **Notes for the Data Analyst:**
> - **`REPORTING_RATE`**: This is the final calculated rate. Its definition > depends on the pipeline parameters:  
>   - If use\_weighted\_reporting\_rates is **True**, this contains the weighted rate.  
>    - If **False**, it contains the unweighted rate.  
>    - The denominator logic follows the dataelement\_method\_denominator selection.  
> - **`ADM2_ID`**: The unique identifier for the administrative level 2 area.  
> - **`YEAR`** & **`MONTH`**: The time period for the reporting rate.  
