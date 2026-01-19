## Reporting Rate Calculation: Data Element Availability Method

### Short description
This pipeline estimates routine health facility reporting rates using HMIS data and facility metadata.

**Numerator Calculation**
The numerator is calculated as the number of facilities reporting in the specific month. A facility is counted if it submitted a non-missing, positive value for at least one **Activity Indicator** during that period.

**Denominator Calculation:**<br>
The denominator defines the number of facilities _expected_ to report. The user can choose between two approaches:
- **Active Facilities (Annual):** The denominator is the total number of unique facilities that reported **at least once during the entire year**.
- **Operational Status (Monthly):** The denominator is the number of facilities considered **operational** during the specific month. Operational status is based on pyramid opening/closing dates, excludes facilities explicitly named "closed", and requires the facility to have **ever reported data** in the past.

---

### Pipeline parameters

- **Outliers detection method**: Specify which method was used to detect outliers in routine data. Choose "Routine data (Raw)" to use raw routine data.
    
- **Use routine with outliers removed**: Toggle this on to use the routine data after outliers have been removed (using the outliers detection method selected above). Else, this pipeline will use either the imputed routine data (to replace the outlier values removed) or the raw routine data if you selected "Routine data (Raw)" as your choice of “Outlier processing method”.
    
- **Use weighted reporting rates**: Toggle this on to apply weights to the reporting rates based on the volume of activity. If enabled, facilities with higher volume (based on "Volume activity indicators") contribute more to the final rate.
    
- **Activity indicators**: Define which data elements will be used to determine the activity of a facility. A facility–period is considered “active” (code variable: `ACTIVE_THIS_PERIOD`) if at least one of these indicators has a non-missing value greater than zero.
    
- **Volume activity indicators**: Indicators selected to determine the volume of activity. These indicators are used to compute a **weight** for each health facility based on its average reported volume.
    - _Note:_ These weights are applied to **both** the numerator and denominator when "Use weighted reporting rates" is enabled.
        
- **Denominator method**: Choose which method to use to determine the denominator (number of facilities expected to report).
    - `ROUTINE_ACTIVE_FACILITIES`: The denominator is the number of facilities that were active (reported at least once) during the current year.
    - `PYRAMID_OPEN_FACILITIES`: The denominator is the number of facilities considered "Open" (operational) during the period based on metadata and history.