# **Dataset Reporting Rate: Calculation Based on DHIS2 Extracted Data**

The **reporting rate** measures the proportion of registered health facilities that submit data. It is calculated for each administrative level 2 (`ADM2`) area and for each reporting period (`PERIOD` in YYYYMM format).
<br>

**Dataset Selection**<br>
The choice of dataset(s) used for reporting rate calculation is controlled by modifying the <code>SNT_config.json</code> configuration file. This allows flexible selection among multiple datasets extracted from the same DHIS2 instance.

**Calculation Logic**<br>
From the selected dataset(s):
- **Numerator:** Number of facilities that _actually_ reported, derived from the element <code>"ACTUAL_REPORTS"</code>.
- **Denominator:** Number of facilities _expected_ to report, derived from the element <code>"EXPECTED_REPORTS"</code>.

After aggregating these counts at the ADM2 level, the reporting rate is computed as:
<br>
<code>REPORTING RATE = ACTUAL_REPORTS / EXPECTED_REPORTS</code>
<br>
and expressed as a **proportion** between 0 and 1.
<br>

-----

### Additional Data Processing Steps

- **Handling Multiple Datasets:**  
  When multiple datasets are available, the pipeline uses only those specified in <code>SNT_config.json</code>. For these selected datasets, the counts of actual and expected reports are summed by ADM2 area.

- **Deduplication of Entries:**  
  Sometimes, the same organizational unit (<code>OU_ID</code>) may appear in multiple datasets for the same period, risking double counting. To address this, deduplication is performed by keeping only the entry with the **highest** <code>ACTUAL_REPORTS</code> value for each unique combination of <code>OU_ID</code> and <code>PERIOD</code>.  
  <ul>
    <li><strong>Why keep the highest?</strong> Because <code>ACTUAL_REPORTS</code> values are binary (0 or 1). If duplicates agree (all 0 or all 1), keeping one suffices. If they differ (some 0, some 1), keeping the 1 ensures that presence of a report is not missed.</li>
    <li><strong>🚨Important:</strong> Deduplication only proceeds if all duplicated values are within {0,1}. If other values are present, deduplication is skipped with a warning to avoid incorrect data handling.</li>
  </ul>

-----


### 🇳🇪 <strong>Niger-Specific Processing:</strong>  
  In Niger, datasets for <strong>HOP</strong> (hospital) facilities are already **pre-aggregated** and may contain values greater than 1 for actual or expected reports, reflecting subunits or departments within a hospital. 
  <br>
  To accurately represent reporting at the facility level and avoid overcounting, all values greater than 1 are converted to 1 (presence/absence). This ensures that the reporting rate reflects whether the hospital as a whole reported, rather than counting multiple subunits separately. This step also prevents cases where <code>ACTUAL_REPORTS</code> exceeds <code>EXPECTED_REPORTS</code>.

------

### Pipeline parameters

- **Outliers detection method**: Specify which method was used to detect outliers in routine data. Choose "Routine data (Raw)" to use raw routine data.
    
- **Use routine with outliers removed**: Toggle this on to use the routine data after outliers have been removed (using the outliers detection method selected above). Else, this pipeline will use either the imputed routine data (to replace the outlier values removed) or the raw routine data if you selected "Routine data (Raw)" as your choice of “Outlier processing method”.