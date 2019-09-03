---
interact_link: content/portfolio/dengai/00-dengai.ipynb
kernel_name: python3
has_widgets: false
title: '00 - Data Preprocessing'
prev_page:
  url: /portfolio/dengai/README
  title: 'DengAI'
next_page:
  url: /portfolio/dengai/01-dengai
  title: '01 - Naive Regressors'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# DengAI Analysis Part 00 - Data Preprocessing

By: Chengyi (Jeff) Chen, under guidance of CSCI499: AI for Social Good Teaching Assistant - Aaron Ferber

---
## Description

Predicting Dengue based on environment features in San Juan and Iquitos

https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/

---
## Download Links

Here are the training features and labels given by the competition link.
The training data has 1,456 rows and 24 columns with some nan entries.

- features: https://www.dropbox.com/s/1kuf94b4mk6axyy/dengue_features_train.csv?dl=1
- labels: https://www.dropbox.com/s/626ak8397abonv4/dengue_labels_train.csv?dl=1
- test features: https://s3.amazonaws.com:443/drivendata/data/44/public/dengue_features_test.csv

---
## Objective

- The goal is to predict the `total_cases` label for each `(city, year, weekofyear)` in the test set. There are two cities, San Juan and Iquitos, with test data for each `city` spanning 5 and 3 years respectively. You will make one submission that contains predictions for both cities. The data for each city have been concatenated along with a city column indicating the source: `sj` for San Juan and `iq` for Iquitos. The test set is a pure future hold-out, meaning the test data are sequential and non-overlapping with any of the training data. Throughout, missing values have been filled as `NaN`s.



<a id="imports"></a>

---
## Library Imports



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Library Imports
import pandas as pd
import numpy as np
import subprocess
import statsmodels.formula.api as sm
from statsmodels import stats as sms
import statsmodels as statsmodels
from scipy import stats
import os
from collections import Counter
from sklearn import model_selection, kernel_ridge, linear_model, metrics, feature_selection, preprocessing
from os import listdir
from os.path import isfile, join, isdir

# plotting libraries
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import seaborn as sns
sns.set(style="ticks")
from pylab import rcParams
%matplotlib inline

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# DengAI dataset URLs
dengai_features_url = 'https://www.dropbox.com/s/1kuf94b4mk6axyy/dengue_features_train.csv'
dengai_labels_url = 'https://www.dropbox.com/s/626ak8397abonv4/dengue_labels_train.csv'
dengai_test_features_url = 'https://s3.amazonaws.com:443/drivendata/data/44/public/dengue_features_test.csv'
dengai_submission_format_url = 'https://s3.amazonaws.com:443/drivendata/data/44/public/submission_format.csv'

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Helper function to create a new folder
def mkdir(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
        else:
            print("(%s) already exists" % (path))

```
</div>

</div>



<a id="data"></a>

---
## Downloading Dataset



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Directory of dengAI data
DENGAI_PATH = 'data/dengai'

# Remove any old data for dengAI
subprocess.call(['rm', '-rf', DENGAI_PATH])

# Make the directory for the dengAI data
mkdir(DENGAI_PATH + '/features')
mkdir(DENGAI_PATH + '/labels')
mkdir(DENGAI_PATH + '/test_features')

# Download data into data/dengai in the current directory
# run !ls data/dengai to see the csv it downloaded
!wget --directory-prefix=data/dengai/features -Nq https://www.dropbox.com/s/1kuf94b4mk6axyy/dengue_features_train.csv
!wget --directory-prefix=data/dengai/labels -Nq https://www.dropbox.com/s/626ak8397abonv4/dengue_labels_train.csv
!wget --directory-prefix=data/dengai/test_features -Nq https://s3.amazonaws.com:443/drivendata/data/44/public/dengue_features_test.csv
!wget --directory-prefix=data/dengai -Nq https://s3.amazonaws.com:443/drivendata/data/44/public/submission_format.csv

```
</div>

</div>



<a id="feats"></a>

---
## Features



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
raw_features_df = pd.read_csv('./data/dengai/features/dengue_features_train.csv', index_col='week_start_date')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
raw_features_df.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>...</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
    <tr>
      <th>week_start_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-04-30</th>
      <td>sj</td>
      <td>1990</td>
      <td>18</td>
      <td>0.122600</td>
      <td>0.103725</td>
      <td>0.198483</td>
      <td>0.177617</td>
      <td>12.42</td>
      <td>297.572857</td>
      <td>297.742857</td>
      <td>...</td>
      <td>32.00</td>
      <td>73.365714</td>
      <td>12.42</td>
      <td>14.012857</td>
      <td>2.628571</td>
      <td>25.442857</td>
      <td>6.900000</td>
      <td>29.4</td>
      <td>20.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>1990-05-07</th>
      <td>sj</td>
      <td>1990</td>
      <td>19</td>
      <td>0.169900</td>
      <td>0.142175</td>
      <td>0.162357</td>
      <td>0.155486</td>
      <td>22.82</td>
      <td>298.211429</td>
      <td>298.442857</td>
      <td>...</td>
      <td>17.94</td>
      <td>77.368571</td>
      <td>22.82</td>
      <td>15.372857</td>
      <td>2.371429</td>
      <td>26.714286</td>
      <td>6.371429</td>
      <td>31.7</td>
      <td>22.2</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>1990-05-14</th>
      <td>sj</td>
      <td>1990</td>
      <td>20</td>
      <td>0.032250</td>
      <td>0.172967</td>
      <td>0.157200</td>
      <td>0.170843</td>
      <td>34.54</td>
      <td>298.781429</td>
      <td>298.878571</td>
      <td>...</td>
      <td>26.10</td>
      <td>82.052857</td>
      <td>34.54</td>
      <td>16.848571</td>
      <td>2.300000</td>
      <td>26.714286</td>
      <td>6.485714</td>
      <td>32.2</td>
      <td>22.8</td>
      <td>41.4</td>
    </tr>
    <tr>
      <th>1990-05-21</th>
      <td>sj</td>
      <td>1990</td>
      <td>21</td>
      <td>0.128633</td>
      <td>0.245067</td>
      <td>0.227557</td>
      <td>0.235886</td>
      <td>15.36</td>
      <td>298.987143</td>
      <td>299.228571</td>
      <td>...</td>
      <td>13.90</td>
      <td>80.337143</td>
      <td>15.36</td>
      <td>16.672857</td>
      <td>2.428571</td>
      <td>27.471429</td>
      <td>6.771429</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1990-05-28</th>
      <td>sj</td>
      <td>1990</td>
      <td>22</td>
      <td>0.196200</td>
      <td>0.262200</td>
      <td>0.251200</td>
      <td>0.247340</td>
      <td>7.52</td>
      <td>299.518571</td>
      <td>299.664286</td>
      <td>...</td>
      <td>12.20</td>
      <td>80.460000</td>
      <td>7.52</td>
      <td>17.210000</td>
      <td>3.014286</td>
      <td>28.942857</td>
      <td>9.371429</td>
      <td>35.0</td>
      <td>23.9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
raw_features_df.columns

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Index(['city', 'year', 'weekofyear', 'ndvi_ne', 'ndvi_nw', 'ndvi_se',
       'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm'],
      dtype='object')
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
raw_features_df.describe()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>weekofyear</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>...</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1456.000000</td>
      <td>1456.000000</td>
      <td>1262.000000</td>
      <td>1404.000000</td>
      <td>1434.000000</td>
      <td>1434.000000</td>
      <td>1443.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>...</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1443.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1413.000000</td>
      <td>1413.000000</td>
      <td>1436.000000</td>
      <td>1442.000000</td>
      <td>1434.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2001.031593</td>
      <td>26.503434</td>
      <td>0.142294</td>
      <td>0.130553</td>
      <td>0.203783</td>
      <td>0.202305</td>
      <td>45.760388</td>
      <td>298.701852</td>
      <td>299.225578</td>
      <td>295.246356</td>
      <td>...</td>
      <td>40.151819</td>
      <td>82.161959</td>
      <td>45.760388</td>
      <td>16.746427</td>
      <td>4.903754</td>
      <td>27.185783</td>
      <td>8.059328</td>
      <td>32.452437</td>
      <td>22.102150</td>
      <td>39.326360</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.408314</td>
      <td>15.019437</td>
      <td>0.140531</td>
      <td>0.119999</td>
      <td>0.073860</td>
      <td>0.083903</td>
      <td>43.715537</td>
      <td>1.362420</td>
      <td>1.261715</td>
      <td>1.527810</td>
      <td>...</td>
      <td>43.434399</td>
      <td>7.153897</td>
      <td>43.715537</td>
      <td>1.542494</td>
      <td>3.546445</td>
      <td>1.292347</td>
      <td>2.128568</td>
      <td>1.959318</td>
      <td>1.574066</td>
      <td>47.455314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>1.000000</td>
      <td>-0.406250</td>
      <td>-0.456100</td>
      <td>-0.015533</td>
      <td>-0.063457</td>
      <td>0.000000</td>
      <td>294.635714</td>
      <td>294.892857</td>
      <td>289.642857</td>
      <td>...</td>
      <td>0.000000</td>
      <td>57.787143</td>
      <td>0.000000</td>
      <td>11.715714</td>
      <td>1.357143</td>
      <td>21.400000</td>
      <td>4.528571</td>
      <td>26.700000</td>
      <td>14.700000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1997.000000</td>
      <td>13.750000</td>
      <td>0.044950</td>
      <td>0.049217</td>
      <td>0.155087</td>
      <td>0.144209</td>
      <td>9.800000</td>
      <td>297.658929</td>
      <td>298.257143</td>
      <td>294.118929</td>
      <td>...</td>
      <td>13.055000</td>
      <td>77.177143</td>
      <td>9.800000</td>
      <td>15.557143</td>
      <td>2.328571</td>
      <td>26.300000</td>
      <td>6.514286</td>
      <td>31.100000</td>
      <td>21.100000</td>
      <td>8.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2002.000000</td>
      <td>26.500000</td>
      <td>0.128817</td>
      <td>0.121429</td>
      <td>0.196050</td>
      <td>0.189450</td>
      <td>38.340000</td>
      <td>298.646429</td>
      <td>299.289286</td>
      <td>295.640714</td>
      <td>...</td>
      <td>27.245000</td>
      <td>80.301429</td>
      <td>38.340000</td>
      <td>17.087143</td>
      <td>2.857143</td>
      <td>27.414286</td>
      <td>7.300000</td>
      <td>32.800000</td>
      <td>22.200000</td>
      <td>23.850000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2005.000000</td>
      <td>39.250000</td>
      <td>0.248483</td>
      <td>0.216600</td>
      <td>0.248846</td>
      <td>0.246982</td>
      <td>70.235000</td>
      <td>299.833571</td>
      <td>300.207143</td>
      <td>296.460000</td>
      <td>...</td>
      <td>52.200000</td>
      <td>86.357857</td>
      <td>70.235000</td>
      <td>17.978214</td>
      <td>7.625000</td>
      <td>28.157143</td>
      <td>9.566667</td>
      <td>33.900000</td>
      <td>23.300000</td>
      <td>53.900000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2010.000000</td>
      <td>53.000000</td>
      <td>0.508357</td>
      <td>0.454429</td>
      <td>0.538314</td>
      <td>0.546017</td>
      <td>390.600000</td>
      <td>302.200000</td>
      <td>302.928571</td>
      <td>298.450000</td>
      <td>...</td>
      <td>570.500000</td>
      <td>98.610000</td>
      <td>390.600000</td>
      <td>20.461429</td>
      <td>16.028571</td>
      <td>30.800000</td>
      <td>15.800000</td>
      <td>42.200000</td>
      <td>25.600000</td>
      <td>543.300000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 22 columns</p>
</div>
</div>


</div>
</div>
</div>



<a id="missing_data"></a>

---
## Missing Data



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def feat_col_nans(df):
    '''
    FUNCTION: To display stacked bar chart of the NaNs vs Non-NaNs of each column in dataframe provided.
    
    Parameters
    ----------
    df: (pandas.DataFrame) Dataframe
    
    Returns
    -------
    Nothing. Displays the Stacked Bar chart
    '''
    # Get number of NaNs for each column
    nan_count_per_col = len(df) - df.count(axis=0)
    
    # Graph configs
    rcParams['figure.figsize'] = 15, 8
    sns.set()
    
    # Create NaNs
    plt.bar(np.arange(len(df.columns)), nan_count_per_col, color='red', edgecolor='white', label='NaNs')
    
    # Create Non-NaNs
    plt.bar(np.arange(len(df.columns)), df.count(), bottom=nan_count_per_col, color='skyblue', edgecolor='white', label='Non-NaNs')
    
    plt.title('Stacked Bar Chart of NaNs and Non-NaNs in each Feature', fontsize=20)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Features', fontsize=16)
    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=90, fontsize=7)
    plt.grid()
    plt.legend(loc='upper left');
    
feat_col_nans(raw_features_df)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/00-dengai_13_0.png)

</div>
</div>
</div>



__Seems like all the features don't have much missing data, so we don't have to remove any feature columns__



__Replace missing values with most frequent value__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Which columns are the ones with NaN
raw_features_df.isna().any()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
city                                     False
year                                     False
weekofyear                               False
ndvi_ne                                   True
ndvi_nw                                   True
ndvi_se                                   True
ndvi_sw                                   True
precipitation_amt_mm                      True
reanalysis_air_temp_k                     True
reanalysis_avg_temp_k                     True
reanalysis_dew_point_temp_k               True
reanalysis_max_air_temp_k                 True
reanalysis_min_air_temp_k                 True
reanalysis_precip_amt_kg_per_m2           True
reanalysis_relative_humidity_percent      True
reanalysis_sat_precip_amt_mm              True
reanalysis_specific_humidity_g_per_kg     True
reanalysis_tdtr_k                         True
station_avg_temp_c                        True
station_diur_temp_rng_c                   True
station_max_temp_c                        True
station_min_temp_c                        True
station_precip_mm                         True
dtype: bool
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Taking care of missing data
raw_features_df.fillna(raw_features_df.mode().iloc[0], inplace=True)
feat_col_nans(raw_features_df)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/00-dengai_17_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Check the datatypes of the feature columns
raw_features_df.info()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
<class 'pandas.core.frame.DataFrame'>
Index: 1456 entries, 1990-04-30 to 2010-06-25
Data columns (total 23 columns):
city                                     1456 non-null object
year                                     1456 non-null int64
weekofyear                               1456 non-null int64
ndvi_ne                                  1456 non-null float64
ndvi_nw                                  1456 non-null float64
ndvi_se                                  1456 non-null float64
ndvi_sw                                  1456 non-null float64
precipitation_amt_mm                     1456 non-null float64
reanalysis_air_temp_k                    1456 non-null float64
reanalysis_avg_temp_k                    1456 non-null float64
reanalysis_dew_point_temp_k              1456 non-null float64
reanalysis_max_air_temp_k                1456 non-null float64
reanalysis_min_air_temp_k                1456 non-null float64
reanalysis_precip_amt_kg_per_m2          1456 non-null float64
reanalysis_relative_humidity_percent     1456 non-null float64
reanalysis_sat_precip_amt_mm             1456 non-null float64
reanalysis_specific_humidity_g_per_kg    1456 non-null float64
reanalysis_tdtr_k                        1456 non-null float64
station_avg_temp_c                       1456 non-null float64
station_diur_temp_rng_c                  1456 non-null float64
station_max_temp_c                       1456 non-null float64
station_min_temp_c                       1456 non-null float64
station_precip_mm                        1456 non-null float64
dtypes: float64(20), int64(2), object(1)
memory usage: 273.0+ KB
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# The two cities in our dataset
raw_features_df['city'].unique()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array(['sj', 'iq'], dtype=object)
```


</div>
</div>
</div>



__Let's remove the time-related variables like `year` and `weekofyear` from the dataset and store the rest as our input variables for our model__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
features_df = raw_features_df.drop(['year', 'weekofyear'], axis=1)

```
</div>

</div>



__Let's keep a dataframe for each city's data separately so that we are building 2 models for each machine learning technique we use, 1. Only City sj and 2. Only City iq__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Let's split our data to one for city sj and one for city iq
sj_X = features_df[features_df['city'] == 'sj'].drop(['city'], axis=1)
sj_X.index = pd.to_datetime(sj_X.index)
sj_X.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>reanalysis_min_air_temp_k</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
    <tr>
      <th>week_start_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-04-30</th>
      <td>0.122600</td>
      <td>0.103725</td>
      <td>0.198483</td>
      <td>0.177617</td>
      <td>12.42</td>
      <td>297.572857</td>
      <td>297.742857</td>
      <td>292.414286</td>
      <td>299.8</td>
      <td>295.9</td>
      <td>32.00</td>
      <td>73.365714</td>
      <td>12.42</td>
      <td>14.012857</td>
      <td>2.628571</td>
      <td>25.442857</td>
      <td>6.900000</td>
      <td>29.4</td>
      <td>20.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>1990-05-07</th>
      <td>0.169900</td>
      <td>0.142175</td>
      <td>0.162357</td>
      <td>0.155486</td>
      <td>22.82</td>
      <td>298.211429</td>
      <td>298.442857</td>
      <td>293.951429</td>
      <td>300.9</td>
      <td>296.4</td>
      <td>17.94</td>
      <td>77.368571</td>
      <td>22.82</td>
      <td>15.372857</td>
      <td>2.371429</td>
      <td>26.714286</td>
      <td>6.371429</td>
      <td>31.7</td>
      <td>22.2</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>1990-05-14</th>
      <td>0.032250</td>
      <td>0.172967</td>
      <td>0.157200</td>
      <td>0.170843</td>
      <td>34.54</td>
      <td>298.781429</td>
      <td>298.878571</td>
      <td>295.434286</td>
      <td>300.5</td>
      <td>297.3</td>
      <td>26.10</td>
      <td>82.052857</td>
      <td>34.54</td>
      <td>16.848571</td>
      <td>2.300000</td>
      <td>26.714286</td>
      <td>6.485714</td>
      <td>32.2</td>
      <td>22.8</td>
      <td>41.4</td>
    </tr>
    <tr>
      <th>1990-05-21</th>
      <td>0.128633</td>
      <td>0.245067</td>
      <td>0.227557</td>
      <td>0.235886</td>
      <td>15.36</td>
      <td>298.987143</td>
      <td>299.228571</td>
      <td>295.310000</td>
      <td>301.4</td>
      <td>297.0</td>
      <td>13.90</td>
      <td>80.337143</td>
      <td>15.36</td>
      <td>16.672857</td>
      <td>2.428571</td>
      <td>27.471429</td>
      <td>6.771429</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1990-05-28</th>
      <td>0.196200</td>
      <td>0.262200</td>
      <td>0.251200</td>
      <td>0.247340</td>
      <td>7.52</td>
      <td>299.518571</td>
      <td>299.664286</td>
      <td>295.821429</td>
      <td>301.9</td>
      <td>297.5</td>
      <td>12.20</td>
      <td>80.460000</td>
      <td>7.52</td>
      <td>17.210000</td>
      <td>3.014286</td>
      <td>28.942857</td>
      <td>9.371429</td>
      <td>35.0</td>
      <td>23.9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
iq_X = features_df[features_df['city'] == 'iq'].drop(['city'], axis=1)
iq_X.index = pd.to_datetime(iq_X.index)
iq_X.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>reanalysis_min_air_temp_k</th>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
    </tr>
    <tr>
      <th>week_start_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-07-01</th>
      <td>0.192886</td>
      <td>0.132257</td>
      <td>0.340886</td>
      <td>0.247200</td>
      <td>25.41</td>
      <td>296.740000</td>
      <td>298.450000</td>
      <td>295.184286</td>
      <td>307.3</td>
      <td>293.1</td>
      <td>43.19</td>
      <td>92.418571</td>
      <td>25.41</td>
      <td>16.651429</td>
      <td>8.928571</td>
      <td>26.400000</td>
      <td>10.775000</td>
      <td>32.5</td>
      <td>20.7</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2000-07-08</th>
      <td>0.216833</td>
      <td>0.276100</td>
      <td>0.289457</td>
      <td>0.241657</td>
      <td>60.61</td>
      <td>296.634286</td>
      <td>298.428571</td>
      <td>295.358571</td>
      <td>306.6</td>
      <td>291.1</td>
      <td>46.00</td>
      <td>93.581429</td>
      <td>60.61</td>
      <td>16.862857</td>
      <td>10.314286</td>
      <td>26.900000</td>
      <td>11.566667</td>
      <td>34.0</td>
      <td>20.8</td>
      <td>55.6</td>
    </tr>
    <tr>
      <th>2000-07-15</th>
      <td>0.176757</td>
      <td>0.173129</td>
      <td>0.204114</td>
      <td>0.128014</td>
      <td>55.52</td>
      <td>296.415714</td>
      <td>297.392857</td>
      <td>295.622857</td>
      <td>304.5</td>
      <td>292.6</td>
      <td>64.77</td>
      <td>95.848571</td>
      <td>55.52</td>
      <td>17.120000</td>
      <td>7.385714</td>
      <td>26.800000</td>
      <td>11.466667</td>
      <td>33.0</td>
      <td>20.7</td>
      <td>38.1</td>
    </tr>
    <tr>
      <th>2000-07-22</th>
      <td>0.227729</td>
      <td>0.145429</td>
      <td>0.254200</td>
      <td>0.200314</td>
      <td>5.60</td>
      <td>295.357143</td>
      <td>296.228571</td>
      <td>292.797143</td>
      <td>303.6</td>
      <td>288.6</td>
      <td>23.96</td>
      <td>87.234286</td>
      <td>5.60</td>
      <td>14.431429</td>
      <td>9.114286</td>
      <td>25.766667</td>
      <td>10.533333</td>
      <td>31.5</td>
      <td>14.7</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2000-07-29</th>
      <td>0.328643</td>
      <td>0.322129</td>
      <td>0.254371</td>
      <td>0.361043</td>
      <td>62.76</td>
      <td>296.432857</td>
      <td>297.635714</td>
      <td>293.957143</td>
      <td>307.0</td>
      <td>291.5</td>
      <td>31.80</td>
      <td>88.161429</td>
      <td>62.76</td>
      <td>15.444286</td>
      <td>9.500000</td>
      <td>26.600000</td>
      <td>11.480000</td>
      <td>33.3</td>
      <td>19.1</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



<a id="labels"></a>

---
## Labels



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
raw_labels_df = pd.read_csv('./data/dengai/labels/dengue_labels_train.csv')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
raw_labels_df.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sj</td>
      <td>1990</td>
      <td>18</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>1990</td>
      <td>19</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>1990</td>
      <td>20</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>1990</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>1990</td>
      <td>22</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# The total dengue cases for City sj
sj_y = raw_labels_df[raw_labels_df['city'] == 'sj']['total_cases']
sj_y.index = sj_X.index

# The total dengue cases for city iq
iq_y = raw_labels_df[raw_labels_df['city'] == 'iq']['total_cases']
iq_y.index = iq_X.index

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_y.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
week_start_date
1990-04-30    4
1990-05-07    5
1990-05-14    4
1990-05-21    3
1990-05-28    6
Name: total_cases, dtype: int64
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
iq_y.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
week_start_date
2000-07-01    0
2000-07-08    0
2000-07-15    0
2000-07-22    0
2000-07-29    0
Name: total_cases, dtype: int64
```


</div>
</div>
</div>



### Visualize target distribution
__Let's first start to visualize the frequency of the dengue cases against the time__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Output Images Settings
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
fig, ax = plt.subplots(2,1)

ax[0].plot(sj_X.index, sj_y)
ax[0].grid(True)
ax[0].set_title("Distribution of total cases (City sj)", fontsize=22, color='k')
ax[0].set_ylabel("Frequency", fontsize=22)
ax[0].set_xlabel("Date", fontsize=22)

ax[1].plot(iq_X.index, iq_y)
ax[1].grid(True)
ax[1].set_title("Distribution of total cases (City iq)", fontsize=22, color='k')
ax[1].set_ylabel("Frequency", fontsize=22)
ax[1].set_xlabel("Date", fontsize=22)

plt.tight_layout()
plt.show();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/00-dengai_32_0.png)

</div>
</div>
</div>



__Seems like Dengue cases spiked in the second half of 1994, second half of 1998, second half of 2005, second half of 2007 for City sj and spiked in the second half of 2004, start of 2008, and second half of 2008 for City iq. However, there is some seasonal pattern to the data, which is logical given that dengue mosquitos breed more often in certain climates (hot and humid), so maybe a Seasonal Arima later on would reveal more information about the underlying trend.__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Create a directory to store the cleaned data
mkdir('./data/dengai/cleaned')

# Save the X and y for each City
sj_X.to_csv(r'./data/dengai/cleaned/sj_X.csv')
sj_y.to_csv(r'./data/dengai/cleaned/sj_y.csv')
iq_X.to_csv(r'./data/dengai/cleaned/iq_X.csv')
iq_y.to_csv(r'./data/dengai/cleaned/iq_y.csv')

```
</div>

</div>



__In the next part of this project, we will work on building and training some naive regressors on this cleaned dataset__

