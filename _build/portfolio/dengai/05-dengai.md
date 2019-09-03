---
interact_link: content/portfolio/dengai/05-dengai.ipynb
kernel_name: python3
has_widgets: false
title: '05 - TPOT'
prev_page:
  url: /portfolio/dengai/04-dengai
  title: '04 - Looking at the Benchmark'
next_page:
  url: /portfolio/dengai/06a-dengai
  title: '06a - Time Series Analysis'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# DengAI Analysis Part 05 - TPOT

By: Chengyi (Jeff) Chen, under guidance of CSCI499: AI for Social Good Teaching Assistant - Aaron Ferber

---
## Content

In this notebook, we will explore using an external API called [tpot](https://github.com/EpistasisLab/tpot) that automatically optimizes machine learning pipelines using genetic programming to help us find the best regressor for our dengue prediction.



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
import os
from collections import Counter
from sklearn import model_selection, kernel_ridge, linear_model, metrics, feature_selection, preprocessing
from os import listdir
from os.path import isfile, join, isdir
import warnings
warnings.filterwarnings('ignore')

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



---

## TPOT

__Let's run tpot on both City sj and iq data to generate the best possible regressors to be used for our prediction.__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

def preprocess_data(data_path, labels_path=None):
    """
    Fills all NaNs with the most recent value
    
    Parameters: 
    -----------
    data_path: (str) Path to location of the DengAI training set features
    labels_path: (str) Path to location of the DengAI training set labels
    
    Returns:
    --------
    The pandas dataframes of City SJ and IQ features with the labels at the last column
    """
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

def get_tpot_best(X_train, X_test, y_train, y_test, city):
    """
    Uses TPOT's Regressor optimizer to find the best regressor given the data provided
    
    Parameters: 
    -----------
    train_test_split numpy arrays and specification of which city's data this belongs to
    
    Returns:
    --------
    Nothing. It saves the optimum regressor into a python script
    """
    tpot = TPOTRegressor(generations=10, population_size=100,
                         offspring_size=None, mutation_rate=0.9,
                         crossover_rate=0.1,
                         scoring='neg_mean_absolute_error', cv=5,
                         subsample=1.0, n_jobs=-1,
                         max_time_mins=None, max_eval_time_mins=5,
                         random_state=None, config_dict=None,
                         template="RandomTree",
                         warm_start=False,
                         memory=None,
                         use_dask=False,
                         periodic_checkpoint_folder=None,
                         early_stop=None,
                         verbosity=0,
                         disable_update_check=False)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('{}_tpot_dengai_pipeline.py'.format(city))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Get City SJ and IQ data
sj_train, iq_train = preprocess_data(data_path='./data/dengai/features/dengue_features_train.csv', 
                                     labels_path='./data/dengai/labels/dengue_labels_train.csv')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_train.head()

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
      <th></th>
      <th>week_start_date</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>...</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
      <th>total_cases</th>
    </tr>
    <tr>
      <th>year</th>
      <th>weekofyear</th>
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
      <th rowspan="5" valign="top">1990</th>
      <th>18</th>
      <td>1990-04-30</td>
      <td>0.122600</td>
      <td>0.103725</td>
      <td>0.198483</td>
      <td>0.177617</td>
      <td>12.42</td>
      <td>297.572857</td>
      <td>297.742857</td>
      <td>292.414286</td>
      <td>299.8</td>
      <td>...</td>
      <td>73.365714</td>
      <td>12.42</td>
      <td>14.012857</td>
      <td>2.628571</td>
      <td>25.442857</td>
      <td>6.900000</td>
      <td>29.4</td>
      <td>20.0</td>
      <td>16.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1990-05-07</td>
      <td>0.169900</td>
      <td>0.142175</td>
      <td>0.162357</td>
      <td>0.155486</td>
      <td>22.82</td>
      <td>298.211429</td>
      <td>298.442857</td>
      <td>293.951429</td>
      <td>300.9</td>
      <td>...</td>
      <td>77.368571</td>
      <td>22.82</td>
      <td>15.372857</td>
      <td>2.371429</td>
      <td>26.714286</td>
      <td>6.371429</td>
      <td>31.7</td>
      <td>22.2</td>
      <td>8.6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1990-05-14</td>
      <td>0.032250</td>
      <td>0.172967</td>
      <td>0.157200</td>
      <td>0.170843</td>
      <td>34.54</td>
      <td>298.781429</td>
      <td>298.878571</td>
      <td>295.434286</td>
      <td>300.5</td>
      <td>...</td>
      <td>82.052857</td>
      <td>34.54</td>
      <td>16.848571</td>
      <td>2.300000</td>
      <td>26.714286</td>
      <td>6.485714</td>
      <td>32.2</td>
      <td>22.8</td>
      <td>41.4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1990-05-21</td>
      <td>0.128633</td>
      <td>0.245067</td>
      <td>0.227557</td>
      <td>0.235886</td>
      <td>15.36</td>
      <td>298.987143</td>
      <td>299.228571</td>
      <td>295.310000</td>
      <td>301.4</td>
      <td>...</td>
      <td>80.337143</td>
      <td>15.36</td>
      <td>16.672857</td>
      <td>2.428571</td>
      <td>27.471429</td>
      <td>6.771429</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>4.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1990-05-28</td>
      <td>0.196200</td>
      <td>0.262200</td>
      <td>0.251200</td>
      <td>0.247340</td>
      <td>7.52</td>
      <td>299.518571</td>
      <td>299.664286</td>
      <td>295.821429</td>
      <td>301.9</td>
      <td>...</td>
      <td>80.460000</td>
      <td>7.52</td>
      <td>17.210000</td>
      <td>3.014286</td>
      <td>28.942857</td>
      <td>9.371429</td>
      <td>35.0</td>
      <td>23.9</td>
      <td>5.8</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
iq_train.head()

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
      <th></th>
      <th>week_start_date</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>reanalysis_dew_point_temp_k</th>
      <th>reanalysis_max_air_temp_k</th>
      <th>...</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
      <th>total_cases</th>
    </tr>
    <tr>
      <th>year</th>
      <th>weekofyear</th>
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
      <th rowspan="5" valign="top">2000</th>
      <th>26</th>
      <td>2000-07-01</td>
      <td>0.192886</td>
      <td>0.132257</td>
      <td>0.340886</td>
      <td>0.247200</td>
      <td>25.41</td>
      <td>296.740000</td>
      <td>298.450000</td>
      <td>295.184286</td>
      <td>307.3</td>
      <td>...</td>
      <td>92.418571</td>
      <td>25.41</td>
      <td>16.651429</td>
      <td>8.928571</td>
      <td>26.400000</td>
      <td>10.775000</td>
      <td>32.5</td>
      <td>20.7</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2000-07-08</td>
      <td>0.216833</td>
      <td>0.276100</td>
      <td>0.289457</td>
      <td>0.241657</td>
      <td>60.61</td>
      <td>296.634286</td>
      <td>298.428571</td>
      <td>295.358571</td>
      <td>306.6</td>
      <td>...</td>
      <td>93.581429</td>
      <td>60.61</td>
      <td>16.862857</td>
      <td>10.314286</td>
      <td>26.900000</td>
      <td>11.566667</td>
      <td>34.0</td>
      <td>20.8</td>
      <td>55.6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2000-07-15</td>
      <td>0.176757</td>
      <td>0.173129</td>
      <td>0.204114</td>
      <td>0.128014</td>
      <td>55.52</td>
      <td>296.415714</td>
      <td>297.392857</td>
      <td>295.622857</td>
      <td>304.5</td>
      <td>...</td>
      <td>95.848571</td>
      <td>55.52</td>
      <td>17.120000</td>
      <td>7.385714</td>
      <td>26.800000</td>
      <td>11.466667</td>
      <td>33.0</td>
      <td>20.7</td>
      <td>38.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2000-07-22</td>
      <td>0.227729</td>
      <td>0.145429</td>
      <td>0.254200</td>
      <td>0.200314</td>
      <td>5.60</td>
      <td>295.357143</td>
      <td>296.228571</td>
      <td>292.797143</td>
      <td>303.6</td>
      <td>...</td>
      <td>87.234286</td>
      <td>5.60</td>
      <td>14.431429</td>
      <td>9.114286</td>
      <td>25.766667</td>
      <td>10.533333</td>
      <td>31.5</td>
      <td>14.7</td>
      <td>30.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2000-07-29</td>
      <td>0.328643</td>
      <td>0.322129</td>
      <td>0.254371</td>
      <td>0.361043</td>
      <td>62.76</td>
      <td>296.432857</td>
      <td>297.635714</td>
      <td>293.957143</td>
      <td>307.0</td>
      <td>...</td>
      <td>88.161429</td>
      <td>62.76</td>
      <td>15.444286</td>
      <td>9.500000</td>
      <td>26.600000</td>
      <td>11.480000</td>
      <td>33.3</td>
      <td>19.1</td>
      <td>4.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
</div>


</div>
</div>
</div>



### Optimize Regressor for City SJ data



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_X_train, sj_X_test, sj_y_train, sj_y_test = train_test_split(sj_train.drop(['week_start_date', 'total_cases'], axis=1).astype(float), 
                                                                sj_train['total_cases'].astype(float), 
                                                                train_size=0.75, 
                                                                test_size=0.25)

get_tpot_best(sj_X_train, sj_X_test, sj_y_train, sj_y_test, city='sj')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
-23.16131926561329
```
</div>
</div>
</div>



### Optimize Regressor for City IQ data



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
iq_X_train, iq_X_test, iq_y_train, iq_y_test = train_test_split(iq_train.drop(['week_start_date', 'total_cases'], axis=1).astype(float), 
                                                                iq_train['total_cases'].astype(float), 
                                                                train_size=0.75, 
                                                                test_size=0.25)

get_tpot_best(iq_X_train, iq_X_test, iq_y_train, iq_y_test, city='iq')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
-5.45120052924523
```
</div>
</div>
</div>



---

## Optimized Models



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error

```
</div>

</div>



__City San Juan Regressor__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# Average CV score on the training set was:-20.202304763339182
sj_exported_pipeline = make_pipeline(
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.01, loss="exponential", n_estimators=100)),
    AdaBoostRegressor(learning_rate=0.001, loss="linear", n_estimators=100)
)

sj_exported_pipeline.fit(sj_X_train, sj_y_train)
sj_y_pred = sj_exported_pipeline.predict(sj_X_test)
mean_absolute_error(sj_y_test, sj_y_pred)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
23.165315854645822
```


</div>
</div>
</div>



__City Iquitos Regressor__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.feature_selection import SelectPercentile, f_regression
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# Average CV score on the training set was:-5.872964338767224
iq_exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    SelectPercentile(score_func=f_regression, percentile=24),
    XGBRegressor(learning_rate=0.01, max_depth=5, min_child_weight=15, n_estimators=100, nthread=1, subsample=0.1)
)

iq_exported_pipeline.fit(iq_X_train, iq_y_train)
iq_y_pred = iq_exported_pipeline.predict(iq_X_test)
mean_absolute_error(iq_y_test, iq_y_pred)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
5.45120052924523
```


</div>
</div>
</div>



__The MAE scores don't really look much better than the vanilla models we used previously though, I think we will have to choose a time series analysis model that captures the time aspect of the data instead.__



---

## Competition Predictions



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_test, iq_test = preprocess_data('./data/dengai/test_features/dengue_features_test.csv')

sj_predictions = sj_exported_pipeline.predict(sj_test.drop(['week_start_date'], axis=1).astype(float)).astype(int)
iq_predictions = iq_exported_pipeline.predict(iq_test.drop(['week_start_date'], axis=1).astype(float)).astype(int)

submission = pd.read_csv('./data/dengai/submission_format.csv',
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv('./data/dengai/tpot_submission.csv')

```
</div>

</div>

