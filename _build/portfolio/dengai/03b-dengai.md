---
interact_link: content/portfolio/dengai/03b-dengai.ipynb
kernel_name: python3
has_widgets: false
title: '03b - Reducing number of PCA components'
prev_page:
  url: /portfolio/dengai/03a-dengai.html
  title: '03a - Feature Scaling and PCA'
next_page:
  url: /portfolio/dengai/04-dengai.html
  title: '04 - Looking at the Benchmark'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# DengAI Analysis Part 03b - Reducing number of PCA components

By: Chengyi (Jeff) Chen, under guidance of CSCI499: AI for Social Good Teaching Assistant - Aaron Ferber

---
## Content

In this notebook, we will apply some feature selection and engineering techniques in order to boost out predictions.



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



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# DengAI dataset URLs
dengai_features_url = 'https://www.dropbox.com/s/1kuf94b4mk6axyy/dengue_features_train.csv'
dengai_labels_url = 'https://www.dropbox.com/s/626ak8397abonv4/dengue_labels_train.csv'
dengai_test_features_url = 'https://s3.amazonaws.com:443/drivendata/data/44/public/dengue_features_test.csv'

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



<a id="clean_feats"></a>

---
## Cleaned Features



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_X = pd.read_csv('./data/dengai/cleaned/sj_X.csv', index_col='week_start_date')
sj_y = pd.read_csv('./data/dengai/cleaned/sj_y.csv', header=None, names=['week_start_date', 'num_cases'], index_col='week_start_date')
iq_X = pd.read_csv('./data/dengai/cleaned/iq_X.csv', index_col='week_start_date')
iq_y = pd.read_csv('./data/dengai/cleaned/iq_y.csv', header=None, names=['week_start_date', 'num_cases'], index_col='week_start_date')

```
</div>

</div>



<a id="power_transform"></a>

---
## Yeo-Johnson Transform the Labels



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.preprocessing import PowerTransformer

# Applying yeo-johnson transform on the labels of City sj 
# REMEMBER TO INVERSE TRANSFORM YOUR Y_PREDS
# sj_pwr = PowerTransformer()
# sj_y = pd.Series(sj_pwr.fit_transform(sj_y).flatten(), index=sj_y.index)

# Applying yeo-johnson transform on the labels of City iq 
# REMEMBER TO INVERSE TRANSFORM YOUR Y_PREDS
# iq_pwr = PowerTransformer()
# iq_y = pd.Series(iq_pwr.fit_transform(iq_y).flatten(), index=iq_y.index)

```
</div>

</div>



---
<a id="feateng"></a>

## Feature Engineering / Scaling

__Now that we have checked each raw feature for any violation of linear regression assumptions, let's feature engineer to perform [transformations](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/) to create new synthetic features that might help our models perform better.__



### 1. Power Transform (Make features normally distributed) / Log Transform (Reduce skewness)



__Yeo-Johnson Transform / Quantile Transform:__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.preprocessing import power_transform

# Helper function to plot the power transformed
# versions of the features
def plot_pwr_dist(X):
    rcParams['figure.figsize'] = 20, 60
    fig, ax = plt.subplots(len(X.columns[2:]) // 2,2)

    for idx, name in enumerate(X.columns[2:]):
        if idx < len(X.columns[2:]) // 2:
            sns.distplot(X[name], ax=ax[idx, 0])
            sns.kdeplot(np.random.normal(size=len(X[name])), ax=ax[idx, 0])
            ax[idx, 0].grid(True)
            ax[idx, 0].set_title("Distribution of Yeo-Johnson Transformed {}".format(name), color='k')
        else:
            sns.distplot(X[name], ax=ax[idx - (len(X.columns[2:]) // 2), 1])
            sns.kdeplot(np.random.normal(size=len(X[name])), ax=ax[idx - (len(X.columns[2:]) // 2), 1])
            ax[idx - (len(X.columns[2:]) // 2), 1].grid(True)
            ax[idx - (len(X.columns[2:]) // 2), 1].set_title("Distribution of Yeo-Johnson Transformed  {}".format(name), color='k')

    plt.tight_layout()
    plt.show();

```
</div>

</div>



- __City sj:__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Applying yeo-johnson transform on the features of City sj 
sj_X_pwr = PowerTransformer()
sj_X = pd.DataFrame(sj_X_pwr.fit_transform(sj_X), index=sj_X.index, columns=sj_X.columns)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_pwr_dist(sj_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/03b-dengai_15_0.png)

</div>
</div>
</div>



- __City iq:__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Applying yeo-johnson transform on the features of City iq 
iq_X_pwr = PowerTransformer()
iq_X = pd.DataFrame(iq_X_pwr.fit_transform(iq_X), index=iq_X.index, columns=iq_X.columns)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_pwr_dist(iq_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/03b-dengai_18_0.png)

</div>
</div>
</div>



### 2. Seasonal Difference
__We will add the `ROLLING MOVING AVERAGE` and `PREVIOUS TIME STEP TOTAL NUMBER OF DENGUE CASES` of each feature from the previous timesteps in order to account for how data from previous timesteps also affect the total number of dengue cases in the current timestep.__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
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
      <td>0.703406</td>
      <td>0.421506</td>
      <td>0.420211</td>
      <td>0.225938</td>
      <td>-0.133201</td>
      <td>-1.267007</td>
      <td>-1.243194</td>
      <td>-1.551243</td>
      <td>-1.257668</td>
      <td>-1.111277</td>
      <td>0.460331</td>
      <td>-1.502762</td>
      <td>-0.133201</td>
      <td>-1.513467</td>
      <td>0.355365</td>
      <td>-1.112537</td>
      <td>0.189064</td>
      <td>-1.268181</td>
      <td>-1.611465</td>
      <td>-0.041619</td>
    </tr>
    <tr>
      <th>1990-05-07</th>
      <td>1.196206</td>
      <td>0.848595</td>
      <td>-0.201284</td>
      <td>-0.173025</td>
      <td>0.226922</td>
      <td>-0.805626</td>
      <td>-0.726555</td>
      <td>-0.857927</td>
      <td>-0.434258</td>
      <td>-0.791699</td>
      <td>-0.133577</td>
      <td>-0.385129</td>
      <td>0.226922</td>
      <td>-0.843269</td>
      <td>-0.180430</td>
      <td>-0.289741</td>
      <td>-0.446716</td>
      <td>-0.049699</td>
      <td>-0.358533</td>
      <td>-0.525588</td>
    </tr>
    <tr>
      <th>1990-05-14</th>
      <td>-0.240230</td>
      <td>1.191405</td>
      <td>-0.292164</td>
      <td>0.104414</td>
      <td>0.500156</td>
      <td>-0.369180</td>
      <td>-0.386707</td>
      <td>0.059014</td>
      <td>-0.740325</td>
      <td>-0.131313</td>
      <td>0.245261</td>
      <td>1.039584</td>
      <td>0.500156</td>
      <td>0.075226</td>
      <td>-0.342842</td>
      <td>-0.289741</td>
      <td>-0.308256</td>
      <td>0.261982</td>
      <td>0.044281</td>
      <td>0.840314</td>
    </tr>
    <tr>
      <th>1990-05-21</th>
      <td>0.766309</td>
      <td>1.996750</td>
      <td>0.902026</td>
      <td>1.250635</td>
      <td>-0.012904</td>
      <td>-0.205712</td>
      <td>-0.103079</td>
      <td>-0.029182</td>
      <td>-0.040648</td>
      <td>-0.364483</td>
      <td>-0.378643</td>
      <td>0.502897</td>
      <td>-0.012904</td>
      <td>-0.045552</td>
      <td>-0.054986</td>
      <td>0.261100</td>
      <td>0.035477</td>
      <td>1.013376</td>
      <td>0.401642</td>
      <td>-1.022056</td>
    </tr>
    <tr>
      <th>1990-05-28</th>
      <td>1.469873</td>
      <td>2.188653</td>
      <td>1.282453</td>
      <td>1.447868</td>
      <td>-0.393966</td>
      <td>0.231798</td>
      <td>0.263764</td>
      <td>0.348824</td>
      <td>0.365561</td>
      <td>0.031905</td>
      <td>-0.499831</td>
      <td>0.540753</td>
      <td>-0.393966</td>
      <td>0.333943</td>
      <td>1.039884</td>
      <td>1.473942</td>
      <td>3.026931</td>
      <td>2.369253</td>
      <td>0.857563</td>
      <td>-0.795083</td>
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
sj_X.rolling(window=2).mean().head()

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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1990-05-07</th>
      <td>0.949806</td>
      <td>0.635050</td>
      <td>0.109464</td>
      <td>0.026456</td>
      <td>0.046860</td>
      <td>-1.036316</td>
      <td>-0.984875</td>
      <td>-1.204585</td>
      <td>-0.845963</td>
      <td>-0.951488</td>
      <td>0.163377</td>
      <td>-0.943946</td>
      <td>0.046860</td>
      <td>-1.178368</td>
      <td>0.087467</td>
      <td>-0.701139</td>
      <td>-0.128826</td>
      <td>-0.658940</td>
      <td>-0.984999</td>
      <td>-0.283604</td>
    </tr>
    <tr>
      <th>1990-05-14</th>
      <td>0.477988</td>
      <td>1.020000</td>
      <td>-0.246724</td>
      <td>-0.034306</td>
      <td>0.363539</td>
      <td>-0.587403</td>
      <td>-0.556631</td>
      <td>-0.399456</td>
      <td>-0.587292</td>
      <td>-0.461506</td>
      <td>0.055842</td>
      <td>0.327227</td>
      <td>0.363539</td>
      <td>-0.384022</td>
      <td>-0.261636</td>
      <td>-0.289741</td>
      <td>-0.377486</td>
      <td>0.106142</td>
      <td>-0.157126</td>
      <td>0.157363</td>
    </tr>
    <tr>
      <th>1990-05-21</th>
      <td>0.263039</td>
      <td>1.594077</td>
      <td>0.304931</td>
      <td>0.677524</td>
      <td>0.243626</td>
      <td>-0.287446</td>
      <td>-0.244893</td>
      <td>0.014916</td>
      <td>-0.390486</td>
      <td>-0.247898</td>
      <td>-0.066691</td>
      <td>0.771241</td>
      <td>0.243626</td>
      <td>0.014837</td>
      <td>-0.198914</td>
      <td>-0.014321</td>
      <td>-0.136390</td>
      <td>0.637679</td>
      <td>0.222962</td>
      <td>-0.090871</td>
    </tr>
    <tr>
      <th>1990-05-28</th>
      <td>1.118091</td>
      <td>2.092701</td>
      <td>1.092240</td>
      <td>1.349251</td>
      <td>-0.203435</td>
      <td>0.013043</td>
      <td>0.080342</td>
      <td>0.159821</td>
      <td>0.162457</td>
      <td>-0.166289</td>
      <td>-0.439237</td>
      <td>0.521825</td>
      <td>-0.203435</td>
      <td>0.144195</td>
      <td>0.492449</td>
      <td>0.867521</td>
      <td>1.531204</td>
      <td>1.691314</td>
      <td>0.629602</td>
      <td>-0.908569</td>
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
# # Create the new columns we want to incorporate into model
# # Add the rolling means for 1 week window to 8 week window
# for window in range(1, 9):
#     sj_X_n_week_rolling_mean = sj_X.rolling(window=window+1).mean()
#     sj_X_n_week_rolling_mean.columns = [col_name + '_rolling_{}_week'.format(window) for col_name in sj_X_n_week_rolling_mean.columns]

# #     sj_y_n_week_prior_cases = sj_y.shift(window)
# #     sj_y_n_week_prior_cases.name = '{}_week_prior_cases'.format(window)

#     sj_X = pd.concat([sj_X, 
#                       sj_X_n_week_rolling_mean], axis=1)

# # Rolling means with prior number of cases
# # sj_X = pd.concat([sj_X, 
# #                   sj_X_1_week_rolling_mean, 
# #                   sj_X_2_week_rolling_mean, 
# #                   sj_y_1_week_prior_cases, 
# #                   sj_y_2_week_prior_cases], axis=1).dropna(axis=0)

# sj_X = sj_X.dropna(axis=0)
# sj_X.head()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# # Create the new columns we want to incorporate into model
# # Add the rolling means for 1 week window to 8 week window
# for window in range(1, 9):
#     iq_X_n_week_rolling_mean = iq_X.rolling(window=window+1).mean()
#     iq_X_n_week_rolling_mean.columns = [col_name + '_rolling_{}_week'.format(window) for col_name in iq_X_n_week_rolling_mean.columns]

# #     sj_y_n_week_prior_cases = sj_y.shift(window)
# #     sj_y_n_week_prior_cases.name = '{}_week_prior_cases'.format(window)

#     iq_X = pd.concat([iq_X, 
#                       iq_X_n_week_rolling_mean], axis=1)

# # Rolling means with prior number of cases
# # sj_X = pd.concat([sj_X, 
# #                   sj_X_1_week_rolling_mean, 
# #                   sj_X_2_week_rolling_mean, 
# #                   sj_y_1_week_prior_cases, 
# #                   sj_y_2_week_prior_cases], axis=1).dropna(axis=0)

# iq_X = iq_X.dropna(axis=0)
# iq_X.head()

```
</div>

</div>



### 3. Trend Difference



---
<a id="featselect"></a>

## Feature Selection

__Before continuing to normalize and then standardize our data, let's squash our features to a lower dimensional space. This could be done using either Principal Component Analysis (PCA) or Linear Discriminant Analysis (LDA), but here we will use PCA.__



### Principal Component Analysis



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# We will perform PCA here using the original features,
# not the new feature set after removing features for
# multicollinearity using VIF
from sklearn import decomposition

sj_pca = decomposition.PCA(n_components=1)

# PCA on City sj features
sj_X_pca = sj_pca.fit_transform(sj_X)

iq_pca = decomposition.PCA(n_components=1)

# PCA on City iq features
iq_X_pca = iq_pca.fit_transform(iq_X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print("City sj features shape BEFORE: {}, after: {}".format(sj_X.shape, sj_X_pca.shape))
print("City sj target shape:", sj_y.shape)

print("City iq features shape BEFORE: {}, after: {}".format(iq_X.shape, iq_X_pca.shape))
print("City iq target shape:", iq_y.shape)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
City sj features shape BEFORE: (936, 20), after: (936, 1)
City sj target shape: (936,)
City iq features shape BEFORE: (520, 20), after: (520, 1)
City iq target shape: (520,)
```
</div>
</div>
</div>



### 4. Standardization



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Import standard scaler module
from sklearn.preprocessing import StandardScaler

# Scale City sj features
sj_X_std_scaler = StandardScaler()
sj_X_scaled = sj_X_std_scaler.fit_transform(sj_X_pca)

# Scale City iq features
iq_X_std_scaler = StandardScaler()
iq_X_scaled = iq_X_std_scaler.fit_transform(iq_X_pca)

```
</div>

</div>



### 5. Normalization



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Import minmax scaler module
from sklearn.preprocessing import MinMaxScaler

# Scale City sj features
sj_X_norm_scaler = MinMaxScaler()
sj_X_normed = sj_X_norm_scaler.fit_transform(sj_X_scaled)

# Scale City iq features
iq_X_norm_scaler = MinMaxScaler()
iq_X_normed = iq_X_norm_scaler.fit_transform(iq_X_scaled)

```
</div>

</div>



---

## Training with Transformed Features
__Now let's start training all the previous models we used in Part 2 with the transformed features__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Import models and utilities from sklearn
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict
from time import time

# Let's define a function that'll settle the training
# testing pipeline for us
def train(X, y, mods, pwr):
    """
    Handles the entire train and testing pipeline
    
    Parameters:
    -----------
    X: (pandas.DataFrame) Feature columns
    y: (pandas.DataFrame) Labels
    mods: (list) List of sklearn models to be trained on
    pwr: yeo-johnson Transformer that was used to transform y
    
    Returns:
    --------
    DataFrame of results of training and also a dictionary of the trained models
    """
    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Initialize models dictionary
    trained_mods = {str(mod())[:str(mod()).find('(')]: mod for mod in mods}
    
    # Initialize model performance dictionary
    performance_mods = {str(mod())[:str(mod()).find('(')]: defaultdict(float) for mod in mods}
    
    # Split into training and testing sets using KFold cross validation
    kf = KFold(n_splits=20, shuffle=False, random_state=42)
    
    # Loop through all models
    for idx, (mod_name, mod) in enumerate(trained_mods.items()):
        
        # Trained Model
        trained_mod = None
        
        # List of each score we get from each fold training / testing
        mse_train_scores = []
        mae_train_scores = []
        r2_train_scores = []
        mse_test_scores = []
        mae_test_scores = []
        r2_test_scores = []
        
        # Go through each fold in the KFold cross validation
        for train_index, test_index in kf.split(X):
            
            # Train Test Splits
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            ################
            ### TRAINING ###
            ################
            # Initialize current model
            curr_mod = mod()

            # Fit the model
            trained_mod = curr_mod.fit(X_train, y_train) 
            
            # Prediction scores for training set
            y_train_pred = pwr.inverse_transform(trained_mod.predict(X_train)[:,None]).astype(int)
            mse_train_scores.append(mean_squared_error(y_train, y_train_pred))
            mae_train_scores.append(mean_absolute_error(y_train, y_train_pred))
            r2_train_scores.append(r2_score(y_train, y_train_pred)) 
        
            ###############
            ### TESTING ###
            ###############
            # Prediction scores for testing set
            y_test_pred = pwr.inverse_transform(trained_mod.predict(X_test)[:,None]).astype(int)
            mse_test_scores.append(mean_squared_error(y_test, y_test_pred))
            mae_test_scores.append(mean_absolute_error(y_test, y_test_pred))
            r2_test_scores.append(r2_score(y_test, y_test_pred)) 
        
        # Saving average train scores
        performance_mods[mod_name]['train_' + str(mean_squared_error.__name__)] = np.mean(mse_train_scores)
        performance_mods[mod_name]['train_' + str(mean_absolute_error.__name__)] = np.mean(mae_train_scores)
        performance_mods[mod_name]['train_' + str(r2_score.__name__)] = np.mean(r2_train_scores)
        
        # Saving average test scores
        performance_mods[mod_name]['test_' + str(mean_squared_error.__name__)] = np.mean(mse_test_scores)
        performance_mods[mod_name]['test_' + str(mean_absolute_error.__name__)] = np.mean(mae_test_scores)
        performance_mods[mod_name]['test_' + str(r2_score.__name__)] = np.mean(r2_test_scores)
        
        # Saving last trained model
        trained_mods[mod_name] = trained_mod
            
    return performance_mods, trained_mods

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Let's declare the list of models we want to train
mods_to_train = [LinearRegression, BayesianRidge, ElasticNet, GaussianProcessRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, KernelRidge, SVR, MLPRegressor]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# City SJ
sj_performance_mods, sj_trained_mods = train(sj_X_normed, sj_y, mods_to_train, sj_pwr)
pd.DataFrame.from_dict(sj_performance_mods).transpose()

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
      <th>test_mean_absolute_error</th>
      <th>test_mean_squared_error</th>
      <th>test_r2_score</th>
      <th>train_mean_absolute_error</th>
      <th>train_mean_squared_error</th>
      <th>train_r2_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LinearRegression</th>
      <td>18.861655</td>
      <td>377.965704</td>
      <td>-410.565839</td>
      <td>18.881462</td>
      <td>378.598740</td>
      <td>-377.666691</td>
    </tr>
    <tr>
      <th>BayesianRidge</th>
      <td>18.845837</td>
      <td>376.392880</td>
      <td>-409.044621</td>
      <td>18.862231</td>
      <td>377.034093</td>
      <td>-376.101413</td>
    </tr>
    <tr>
      <th>ElasticNet</th>
      <td>18.050046</td>
      <td>326.874573</td>
      <td>-354.340496</td>
      <td>18.050000</td>
      <td>326.848744</td>
      <td>-325.910704</td>
    </tr>
    <tr>
      <th>GaussianProcessRegressor</th>
      <td>19.094865</td>
      <td>404.441778</td>
      <td>-440.118352</td>
      <td>19.175321</td>
      <td>411.608891</td>
      <td>-410.674058</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>25.499168</td>
      <td>1360.138378</td>
      <td>-1483.003349</td>
      <td>25.441261</td>
      <td>1262.796116</td>
      <td>-1261.611180</td>
    </tr>
    <tr>
      <th>AdaBoostRegressor</th>
      <td>19.005480</td>
      <td>409.149538</td>
      <td>-446.890823</td>
      <td>19.080035</td>
      <td>410.942970</td>
      <td>-409.944486</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>19.799791</td>
      <td>489.938971</td>
      <td>-544.053235</td>
      <td>19.936497</td>
      <td>487.510916</td>
      <td>-486.574667</td>
    </tr>
    <tr>
      <th>KernelRidge</th>
      <td>16.480342</td>
      <td>272.966722</td>
      <td>-295.642156</td>
      <td>16.487125</td>
      <td>273.175848</td>
      <td>-272.238558</td>
    </tr>
    <tr>
      <th>SVR</th>
      <td>19.492761</td>
      <td>402.530416</td>
      <td>-437.649922</td>
      <td>19.506349</td>
      <td>403.276845</td>
      <td>-402.369042</td>
    </tr>
    <tr>
      <th>MLPRegressor</th>
      <td>18.886146</td>
      <td>382.446350</td>
      <td>-416.468768</td>
      <td>18.898629</td>
      <td>382.799230</td>
      <td>-381.877429</td>
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
# City IQ
iq_performance_mods, iq_trained_mods = train(iq_X_normed, iq_y, mods_to_train, iq_pwr)
pd.DataFrame.from_dict(iq_performance_mods).transpose()

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
      <th>test_mean_absolute_error</th>
      <th>test_mean_squared_error</th>
      <th>test_r2_score</th>
      <th>train_mean_absolute_error</th>
      <th>train_mean_squared_error</th>
      <th>train_r2_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LinearRegression</th>
      <td>3.544215</td>
      <td>14.495449</td>
      <td>-15.288360</td>
      <td>3.550288</td>
      <td>14.500890</td>
      <td>-13.505659</td>
    </tr>
    <tr>
      <th>BayesianRidge</th>
      <td>3.530754</td>
      <td>14.314003</td>
      <td>-15.033050</td>
      <td>3.535916</td>
      <td>14.310972</td>
      <td>-13.315649</td>
    </tr>
    <tr>
      <th>ElasticNet</th>
      <td>3.150000</td>
      <td>11.166398</td>
      <td>-11.316103</td>
      <td>3.150000</td>
      <td>11.043874</td>
      <td>-10.050910</td>
    </tr>
    <tr>
      <th>GaussianProcessRegressor</th>
      <td>3.650175</td>
      <td>16.291489</td>
      <td>-17.111802</td>
      <td>3.620086</td>
      <td>15.153043</td>
      <td>-14.157481</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>5.457576</td>
      <td>70.156277</td>
      <td>-79.111722</td>
      <td>5.090472</td>
      <td>48.902960</td>
      <td>-47.918521</td>
    </tr>
    <tr>
      <th>AdaBoostRegressor</th>
      <td>3.634600</td>
      <td>14.892825</td>
      <td>-15.767551</td>
      <td>3.610432</td>
      <td>14.518433</td>
      <td>-13.525008</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>3.858595</td>
      <td>20.637736</td>
      <td>-22.114269</td>
      <td>3.812247</td>
      <td>17.230560</td>
      <td>-16.235364</td>
    </tr>
    <tr>
      <th>KernelRidge</th>
      <td>2.994231</td>
      <td>9.971601</td>
      <td>-10.017791</td>
      <td>2.994332</td>
      <td>9.967875</td>
      <td>-8.970040</td>
    </tr>
    <tr>
      <th>SVR</th>
      <td>4.084600</td>
      <td>18.939334</td>
      <td>-20.290275</td>
      <td>4.081462</td>
      <td>18.877207</td>
      <td>-17.884865</td>
    </tr>
    <tr>
      <th>MLPRegressor</th>
      <td>3.526908</td>
      <td>14.192131</td>
      <td>-14.881601</td>
      <td>3.531273</td>
      <td>14.209774</td>
      <td>-13.214851</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



__Seems like with the rolling means and prior number of cases for previous timesteps added, KernelRidge is now performing the best overall, so let's use that on the competition test features and use the KernelRidge predictions as our next submission.__



<a id="test"></a>

---
## Competition Test Prediction

__Here we will use the KernelRidge model in order to predict the labels given the competition's test features__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
test_feats_df = pd.read_csv('./data/dengai/test_features/dengue_features_test.csv', index_col='week_start_date')
test_feats_df.head()

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
      <th>2008-04-29</th>
      <td>sj</td>
      <td>2008</td>
      <td>18</td>
      <td>-0.0189</td>
      <td>-0.018900</td>
      <td>0.102729</td>
      <td>0.091200</td>
      <td>78.60</td>
      <td>298.492857</td>
      <td>298.550000</td>
      <td>...</td>
      <td>25.37</td>
      <td>78.781429</td>
      <td>78.60</td>
      <td>15.918571</td>
      <td>3.128571</td>
      <td>26.528571</td>
      <td>7.057143</td>
      <td>33.3</td>
      <td>21.7</td>
      <td>75.2</td>
    </tr>
    <tr>
      <th>2008-05-06</th>
      <td>sj</td>
      <td>2008</td>
      <td>19</td>
      <td>-0.0180</td>
      <td>-0.012400</td>
      <td>0.082043</td>
      <td>0.072314</td>
      <td>12.56</td>
      <td>298.475714</td>
      <td>298.557143</td>
      <td>...</td>
      <td>21.83</td>
      <td>78.230000</td>
      <td>12.56</td>
      <td>15.791429</td>
      <td>2.571429</td>
      <td>26.071429</td>
      <td>5.557143</td>
      <td>30.0</td>
      <td>22.2</td>
      <td>34.3</td>
    </tr>
    <tr>
      <th>2008-05-13</th>
      <td>sj</td>
      <td>2008</td>
      <td>20</td>
      <td>-0.0015</td>
      <td>NaN</td>
      <td>0.151083</td>
      <td>0.091529</td>
      <td>3.66</td>
      <td>299.455714</td>
      <td>299.357143</td>
      <td>...</td>
      <td>4.12</td>
      <td>78.270000</td>
      <td>3.66</td>
      <td>16.674286</td>
      <td>4.428571</td>
      <td>27.928571</td>
      <td>7.785714</td>
      <td>32.8</td>
      <td>22.8</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2008-05-20</th>
      <td>sj</td>
      <td>2008</td>
      <td>21</td>
      <td>NaN</td>
      <td>-0.019867</td>
      <td>0.124329</td>
      <td>0.125686</td>
      <td>0.00</td>
      <td>299.690000</td>
      <td>299.728571</td>
      <td>...</td>
      <td>2.20</td>
      <td>73.015714</td>
      <td>0.00</td>
      <td>15.775714</td>
      <td>4.342857</td>
      <td>28.057143</td>
      <td>6.271429</td>
      <td>33.3</td>
      <td>24.4</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>2008-05-27</th>
      <td>sj</td>
      <td>2008</td>
      <td>22</td>
      <td>0.0568</td>
      <td>0.039833</td>
      <td>0.062267</td>
      <td>0.075914</td>
      <td>0.76</td>
      <td>299.780000</td>
      <td>299.671429</td>
      <td>...</td>
      <td>4.36</td>
      <td>74.084286</td>
      <td>0.76</td>
      <td>16.137143</td>
      <td>3.542857</td>
      <td>27.614286</td>
      <td>7.085714</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>84.1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
</div>


</div>
</div>
</div>



### Clean Data Pipeline



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Taking care of missing data
test_feats_df.fillna(test_feats_df.mode().iloc[0], inplace=True)

# Drop unecessary feature columns
test_feats_df = test_feats_df.drop(['year', 'weekofyear'], axis=1)

# Split dataset to City sj and City iq
test_sj_X = test_feats_df[test_feats_df['city'] == 'sj'].drop(['city'], axis=1)
test_sj_X.index = pd.to_datetime(test_sj_X.index)

test_iq_X = test_feats_df[test_feats_df['city'] == 'iq'].drop(['city'], axis=1)
test_iq_X.index = pd.to_datetime(test_iq_X.index)

```
</div>

</div>



### Feature Transformations



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def transform_feats(X, pwr, pca, std_scaler, norm_scaler):
    """
    Transforms the features and returns the transformed numpy array
    
    Parameters:
    -----------
    X: (numpy.array) Feature columns that need to be transformed
    
    Returns:
    --------
    A numpy array with transformed features
    """
    # Transform features to be more normal
    X = pwr.transform(X)
    
    # Apply PCA
    X_pca = pca.transform(X)
    
    # Standardize
    X_scaled = std_scaler.transform(X_pca)
    
    # Normalize
    X_normed = norm_scaler.transform(X_scaled)
    
    return X_normed

```
</div>

</div>



### Prediction



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
test_sj_y_pred = sj_pwr.inverse_transform(sj_trained_mods['KernelRidge'].predict(transform_feats(test_sj_X, sj_X_pwr, sj_pca, sj_X_std_scaler, sj_X_norm_scaler))[:,None])
test_iq_y_pred = iq_pwr.inverse_transform(iq_trained_mods['KernelRidge'].predict(transform_feats(test_iq_X, iq_X_pwr, iq_pca, iq_X_std_scaler, iq_X_norm_scaler))[:,None])

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Save the results to csv and upload to competition
submission_df = pd.read_csv('./data/dengai/submission_format.csv')
submission_df.head()

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
      <td>2008</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>2008</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>2008</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>2008</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>2008</td>
      <td>22</td>
      <td>0</td>
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
sj_submission_df = submission_df[submission_df['city'] == 'sj'].drop(['total_cases'], axis=1)
sj_submission_df.reset_index(inplace=True)
sj_submission_df = pd.concat([sj_submission_df, pd.DataFrame(test_sj_y_pred.flatten().astype(int), columns=['total_cases'])], axis=1)
sj_submission_df.index = sj_submission_df['index']
sj_submission_df.drop(['index'], axis=1, inplace=True)
sj_submission_df.head()

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
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sj</td>
      <td>2008</td>
      <td>18</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>2008</td>
      <td>19</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>2008</td>
      <td>20</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>2008</td>
      <td>21</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>2008</td>
      <td>22</td>
      <td>16</td>
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
iq_submission_df = submission_df[submission_df['city'] == 'iq'].drop(['total_cases'], axis=1)
iq_submission_df.reset_index(inplace=True)
iq_submission_df = pd.concat([iq_submission_df, pd.DataFrame(test_iq_y_pred.flatten().astype(int), columns=['total_cases'])], axis=1)
iq_submission_df.index = iq_submission_df['index']
iq_submission_df.drop(['index'], axis=1, inplace=True)
iq_submission_df.head()

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
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>260</th>
      <td>iq</td>
      <td>2010</td>
      <td>26</td>
      <td>3</td>
    </tr>
    <tr>
      <th>261</th>
      <td>iq</td>
      <td>2010</td>
      <td>27</td>
      <td>3</td>
    </tr>
    <tr>
      <th>262</th>
      <td>iq</td>
      <td>2010</td>
      <td>28</td>
      <td>3</td>
    </tr>
    <tr>
      <th>263</th>
      <td>iq</td>
      <td>2010</td>
      <td>29</td>
      <td>3</td>
    </tr>
    <tr>
      <th>264</th>
      <td>iq</td>
      <td>2010</td>
      <td>30</td>
      <td>3</td>
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
submission_df = pd.concat([sj_submission_df, iq_submission_df], axis=0).reset_index().drop(['index'], axis=1)
submission_df.head()

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
      <td>2008</td>
      <td>18</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>2008</td>
      <td>19</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>2008</td>
      <td>20</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>2008</td>
      <td>21</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>2008</td>
      <td>22</td>
      <td>16</td>
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
# Save to csv
submission_df.to_csv('./data/dengai/kernelridge_submission.csv', index=False)

```
</div>

</div>



__This received an MAE score of 29.5577 in the competition, hinting that we might have overfitted our model...__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# test_sj_y_pred = sj_pwr.inverse_transform(sj_trained_mods['ElasticNet'].predict(transform_feats(test_sj_X))[:,None])
# test_iq_y_pred = iq_pwr.inverse_transform(iq_trained_mods['ElasticNet'].predict(transform_feats(test_iq_X))[:,None])

# # Save the results to csv and upload to competition
# submission_df = pd.read_csv('./data/dengai/submission_format.csv')

# sj_submission_df = submission_df[submission_df['city'] == 'sj'].drop(['total_cases'], axis=1)
# sj_submission_df.reset_index(inplace=True)
# sj_submission_df = pd.concat([sj_submission_df, pd.DataFrame(test_sj_y_pred.flatten().astype(int), columns=['total_cases'])], axis=1)
# sj_submission_df.index = sj_submission_df['index']
# sj_submission_df.drop(['index'], axis=1, inplace=True)

# iq_submission_df = submission_df[submission_df['city'] == 'iq'].drop(['total_cases'], axis=1)
# iq_submission_df.reset_index(inplace=True)
# iq_submission_df = pd.concat([iq_submission_df, pd.DataFrame(test_iq_y_pred.flatten().astype(int), columns=['total_cases'])], axis=1)
# iq_submission_df.index = iq_submission_df['index']
# iq_submission_df.drop(['index'], axis=1, inplace=True)

# submission_df = pd.concat([sj_submission_df, iq_submission_df], axis=0).reset_index().drop(['index'], axis=1)

# # Save to csv
# submission_df.to_csv('./data/dengai/elasticnet_submission.csv', index=False)

```
</div>

</div>



__Using the Elasticnet model, we got an MAE of 29.2212, supporting the fact that we are definitely overfitting our models to the current distribution of data that we have, perhaps we should increase the number of components after PCA...__

