---
interact_link: content/portfolio/dengai/07-dengai.ipynb
kernel_name: python3
has_widgets: false
title: 'DengAI (Latest)'
prev_page:
  url: /portfolio/dengai/README
  title: 'DengAI'
next_page:
  url: /machine-learning/01-supervised-learning/classification/README
  title: 'Classification'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# DengAI Analysis Part 8 - Eliminating Seasonality

By: Chengyi (Jeff) Chen, under guidance of CSCI499: AI for Social Good Teaching Assistant - Aaron Ferber

---
## Content

In this notebook, we will use the seasonality feature created in the last notebook to de-seasonalize our total dengue cases so that our features are purely used to predict the residuals and not the seasonal trend.



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
## Cleaned Features & Seasonality  Feature



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Training Features and Labels
sj_X_train = pd.read_csv('./data/dengai/cleaned/sj_X.csv', index_col='week_start_date')
sj_y_train = pd.read_csv('./data/dengai/cleaned/sj_y.csv', header=None, names=['week_start_date', 'num_cases'], index_col='week_start_date')
iq_X_train = pd.read_csv('./data/dengai/cleaned/iq_X.csv', index_col='week_start_date')
iq_y_train = pd.read_csv('./data/dengai/cleaned/iq_y.csv', header=None, names=['week_start_date', 'num_cases'], index_col='week_start_date')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
test_feats_df = pd.read_csv('./data/dengai/test_features/dengue_features_test.csv', index_col='week_start_date')

# Taking care of missing data
test_feats_df.fillna(test_feats_df.mode().iloc[0], inplace=True)

# Drop unecessary feature columns
test_feats_df = test_feats_df.drop(['year', 'weekofyear'], axis=1)

# Split dataset to City sj and City iq
sj_X_test = test_feats_df[test_feats_df['city'] == 'sj'].drop(['city'], axis=1)
iq_X_test = test_feats_df[test_feats_df['city'] == 'iq'].drop(['city'], axis=1)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Seasonality Feature from SARIMA
sj_seasonality = pd.read_csv('./data/dengai/cleaned/sj_seasonality.csv', index_col=[0]).rename({'num_cases': 'seasonality'}, axis='columns')
iq_seasonality = pd.read_csv('./data/dengai/cleaned/iq_seasonality.csv', index_col=[0]).rename({'num_cases': 'seasonality'}, axis='columns')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Truncate Training and Test set so as to include seasonality
sj_X_train = pd.concat([sj_X_train.loc[sj_seasonality.first_valid_index():], sj_seasonality.loc[sj_seasonality.first_valid_index():sj_X_train.last_valid_index()]], axis=1)
sj_y_train = sj_y_train.loc[sj_seasonality.first_valid_index():]
iq_X_train = pd.concat([iq_X_train.loc[iq_seasonality.first_valid_index():], iq_seasonality.loc[iq_seasonality.first_valid_index():iq_X_train.last_valid_index()]], axis=1)
iq_y_train = iq_y_train.loc[iq_seasonality.first_valid_index():]

sj_X_test = pd.concat([sj_X_test, sj_seasonality.loc[sj_X_test.first_valid_index():]], axis=1)
iq_X_test = pd.concat([iq_X_test, iq_seasonality.loc[iq_X_test.first_valid_index():]], axis=1)

```
</div>

</div>



---
## Problem of Covariate Shifting (Accounting for Seasonality in Features)

In addition to adding the new seasonality feature we created with SARIMA in the previous notebook, I also wanted to further examine whether the features which we were training on come from the same distribution as our testing features. There is a possibility of this [covariate shift problem](http://blog.smola.org/post/4110255196/real-simple-covariate-shift-correction) because in the previous parts, our MAE scores on the training features seemed to be doing quite well, but when we submitted them, the model got a horrible MAE. Submitting the Seasonality feature alone as the predicted number of cases got an EVEN BETTER MAE score than using a machine learning model, so let's check to see if the distributions are different and explore what we could do to correct this. Furthermore, even though we've accounted for the seasonality in the total number of dengue cases using the new seasonality feature from SARIMA previously, there might still be other seasonal differences hidden in our feature space that we also need to  account  for, such as "higher-order" seasonalities in our features and seasonalities on our seasonalities. Remember, many of the machine learning models we use assume that the data points are [__independent, and identically distributed__](https://stats.stackexchange.com/questions/213464/on-the-importance-of-the-i-i-d-assumption-in-statistical-learning), making this process of de-seasonalizing very important for them to perform well.



### Correlations of Exogenous with Endogenous



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# City SJ
(pd.concat([sj_X_train, sj_y_train], axis=1)
 .corr()
 .num_cases
 .drop('num_cases') # Don't compare with itself
 .sort_values(ascending=False)
 .plot
 .barh());

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_12_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# City IQ
(pd.concat([iq_X_train, iq_y_train], axis=1)
 .corr()
 .num_cases
 .drop('num_cases') # Don't compare with itself
 .sort_values(ascending=False)
 .plot
 .barh());

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_13_0.png)

</div>
</div>
</div>



Looking at the correlations, it does seem like we have to pay special attention at the temperature and humidity features



### Check if the Competition Training Features and Test Features come from the Same Distribution
We will use the 2 sample Kolmogorov-Smirnov test statistic to determine whether the training features and testing features come from the same distribution



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def KS_test(X_train, X_test, alpha=0.05):
    """
    ============
    = FUNCTION =
    ============
    Performs KS test on Train and Test features to see if they are drawn from different distributions
    
    ==============
    = PARAMETERS =
    ==============
    X_train: The Training features
    X_test:  The Testing features
    alpha: Significance level to reject the null that X_train and X_test are drawn from same distribution
    
    ===========
    = RETURNS =
    ===========
    Pandas DataFrame of KS test results for each feature
    """
    ks = {feat: {'ks_stat': None, 'p-value': None} for feat in X_train.drop(['seasonality'], axis=1).columns}

    # Check if the distribution of features 
    # for the Training and Testing set is  
    # significantly different
    for feat in X_train.drop(['seasonality'], axis=1).columns:

        ks_stat, p_val = stats.ks_2samp(X_train[feat].values, X_test[feat].values)

        if p_val < alpha:
            print('Training and Testing distributions of {} are significantly different with {}% confidence'.format(feat, (1-alpha)*100))
        
        ks[feat]['ks_stat'] = ks_stat
        ks[feat]['p-value'] = p_val
        
    return pd.DataFrame.from_dict(ks).transpose()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# KS Test on City SJ features
KS_test(sj_X_train, sj_X_test, alpha=0.01)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Training and Testing distributions of ndvi_ne are significantly different with 99.0% confidence
Training and Testing distributions of ndvi_nw are significantly different with 99.0% confidence
Training and Testing distributions of ndvi_sw are significantly different with 99.0% confidence
Training and Testing distributions of precipitation_amt_mm are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_air_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_avg_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_dew_point_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_min_air_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_precip_amt_kg_per_m2 are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_sat_precip_amt_mm are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_specific_humidity_g_per_kg are significantly different with 99.0% confidence
Training and Testing distributions of station_diur_temp_rng_c are significantly different with 99.0% confidence
Training and Testing distributions of station_min_temp_c are significantly different with 99.0% confidence
```
</div>
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
      <th>ks_stat</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ndvi_ne</th>
      <td>0.345924</td>
      <td>1.293898e-21</td>
    </tr>
    <tr>
      <th>ndvi_nw</th>
      <td>0.204535</td>
      <td>7.821258e-08</td>
    </tr>
    <tr>
      <th>ndvi_se</th>
      <td>0.068886</td>
      <td>2.880394e-01</td>
    </tr>
    <tr>
      <th>ndvi_sw</th>
      <td>0.132836</td>
      <td>1.501490e-03</td>
    </tr>
    <tr>
      <th>precipitation_amt_mm</th>
      <td>0.120666</td>
      <td>5.282118e-03</td>
    </tr>
    <tr>
      <th>reanalysis_air_temp_k</th>
      <td>0.184673</td>
      <td>1.828697e-06</td>
    </tr>
    <tr>
      <th>reanalysis_avg_temp_k</th>
      <td>0.171297</td>
      <td>1.274175e-05</td>
    </tr>
    <tr>
      <th>reanalysis_dew_point_temp_k</th>
      <td>0.115442</td>
      <td>8.733921e-03</td>
    </tr>
    <tr>
      <th>reanalysis_max_air_temp_k</th>
      <td>0.112801</td>
      <td>1.116690e-02</td>
    </tr>
    <tr>
      <th>reanalysis_min_air_temp_k</th>
      <td>0.151780</td>
      <td>1.666495e-04</td>
    </tr>
    <tr>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <td>0.163892</td>
      <td>3.505528e-05</td>
    </tr>
    <tr>
      <th>reanalysis_relative_humidity_percent</th>
      <td>0.067451</td>
      <td>3.117046e-01</td>
    </tr>
    <tr>
      <th>reanalysis_sat_precip_amt_mm</th>
      <td>0.120666</td>
      <td>5.282118e-03</td>
    </tr>
    <tr>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <td>0.123881</td>
      <td>3.833574e-03</td>
    </tr>
    <tr>
      <th>reanalysis_tdtr_k</th>
      <td>0.089093</td>
      <td>7.861031e-02</td>
    </tr>
    <tr>
      <th>station_avg_temp_c</th>
      <td>0.107520</td>
      <td>1.794611e-02</td>
    </tr>
    <tr>
      <th>station_diur_temp_rng_c</th>
      <td>0.329908</td>
      <td>1.067985e-19</td>
    </tr>
    <tr>
      <th>station_max_temp_c</th>
      <td>0.045982</td>
      <td>7.818338e-01</td>
    </tr>
    <tr>
      <th>station_min_temp_c</th>
      <td>0.117566</td>
      <td>7.137969e-03</td>
    </tr>
    <tr>
      <th>station_precip_mm</th>
      <td>0.109472</td>
      <td>1.509999e-02</td>
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
# KS Test on City IQ features
KS_test(iq_X_train, iq_X_test, alpha=0.01)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Training and Testing distributions of ndvi_nw are significantly different with 99.0% confidence
Training and Testing distributions of precipitation_amt_mm are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_precip_amt_kg_per_m2 are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_sat_precip_amt_mm are significantly different with 99.0% confidence
Training and Testing distributions of station_precip_mm are significantly different with 99.0% confidence
```
</div>
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
      <th>ks_stat</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ndvi_ne</th>
      <td>0.081620</td>
      <td>4.326702e-01</td>
    </tr>
    <tr>
      <th>ndvi_nw</th>
      <td>0.221904</td>
      <td>2.631674e-05</td>
    </tr>
    <tr>
      <th>ndvi_se</th>
      <td>0.134203</td>
      <td>3.279584e-02</td>
    </tr>
    <tr>
      <th>ndvi_sw</th>
      <td>0.103141</td>
      <td>1.763144e-01</td>
    </tr>
    <tr>
      <th>precipitation_amt_mm</th>
      <td>0.155099</td>
      <td>8.253519e-03</td>
    </tr>
    <tr>
      <th>reanalysis_air_temp_k</th>
      <td>0.086761</td>
      <td>3.567729e-01</td>
    </tr>
    <tr>
      <th>reanalysis_avg_temp_k</th>
      <td>0.093072</td>
      <td>2.762176e-01</td>
    </tr>
    <tr>
      <th>reanalysis_dew_point_temp_k</th>
      <td>0.074830</td>
      <td>5.451579e-01</td>
    </tr>
    <tr>
      <th>reanalysis_max_air_temp_k</th>
      <td>0.067398</td>
      <td>6.777616e-01</td>
    </tr>
    <tr>
      <th>reanalysis_min_air_temp_k</th>
      <td>0.126623</td>
      <td>5.149759e-02</td>
    </tr>
    <tr>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <td>0.154456</td>
      <td>8.636909e-03</td>
    </tr>
    <tr>
      <th>reanalysis_relative_humidity_percent</th>
      <td>0.049338</td>
      <td>9.439700e-01</td>
    </tr>
    <tr>
      <th>reanalysis_sat_precip_amt_mm</th>
      <td>0.155099</td>
      <td>8.253519e-03</td>
    </tr>
    <tr>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <td>0.067085</td>
      <td>6.833867e-01</td>
    </tr>
    <tr>
      <th>reanalysis_tdtr_k</th>
      <td>0.110523</td>
      <td>1.230651e-01</td>
    </tr>
    <tr>
      <th>station_avg_temp_c</th>
      <td>0.140301</td>
      <td>2.238326e-02</td>
    </tr>
    <tr>
      <th>station_diur_temp_rng_c</th>
      <td>0.137911</td>
      <td>2.605042e-02</td>
    </tr>
    <tr>
      <th>station_max_temp_c</th>
      <td>0.045465</td>
      <td>9.723719e-01</td>
    </tr>
    <tr>
      <th>station_min_temp_c</th>
      <td>0.150699</td>
      <td>1.122022e-02</td>
    </tr>
    <tr>
      <th>station_precip_mm</th>
      <td>0.294756</td>
      <td>4.890300e-09</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



It does seem that we might have found the root problem of why our models are so bad at predicting for the test features - our training features are mostly significantly different from our competition testing features. However, we need to check whether this significant difference might be purely due to seasonality or if there's something else that's causing them to be different. As a proxy, let's check if the features at different timesteps in our training data are significantly different.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Seeing if there is a large difference between one 
# half of our training features and the other half of
# our  training features
KS_test(sj_X_train.iloc[:len(sj_X_train)//2], sj_X_train.iloc[len(sj_X_train)//2:len(sj_X_train)], alpha=0.01)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Training and Testing distributions of ndvi_ne are significantly different with 99.0% confidence
Training and Testing distributions of ndvi_nw are significantly different with 99.0% confidence
Training and Testing distributions of ndvi_se are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_air_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_avg_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_max_air_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_min_air_temp_k are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_precip_amt_kg_per_m2 are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_relative_humidity_percent are significantly different with 99.0% confidence
Training and Testing distributions of reanalysis_tdtr_k are significantly different with 99.0% confidence
Training and Testing distributions of station_diur_temp_rng_c are significantly different with 99.0% confidence
Training and Testing distributions of station_max_temp_c are significantly different with 99.0% confidence
```
</div>
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
      <th>ks_stat</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ndvi_ne</th>
      <td>0.251924</td>
      <td>1.227849e-12</td>
    </tr>
    <tr>
      <th>ndvi_nw</th>
      <td>0.307028</td>
      <td>1.454227e-18</td>
    </tr>
    <tr>
      <th>ndvi_se</th>
      <td>0.143931</td>
      <td>2.064877e-04</td>
    </tr>
    <tr>
      <th>ndvi_sw</th>
      <td>0.058753</td>
      <td>4.289416e-01</td>
    </tr>
    <tr>
      <th>precipitation_amt_mm</th>
      <td>0.029310</td>
      <td>9.912086e-01</td>
    </tr>
    <tr>
      <th>reanalysis_air_temp_k</th>
      <td>0.238353</td>
      <td>2.341310e-11</td>
    </tr>
    <tr>
      <th>reanalysis_avg_temp_k</th>
      <td>0.222187</td>
      <td>6.338625e-10</td>
    </tr>
    <tr>
      <th>reanalysis_dew_point_temp_k</th>
      <td>0.070700</td>
      <td>2.181060e-01</td>
    </tr>
    <tr>
      <th>reanalysis_max_air_temp_k</th>
      <td>0.187752</td>
      <td>3.297729e-07</td>
    </tr>
    <tr>
      <th>reanalysis_min_air_temp_k</th>
      <td>0.153406</td>
      <td>5.926713e-05</td>
    </tr>
    <tr>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <td>0.184709</td>
      <td>5.448141e-07</td>
    </tr>
    <tr>
      <th>reanalysis_relative_humidity_percent</th>
      <td>0.301798</td>
      <td>5.961634e-18</td>
    </tr>
    <tr>
      <th>reanalysis_sat_precip_amt_mm</th>
      <td>0.029310</td>
      <td>9.912086e-01</td>
    </tr>
    <tr>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <td>0.068407</td>
      <td>2.510495e-01</td>
    </tr>
    <tr>
      <th>reanalysis_tdtr_k</th>
      <td>0.341068</td>
      <td>8.273954e-23</td>
    </tr>
    <tr>
      <th>station_avg_temp_c</th>
      <td>0.104772</td>
      <td>1.544738e-02</td>
    </tr>
    <tr>
      <th>station_diur_temp_rng_c</th>
      <td>0.164363</td>
      <td>1.267343e-05</td>
    </tr>
    <tr>
      <th>station_max_temp_c</th>
      <td>0.150849</td>
      <td>8.366082e-05</td>
    </tr>
    <tr>
      <th>station_min_temp_c</th>
      <td>0.042334</td>
      <td>8.221223e-01</td>
    </tr>
    <tr>
      <th>station_precip_mm</th>
      <td>0.045682</td>
      <td>7.443536e-01</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



It does seem like temperature and humidity which are highly correlated with the total number of dengue cases are very different for both halves of the dataset, meaning that there are probably temporal dependencies that we have to account for.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Seeing if there is a large difference between one 
# half of our training features and the other half of
# our  training features
KS_test(sj_X_train.iloc[:52], sj_X_train.iloc[52:52*2], alpha=0.01)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Training and Testing distributions of station_diur_temp_rng_c are significantly different with 99.0% confidence
```
</div>
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
      <th>ks_stat</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ndvi_ne</th>
      <td>0.153846</td>
      <td>0.534016</td>
    </tr>
    <tr>
      <th>ndvi_nw</th>
      <td>0.134615</td>
      <td>0.702120</td>
    </tr>
    <tr>
      <th>ndvi_se</th>
      <td>0.173077</td>
      <td>0.383104</td>
    </tr>
    <tr>
      <th>ndvi_sw</th>
      <td>0.211538</td>
      <td>0.171117</td>
    </tr>
    <tr>
      <th>precipitation_amt_mm</th>
      <td>0.192308</td>
      <td>0.261726</td>
    </tr>
    <tr>
      <th>reanalysis_air_temp_k</th>
      <td>0.134615</td>
      <td>0.702120</td>
    </tr>
    <tr>
      <th>reanalysis_avg_temp_k</th>
      <td>0.153846</td>
      <td>0.534016</td>
    </tr>
    <tr>
      <th>reanalysis_dew_point_temp_k</th>
      <td>0.153846</td>
      <td>0.534016</td>
    </tr>
    <tr>
      <th>reanalysis_max_air_temp_k</th>
      <td>0.153846</td>
      <td>0.534016</td>
    </tr>
    <tr>
      <th>reanalysis_min_air_temp_k</th>
      <td>0.134615</td>
      <td>0.702120</td>
    </tr>
    <tr>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <td>0.192308</td>
      <td>0.261726</td>
    </tr>
    <tr>
      <th>reanalysis_relative_humidity_percent</th>
      <td>0.115385</td>
      <td>0.858021</td>
    </tr>
    <tr>
      <th>reanalysis_sat_precip_amt_mm</th>
      <td>0.192308</td>
      <td>0.261726</td>
    </tr>
    <tr>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <td>0.153846</td>
      <td>0.534016</td>
    </tr>
    <tr>
      <th>reanalysis_tdtr_k</th>
      <td>0.153846</td>
      <td>0.534016</td>
    </tr>
    <tr>
      <th>station_avg_temp_c</th>
      <td>0.115385</td>
      <td>0.858021</td>
    </tr>
    <tr>
      <th>station_diur_temp_rng_c</th>
      <td>0.326923</td>
      <td>0.005642</td>
    </tr>
    <tr>
      <th>station_max_temp_c</th>
      <td>0.096154</td>
      <td>0.961394</td>
    </tr>
    <tr>
      <th>station_min_temp_c</th>
      <td>0.115385</td>
      <td>0.858021</td>
    </tr>
    <tr>
      <th>station_precip_mm</th>
      <td>0.096154</td>
      <td>0.961394</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



Specifically, now checking the distribution of features between a period of 1 year reveals that there is definitely a seasonal pattern of a year in our features.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sns.set()
fig, ax = plt.subplots(1, 1, figsize=(16,10), sharex=True)
sj_X_train.drop(['seasonality'], axis=1).plot(ax=ax);

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_24_0.png)

</div>
</div>
</div>



---
## Feature Extraction

For now, we'll just assume that there exists a single seasonal pattern in the feature space, where the same pattern occurs every 52 weeks, so let's just find the mean of each feature in every 52-week period and subtract it from each 52-week point, this will make more sense later.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def deseasonalize(X, period=52):
    """
    ============
    = FUNCTION =
    ============
    Removes the seasonality from each feature and returns the seasonality of each feature in a new dataframe
    
    ==============
    = PARAMETERS =
    ==============
    X: The features
    period: The period in which seasonality is found
    
    ===========
    = RETURNS =
    ===========
    Returns deseasonalized features and seasonality component in a new dataframe
    """
    
    deseasonalized_feats = pd.DataFrame(index=X.index)
    seasonality = pd.DataFrame(index=X.index)
    
    # Number of seasons inside our dataset
    num_full_seasons = X.shape[0] // period
    leftover = X.shape[0] % period
    
    # Getting the mean of each feature in each season period
    for feat in X.columns:
        
        # A single pattern in the period
        pattern = []
        
        # Getting mean for each 52-week period
        for idx in range(period):
            pattern.append(np.mean([X.iloc[row_idx][feat] for row_idx in range(idx, X.shape[0], period)]))
            
        # Creating seasonality component
        feat_season = []
        for idx in range(num_full_seasons):
            feat_season += pattern
            
        feat_season += pattern[:leftover]
        
        # Adding Seasonality component to dataframe
        seasonality = pd.concat([seasonality, 
                                 pd.DataFrame({feat + '_seasonality': feat_season}, index=X.index)], axis=1)
        
        # Subtracting seasonality from original feature
        deseasonalized_feats = pd.concat([deseasonalized_feats, 
                                          pd.DataFrame({feat + '_deseasonalized': np.subtract(X[[feat]].values, seasonality[[feat + '_seasonality']].values).flatten()}, index=X.index)], axis=1)
        
        
    return seasonality, deseasonalized_feats

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_seasonality_feats, sj_deseasonalize_feats = deseasonalize(pd.concat([sj_X_train, sj_X_test], axis=0))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sns.set()
fig, ax = plt.subplots(2, 1, figsize=(16,16), sharex=True)
ax[0].set_title('City San Juan Feature Seasonality')
sj_seasonality_feats.plot(ax=ax[0])
ax[1].set_title('City San Juan Deseasonalized Features')
sj_deseasonalize_feats.plot(ax=ax[1]);

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_28_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
iq_seasonality_feats, iq_deseasonalize_feats = deseasonalize(pd.concat([iq_X_train, iq_X_test], axis=0))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sns.set()
fig, ax = plt.subplots(2, 1, figsize=(16,16), sharex=True)
ax[0].set_title('City Iquitos Feature Seasonality')
iq_seasonality_feats.plot(ax=ax[0])
ax[1].set_title('City Iquitos Deseasonalized Features')
iq_deseasonalize_feats.plot(ax=ax[1]);

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_30_0.png)

</div>
</div>
</div>



---
## Deseasonalized Features and Labels



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_X_train = sj_deseasonalize_feats.loc[sj_X_train.index]
sj_y_train = pd.DataFrame({'total_cases': np.subtract(sj_y_train.values.flatten(), sj_seasonality.loc[sj_seasonality.first_valid_index():sj_X_train.last_valid_index()].values.flatten())}, index=sj_y_train.index)
iq_X_train = iq_deseasonalize_feats.loc[iq_X_train.index]
iq_y_train = pd.DataFrame({'total_cases': np.subtract(iq_y_train.values.flatten(), iq_seasonality.loc[iq_seasonality.first_valid_index():iq_X_train.last_valid_index()].values.flatten())}, index=iq_y_train.index)

sj_X_test = sj_deseasonalize_feats.loc[sj_X_test.index]
iq_X_test = iq_deseasonalize_feats.loc[iq_X_test.index]

```
</div>

</div>



### Standardization



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Import standard scaler module
from sklearn.preprocessing import StandardScaler

# Scale City sj features
sj_X_std_scaler = StandardScaler()
sj_X_train_scaled = sj_X_std_scaler.fit_transform(sj_X_train)

# Scale City iq features
iq_X_std_scaler = StandardScaler()
iq_X_train_scaled = iq_X_std_scaler.fit_transform(iq_X_train)

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
def train(X, y, mods):
    """
    Handles the entire train and testing pipeline
    
    Parameters:
    -----------
    X: (pandas.DataFrame) Feature columns
    y: (pandas.DataFrame) Labels
    mods: (list) List of sklearn models to be trained on
    
    Returns:
    --------
    DataFrame of results of training and also a dictionary of the trained models
    """
    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y).flatten()
    
    # Initialize models dictionary
    trained_mods = {str(mod())[:str(mod()).find('(')]: mod for mod in mods}
    
    # Initialize model performance dictionary
    performance_mods = {str(mod())[:str(mod()).find('(')]: defaultdict(float) for mod in mods}
        
    # Loop through all models
    for idx, (mod_name, mod) in enumerate(trained_mods.items()):
        
        # Trained Model
        trained_mod = mod()
        
        # MAE score
        mae_train = []
        mae_test = []
        
        for test_size in range(10, 60, 5):
        
            # Train Test Splits
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, shuffle=False)

            ################
            ### TRAINING ###
            ################
            # Fit the model
            trained_mod = trained_mod.fit(X_train, y_train) 

            # Prediction scores for training set
            y_train_pred = trained_mod.predict(X_train).astype(int)
            mae_train.append(mean_absolute_error(y_train, y_train_pred))

            ###############
            ### TESTING ###
            ###############
            # Prediction scores for testing set
            y_test_pred = trained_mod.predict(X_test).astype(int)
            mae_test.append(mean_absolute_error(y_test, y_test_pred))
        
        # Saving average train scores
        performance_mods[mod_name]['train_' + str(mean_absolute_error.__name__)] = np.mean(mae_train)
        
        # Saving average test scores
        performance_mods[mod_name]['test_' + str(mean_absolute_error.__name__)] = np.mean(mae_test)
        
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
sj_performance_mods, sj_trained_mods = train(sj_X_train_scaled, sj_y_train, mods_to_train)
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
      <th>train_mean_absolute_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LinearRegression</th>
      <td>33.683914</td>
      <td>32.787119</td>
    </tr>
    <tr>
      <th>BayesianRidge</th>
      <td>32.043543</td>
      <td>32.005759</td>
    </tr>
    <tr>
      <th>ElasticNet</th>
      <td>30.376247</td>
      <td>31.013492</td>
    </tr>
    <tr>
      <th>GaussianProcessRegressor</th>
      <td>21.383412</td>
      <td>0.877700</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>34.513810</td>
      <td>11.841555</td>
    </tr>
    <tr>
      <th>AdaBoostRegressor</th>
      <td>41.198541</td>
      <td>30.371852</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>31.468851</td>
      <td>13.754183</td>
    </tr>
    <tr>
      <th>KernelRidge</th>
      <td>26.194115</td>
      <td>32.354546</td>
    </tr>
    <tr>
      <th>SVR</th>
      <td>20.255342</td>
      <td>26.895049</td>
    </tr>
    <tr>
      <th>MLPRegressor</th>
      <td>28.834891</td>
      <td>29.408270</td>
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
iq_performance_mods, iq_trained_mods = train(iq_X_train_scaled, iq_y_train, mods_to_train)
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
      <th>train_mean_absolute_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LinearRegression</th>
      <td>7.391553</td>
      <td>5.771250</td>
    </tr>
    <tr>
      <th>BayesianRidge</th>
      <td>7.007555</td>
      <td>5.488698</td>
    </tr>
    <tr>
      <th>ElasticNet</th>
      <td>6.848167</td>
      <td>5.435301</td>
    </tr>
    <tr>
      <th>GaussianProcessRegressor</th>
      <td>6.599916</td>
      <td>0.899536</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>7.073381</td>
      <td>2.557108</td>
    </tr>
    <tr>
      <th>AdaBoostRegressor</th>
      <td>7.543269</td>
      <td>4.726244</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>6.664787</td>
      <td>2.078807</td>
    </tr>
    <tr>
      <th>KernelRidge</th>
      <td>7.713418</td>
      <td>5.791604</td>
    </tr>
    <tr>
      <th>SVR</th>
      <td>6.677913</td>
      <td>5.042665</td>
    </tr>
    <tr>
      <th>MLPRegressor</th>
      <td>7.380016</td>
      <td>4.887611</td>
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
sns.set()
fig, ax = plt.subplots(1, 1, figsize=(14,10), sharex=True)
sj_X_train['reanalysis_max_air_temp_k_deseasonalized'].plot(ax=ax);

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_40_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2,2)
rcParams['figure.figsize'] = 16, 16

sj_acf = plot_acf(sj_X_train['reanalysis_max_air_temp_k_deseasonalized'], ax=ax[0,0], lags=100)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_41_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# TODO: Seems like we need to do more to eliminate the trend

```
</div>

</div>



<a id="test"></a>

---
## Competition Test Prediction



### Prediction



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Save the results to csv and upload to competition
submission_df = pd.read_csv('./data/dengai/submission_format.csv')

# City SJ
sj_submission_df = submission_df[submission_df['city'] == 'sj'].drop(['total_cases'], axis=1)
sj_submission_df.reset_index(inplace=True)
sj_submission_df = pd.concat([sj_submission_df, 
                              pd.DataFrame(np.add(sj_trained_mods['SVR'].predict(sj_X_std_scaler.transform(sj_X_test)), sj_seasonality.loc[sj_X_test.first_valid_index():].values.flatten()).astype(int),
                                           columns=['total_cases'])], 
                             axis=1)
sj_submission_df.index = sj_submission_df['index']
sj_submission_df.drop(['index'], axis=1, inplace=True)

# City IQ
iq_submission_df = submission_df[submission_df['city'] == 'iq'].drop(['total_cases'], axis=1)
iq_submission_df.reset_index(inplace=True)
iq_submission_df = pd.concat([iq_submission_df, 
                              pd.DataFrame(np.add(iq_trained_mods['SVR'].predict(iq_X_std_scaler.transform(iq_X_test)), iq_seasonality.loc[iq_X_test.first_valid_index():].values.flatten()).astype(int), 
                                           columns=['total_cases'])], 
                             axis=1)
iq_submission_df.index = iq_submission_df['index']
iq_submission_df.drop(['index'], axis=1, inplace=True)

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
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>2008</td>
      <td>19</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>2008</td>
      <td>20</td>
      <td>20</td>
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
      <td>21</td>
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
# Benchmark Predictions
benchmark = pd.read_csv('./data/dengai/benchmark.csv')

sj_bench = benchmark[benchmark['city'] == 'sj']['total_cases']
sj_bench = pd.DataFrame(
    [np.nan for i in range(len(sj_X_train))] + list(sj_bench.values),
    index=sj_X_train.index.append(sj_X_test.index), 
    columns=['Benchmark Predictions'])

iq_bench = benchmark[benchmark['city'] == 'iq']['total_cases']
iq_bench = pd.DataFrame(
    [np.nan for i in range(len(iq_X_train))] + list(iq_bench.values),
    index=iq_X_train.index.append(iq_X_test.index),
    columns=['Benchmark Predictions'])

# Deseasonalized Predictions
sj_deseasonalized = sj_trained_mods['SVR'].predict(sj_X_test).astype(int)
sj_deseasonalized = pd.DataFrame([np.nan for i in range(len(sj_X_train))] + # list(sj_deseasonalized),
                                 list(np.add(sj_deseasonalized, sj_seasonality.loc[sj_X_test.first_valid_index():]['seasonality'])),
                         index=sj_X_train.index.append(sj_X_test.index),
                         columns=['Deseasonalized Predictions'])

iq_deseasonalized = iq_trained_mods['SVR'].predict(iq_X_test).astype(int)
iq_deseasonalized = pd.DataFrame([np.nan for i in range(len(iq_X_train))] + # list(iq_deseasonalized),
                                 list(np.add(iq_deseasonalized, iq_seasonality.loc[iq_X_test.first_valid_index():]['seasonality'])),
                         index=iq_X_train.index.append(iq_X_test.index),
                         columns=['Deseasonalized Predictions'])

# Plot 
sns.set()
fig, ax = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

ax[0].set_title("City San Juan")
ax[0].plot(np.add(sj_y_train[['total_cases']].values.flatten(), sj_seasonality.loc[sj_X_train.first_valid_index():sj_X_train.last_valid_index()]['seasonality'].values), label="Training Total Dengue Cases")
sj_bench.plot(ax=ax[0])
sj_deseasonalized.plot(ax=ax[0])

ax[1].set_title("City Iquitos")
ax[1].plot(np.add(iq_y_train[['total_cases']].values.flatten(), iq_seasonality.loc[iq_X_train.first_valid_index():iq_X_train.last_valid_index()]['seasonality'].values), label="Training Total Dengue Cases")
iq_bench.plot(ax=ax[1])
iq_deseasonalized.plot(ax=ax[1])

plt.legend()
plt.show();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_46_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Save to csv
submission_df.to_csv('./data/dengai/deseasonalized_submission.csv', index=False)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sj_y_train.head()

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
      <th>num_cases</th>
    </tr>
    <tr>
      <th>week_start_date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-04-30</th>
      <td>4</td>
    </tr>
    <tr>
      <th>1990-05-07</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1990-05-14</th>
      <td>4</td>
    </tr>
    <tr>
      <th>1990-05-21</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1990-05-28</th>
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
from fbprophet import Prophet

m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
df = pd.merge(sj_X_train, sj_y_train, left_index=True, right_index=True)
df = df.reset_index()
df = df.rename(columns={'week_start_date': 'ds', 'num_cases': 'y'})
df.head()

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
      <th>ds</th>
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
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
sj_X_train.columns

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Index(['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
       'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
       'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
       'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
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
for feat in sj_X_train.columns:
    m.add_regressor(feat)
    
df['floor'] = 0
m.fit(df)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<fbprophet.forecaster.Prophet at 0x1c2ca042e8>
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
future = m.make_future_dataframe(periods=2000)
# future['cap'] = 8.5
fcst = m.predict(df)
fig = m.plot(fcst)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_52_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.scatter(fcst['yhat'], df['y'])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<matplotlib.collections.PathCollection at 0x1c2b87c2e8>
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_53_1.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.plot(y['y']-fcst['yhat'])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[<matplotlib.lines.Line2D at 0x1c287efeb8>]
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/07-dengai_54_1.png)

</div>
</div>
</div>

