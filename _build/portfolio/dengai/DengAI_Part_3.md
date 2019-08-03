---
redirect_from:
  - "/portfolio/dengai/dengai-part-3"
interact_link: content/portfolio/dengai/DengAI_Part_3.ipynb
kernel_name: python3
has_widgets: false
title: 'DengAI Part 3'
prev_page:
  url: /portfolio/dengai/DengAI_Part_2
  title: 'DengAI Part 2'
next_page:
  url: /portfolio/dengai/DengAI_Part_4a
  title: 'DengAI Part 4a'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# DengAI Analysis Part 3 - Linear Regression Assumptions (OPTIONAL)

By: Chengyi (Jeff) Chen, under guidance of CSCI499: AI for Social Good Teaching Assistant - Aaron Ferber

---
## Content

In this notebook, we will look at whether our Linear Regression assumptions hold in order to effectively use a linear model for our predictions.



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
sj_pwr = PowerTransformer()
sj_y = pd.Series(sj_pwr.fit_transform(sj_y).flatten(), index=sj_y.index)

# Applying yeo-johnson transform on the labels of City iq 
# REMEMBER TO INVERSE TRANSFORM YOUR Y_PREDS
iq_pwr = PowerTransformer()
iq_y = pd.Series(iq_pwr.fit_transform(iq_y).flatten(), index=iq_y.index)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.model_selection import train_test_split

# Split our train and test data
sj_X_train, sj_X_test, sj_y_train, sj_y_test = train_test_split(sj_X, sj_y, test_size=0.33, random_state=42)
iq_X_train, iq_X_test, iq_y_train, iq_y_test = train_test_split(iq_X, iq_y, test_size=0.33, random_state=42)

```
</div>

</div>



<a id="linregassumptions"></a>

---
## Linear Regression Model Assumptions

__*According to the Gauss-Markov theorem, the Ordinary Least Squares method in Classical Linear Regression gives the best estimator of best fit, but has quite a number of assumptions, let's address them first to see if it is still an appropriate model and what we can do to make it appropriate*__

We have:
1. The model parameters are linear, meaning the regression coefficients don’t enter the function being estimated as exponents (although the variables can have exponents).

2. The values for the independent variables are derived from a random sample of the population, and they contain variability.

3. The explanatory variables don’t have perfect collinearity (that is, no independent variable can be expressed as a linear function of any other independent variables).

4. The error term has zero conditional mean, meaning that the average error is zero at any specific value of the independent variable(s).

5. The model has no heteroskedasticity (meaning the variance of the error is the same regardless of the independent variable’s value).

6. The model has no autocorrelation (the error term doesn’t exhibit a systematic relationship over time).

__In our analysis, we will address the following key assumptions:__
- [Linearity between the Exogenous and Endogenous variables](#linearity)
- [Normality of Features {Optional}](#normfeat)
- [Lack of Multicollinearity](#multicol)
- [Homoscedasticity](#homo)
- [Normality of Errors](#normerr)
- [Autocorrelation](#autocorr)



---
<a id="linearity"></a>

## 1. Linearity between each of the Independent Variables and the Dependent Variable

- Plotting the Dependent Variable against each of the Indepent Variables



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Helper function to plot scatter plots of 
# independent vars against dependent var
def plot_scatter_feats(X, y):
    rcParams['figure.figsize'] = 10, 40
    fig, ax = plt.subplots(len(X.columns) // 2,2)

    for idx, name in enumerate(X.columns):
        if idx < len(X.columns) // 2:
            ax[idx, 0].scatter(X[name], y, alpha=0.5, linewidths=0.01)
            ax[idx, 0].grid(True)
            ax[idx, 0].set_title("Number of Cases against {}".format(name), color='k')
            ax[idx, 0].set_ylabel("Frequency")
            ax[idx, 0].set_xlabel("{}".format(name))
        else:
            ax[idx - (len(X.columns) // 2), 1].scatter(X[name], y, alpha=0.5, linewidths=0.01)
            ax[idx - (len(X.columns) // 2), 1].grid(True)
            ax[idx - (len(X.columns) // 2), 1].set_title("Number of Cases against {}".format(name), color='k')
            ax[idx - (len(X.columns) // 2), 1].set_ylabel("Frequency")
            ax[idx - (len(X.columns) // 2), 1].set_xlabel("{}".format(name))

    plt.tight_layout()
    plt.show();

```
</div>

</div>



#### City sj:
- __Let's plot the dependent var (Number of Dengue Cases) against each of the independent vars (features)__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_scatter_feats(sj_X, sj_y)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_14_0.png)

</div>
</div>
</div>



#### City iq:
- __Let's plot the dependent var (Number of Dengue Cases) against each of the independent vars (features)__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_scatter_feats(iq_X, iq_y)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_16_0.png)

</div>
</div>
</div>



#### Conclusion:

- __Seems like none of the features have a linear relationship with the number of dengue cases, so the first assumption for the use of Linear Regression doesn't hold__



---
<a id="normfeat"></a>

## 2. Normality of features

- Even though this isn't a requirement for linear regression models, it'll be a good check because we can use transformations like Yeo-Johnson (positive and negative values) / Box-Cox (Only positive values) / Log Transforms to normalize and reduce skewness of our features to tackle the problem of covariate shifting.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Helper function to plot the kernel density 
# estimation plot of normal distribution overlayed
# on the distribution plot of the features
def plot_dist(X):
    rcParams['figure.figsize'] = 10, 40
    fig, ax = plt.subplots(len(X.columns) // 2,2)

    for idx, name in enumerate(X.columns):
        if idx < len(X.columns) // 2:
            dist_plot = sns.distplot(X[name], ax=ax[idx, 0], label=name)
            normal_dist = sns.kdeplot(np.random.normal(size=len(X[name])), ax=ax[idx, 0], label='Normal Distribution')
            ax[idx, 0].grid(True)
            ax[idx, 0].set_title("Distribution of {}".format(name), color='k')
        else:
            dist_plot = sns.distplot(X[name], ax=ax[idx - (len(X.columns) // 2), 1], label=name)
            normal_dist = sns.kdeplot(np.random.normal(size=len(X[name])), ax=ax[idx - (len(X.columns) // 2), 1], label='Normal Distribution')
            ax[idx - (len(X.columns) // 2), 1].grid(True)
            ax[idx - (len(X.columns) // 2), 1].set_title("Distribution of {}".format(name), color='k')

    plt.tight_layout()
    plt.show();

# Helper function to calculate the empirical cumulative 
# distribution function
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys

# Helper function to plot the normal cdf overlayed on the
# empirical cdf of the feature
def plot_ecdf(X):
    rcParams['figure.figsize'] = 10, 40
    fig, ax = plt.subplots(len(X.columns) // 2,2)

    for idx, name in enumerate(X.columns):
        if idx < len(X.columns) // 2:
            normal_cdf_line = ax[idx, 0].scatter(X[name], stats.norm.cdf(X[name]), label='Normal CDF')
            ecdf_line = ax[idx, 0].plot(*ecdf(X[name]), label='ECDF')
            ax[idx, 0].grid(True)
            ax[idx, 0].set_title("Distribution of {}".format(name), color='k')
        else:
            normal_cdf_line = ax[idx - (len(X.columns) // 2), 1].scatter(X[name], stats.norm.cdf(X[name]), label='Normal CDF')
            ecdf_line = ax[idx - (len(X.columns) // 2), 1].plot(*ecdf(X[name]), label='ECDF')
            ax[idx - (len(X.columns) // 2), 1].grid(True)
            ax[idx - (len(X.columns) // 2), 1].set_title("Distribution of {}".format(name), color='k')

    plt.tight_layout()
    plt.show();

```
</div>

</div>



#### City sj:
- __Let's plot the distribution of each of these independent vars (features) overlayed on a normal distribution too to see which of the features most similarly follow a normal distribution__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_dist(sj_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_21_0.png)

</div>
</div>
</div>



__Also, because we have more than 50 samples, we shall use the one-sample Kolmogorov-Smirnov Test for Normality instead of the Shapiro-Wilk Test for Normality__
- __Null Hypothesis: Sample is drawn from the Normal Distribution__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Kolmogorov-Smirnov Test from scipy stats
sj_X_kstest = {}
for idx, name in enumerate(sj_X.columns):
    sj_X_kstest[name] = stats.kstest(sj_X[name], 'norm').pvalue
    print(name + ' p-value: ' + str(sj_X_kstest[name]))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
ndvi_ne p-value: 6.444165324190709e-147
ndvi_nw p-value: 8.018407981845908e-162
ndvi_se p-value: 3.290057114065992e-229
ndvi_sw p-value: 6.5814645663652e-225
precipitation_amt_mm p-value: 0.0
reanalysis_air_temp_k p-value: 0.0
reanalysis_avg_temp_k p-value: 0.0
reanalysis_dew_point_temp_k p-value: 0.0
reanalysis_max_air_temp_k p-value: 0.0
reanalysis_min_air_temp_k p-value: 0.0
reanalysis_precip_amt_kg_per_m2 p-value: 0.0
reanalysis_relative_humidity_percent p-value: 0.0
reanalysis_sat_precip_amt_mm p-value: 0.0
reanalysis_specific_humidity_g_per_kg p-value: 0.0
reanalysis_tdtr_k p-value: 0.0
station_avg_temp_c p-value: 0.0
station_diur_temp_rng_c p-value: 0.0
station_max_temp_c p-value: 0.0
station_min_temp_c p-value: 0.0
station_precip_mm p-value: 0.0
```
</div>
</div>
</div>



__Plotting the ecdf of the feature and the normal distribution cdf__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_ecdf(sj_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_25_0.png)

</div>
</div>
</div>



#### City iq:
- __Let's plot the distribution of each of these independent vars (features) overlayed on a normal distribution too to see which of the features most similarly follow a normal distribution__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_dist(iq_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_27_0.png)

</div>
</div>
</div>



__Again, because we have more than 50 samples, we shall use the one-sample Kolmogorov-Smirnov Test for Normality instead of the Shapiro-Wilk Test for Normality__
- __Null Hypothesis: Sample is drawn from the Normal Distribution__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Kolmogorov-Smirnov Test from scipy stats
iq_X_kstest = {}
for idx, name in enumerate(iq_X.columns):
    iq_X_kstest[name] = stats.kstest(iq_X[name], 'norm').pvalue
    print(name + ' p-value: ' + str(iq_X_kstest[name]))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
ndvi_ne p-value: 1.391204576455867e-135
ndvi_nw p-value: 4.935150098784429e-131
ndvi_se p-value: 4.800097394750673e-137
ndvi_sw p-value: 1.397598704162834e-138
precipitation_amt_mm p-value: 0.0
reanalysis_air_temp_k p-value: 0.0
reanalysis_avg_temp_k p-value: 0.0
reanalysis_dew_point_temp_k p-value: 0.0
reanalysis_max_air_temp_k p-value: 0.0
reanalysis_min_air_temp_k p-value: 0.0
reanalysis_precip_amt_kg_per_m2 p-value: 0.0
reanalysis_relative_humidity_percent p-value: 0.0
reanalysis_sat_precip_amt_mm p-value: 0.0
reanalysis_specific_humidity_g_per_kg p-value: 0.0
reanalysis_tdtr_k p-value: 0.0
station_avg_temp_c p-value: 0.0
station_diur_temp_rng_c p-value: 0.0
station_max_temp_c p-value: 0.0
station_min_temp_c p-value: 0.0
station_precip_mm p-value: 0.0
```
</div>
</div>
</div>



__Plotting the ecdf of the feature and the normal distribution cdf__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_ecdf(iq_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_31_0.png)

</div>
</div>
</div>



#### Conclusion:

- __Seems like none of our features are normally distributed, we would have to perform some transforms later on in the Feature Engineering section in order to improve the performances of our models__



---
<a id="multicol"></a>

## 3. Lack of Multicollinearity



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Helper function that plots a correlation matrix for 
# each pair of independent vars to ensure
# that no 2 features are too highly 
# correlated with each other
def correlation_matrix(X):
    axs = pd.plotting.scatter_matrix(X, figsize=(30, 30), diagonal='kde')
    n = len(X.columns)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axs[x, y]
            # to make x axis name vertical  
            ax.xaxis.label.set_rotation(90)
            # to make y axis name horizontal 
            ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50
    plt.show();

# We will use the variance inflation factor
# metric to determine which columns to drop
# Anything with VIF above 5.0 will be dropped
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Helper function to iteratively drop columns
# that have VIFs > 5.0
def calculate_vif(X, thresh=5.0):
    
        # Columns to be dropped
        dropped_cols = []
    
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(np.array(X), X.columns.get_loc(var)) for var in X.columns]
            
            # Get the highest VIF value
            max_vif = max(vif)
            
            # Drop the column with highest VIF value
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                dropped_cols.append(X.columns[maxloc])
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
                
        return pd.DataFrame(X, index=X.index, columns=X.columns), dropped_cols

```
</div>

</div>



#### City sj:
- __Multicollinearity checks__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
correlation_matrix(sj_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_36_0.png)

</div>
</div>
</div>



__There are definitely a number of features that are highly correlated and so we need to remove those for Linear Regression, let's take a look at the correlation matrix for City sj's data__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Correlation matrix for City sj's data
sj_X.corr()

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
  </thead>
  <tbody>
    <tr>
      <th>ndvi_ne</th>
      <td>1.000000</td>
      <td>0.611061</td>
      <td>0.204144</td>
      <td>0.160510</td>
      <td>-0.039551</td>
      <td>-0.063952</td>
      <td>-0.062322</td>
      <td>-0.034695</td>
      <td>-0.037749</td>
      <td>-0.084744</td>
      <td>0.005093</td>
      <td>0.035057</td>
      <td>-0.039551</td>
      <td>-0.031195</td>
      <td>-0.004793</td>
      <td>0.058791</td>
      <td>0.126759</td>
      <td>0.079690</td>
      <td>0.015922</td>
      <td>-0.070517</td>
    </tr>
    <tr>
      <th>ndvi_nw</th>
      <td>0.611061</td>
      <td>1.000000</td>
      <td>0.203699</td>
      <td>0.209359</td>
      <td>-0.031111</td>
      <td>-0.077111</td>
      <td>-0.075962</td>
      <td>-0.027575</td>
      <td>-0.045942</td>
      <td>-0.075725</td>
      <td>0.008295</td>
      <td>0.073228</td>
      <td>-0.031111</td>
      <td>-0.022876</td>
      <td>-0.044089</td>
      <td>0.080030</td>
      <td>0.177679</td>
      <td>0.124975</td>
      <td>0.009846</td>
      <td>-0.074033</td>
    </tr>
    <tr>
      <th>ndvi_se</th>
      <td>0.204144</td>
      <td>0.203699</td>
      <td>1.000000</td>
      <td>0.800022</td>
      <td>-0.107798</td>
      <td>-0.017256</td>
      <td>-0.014424</td>
      <td>-0.066628</td>
      <td>-0.009224</td>
      <td>-0.048689</td>
      <td>-0.125404</td>
      <td>-0.116932</td>
      <td>-0.107798</td>
      <td>-0.062974</td>
      <td>0.042065</td>
      <td>-0.064756</td>
      <td>0.008685</td>
      <td>-0.079620</td>
      <td>-0.075879</td>
      <td>-0.133574</td>
    </tr>
    <tr>
      <th>ndvi_sw</th>
      <td>0.160510</td>
      <td>0.209359</td>
      <td>0.800022</td>
      <td>1.000000</td>
      <td>-0.117825</td>
      <td>-0.043794</td>
      <td>-0.036385</td>
      <td>-0.087744</td>
      <td>-0.015060</td>
      <td>-0.072415</td>
      <td>-0.125116</td>
      <td>-0.117245</td>
      <td>-0.117825</td>
      <td>-0.080988</td>
      <td>0.051879</td>
      <td>-0.041236</td>
      <td>0.070172</td>
      <td>-0.017788</td>
      <td>-0.074070</td>
      <td>-0.172982</td>
    </tr>
    <tr>
      <th>precipitation_amt_mm</th>
      <td>-0.039551</td>
      <td>-0.031111</td>
      <td>-0.107798</td>
      <td>-0.117825</td>
      <td>1.000000</td>
      <td>0.232449</td>
      <td>0.220561</td>
      <td>0.400364</td>
      <td>0.256661</td>
      <td>0.244826</td>
      <td>0.510971</td>
      <td>0.499257</td>
      <td>1.000000</td>
      <td>0.405957</td>
      <td>-0.091373</td>
      <td>0.196258</td>
      <td>-0.154054</td>
      <td>0.188603</td>
      <td>0.221128</td>
      <td>0.568559</td>
    </tr>
    <tr>
      <th>reanalysis_air_temp_k</th>
      <td>-0.063952</td>
      <td>-0.077111</td>
      <td>-0.017256</td>
      <td>-0.043794</td>
      <td>0.232449</td>
      <td>1.000000</td>
      <td>0.997496</td>
      <td>0.903435</td>
      <td>0.934956</td>
      <td>0.942364</td>
      <td>0.077371</td>
      <td>0.298523</td>
      <td>0.232449</td>
      <td>0.905013</td>
      <td>0.173619</td>
      <td>0.880791</td>
      <td>0.036733</td>
      <td>0.698705</td>
      <td>0.833196</td>
      <td>0.109889</td>
    </tr>
    <tr>
      <th>reanalysis_avg_temp_k</th>
      <td>-0.062322</td>
      <td>-0.075962</td>
      <td>-0.014424</td>
      <td>-0.036385</td>
      <td>0.220561</td>
      <td>0.997496</td>
      <td>1.000000</td>
      <td>0.895272</td>
      <td>0.938752</td>
      <td>0.939230</td>
      <td>0.059390</td>
      <td>0.284517</td>
      <td>0.220561</td>
      <td>0.896446</td>
      <td>0.196604</td>
      <td>0.878929</td>
      <td>0.051095</td>
      <td>0.704125</td>
      <td>0.827554</td>
      <td>0.093831</td>
    </tr>
    <tr>
      <th>reanalysis_dew_point_temp_k</th>
      <td>-0.034695</td>
      <td>-0.027575</td>
      <td>-0.066628</td>
      <td>-0.087744</td>
      <td>0.400364</td>
      <td>0.903435</td>
      <td>0.895272</td>
      <td>1.000000</td>
      <td>0.847312</td>
      <td>0.898975</td>
      <td>0.325861</td>
      <td>0.678608</td>
      <td>0.400364</td>
      <td>0.998332</td>
      <td>-0.036918</td>
      <td>0.868923</td>
      <td>-0.058712</td>
      <td>0.690384</td>
      <td>0.850172</td>
      <td>0.281470</td>
    </tr>
    <tr>
      <th>reanalysis_max_air_temp_k</th>
      <td>-0.037749</td>
      <td>-0.045942</td>
      <td>-0.009224</td>
      <td>-0.015060</td>
      <td>0.256661</td>
      <td>0.934956</td>
      <td>0.938752</td>
      <td>0.847312</td>
      <td>1.000000</td>
      <td>0.828538</td>
      <td>0.089625</td>
      <td>0.288231</td>
      <td>0.256661</td>
      <td>0.852958</td>
      <td>0.349128</td>
      <td>0.852609</td>
      <td>0.111930</td>
      <td>0.760855</td>
      <td>0.770417</td>
      <td>0.102656</td>
    </tr>
    <tr>
      <th>reanalysis_min_air_temp_k</th>
      <td>-0.084744</td>
      <td>-0.075725</td>
      <td>-0.048689</td>
      <td>-0.072415</td>
      <td>0.244826</td>
      <td>0.942364</td>
      <td>0.939230</td>
      <td>0.898975</td>
      <td>0.828538</td>
      <td>1.000000</td>
      <td>0.130047</td>
      <td>0.385279</td>
      <td>0.244826</td>
      <td>0.896369</td>
      <td>-0.053713</td>
      <td>0.841624</td>
      <td>-0.026271</td>
      <td>0.627690</td>
      <td>0.829668</td>
      <td>0.147299</td>
    </tr>
    <tr>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <td>0.005093</td>
      <td>0.008295</td>
      <td>-0.125404</td>
      <td>-0.125116</td>
      <td>0.510971</td>
      <td>0.077371</td>
      <td>0.059390</td>
      <td>0.325861</td>
      <td>0.089625</td>
      <td>0.130047</td>
      <td>1.000000</td>
      <td>0.602291</td>
      <td>0.510971</td>
      <td>0.330466</td>
      <td>-0.306139</td>
      <td>0.132884</td>
      <td>-0.249037</td>
      <td>0.076726</td>
      <td>0.193756</td>
      <td>0.479889</td>
    </tr>
    <tr>
      <th>reanalysis_relative_humidity_percent</th>
      <td>0.035057</td>
      <td>0.073228</td>
      <td>-0.116932</td>
      <td>-0.117245</td>
      <td>0.499257</td>
      <td>0.298523</td>
      <td>0.284517</td>
      <td>0.678608</td>
      <td>0.288231</td>
      <td>0.385279</td>
      <td>0.602291</td>
      <td>1.000000</td>
      <td>0.499257</td>
      <td>0.672710</td>
      <td>-0.374583</td>
      <td>0.427480</td>
      <td>-0.193197</td>
      <td>0.342374</td>
      <td>0.465352</td>
      <td>0.443702</td>
    </tr>
    <tr>
      <th>reanalysis_sat_precip_amt_mm</th>
      <td>-0.039551</td>
      <td>-0.031111</td>
      <td>-0.107798</td>
      <td>-0.117825</td>
      <td>1.000000</td>
      <td>0.232449</td>
      <td>0.220561</td>
      <td>0.400364</td>
      <td>0.256661</td>
      <td>0.244826</td>
      <td>0.510971</td>
      <td>0.499257</td>
      <td>1.000000</td>
      <td>0.405957</td>
      <td>-0.091373</td>
      <td>0.196258</td>
      <td>-0.154054</td>
      <td>0.188603</td>
      <td>0.221128</td>
      <td>0.568559</td>
    </tr>
    <tr>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <td>-0.031195</td>
      <td>-0.022876</td>
      <td>-0.062974</td>
      <td>-0.080988</td>
      <td>0.405957</td>
      <td>0.905013</td>
      <td>0.896446</td>
      <td>0.998332</td>
      <td>0.852958</td>
      <td>0.896369</td>
      <td>0.330466</td>
      <td>0.672710</td>
      <td>0.405957</td>
      <td>1.000000</td>
      <td>-0.030248</td>
      <td>0.869591</td>
      <td>-0.062715</td>
      <td>0.692052</td>
      <td>0.849818</td>
      <td>0.282952</td>
    </tr>
    <tr>
      <th>reanalysis_tdtr_k</th>
      <td>-0.004793</td>
      <td>-0.044089</td>
      <td>0.042065</td>
      <td>0.051879</td>
      <td>-0.091373</td>
      <td>0.173619</td>
      <td>0.196604</td>
      <td>-0.036918</td>
      <td>0.349128</td>
      <td>-0.053713</td>
      <td>-0.306139</td>
      <td>-0.374583</td>
      <td>-0.091373</td>
      <td>-0.030248</td>
      <td>1.000000</td>
      <td>0.135205</td>
      <td>0.371731</td>
      <td>0.277799</td>
      <td>0.005795</td>
      <td>-0.203881</td>
    </tr>
    <tr>
      <th>station_avg_temp_c</th>
      <td>0.058791</td>
      <td>0.080030</td>
      <td>-0.064756</td>
      <td>-0.041236</td>
      <td>0.196258</td>
      <td>0.880791</td>
      <td>0.878929</td>
      <td>0.868923</td>
      <td>0.852609</td>
      <td>0.841624</td>
      <td>0.132884</td>
      <td>0.427480</td>
      <td>0.196258</td>
      <td>0.869591</td>
      <td>0.135205</td>
      <td>1.000000</td>
      <td>0.181709</td>
      <td>0.864203</td>
      <td>0.897568</td>
      <td>0.029107</td>
    </tr>
    <tr>
      <th>station_diur_temp_rng_c</th>
      <td>0.126759</td>
      <td>0.177679</td>
      <td>0.008685</td>
      <td>0.070172</td>
      <td>-0.154054</td>
      <td>0.036733</td>
      <td>0.051095</td>
      <td>-0.058712</td>
      <td>0.111930</td>
      <td>-0.026271</td>
      <td>-0.249037</td>
      <td>-0.193197</td>
      <td>-0.154054</td>
      <td>-0.062715</td>
      <td>0.371731</td>
      <td>0.181709</td>
      <td>1.000000</td>
      <td>0.468789</td>
      <td>-0.127939</td>
      <td>-0.260208</td>
    </tr>
    <tr>
      <th>station_max_temp_c</th>
      <td>0.079690</td>
      <td>0.124975</td>
      <td>-0.079620</td>
      <td>-0.017788</td>
      <td>0.188603</td>
      <td>0.698705</td>
      <td>0.704125</td>
      <td>0.690384</td>
      <td>0.760855</td>
      <td>0.627690</td>
      <td>0.076726</td>
      <td>0.342374</td>
      <td>0.188603</td>
      <td>0.692052</td>
      <td>0.277799</td>
      <td>0.864203</td>
      <td>0.468789</td>
      <td>1.000000</td>
      <td>0.674191</td>
      <td>0.001858</td>
    </tr>
    <tr>
      <th>station_min_temp_c</th>
      <td>0.015922</td>
      <td>0.009846</td>
      <td>-0.075879</td>
      <td>-0.074070</td>
      <td>0.221128</td>
      <td>0.833196</td>
      <td>0.827554</td>
      <td>0.850172</td>
      <td>0.770417</td>
      <td>0.829668</td>
      <td>0.193756</td>
      <td>0.465352</td>
      <td>0.221128</td>
      <td>0.849818</td>
      <td>0.005795</td>
      <td>0.897568</td>
      <td>-0.127939</td>
      <td>0.674191</td>
      <td>1.000000</td>
      <td>0.079721</td>
    </tr>
    <tr>
      <th>station_precip_mm</th>
      <td>-0.070517</td>
      <td>-0.074033</td>
      <td>-0.133574</td>
      <td>-0.172982</td>
      <td>0.568559</td>
      <td>0.109889</td>
      <td>0.093831</td>
      <td>0.281470</td>
      <td>0.102656</td>
      <td>0.147299</td>
      <td>0.479889</td>
      <td>0.443702</td>
      <td>0.568559</td>
      <td>0.282952</td>
      <td>-0.203881</td>
      <td>0.029107</td>
      <td>-0.260208</td>
      <td>0.001858</td>
      <td>0.079721</td>
      <td>1.000000</td>
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
# Let's use statsmodel's api instead for linear regression
# as it'll provide us a better summary of the fit
sj_ols = sm.OLS(endog=sj_y_train, exog=sj_X_train).fit()

sj_y_pred = sj_pwr.inverse_transform(sj_ols.predict(sj_X_test)[:, None])
sj_ols.summary()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.170</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.144</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   6.544</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 12 Apr 2019</td> <th>  Prob (F-statistic):</th> <td>1.09e-15</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:18:42</td>     <th>  Log-Likelihood:    </th> <td> -826.99</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   627</td>      <th>  AIC:               </th> <td>   1692.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   608</td>      <th>  BIC:               </th> <td>   1776.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    19</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ndvi_ne</th>                               <td>    0.5845</td> <td>    0.490</td> <td>    1.193</td> <td> 0.233</td> <td>   -0.378</td> <td>    1.547</td>
</tr>
<tr>
  <th>ndvi_nw</th>                               <td>    0.5400</td> <td>    0.527</td> <td>    1.024</td> <td> 0.306</td> <td>   -0.495</td> <td>    1.575</td>
</tr>
<tr>
  <th>ndvi_se</th>                               <td>   -2.9661</td> <td>    1.084</td> <td>   -2.736</td> <td> 0.006</td> <td>   -5.095</td> <td>   -0.837</td>
</tr>
<tr>
  <th>ndvi_sw</th>                               <td>    1.6052</td> <td>    1.135</td> <td>    1.414</td> <td> 0.158</td> <td>   -0.625</td> <td>    3.835</td>
</tr>
<tr>
  <th>precipitation_amt_mm</th>                  <td>   -0.0009</td> <td>    0.001</td> <td>   -1.597</td> <td> 0.111</td> <td>   -0.002</td> <td>    0.000</td>
</tr>
<tr>
  <th>reanalysis_air_temp_k</th>                 <td>    5.4901</td> <td>    1.797</td> <td>    3.055</td> <td> 0.002</td> <td>    1.961</td> <td>    9.019</td>
</tr>
<tr>
  <th>reanalysis_avg_temp_k</th>                 <td>   -0.5753</td> <td>    0.500</td> <td>   -1.152</td> <td> 0.250</td> <td>   -1.557</td> <td>    0.406</td>
</tr>
<tr>
  <th>reanalysis_dew_point_temp_k</th>           <td>   -5.6063</td> <td>    1.875</td> <td>   -2.990</td> <td> 0.003</td> <td>   -9.289</td> <td>   -1.924</td>
</tr>
<tr>
  <th>reanalysis_max_air_temp_k</th>             <td>    0.2843</td> <td>    0.122</td> <td>    2.328</td> <td> 0.020</td> <td>    0.045</td> <td>    0.524</td>
</tr>
<tr>
  <th>reanalysis_min_air_temp_k</th>             <td>    0.0440</td> <td>    0.129</td> <td>    0.341</td> <td> 0.734</td> <td>   -0.210</td> <td>    0.298</td>
</tr>
<tr>
  <th>reanalysis_precip_amt_kg_per_m2</th>       <td>    0.0026</td> <td>    0.002</td> <td>    1.250</td> <td> 0.212</td> <td>   -0.001</td> <td>    0.007</td>
</tr>
<tr>
  <th>reanalysis_relative_humidity_percent</th>  <td>    1.0384</td> <td>    0.376</td> <td>    2.764</td> <td> 0.006</td> <td>    0.301</td> <td>    1.776</td>
</tr>
<tr>
  <th>reanalysis_sat_precip_amt_mm</th>          <td>   -0.0009</td> <td>    0.001</td> <td>   -1.597</td> <td> 0.111</td> <td>   -0.002</td> <td>    0.000</td>
</tr>
<tr>
  <th>reanalysis_specific_humidity_g_per_kg</th> <td>    0.7782</td> <td>    0.122</td> <td>    6.393</td> <td> 0.000</td> <td>    0.539</td> <td>    1.017</td>
</tr>
<tr>
  <th>reanalysis_tdtr_k</th>                     <td>   -0.5676</td> <td>    0.139</td> <td>   -4.089</td> <td> 0.000</td> <td>   -0.840</td> <td>   -0.295</td>
</tr>
<tr>
  <th>station_avg_temp_c</th>                    <td>   -0.4782</td> <td>    0.128</td> <td>   -3.723</td> <td> 0.000</td> <td>   -0.731</td> <td>   -0.226</td>
</tr>
<tr>
  <th>station_diur_temp_rng_c</th>               <td>    0.2182</td> <td>    0.077</td> <td>    2.842</td> <td> 0.005</td> <td>    0.067</td> <td>    0.369</td>
</tr>
<tr>
  <th>station_max_temp_c</th>                    <td>    0.0428</td> <td>    0.062</td> <td>    0.688</td> <td> 0.492</td> <td>   -0.079</td> <td>    0.165</td>
</tr>
<tr>
  <th>station_min_temp_c</th>                    <td>    0.1212</td> <td>    0.082</td> <td>    1.469</td> <td> 0.142</td> <td>   -0.041</td> <td>    0.283</td>
</tr>
<tr>
  <th>station_precip_mm</th>                     <td>   -0.0034</td> <td>    0.002</td> <td>   -2.050</td> <td> 0.041</td> <td>   -0.007</td> <td>   -0.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.496</td> <th>  Durbin-Watson:     </th> <td>   2.003</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.780</td> <th>  Jarque-Bera (JB):  </th> <td>   0.600</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.034</td> <th>  Prob(JB):          </th> <td>   0.741</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.865</td> <th>  Cond. No.          </th> <td>5.48e+16</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 9.58e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.
</div>


</div>
</div>
</div>



__From this, we can definitely conclude that there is a strong multicollinearity problem as an almost zero eigen value shows minimal variation in the data orthogonal with other eigen vectors. Hence, we will definitely need to remove some of those features later on in the Feature Engineering section.__



#### City iq:
- __Multicollinearity checks__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
correlation_matrix(iq_X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_42_0.png)

</div>
</div>
</div>



__There are definitely a number of features that are highly correlated and so we need to remove those for Linear Regression, let's take a look at the correlation matrix for City iq's data__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Correlation matrix for City iq's data
iq_X.corr()

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
  </thead>
  <tbody>
    <tr>
      <th>ndvi_ne</th>
      <td>1.000000</td>
      <td>0.774034</td>
      <td>0.774317</td>
      <td>0.839890</td>
      <td>0.014218</td>
      <td>0.127790</td>
      <td>0.158418</td>
      <td>-0.037162</td>
      <td>0.239496</td>
      <td>-0.046257</td>
      <td>-0.066471</td>
      <td>-0.116273</td>
      <td>0.014218</td>
      <td>-0.038419</td>
      <td>0.201657</td>
      <td>0.109382</td>
      <td>0.130617</td>
      <td>0.153835</td>
      <td>-0.032599</td>
      <td>0.020981</td>
    </tr>
    <tr>
      <th>ndvi_nw</th>
      <td>0.774034</td>
      <td>1.000000</td>
      <td>0.654885</td>
      <td>0.763458</td>
      <td>-0.033660</td>
      <td>0.129202</td>
      <td>0.160483</td>
      <td>-0.039842</td>
      <td>0.229888</td>
      <td>-0.037980</td>
      <td>-0.060720</td>
      <td>-0.116696</td>
      <td>-0.033660</td>
      <td>-0.037639</td>
      <td>0.200341</td>
      <td>0.121743</td>
      <td>0.169618</td>
      <td>0.168421</td>
      <td>-0.108186</td>
      <td>-0.009004</td>
    </tr>
    <tr>
      <th>ndvi_se</th>
      <td>0.774317</td>
      <td>0.654885</td>
      <td>1.000000</td>
      <td>0.716587</td>
      <td>-0.023394</td>
      <td>0.177077</td>
      <td>0.201097</td>
      <td>-0.067431</td>
      <td>0.278891</td>
      <td>-0.055555</td>
      <td>-0.109648</td>
      <td>-0.176609</td>
      <td>-0.023394</td>
      <td>-0.066830</td>
      <td>0.245867</td>
      <td>0.113640</td>
      <td>0.150521</td>
      <td>0.172615</td>
      <td>-0.072521</td>
      <td>0.014644</td>
    </tr>
    <tr>
      <th>ndvi_sw</th>
      <td>0.839890</td>
      <td>0.763458</td>
      <td>0.716587</td>
      <td>1.000000</td>
      <td>-0.002726</td>
      <td>0.146693</td>
      <td>0.170130</td>
      <td>-0.034068</td>
      <td>0.243211</td>
      <td>-0.026474</td>
      <td>-0.053889</td>
      <td>-0.127395</td>
      <td>-0.002726</td>
      <td>-0.031982</td>
      <td>0.191847</td>
      <td>0.110828</td>
      <td>0.138694</td>
      <td>0.182665</td>
      <td>-0.068972</td>
      <td>0.004591</td>
    </tr>
    <tr>
      <th>precipitation_amt_mm</th>
      <td>0.014218</td>
      <td>-0.033660</td>
      <td>-0.023394</td>
      <td>-0.002726</td>
      <td>1.000000</td>
      <td>-0.071019</td>
      <td>-0.062916</td>
      <td>0.461964</td>
      <td>-0.191111</td>
      <td>0.268256</td>
      <td>0.347904</td>
      <td>0.441872</td>
      <td>1.000000</td>
      <td>0.456326</td>
      <td>-0.325485</td>
      <td>0.104711</td>
      <td>-0.064426</td>
      <td>0.011423</td>
      <td>0.268507</td>
      <td>0.379101</td>
    </tr>
    <tr>
      <th>reanalysis_air_temp_k</th>
      <td>0.127790</td>
      <td>0.129202</td>
      <td>0.177077</td>
      <td>0.146693</td>
      <td>-0.071019</td>
      <td>1.000000</td>
      <td>0.969564</td>
      <td>0.152962</td>
      <td>0.701653</td>
      <td>0.433898</td>
      <td>-0.096471</td>
      <td>-0.555667</td>
      <td>-0.071019</td>
      <td>0.181193</td>
      <td>0.494696</td>
      <td>0.575891</td>
      <td>0.396199</td>
      <td>0.615369</td>
      <td>0.261562</td>
      <td>-0.145418</td>
    </tr>
    <tr>
      <th>reanalysis_avg_temp_k</th>
      <td>0.158418</td>
      <td>0.160483</td>
      <td>0.201097</td>
      <td>0.170130</td>
      <td>-0.062916</td>
      <td>0.969564</td>
      <td>1.000000</td>
      <td>0.143436</td>
      <td>0.756814</td>
      <td>0.399067</td>
      <td>-0.112285</td>
      <td>-0.542291</td>
      <td>-0.062916</td>
      <td>0.167883</td>
      <td>0.569072</td>
      <td>0.551703</td>
      <td>0.420398</td>
      <td>0.606750</td>
      <td>0.215982</td>
      <td>-0.142035</td>
    </tr>
    <tr>
      <th>reanalysis_dew_point_temp_k</th>
      <td>-0.037162</td>
      <td>-0.039842</td>
      <td>-0.067431</td>
      <td>-0.034068</td>
      <td>0.461964</td>
      <td>0.152962</td>
      <td>0.143436</td>
      <td>1.000000</td>
      <td>-0.253297</td>
      <td>0.734408</td>
      <td>0.567774</td>
      <td>0.733533</td>
      <td>0.461964</td>
      <td>0.997687</td>
      <td>-0.596465</td>
      <td>0.313601</td>
      <td>-0.144900</td>
      <td>0.086939</td>
      <td>0.611714</td>
      <td>0.184254</td>
    </tr>
    <tr>
      <th>reanalysis_max_air_temp_k</th>
      <td>0.239496</td>
      <td>0.229888</td>
      <td>0.278891</td>
      <td>0.243211</td>
      <td>-0.191111</td>
      <td>0.701653</td>
      <td>0.756814</td>
      <td>-0.253297</td>
      <td>1.000000</td>
      <td>-0.093083</td>
      <td>-0.236611</td>
      <td>-0.685484</td>
      <td>-0.191111</td>
      <td>-0.238309</td>
      <td>0.807493</td>
      <td>0.359023</td>
      <td>0.511604</td>
      <td>0.587153</td>
      <td>-0.129463</td>
      <td>-0.175966</td>
    </tr>
    <tr>
      <th>reanalysis_min_air_temp_k</th>
      <td>-0.046257</td>
      <td>-0.037980</td>
      <td>-0.055555</td>
      <td>-0.026474</td>
      <td>0.268256</td>
      <td>0.433898</td>
      <td>0.399067</td>
      <td>0.734408</td>
      <td>-0.093083</td>
      <td>1.000000</td>
      <td>0.362474</td>
      <td>0.316890</td>
      <td>0.268256</td>
      <td>0.741351</td>
      <td>-0.438653</td>
      <td>0.372890</td>
      <td>-0.042654</td>
      <td>0.183093</td>
      <td>0.608160</td>
      <td>0.066370</td>
    </tr>
    <tr>
      <th>reanalysis_precip_amt_kg_per_m2</th>
      <td>-0.066471</td>
      <td>-0.060720</td>
      <td>-0.109648</td>
      <td>-0.053889</td>
      <td>0.347904</td>
      <td>-0.096471</td>
      <td>-0.112285</td>
      <td>0.567774</td>
      <td>-0.236611</td>
      <td>0.362474</td>
      <td>1.000000</td>
      <td>0.555827</td>
      <td>0.347904</td>
      <td>0.572617</td>
      <td>-0.502188</td>
      <td>0.040132</td>
      <td>-0.125519</td>
      <td>-0.048510</td>
      <td>0.232738</td>
      <td>0.163351</td>
    </tr>
    <tr>
      <th>reanalysis_relative_humidity_percent</th>
      <td>-0.116273</td>
      <td>-0.116696</td>
      <td>-0.176609</td>
      <td>-0.127395</td>
      <td>0.441872</td>
      <td>-0.555667</td>
      <td>-0.542291</td>
      <td>0.733533</td>
      <td>-0.685484</td>
      <td>0.316890</td>
      <td>0.555827</td>
      <td>1.000000</td>
      <td>0.441872</td>
      <td>0.712957</td>
      <td>-0.836152</td>
      <td>-0.126244</td>
      <td>-0.382709</td>
      <td>-0.339640</td>
      <td>0.329833</td>
      <td>0.258117</td>
    </tr>
    <tr>
      <th>reanalysis_sat_precip_amt_mm</th>
      <td>0.014218</td>
      <td>-0.033660</td>
      <td>-0.023394</td>
      <td>-0.002726</td>
      <td>1.000000</td>
      <td>-0.071019</td>
      <td>-0.062916</td>
      <td>0.461964</td>
      <td>-0.191111</td>
      <td>0.268256</td>
      <td>0.347904</td>
      <td>0.441872</td>
      <td>1.000000</td>
      <td>0.456326</td>
      <td>-0.325485</td>
      <td>0.104711</td>
      <td>-0.064426</td>
      <td>0.011423</td>
      <td>0.268507</td>
      <td>0.379101</td>
    </tr>
    <tr>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <td>-0.038419</td>
      <td>-0.037639</td>
      <td>-0.066830</td>
      <td>-0.031982</td>
      <td>0.456326</td>
      <td>0.181193</td>
      <td>0.167883</td>
      <td>0.997687</td>
      <td>-0.238309</td>
      <td>0.741351</td>
      <td>0.572617</td>
      <td>0.712957</td>
      <td>0.456326</td>
      <td>1.000000</td>
      <td>-0.585858</td>
      <td>0.329024</td>
      <td>-0.136610</td>
      <td>0.102790</td>
      <td>0.611363</td>
      <td>0.175124</td>
    </tr>
    <tr>
      <th>reanalysis_tdtr_k</th>
      <td>0.201657</td>
      <td>0.200341</td>
      <td>0.245867</td>
      <td>0.191847</td>
      <td>-0.325485</td>
      <td>0.494696</td>
      <td>0.569072</td>
      <td>-0.596465</td>
      <td>0.807493</td>
      <td>-0.438653</td>
      <td>-0.502188</td>
      <td>-0.836152</td>
      <td>-0.325485</td>
      <td>-0.585858</td>
      <td>1.000000</td>
      <td>0.159228</td>
      <td>0.449902</td>
      <td>0.375843</td>
      <td>-0.372840</td>
      <td>-0.222780</td>
    </tr>
    <tr>
      <th>station_avg_temp_c</th>
      <td>0.109382</td>
      <td>0.121743</td>
      <td>0.113640</td>
      <td>0.110828</td>
      <td>0.104711</td>
      <td>0.575891</td>
      <td>0.551703</td>
      <td>0.313601</td>
      <td>0.359023</td>
      <td>0.372890</td>
      <td>0.040132</td>
      <td>-0.126244</td>
      <td>0.104711</td>
      <td>0.329024</td>
      <td>0.159228</td>
      <td>1.000000</td>
      <td>0.423576</td>
      <td>0.642971</td>
      <td>0.416793</td>
      <td>-0.063083</td>
    </tr>
    <tr>
      <th>station_diur_temp_rng_c</th>
      <td>0.130617</td>
      <td>0.169618</td>
      <td>0.150521</td>
      <td>0.138694</td>
      <td>-0.064426</td>
      <td>0.396199</td>
      <td>0.420398</td>
      <td>-0.144900</td>
      <td>0.511604</td>
      <td>-0.042654</td>
      <td>-0.125519</td>
      <td>-0.382709</td>
      <td>-0.064426</td>
      <td>-0.136610</td>
      <td>0.449902</td>
      <td>0.423576</td>
      <td>1.000000</td>
      <td>0.626126</td>
      <td>-0.217913</td>
      <td>-0.152692</td>
    </tr>
    <tr>
      <th>station_max_temp_c</th>
      <td>0.153835</td>
      <td>0.168421</td>
      <td>0.172615</td>
      <td>0.182665</td>
      <td>0.011423</td>
      <td>0.615369</td>
      <td>0.606750</td>
      <td>0.086939</td>
      <td>0.587153</td>
      <td>0.183093</td>
      <td>-0.048510</td>
      <td>-0.339640</td>
      <td>0.011423</td>
      <td>0.102790</td>
      <td>0.375843</td>
      <td>0.642971</td>
      <td>0.626126</td>
      <td>1.000000</td>
      <td>0.087730</td>
      <td>-0.138054</td>
    </tr>
    <tr>
      <th>station_min_temp_c</th>
      <td>-0.032599</td>
      <td>-0.108186</td>
      <td>-0.072521</td>
      <td>-0.068972</td>
      <td>0.268507</td>
      <td>0.261562</td>
      <td>0.215982</td>
      <td>0.611714</td>
      <td>-0.129463</td>
      <td>0.608160</td>
      <td>0.232738</td>
      <td>0.329833</td>
      <td>0.268507</td>
      <td>0.611363</td>
      <td>-0.372840</td>
      <td>0.416793</td>
      <td>-0.217913</td>
      <td>0.087730</td>
      <td>1.000000</td>
      <td>0.150745</td>
    </tr>
    <tr>
      <th>station_precip_mm</th>
      <td>0.020981</td>
      <td>-0.009004</td>
      <td>0.014644</td>
      <td>0.004591</td>
      <td>0.379101</td>
      <td>-0.145418</td>
      <td>-0.142035</td>
      <td>0.184254</td>
      <td>-0.175966</td>
      <td>0.066370</td>
      <td>0.163351</td>
      <td>0.258117</td>
      <td>0.379101</td>
      <td>0.175124</td>
      <td>-0.222780</td>
      <td>-0.063083</td>
      <td>-0.152692</td>
      <td>-0.138054</td>
      <td>0.150745</td>
      <td>1.000000</td>
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
# Let's use statsmodel's api instead for linear regression
# as it'll provide us a better summary of the fit
iq_ols = sm.OLS(endog=iq_y_train, exog=iq_X_train).fit()

iq_y_pred = iq_pwr.inverse_transform(iq_ols.predict(iq_X_test)[:, None])
iq_ols.summary()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.167</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.119</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.462</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 12 Apr 2019</td> <th>  Prob (F-statistic):</th> <td>2.25e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:18:56</td>     <th>  Log-Likelihood:    </th> <td> -461.29</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   348</td>      <th>  AIC:               </th> <td>   960.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   329</td>      <th>  BIC:               </th> <td>   1034.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    19</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ndvi_ne</th>                               <td>    0.4359</td> <td>    1.321</td> <td>    0.330</td> <td> 0.742</td> <td>   -2.163</td> <td>    3.035</td>
</tr>
<tr>
  <th>ndvi_nw</th>                               <td>    0.2015</td> <td>    1.119</td> <td>    0.180</td> <td> 0.857</td> <td>   -2.000</td> <td>    2.403</td>
</tr>
<tr>
  <th>ndvi_se</th>                               <td>   -0.7865</td> <td>    1.080</td> <td>   -0.728</td> <td> 0.467</td> <td>   -2.911</td> <td>    1.338</td>
</tr>
<tr>
  <th>ndvi_sw</th>                               <td>   -0.1954</td> <td>    1.153</td> <td>   -0.169</td> <td> 0.866</td> <td>   -2.463</td> <td>    2.073</td>
</tr>
<tr>
  <th>precipitation_amt_mm</th>                  <td>   -0.0011</td> <td>    0.001</td> <td>   -1.312</td> <td> 0.190</td> <td>   -0.003</td> <td>    0.001</td>
</tr>
<tr>
  <th>reanalysis_air_temp_k</th>                 <td>    0.5016</td> <td>    0.496</td> <td>    1.011</td> <td> 0.313</td> <td>   -0.475</td> <td>    1.478</td>
</tr>
<tr>
  <th>reanalysis_avg_temp_k</th>                 <td>   -0.2188</td> <td>    0.250</td> <td>   -0.877</td> <td> 0.381</td> <td>   -0.710</td> <td>    0.272</td>
</tr>
<tr>
  <th>reanalysis_dew_point_temp_k</th>           <td>   -0.3052</td> <td>    0.523</td> <td>   -0.583</td> <td> 0.560</td> <td>   -1.334</td> <td>    0.724</td>
</tr>
<tr>
  <th>reanalysis_max_air_temp_k</th>             <td>   -0.0397</td> <td>    0.050</td> <td>   -0.797</td> <td> 0.426</td> <td>   -0.138</td> <td>    0.058</td>
</tr>
<tr>
  <th>reanalysis_min_air_temp_k</th>             <td>    0.0196</td> <td>    0.070</td> <td>    0.281</td> <td> 0.779</td> <td>   -0.118</td> <td>    0.157</td>
</tr>
<tr>
  <th>reanalysis_precip_amt_kg_per_m2</th>       <td>   -0.0019</td> <td>    0.001</td> <td>   -1.362</td> <td> 0.174</td> <td>   -0.005</td> <td>    0.001</td>
</tr>
<tr>
  <th>reanalysis_relative_humidity_percent</th>  <td>    0.0521</td> <td>    0.108</td> <td>    0.484</td> <td> 0.629</td> <td>   -0.160</td> <td>    0.264</td>
</tr>
<tr>
  <th>reanalysis_sat_precip_amt_mm</th>          <td>   -0.0011</td> <td>    0.001</td> <td>   -1.312</td> <td> 0.190</td> <td>   -0.003</td> <td>    0.001</td>
</tr>
<tr>
  <th>reanalysis_specific_humidity_g_per_kg</th> <td>    0.3692</td> <td>    0.085</td> <td>    4.338</td> <td> 0.000</td> <td>    0.202</td> <td>    0.537</td>
</tr>
<tr>
  <th>reanalysis_tdtr_k</th>                     <td>    0.0731</td> <td>    0.078</td> <td>    0.937</td> <td> 0.350</td> <td>   -0.080</td> <td>    0.227</td>
</tr>
<tr>
  <th>station_avg_temp_c</th>                    <td>    0.0021</td> <td>    0.087</td> <td>    0.024</td> <td> 0.981</td> <td>   -0.170</td> <td>    0.174</td>
</tr>
<tr>
  <th>station_diur_temp_rng_c</th>               <td>   -0.0554</td> <td>    0.037</td> <td>   -1.483</td> <td> 0.139</td> <td>   -0.129</td> <td>    0.018</td>
</tr>
<tr>
  <th>station_max_temp_c</th>                    <td>    0.0387</td> <td>    0.062</td> <td>    0.623</td> <td> 0.534</td> <td>   -0.083</td> <td>    0.161</td>
</tr>
<tr>
  <th>station_min_temp_c</th>                    <td>    0.0255</td> <td>    0.062</td> <td>    0.410</td> <td> 0.682</td> <td>   -0.097</td> <td>    0.148</td>
</tr>
<tr>
  <th>station_precip_mm</th>                     <td>    0.0010</td> <td>    0.001</td> <td>    1.201</td> <td> 0.231</td> <td>   -0.001</td> <td>    0.003</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.702</td> <th>  Durbin-Watson:     </th> <td>   2.083</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.058</td> <th>  Jarque-Bera (JB):  </th> <td>   3.591</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.000</td> <th>  Prob(JB):          </th> <td>   0.166</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.502</td> <th>  Cond. No.          </th> <td>9.99e+16</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.64e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.
</div>


</div>
</div>
</div>



__From this, we can definitely conclude that there is a strong multicollinearity problem as an almost zero eigen value shows minimal variation in the data orthogonal with other eigen vectors. Hence, we will definitely need to remove some of those features later on in the Feature Engineering section.__



#### Conclusion:
- __Seems like these feature groups have very high correlation > 0.7:__
- `ndvi_sw` and `ndvi_se`
- `reanalysis_air_temp_k` and `reanalysis_avg_temp_k` and `reanalysis_dew_point_temp_k` and `reanalysis_min_air_temp_k` and `reanalysis_specific_humidity_g_per_kg` and `station_avg_temp_c` and `	station_max_temp_c` and `station_min_temp_c`



---
<a id="homo"></a>

## 4. Homoscedasticity (Errors have similar variance and no trend with predicted values)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Helper function to plot the residuals 
# of the Ordinary Least Squares Regression
# against each feature
def plot_residuals_feat(residuals, X):
    # Plot residuals
    rcParams['figure.figsize'] = 10, 40
    fig, ax = plt.subplots(len(X.columns) // 2,2)

    for idx, name in enumerate(X.columns):
        if idx < len(X.columns) // 2:
            ax[idx, 0].scatter(X[name], residuals)
            ax[idx, 0].grid(True)
            ax[idx, 0].set_title("Residuals against {}".format(name), color='k')
            ax[idx, 0].set_ylabel("Residuals")
            ax[idx, 0].set_xlabel("{}".format(name))
        else:
            ax[idx - (len(X.columns) // 2), 1].scatter(X[name], residuals)
            ax[idx - (len(X.columns) // 2), 1].grid(True)
            ax[idx - (len(X.columns) // 2), 1].set_title("Residuals against {}".format(name), color='k')
            ax[idx - (len(X.columns) // 2), 1].set_ylabel("Residuals")
            ax[idx - (len(X.columns) // 2), 1].set_xlabel("{}".format(name))

    plt.tight_layout()
    plt.show();
    
# Helper function to plot the residuals
# of the Ordinary Least Squares Regression
# against the label
def plot_residuals_label(residuals, y_pred):
    # Plot residuals
    rcParams['figure.figsize'] = 10, 5
    fig, ax = plt.subplots(1,2)

    ax[0].scatter(y_pred, residuals)
    ax[0].grid(True)
    ax[0].set_title("Residuals against {}".format('OLS_y_pred'), color='k')
    ax[0].set_ylabel("Residuals")
    ax[0].set_xlabel("{}".format('OLS_y_pred'))

    ax[1].hist(residuals)
    ax[1].grid(True)
    ax[1].set_title("Histogram of Residuals", color='k')
    ax[1].set_ylabel("Frequency of Residuals")
    ax[1].set_xlabel("{}".format('OLS_y_pred'))

    plt.tight_layout()
    plt.show();

```
</div>

</div>



- __City sj: Create Standardized residual plot against each one of the predictor variables__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Get the residuals for City sj's OLS prediction
sj_residuals = sj_y_test - sj_y_pred.flatten()

plot_residuals_feat(sj_residuals, sj_X_test)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_51_0.png)

</div>
</div>
</div>



- __Let's also see the Residuals against predicted values plot and its Histogram__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_residuals_label(sj_residuals, sj_y_pred)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_53_0.png)

</div>
</div>
</div>



__We'll now perform the Breusch Pagan test to check for heteroskedasticity__
- __Null Hypothesis: All observations have the same error variance, hence homoskedastic.__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Can't find diagnostic
# _, sj_residuals_bp_pvalue, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(sj_residuals)

```
</div>

</div>



- __City iq: Create Standardized residual plot against each one of the predictor variables__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Get the residuals for City iq's OLS prediction
iq_residuals = iq_y_test - iq_y_pred.flatten()

plot_residuals_feat(iq_residuals, iq_X_test)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_57_0.png)

</div>
</div>
</div>



- __Let's also see the Residuals against predicted values plot and its Histogram__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_residuals_label(iq_residuals, iq_y_pred)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_59_0.png)

</div>
</div>
</div>



__We'll now perform the Breusch Pagan test to check for heteroskedasticity__
- __Null Hypothesis: All observations have the same error variance, hence homoskedastic.__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Can't find diagnostic
# _, iq_residuals_bp_pvalue, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(iq_residuals)

```
</div>

</div>



#### Conclusion:
- 



---
<a id="normerr"></a>

## 5. Normality of errors



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Helper function to plot the distribution
# of residuals overlayed on the normal distribution
# to gauge if the errors are normally distributed
def plot_residuals_dist(residuals):
    rcParams['figure.figsize'] = 4, 4

    ax = sns.distplot(residuals)
    sns.kdeplot(np.random.normal(size=len(residuals)))
    ax.grid(True)
    ax.set_title("Distribution of {}".format('OLS Residuals'), color='k')

    plt.tight_layout()
    plt.show();

```
</div>

</div>



- __City sj Error distribution plot:__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_residuals_dist(sj_residuals)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_66_0.png)

</div>
</div>
</div>



__Let's use the Jarque-Bera test (unreliable though, since we have less than 2000 data samples) to check whether the residuals are normally distributed__
- __Null Hypothesis: Joint hypothesis of the skewness being zero and the excess kurtosis being zero (Normally Distributed).__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_, sj_residuals_pvalue = stats.jarque_bera(sj_residuals)
print(sj_residuals_pvalue)

_, norm_residuals_pvalue = stats.jarque_bera(np.random.normal(size=len(sj_residuals)))
print(norm_residuals_pvalue)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
0.0
0.8303925956946897
```
</div>
</div>
</div>



__Seems like our residuals are obviously not normally distributed, we'll have to feature engineer some new features in order to continue using linear regression__



- __City iq Error distribution plot:__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plot_residuals_dist(iq_residuals)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/portfolio/dengai/DengAI_Part_3_71_0.png)

</div>
</div>
</div>



__Let's use the Jarque-Bera test (unreliable though, since we have less than 2000 data samples) to check whether the residuals are normally distributed__
- __Null Hypothesis: Joint hypothesis of the skewness being zero and the excess kurtosis being zero (Normally Distributed).__



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_, iq_residuals_pvalue = stats.jarque_bera(iq_residuals)
print(iq_residuals_pvalue)

_, norm_residuals_pvalue = stats.jarque_bera(np.random.normal(size=len(iq_residuals)))
print(norm_residuals_pvalue)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
0.001110548015989865
0.7732898927241776
```
</div>
</div>
</div>



__Seems like our residuals are obviously not normally distributed, we'll have to feature engineer some new features in order to continue using linear regression__



---
<a id="autocorr"></a>

## 6. Autocorrelation

