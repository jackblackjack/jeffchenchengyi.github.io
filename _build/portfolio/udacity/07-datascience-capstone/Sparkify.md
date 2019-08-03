---
redirect_from:
  - "/portfolio/udacity/07-datascience-capstone/sparkify"
interact_link: content/portfolio/udacity/07-datascience-capstone/Sparkify.ipynb
kernel_name: python3
has_widgets: false
title: 'Predicting Churn Rates for "Sparkify"'
prev_page:
  url: /portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM
  title: 'Recommendation Systems for Articles with IBM'
next_page:
  url: /portfolio/dengai/dengai
  title: 'DengAI'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Sparkify Project Workspace
This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.

You can follow the steps below to guide your data analysis and model building portion of this project.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# import libraries
# General Utilities for Web Scraping
import re
import sys
import os
from os import listdir
from os.path import isfile, join, isdir
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import date, datetime
from dateutil import relativedelta
import csv
import json
import collections
from collections import defaultdict
from collections import Counter

# Machine Learning Utitilies
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, roc_auc_score, recall_score

# plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (18, 12)
get_colors = lambda length: plt.get_cmap('Spectral')(np.linspace(0, 1.0, length))

# Pyspark
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    udf, 
    desc, 
    asc, 
    sum as Fsum,
    lag,
    row_number,
    col,
    count,
    when
)
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import (
    LogisticRegression, 
    RandomForestClassifier, 
    GBTClassifier,
    MultilayerPerceptronClassifier
)
from pyspark.ml.feature import VectorAssembler, StandardScaler

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (18, 12)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# create a Spark session
spark = SparkSession.builder.appName("Sparkify").getOrCreate()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
spark.sparkContext.getConf().getAll()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('spark.app.id', 'local-1564758252671'),
 ('spark.rdd.compress', 'True'),
 ('spark.app.name', 'Sparkify'),
 ('spark.driver.port', '53372'),
 ('spark.serializer.objectStreamReset', '100'),
 ('spark.master', 'local[*]'),
 ('spark.executor.id', 'driver'),
 ('spark.submit.deployMode', 'client'),
 ('spark.driver.host', 'chengyis-mbp'),
 ('spark.ui.showConsoleProgress', 'true')]
```


</div>
</div>
</div>



# Preliminaries

What could cause users to **CHURN**? There can be multiple reasons why users would churn, including the following:
1. Ineffective Recommendation System of Sparkify
    - e.g. Users listen to mostly rock genres, but recommendation system recommends songs way too out of the scope of rock 
2. Bad UI / UX Design of Sparkify
    - e.g. Navigation through the application might be tedious, causing some users to churn and turn to substitutes
3. Users might be interested in listening to songs that are not available on Sparkify
    - e.g. If the main demographic that's churning are parents that only want to have Sesame  Street and Baby songs on their phones, they might end up churning because Sparkify does not support those songs
4. The Premium tier of Sparkify might be too expensive, while the free tier might support fewer services than alternatives

Depending on which of these problems (or combination of problems) caused the customer to churn, it would affect 
1. How we predict whether a customer will churn
    - We could formulate this as either a binary classification task like "**Will** a user churn or not?"
    - We could further ask "**When** a user will churn if they will churn at all
        - Furthermore, instead of predicting a single value as to when the customer will churn, we can predict a probability distribution such as [exponential](https://www.digitalmarketer.com/blog/customer-retention-rates/) to model how long it'll take for the customer to churn given past behavior
    - Also, we could further adapt this with a probablistic perspective - "**What** is the probability that a user will churn given his/her past activity"
2. Solution to reduce churn rates
    - If the problem was related to 1 or 2, we might use an A/B Test to see whether changing the UI / UX  or Recommendation System might decrease churn rates
    - If the problem was more of 3, we could possibly add support for those songs if the demographic is large enough or provide family plan discounts to entice them to continue with the service
    
Because of how complicated we can frame the problem, we will first build a simplified model - a soft classification as to whether a customer will churn or not. Subsequently, if we have the time, we will build a model that'll [predict a probability distribution](https://icml.cc/Conferences/2005/proceedings/papers/015_Predicting_CarneyEtAl.pdf) as to when the customer will churn - give probabilities as to whether a customer will churn as time passes



# Load and Clean Dataset
In this workspace, the mini-dataset file is `mini_user_log.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
path = "./data/mini_sparkify_event_data.json"
user_log = spark.read.json(path)
user_log.printSchema()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
root
 |-- artist: string (nullable = true)
 |-- auth: string (nullable = true)
 |-- firstName: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- itemInSession: long (nullable = true)
 |-- lastName: string (nullable = true)
 |-- length: double (nullable = true)
 |-- level: string (nullable = true)
 |-- location: string (nullable = true)
 |-- method: string (nullable = true)
 |-- page: string (nullable = true)
 |-- registration: long (nullable = true)
 |-- sessionId: long (nullable = true)
 |-- song: string (nullable = true)
 |-- status: long (nullable = true)
 |-- ts: long (nullable = true)
 |-- userAgent: string (nullable = true)
 |-- userId: string (nullable = true)

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log.describe()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
DataFrame[summary: string, artist: string, auth: string, firstName: string, gender: string, itemInSession: string, lastName: string, length: string, level: string, location: string, method: string, page: string, registration: string, sessionId: string, song: string, status: string, ts: string, userAgent: string, userId: string]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log.show(n=1)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+--------------+---------+---------+------+-------------+--------+---------+-----+---------------+------+--------+-------------+---------+---------+------+-------------+--------------------+------+
|        artist|     auth|firstName|gender|itemInSession|lastName|   length|level|       location|method|    page| registration|sessionId|     song|status|           ts|           userAgent|userId|
+--------------+---------+---------+------+-------------+--------+---------+-----+---------------+------+--------+-------------+---------+---------+------+-------------+--------------------+------+
|Martha Tilston|Logged In|    Colin|     M|           50| Freeman|277.89016| paid|Bakersfield, CA|   PUT|NextSong|1538173362000|       29|Rockpools|   200|1538352117000|Mozilla/5.0 (Wind...|    30|
+--------------+---------+---------+------+-------------+--------+---------+-----+---------------+------+--------+-------------+---------+---------+------+-------------+--------------------+------+
only showing top 1 row

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log.take(5)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[Row(artist='Martha Tilston', auth='Logged In', firstName='Colin', gender='M', itemInSession=50, lastName='Freeman', length=277.89016, level='paid', location='Bakersfield, CA', method='PUT', page='NextSong', registration=1538173362000, sessionId=29, song='Rockpools', status=200, ts=1538352117000, userAgent='Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', userId='30'),
 Row(artist='Five Iron Frenzy', auth='Logged In', firstName='Micah', gender='M', itemInSession=79, lastName='Long', length=236.09424, level='free', location='Boston-Cambridge-Newton, MA-NH', method='PUT', page='NextSong', registration=1538331630000, sessionId=8, song='Canada', status=200, ts=1538352180000, userAgent='"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"', userId='9'),
 Row(artist='Adam Lambert', auth='Logged In', firstName='Colin', gender='M', itemInSession=51, lastName='Freeman', length=282.8273, level='paid', location='Bakersfield, CA', method='PUT', page='NextSong', registration=1538173362000, sessionId=29, song='Time For Miracles', status=200, ts=1538352394000, userAgent='Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', userId='30'),
 Row(artist='Enigma', auth='Logged In', firstName='Micah', gender='M', itemInSession=80, lastName='Long', length=262.71302, level='free', location='Boston-Cambridge-Newton, MA-NH', method='PUT', page='NextSong', registration=1538331630000, sessionId=8, song='Knocking On Forbidden Doors', status=200, ts=1538352416000, userAgent='"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"', userId='9'),
 Row(artist='Daft Punk', auth='Logged In', firstName='Colin', gender='M', itemInSession=52, lastName='Freeman', length=223.60771, level='paid', location='Bakersfield, CA', method='PUT', page='NextSong', registration=1538173362000, sessionId=29, song='Harder Better Faster Stronger', status=200, ts=1538352676000, userAgent='Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', userId='30')]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# # If you need to save it into csv
# out_path = "./data/mini_sparkify_event_data.csv"
# user_log.write.save(out_path, format='csv', header=True)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log.describe("length").show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+-------+------------------+
|summary|            length|
+-------+------------------+
|  count|            228108|
|   mean|249.11718197783478|
| stddev| 99.23517921058324|
|    min|           0.78322|
|    max|        3024.66567|
+-------+------------------+

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log.select("page").dropDuplicates().sort("page").show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+--------------------+
|                page|
+--------------------+
|               About|
|          Add Friend|
|     Add to Playlist|
|              Cancel|
|Cancellation Conf...|
|           Downgrade|
|               Error|
|                Help|
|                Home|
|               Login|
|              Logout|
|            NextSong|
|            Register|
|         Roll Advert|
|       Save Settings|
|            Settings|
|    Submit Downgrade|
| Submit Registration|
|      Submit Upgrade|
|         Thumbs Down|
+--------------------+
only showing top 20 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log.describe("userId").show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+-------+------------------+
|summary|            userId|
+-------+------------------+
|  count|            286500|
|   mean| 59682.02278593872|
| stddev|109091.94999910604|
|    min|                  |
|    max|                99|
+-------+------------------+

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log.select(["userId", "firstName", "page", "song"]) \
    .where(user_log.userId == "99") \
    .collect()[:10]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[Row(userId='99', firstName='Madison', page='Home', song=None),
 Row(userId='99', firstName='Madison', page='NextSong', song='Welcome (Album Version)'),
 Row(userId='99', firstName='Madison', page='NextSong', song='Si No Existo (Live)'),
 Row(userId='99', firstName='Madison', page='NextSong', song='Moments with Oliver'),
 Row(userId='99', firstName='Madison', page='NextSong', song='Si Do Mhaimeo I'),
 Row(userId='99', firstName='Madison', page='Thumbs Down', song=None),
 Row(userId='99', firstName='Madison', page='NextSong', song='Coleccionista De Canciones'),
 Row(userId='99', firstName='Madison', page='Roll Advert', song=None),
 Row(userId='99', firstName='Madison', page='NextSong', song='Opposite Of Adults'),
 Row(userId='99', firstName='Madison', page='Thumbs Up', song=None)]
```


</div>
</div>
</div>



Let's find how many songs a user listens to in an hour.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Spark User-defined Function toget hours from timestamp
get_hour = udf(lambda x: datetime.fromtimestamp(x / 1000.0).hour)

# Convert ts column to hours
user_log = user_log.withColumn("hour", get_hour(user_log.ts))
user_log.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Row(artist='Martha Tilston', auth='Logged In', firstName='Colin', gender='M', itemInSession=50, lastName='Freeman', length=277.89016, level='paid', location='Bakersfield, CA', method='PUT', page='NextSong', registration=1538173362000, sessionId=29, song='Rockpools', status=200, ts=1538352117000, userAgent='Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', userId='30', hour='8')
```


</div>
</div>
</div>



See that Colin listened to Rockpools at 8am. We'll now group the songs by the hour to determine how many songs was listened to in each  hour by each user.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
songs_in_hour = user_log.filter(
    user_log.page == "NextSong"
).groupby(user_log.hour).count().orderBy(user_log.hour.cast("float"))
songs_in_hour.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+----+-----+
|hour|count|
+----+-----+
|   0|11636|
|   1|11460|
|   2|11124|
|   3|10930|
|   4|10856|
|   5|10458|
|   6|10122|
|   7| 9630|
|   8| 9062|
|   9| 8388|
|  10| 8336|
|  11| 8355|
|  12| 8147|
|  13| 8178|
|  14| 8054|
|  15| 8091|
|  16| 8253|
|  17| 8498|
|  18| 8882|
|  19| 8830|
+----+-----+
only showing top 20 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
songs_in_hour_pd = songs_in_hour.toPandas()

fig, ax = plt.subplots(1,1)
ax.scatter(songs_in_hour_pd["hour"], songs_in_hour_pd["count"])
ax.set_xlabel("Hour")
ax.set_ylabel("Songs Played")
ax.set_title("Songs Played Vs. Hour")
ax.grid()
plt.show();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../../images/portfolio/udacity/07-datascience-capstone/Sparkify_20_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log_valid = user_log.dropna(how='any', subset=["userId", "sessionId"])
user_log_valid.count()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
286500
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log_valid.select("userId").dropDuplicates().sort("userId").show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+
|userId|
+------+
|      |
|    10|
|   100|
|100001|
|100002|
|100003|
|100004|
|100005|
|100006|
|100007|
|100008|
|100009|
|100010|
|100011|
|100012|
|100013|
|100014|
|100015|
|100016|
|100017|
+------+
only showing top 20 rows

```
</div>
</div>
</div>



Weird, there is an empty string as the `userId`, let's remove all the `userId`s with an empty string and count again.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log_valid = user_log_valid.filter(user_log_valid["userId"] != "")
user_log_valid.count()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
278154
```


</div>
</div>
</div>



This way, we've only included `userId`s that have a valid value.



# Exploratory Data Analysis
When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.

### Define Churn

Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.

### Explore Data
Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.



We want to distinguish a user's activity before and after a particular event such as `Cancellation Confirmation` and `Submit Downgrade`.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
user_log_valid.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+--------------------+---------+---------+------+-------------+--------+---------+-----+--------------------+------+---------------+-------------+---------+--------------------+------+-------------+--------------------+------+----+
|              artist|     auth|firstName|gender|itemInSession|lastName|   length|level|            location|method|           page| registration|sessionId|                song|status|           ts|           userAgent|userId|hour|
+--------------------+---------+---------+------+-------------+--------+---------+-----+--------------------+------+---------------+-------------+---------+--------------------+------+-------------+--------------------+------+----+
|      Martha Tilston|Logged In|    Colin|     M|           50| Freeman|277.89016| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|           Rockpools|   200|1538352117000|Mozilla/5.0 (Wind...|    30|   8|
|    Five Iron Frenzy|Logged In|    Micah|     M|           79|    Long|236.09424| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8|              Canada|   200|1538352180000|"Mozilla/5.0 (Win...|     9|   8|
|        Adam Lambert|Logged In|    Colin|     M|           51| Freeman| 282.8273| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|   Time For Miracles|   200|1538352394000|Mozilla/5.0 (Wind...|    30|   8|
|              Enigma|Logged In|    Micah|     M|           80|    Long|262.71302| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8|Knocking On Forbi...|   200|1538352416000|"Mozilla/5.0 (Win...|     9|   8|
|           Daft Punk|Logged In|    Colin|     M|           52| Freeman|223.60771| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|Harder Better Fas...|   200|1538352676000|Mozilla/5.0 (Wind...|    30|   8|
|The All-American ...|Logged In|    Micah|     M|           81|    Long|208.29995| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8|      Don't Leave Me|   200|1538352678000|"Mozilla/5.0 (Win...|     9|   8|
|The Velvet Underg...|Logged In|    Micah|     M|           82|    Long|260.46649| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8|         Run Run Run|   200|1538352886000|"Mozilla/5.0 (Win...|     9|   8|
|        Starflyer 59|Logged In|    Colin|     M|           53| Freeman|185.44281| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|Passengers (Old A...|   200|1538352899000|Mozilla/5.0 (Wind...|    30|   8|
|                null|Logged In|    Colin|     M|           54| Freeman|     null| paid|     Bakersfield, CA|   PUT|Add to Playlist|1538173362000|       29|                null|   200|1538352905000|Mozilla/5.0 (Wind...|    30|   8|
|            Frumpies|Logged In|    Colin|     M|           55| Freeman|134.47791| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|          Fuck Kitty|   200|1538353084000|Mozilla/5.0 (Wind...|    30|   8|
|        Britt Nicole|Logged In|    Micah|     M|           83|    Long| 229.8771| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8|   Walk On The Water|   200|1538353146000|"Mozilla/5.0 (Win...|     9|   8|
|                null|Logged In|    Micah|     M|           84|    Long|     null| free|Boston-Cambridge-...|   GET|    Roll Advert|1538331630000|        8|                null|   200|1538353150000|"Mozilla/5.0 (Win...|     9|   8|
|Edward Sharpe & T...|Logged In|    Colin|     M|           56| Freeman|223.58159| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|                Jade|   200|1538353218000|Mozilla/5.0 (Wind...|    30|   8|
|               Tesla|Logged In|    Micah|     M|           85|    Long|201.06404| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8|      Gettin' Better|   200|1538353375000|"Mozilla/5.0 (Win...|     9|   8|
|                null|Logged In|    Micah|     M|           86|    Long|     null| free|Boston-Cambridge-...|   PUT|      Thumbs Up|1538331630000|        8|                null|   307|1538353376000|"Mozilla/5.0 (Win...|     9|   8|
|         Stan Mosley|Logged In|    Colin|     M|           57| Freeman|246.69995| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|   So-Called Friends|   200|1538353441000|Mozilla/5.0 (Wind...|    30|   8|
|Florence + The Ma...|Logged In|    Micah|     M|           87|    Long|168.64608| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8| You've Got The Love|   200|1538353576000|"Mozilla/5.0 (Win...|     9|   8|
|   Tokyo Police Club|Logged In|  Ashlynn|     F|            0|Williams| 166.1122| free|     Tallahassee, FL|   PUT|       NextSong|1537365219000|      217|Citizens Of Tomorrow|   200|1538353668000|"Mozilla/5.0 (Mac...|    74|   8|
|             Orishas|Logged In|    Colin|     M|           58| Freeman|222.22322| paid|     Bakersfield, CA|   PUT|       NextSong|1538173362000|       29|           Represent|   200|1538353687000|Mozilla/5.0 (Wind...|    30|   8|
|             Ratatat|Logged In|    Micah|     M|           88|    Long|229.77261| free|Boston-Cambridge-...|   PUT|       NextSong|1538331630000|        8|              Swisha|   200|1538353744000|"Mozilla/5.0 (Win...|     9|   8|
+--------------------+---------+---------+------+-------------+--------+---------+-----+--------------------+------+---------------+-------------+---------+--------------------+------+-------------+--------------------+------+----+
only showing top 20 rows

```
</div>
</div>
</div>



### Churn Rate Part 1: `Cancellation Confirmation`

We will utilize the `Cancellation Confirmation` event to define our churn rate.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Find all users that have confirmed to cancel the service
user_log_valid.filter("page = 'Cancellation Confirmation'").show(5)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------+---------+------+-------------+--------+------+-----+--------------------+------+--------------------+-------------+---------+----+------+-------------+--------------------+------+----+
|artist|     auth|firstName|gender|itemInSession|lastName|length|level|            location|method|                page| registration|sessionId|song|status|           ts|           userAgent|userId|hour|
+------+---------+---------+------+-------------+--------+------+-----+--------------------+------+--------------------+-------------+---------+----+------+-------------+--------------------+------+----+
|  null|Cancelled|   Adriel|     M|          104| Mendoza|  null| paid|  Kansas City, MO-KS|   GET|Cancellation Conf...|1535623466000|      514|null|   200|1538943990000|"Mozilla/5.0 (Mac...|    18|   4|
|  null|Cancelled|    Diego|     M|           56|   Mckee|  null| paid|Phoenix-Mesa-Scot...|   GET|Cancellation Conf...|1537167593000|      540|null|   200|1539033046000|"Mozilla/5.0 (iPh...|    32|   5|
|  null|Cancelled|    Mason|     M|           10|    Hart|  null| free|  Corpus Christi, TX|   GET|Cancellation Conf...|1533157139000|      174|null|   200|1539318918000|"Mozilla/5.0 (Mac...|   125|  12|
|  null|Cancelled|Alexander|     M|          332|  Garcia|  null| paid|Indianapolis-Carm...|   GET|Cancellation Conf...|1536817381000|      508|null|   200|1539375441000|Mozilla/5.0 (Wind...|   105|   4|
|  null|Cancelled|    Kayla|     F|          273| Johnson|  null| paid|Philadelphia-Camd...|   GET|Cancellation Conf...|1538333829000|      797|null|   200|1539465584000|Mozilla/5.0 (Wind...|    17|   5|
+------+---------+---------+------+-------------+--------+------+-----+--------------------+------+--------------------+-------------+---------+----+------+-------------+--------------------+------+----+
only showing top 5 rows

```
</div>
</div>
</div>



How many users have cancelled the service?



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Number of users that have cancelled the service
user_log_valid \
    .filter("page = 'Cancellation Confirmation'") \
    .select("userId") \
    .dropDuplicates() \
    .count()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
52
```


</div>
</div>
</div>



Let's take a look at Adriel's activity using Spark Dataframes to understand why he cancelled the service.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
adriel_log = user_log_valid.select([
        "userId", 
        "sessionId",
        "firstName", 
        "page", 
        "level", 
        "hour"]) \
    .where(user_log_valid.userId == "18") \
    .withColumn("row_num", 
                row_number().over(Window.partitionBy("userId").orderBy("userId"))) \
    .withColumn("prev_hour",
                lag(user_log_valid.hour.cast("float"), count=1).over(Window.partitionBy("userId").orderBy("row_num"))) \
    .fillna(1.0) \
    .withColumn("change_hour_row_num", (col("hour") != col("prev_hour")).cast("float") * col("row_num")) \
    .filter(col("change_hour_row_num") != 0.0) \
    .withColumn("change_hour_row_num_prev",
                lag(col("change_hour_row_num").cast("float"), count=1).over(Window.partitionBy("userId").orderBy("row_num"))) \
    .fillna(0.0) \
    .withColumn("total_songs_per_hour", 
                lag(col("change_hour_row_num") - col("change_hour_row_num_prev"), count=-1).over(Window.partitionBy("userId").orderBy("row_num"))) \
    .fillna(1.0) \
    .drop("prev_hour").drop("change_hour_row_num").drop("change_hour_row_num_prev")

adriel_log.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------+---------+---------------+-----+----+-------+--------------------+
|userId|sessionId|firstName|           page|level|hour|row_num|total_songs_per_hour|
+------+---------+---------+---------------+-----+----+-------+--------------------+
|    18|      242|   Adriel|       NextSong| paid|   2|     19|                19.0|
|    18|      242|   Adriel|       NextSong| paid|   3|     38|                16.0|
|    18|      242|   Adriel|       NextSong| paid|   4|     54|                20.0|
|    18|      242|   Adriel|       NextSong| paid|   5|     74|                18.0|
|    18|      360|   Adriel|       NextSong| paid|   6|     92|                17.0|
|    18|      360|   Adriel|       NextSong| paid|   7|    109|                13.0|
|    18|      363|   Adriel|       NextSong| paid|  14|    122|                 7.0|
|    18|      363|   Adriel|       NextSong| paid|  15|    129|                15.0|
|    18|      363|   Adriel|       NextSong| paid|  16|    144|                 8.0|
|    18|      384|   Adriel|       NextSong| paid|  21|    152|                15.0|
|    18|      384|   Adriel|       NextSong| paid|  22|    167|                15.0|
|    18|      384|   Adriel|       NextSong| paid|  23|    182|                18.0|
|    18|      384|   Adriel|Add to Playlist| paid|   0|    200|                12.0|
|    18|      384|   Adriel|       NextSong| paid|   1|    212|                20.0|
|    18|      409|   Adriel|       NextSong| paid|  13|    232|                 8.0|
|    18|      409|   Adriel|       NextSong| paid|  14|    240|                13.0|
|    18|      409|   Adriel|       NextSong| paid|  15|    253|                11.0|
|    18|      409|   Adriel|       NextSong| paid|  16|    264|                19.0|
|    18|      409|   Adriel|       NextSong| paid|  17|    283|                16.0|
|    18|      409|   Adriel|       NextSong| paid|  18|    299|                17.0|
+------+---------+---------+---------------+-----+----+-------+--------------------+
only showing top 20 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Create list of tuples
# (hour, total songs in that hour)
total_songs_each_hour = [
    (int(row["hour"]), int(row["total_songs_per_hour"])) \
    for row in adriel_log.select("hour", "total_songs_per_hour").collect()]

# Find First hour user started using Sparkify
# and prepend hours before that hour until 12 midnight
first_hour_on_sparkify = total_songs_each_hour[0][0]

# Find Last hour user started using Sparkify
# and append hours after that hour until 12 midnight
last_hour_on_sparkify = total_songs_each_hour[-1][0]

total_songs_each_hour = [(hour, 0) for hour in list(np.arange(24))[:first_hour_on_sparkify]] \
                        + total_songs_each_hour \
                        + [(hour, 0) for hour in list(np.arange(24))[last_hour_on_sparkify+1:]] 

# Insert tuples of (hour, 0)
# for all the times user was inactive
# on Sparkify
hour = 0
while True:
    if hour < len(total_songs_each_hour):
        if hour % 24 != total_songs_each_hour[hour][0]:
            total_songs_each_hour.insert(hour, (hour % 24, 0))
        hour += 1
    else:
        break
        
# Get the user activity from the first hour 
# they started Sparkify
hours, total_songs_per_hour = list(zip(*total_songs_each_hour))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Plot the user activity per day
n = 24 # Number of hours for each plot
plt.rcParams['figure.figsize'] = (18, 18)
fig, axs = plt.subplots(int(np.ceil(len(total_songs_per_hour) / n)), 1)

for idx, ax in enumerate(axs):
    if (idx+1)*n < len(total_songs_per_hour):
        ax.plot(np.arange(n), total_songs_per_hour[idx*n:(idx+1)*n])
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(hours[idx*n:(idx+1)*n])
    else:
        ax.plot(np.arange(len(total_songs_per_hour[idx*n:])), total_songs_per_hour[idx*n:])
        ax.set_xticks(np.arange(len(total_songs_per_hour[idx*n:])))
        ax.set_xticklabels(hours[idx*n:])
    ax.grid()
    ax.set_xlabel("Hours in Day {}".format(idx+1))
    ax.set_ylim(0, np.max(total_songs_per_hour))
    ax.set_ylabel("Total Songs Played")
    
plt.tight_layout()
plt.show();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../../images/portfolio/udacity/07-datascience-capstone/Sparkify_36_0.png)

</div>
</div>
</div>



Let's create a column `Churn` using Spark SQL (Both Spark SQL and Spark Data Frames use the Spark SQL Catalyst Optimizer to optimize queries) to use as the label for your model



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Create temporary SQL Table
user_log_valid.createOrReplaceTempView("log_table")

# Let's get all the users that have cancelled 
# the service and add a boolean column with 1 
# if they churned eventually and 0 if they did not
user_churn_df = spark.sql(
'''
SELECT distinct_users.userId, 
CASE 
    WHEN CAST(churned_users.userId AS INT) IS NULL THEN 0
    ELSE 1
END AS churned
FROM (
    SELECT DISTINCT CAST(log_table.userId AS INT)
    FROM log_table
    ORDER BY 1
) AS distinct_users
FULL OUTER JOIN (
    SELECT CAST(log_table.userId AS INT)
    FROM log_table
    WHERE page = 'Cancellation Confirmation'
    ORDER BY 1
) AS churned_users
ON distinct_users.userId = churned_users.userId
ORDER BY 1
'''
)

user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+
|userId|churned|
+------+-------+
|     2|      0|
|     3|      1|
|     4|      0|
|     5|      0|
|     6|      0|
|     7|      0|
|     8|      0|
|     9|      0|
|    10|      0|
|    11|      0|
+------+-------+
only showing top 10 rows

```
</div>
</div>
</div>



### Churn Rate Part 2: `Submit Downgrade`

We will utilize the `Submit Downgrade` event to define our churn rate.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Find all users that have downgraded their service from paid tier to free
user_log_valid.filter("page = 'Submit Downgrade'").show(5)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------+---------+------+-------------+--------+------+-----+--------------------+------+----------------+-------------+---------+----+------+-------------+--------------------+------+----+
|artist|     auth|firstName|gender|itemInSession|lastName|length|level|            location|method|            page| registration|sessionId|song|status|           ts|           userAgent|userId|hour|
+------+---------+---------+------+-------------+--------+------+-----+--------------------+------+----------------+-------------+---------+----+------+-------------+--------------------+------+----+
|  null|Logged In|     Kael|     M|           47|   Baker|  null| paid|Kingsport-Bristol...|   PUT|Submit Downgrade|1533102330000|      249|null|   307|1538393619000|"Mozilla/5.0 (Mac...|   131|  19|
|  null|Logged In|   Calvin|     M|           17|Marshall|  null| paid|      Pittsburgh, PA|   PUT|Submit Downgrade|1537120757000|      313|null|   307|1538516445000|"Mozilla/5.0 (Mac...|    38|   5|
|  null|Logged In|  Kaylenn|     F|          354| Jenkins|  null| paid|           Price, UT|   PUT|Submit Downgrade|1535903878000|      479|null|   307|1538835479000|"Mozilla/5.0 (Mac...|   141|  22|
|  null|Logged In|    Colin|     M|          214| Freeman|  null| paid|     Bakersfield, CA|   PUT|Submit Downgrade|1538173362000|      532|null|   307|1538995454000|Mozilla/5.0 (Wind...|    30|  18|
|  null|Logged In|  Destiny|     F|            2| Herring|  null| paid|Miami-Fort Lauder...|   PUT|Submit Downgrade|1536844410000|      378|null|   307|1538997356000|"Mozilla/5.0 (Win...|    20|  19|
+------+---------+---------+------+-------------+--------+------+-----+--------------------+------+----------------+-------------+---------+----+------+-------------+--------------------+------+----+
only showing top 5 rows

```
</div>
</div>
</div>



How many users have downgraded their plan?



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Number of users that have downgraded the service
user_log_valid \
    .filter("page = 'Submit Downgrade'") \
    .select("userId") \
    .dropDuplicates() \
    .count()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
49
```


</div>
</div>
</div>



Let's take a look at Kael's activity to understand why he downgraded his plan.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
kael_log = user_log_valid.select([
        "userId", 
        "sessionId",
        "firstName", 
        "page", 
        "level", 
        "hour"]) \
    .where(user_log_valid.userId == "131") \
    .withColumn("row_num", 
                row_number().over(Window.partitionBy().orderBy("userId"))) \
    .withColumn("prev_hour",
                lag(user_log_valid.hour.cast("float"), count=1).over(Window.partitionBy().orderBy("row_num"))) \
    .fillna(1.0) \
    .withColumn("change_hour_row_num", (col("hour") != col("prev_hour")).cast("float") * col("row_num")) \
    .filter(col("change_hour_row_num") != 0.0) \
    .withColumn("change_hour_row_num_prev",
                lag(col("change_hour_row_num").cast("float"), count=1).over(Window.partitionBy().orderBy("row_num"))) \
    .fillna(0.0) \
    .withColumn("total_songs_per_hour", 
                lag(col("change_hour_row_num") - col("change_hour_row_num_prev"), count=-1).over(Window.partitionBy().orderBy("row_num"))) \
    .fillna(1.0) \
    .drop("prev_hour").drop("change_hour_row_num").drop("change_hour_row_num_prev")

kael_log.show(20)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------+---------+--------+-----+----+-------+--------------------+
|userId|sessionId|firstName|    page|level|hour|row_num|total_songs_per_hour|
+------+---------+---------+--------+-----+----+-------+--------------------+
|   131|      249|     Kael|NextSong| free|  16|      1|                 1.0|
|   131|      249|     Kael|NextSong| free|  17|      2|                13.0|
|   131|      249|     Kael|NextSong| free|  18|     15|                23.0|
|   131|      249|     Kael|NextSong| paid|  19|     38|                17.0|
|   131|      249|     Kael|NextSong| paid|  20|     55|                17.0|
|   131|      249|     Kael|NextSong| paid|  21|     72|                18.0|
|   131|      249|     Kael|NextSong| paid|  22|     90|                16.0|
|   131|      249|     Kael|NextSong| paid|  23|    106|                18.0|
|   131|      249|     Kael|    Home| paid|   0|    124|                18.0|
|   131|      249|     Kael|NextSong| paid|   1|    142|                17.0|
|   131|      249|     Kael|NextSong| paid|   2|    159|                19.0|
|   131|      249|     Kael|NextSong| paid|   3|    178|                10.0|
|   131|      249|     Kael|NextSong| paid|   4|    188|                15.0|
|   131|      249|     Kael|NextSong| paid|   5|    203|                16.0|
|   131|      249|     Kael|NextSong| paid|   6|    219|                18.0|
|   131|      249|     Kael|NextSong| paid|   7|    237|                 2.0|
|   131|      320|     Kael|NextSong| paid|  16|    239|                 1.0|
|   131|      320|     Kael|NextSong| paid|  17|    240|                18.0|
|   131|      320|     Kael|NextSong| paid|  18|    258|                10.0|
|   131|      440|     Kael|NextSong| paid|  11|    268|                11.0|
+------+---------+---------+--------+-----+----+-------+--------------------+
only showing top 20 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Create list of tuples
# (hour, total songs in that hour)
total_songs_each_hour = [
    (int(row["hour"]), int(row["total_songs_per_hour"])) \
    for row in kael_log.select("hour", "total_songs_per_hour").collect()]

# Find First hour user started using Sparkify
# and prepend hours before that hour until 12 midnight
first_hour_on_sparkify = total_songs_each_hour[0][0]

# Find Last hour user started using Sparkify
# and append hours after that hour until 12 midnight
last_hour_on_sparkify = total_songs_each_hour[-1][0]

total_songs_each_hour = [(hour, 0) for hour in list(np.arange(24))[:first_hour_on_sparkify]] \
                        + total_songs_each_hour \
                        + [(hour, 0) for hour in list(np.arange(24))[last_hour_on_sparkify+1:]] 

# Insert tuples of (hour, 0)
# for all the times user was inactive
# on Sparkify
hour = 0
while True:
    if hour < len(total_songs_each_hour):
        if hour % 24 != total_songs_each_hour[hour][0]:
            total_songs_each_hour.insert(hour, (hour % 24, 0))
        hour += 1
    else:
        break
        
# Get the user activity from the first hour 
# they started Sparkify
hours, total_songs_per_hour = list(zip(*total_songs_each_hour))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Plot the user activity per day
n = 24 # Number of hours for each plot
plt.rcParams['figure.figsize'] = (18, 60)
fig, axs = plt.subplots(int(np.ceil(len(total_songs_per_hour) / n)), 1)

for idx, ax in enumerate(axs):
    if (idx+1)*n < len(total_songs_per_hour):
        ax.plot(np.arange(n), total_songs_per_hour[idx*n:(idx+1)*n])
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(hours[idx*n:(idx+1)*n])
    else:
        ax.plot(np.arange(len(total_songs_per_hour[idx*n:])), total_songs_per_hour[idx*n:])
        ax.set_xticks(np.arange(len(total_songs_per_hour[idx*n:])))
        ax.set_xticklabels(hours[idx*n:])
    ax.grid()
    ax.set_xlabel("Hours in Day {}".format(idx+1))
    ax.set_ylim(0, np.max(total_songs_per_hour))
    ax.set_ylabel("Total Songs Played")
    
plt.tight_layout()
plt.show();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../../images/portfolio/udacity/07-datascience-capstone/Sparkify_46_0.png)

</div>
</div>
</div>



As seen from these patterns, Kael started listening to Sparkify songs generally in the afternoon and transitioned to listening songs in the early mornings instead.



# Feature Engineering
Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
- Write a script to extract the necessary features from the smaller subset of data
- Ensure that your script is scalable, using the best practices discussed in Lesson 3
- Try your script on the full data set, debugging your script if necessary

If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Let's define the list of churn labels we have
# as our data and sequentially add features to it
user_churn_df.createOrReplaceTempView("data")

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Let's take a look at some of the 
# pages we have avaialble
spark.sql(
    '''
    SELECT DISTINCT log_table.page
    FROM log_table
    '''
).show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+--------------------+
|                page|
+--------------------+
|              Cancel|
|    Submit Downgrade|
|         Thumbs Down|
|                Home|
|           Downgrade|
|         Roll Advert|
|              Logout|
|       Save Settings|
|Cancellation Conf...|
|               About|
|            Settings|
|     Add to Playlist|
|          Add Friend|
|            NextSong|
|           Thumbs Up|
|                Help|
|             Upgrade|
|               Error|
|      Submit Upgrade|
+--------------------+

```
</div>
</div>
</div>



## 1. Measures of User Familiarity and Experience on Sparkify



**Feature 1a: Number of Hours Online**

A user that accesses the application frequently is less likely to churn than one who uses it infrequently



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
hours_online = spark.sql(
    '''
    WITH t1 AS (
        SELECT
            CAST(log_table.userId AS INT),
            DATE_TRUNC('HOUR', DATE(TIMESTAMP(log_table.ts))) AS hour
        FROM log_table
        ORDER BY 1, 2
    )
    SELECT 
        t1.userId,
        COUNT(t1.hour) AS hours_online
    FROM t1
    GROUP BY 1
    ORDER BY 1
    '''
)

hours_online.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+------------+
|userId|hours_online|
+------+------------+
|     2|         899|
|     3|         254|
|     4|        2442|
|     5|         218|
|     6|        3761|
|     7|         201|
|     8|         334|
|     9|        3191|
|    10|         795|
|    11|         848|
+------+------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    hours_online, 
    on=user_churn_df.userId == hours_online.userId, 
    how='left_outer') \
    .drop(hours_online.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+
|userId|churned|hours_online|
+------+-------+------------+
|     2|      0|         899|
|     3|      1|         254|
|     4|      0|        2442|
|     5|      0|         218|
|     6|      0|        3761|
|     7|      0|         201|
|     8|      0|         334|
|     9|      0|        3191|
|    10|      0|         795|
|    11|      0|         848|
+------+-------+------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 1b: Number of Days Online**

A user that accesses the application frequently is less likely to churn than one who uses it infrequently



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
days_online = spark.sql(
    '''
    WITH t1 AS (
        SELECT DISTINCT
            CAST(log_table.userId AS INT),
            DATE(TIMESTAMP(log_table.ts)) AS date
        FROM log_table
        ORDER BY 1, 2
    )
    SELECT 
        t1.userId,
        COUNT(*) AS days_online
    FROM t1
    GROUP BY 1
    ORDER BY 1
    '''
)

days_online.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-----------+
|userId|days_online|
+------+-----------+
|     2|        778|
|     3|        230|
|     4|       2138|
|     5|        169|
|     6|       3331|
|     7|        168|
|     8|        261|
|     9|       2740|
|    10|        693|
|    11|        665|
+------+-----------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    days_online, 
    on=user_churn_df.userId == days_online.userId, 
    how='left_outer') \
    .drop(days_online.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+
|userId|churned|hours_online|days_online|
+------+-------+------------+-----------+
|     2|      0|         899|        778|
|     3|      1|         254|        230|
|     4|      0|        2442|       2138|
|     5|      0|         218|        169|
|     6|      0|        3761|       3331|
|     7|      0|         201|        168|
|     8|      0|         334|        261|
|     9|      0|        3191|       2740|
|    10|      0|         795|        693|
|    11|      0|         848|        665|
+------+-------+------------+-----------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 1c: (Average Number Interactions / Hours Online) / Day Online**

A user that accesses the application frequently is less likely to churn than one who uses it infrequently



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
avg_num_interactions_per_hour_online_per_day_online = spark.sql(
    '''
    WITH t1 AS (
        SELECT
            CAST(log_table.userId AS INT),
            DATE_TRUNC('HOUR', DATE(TIMESTAMP(log_table.ts))) AS hour,
            COUNT(*) AS num_interactions
        FROM log_table
        GROUP BY 1, 2
        ORDER BY 1, 2
    )
    SELECT 
        t1.userId,
        AVG(t1.num_interactions) AS avg_num_interactions_per_hour_online_per_day_online
    FROM t1
    GROUP BY 1
    ORDER BY 1
    '''
)

avg_num_interactions_per_hour_online_per_day_online.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------------------------------------------------+
|userId|avg_num_interactions_per_hour_online_per_day_online|
+------+---------------------------------------------------+
|     2|                                 1.1555269922879177|
|     3|                                 1.1043478260869566|
|     4|                                 1.1421889616463985|
|     5|                                 1.2899408284023668|
|     6|                                  1.129090363254278|
|     7|                                 1.1964285714285714|
|     8|                                 1.2796934865900382|
|     9|                                 1.1645985401459853|
|    10|                                  1.147186147186147|
|    11|                                  1.275187969924812|
+------+---------------------------------------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    avg_num_interactions_per_hour_online_per_day_online, 
    on=user_churn_df.userId == avg_num_interactions_per_hour_online_per_day_online.userId, 
    how='left_outer') \
    .drop(avg_num_interactions_per_hour_online_per_day_online.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|
+------+-------+------------+-----------+---------------------------------------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|
|     3|      1|         254|        230|                                 1.1043478260869566|
|     4|      0|        2442|       2138|                                 1.1421889616463985|
|     5|      0|         218|        169|                                 1.2899408284023668|
|     6|      0|        3761|       3331|                                  1.129090363254278|
|     7|      0|         201|        168|                                 1.1964285714285714|
|     8|      0|         334|        261|                                 1.2796934865900382|
|     9|      0|        3191|       2740|                                 1.1645985401459853|
|    10|      0|         795|        693|                                  1.147186147186147|
|    11|      0|         848|        665|                                  1.275187969924812|
+------+-------+------------+-----------+---------------------------------------------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 1d: Total Days since Joining**

A long-time user might be less likely to churn than someone who just joined the application



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
days_since_joining = spark.sql(
    '''
    WITH t2 AS (
        SELECT 
            t1.userId,
            t1.date,
            ROW_NUMBER() OVER (
                PARTITION BY t1.userId
                ORDER BY t1.date ASC
            ) AS row_num_1,
            ROW_NUMBER() OVER (
                PARTITION BY t1.userId
                ORDER BY t1.date DESC
            ) AS row_num_2
        FROM (
            SELECT DISTINCT
                CAST(log_table.userId AS INT) AS userId,
                DATE(TIMESTAMP(log_table.ts)) AS date
            FROM log_table
            ORDER BY 1, 2
        ) AS t1
        ORDER BY 1, 2
    ),
    t3 AS (
        SELECT
            t2.userId,
            t2.date
        FROM t2
        WHERE t2.row_num_1 = 1
    ),
    t4 AS (
        SELECT
            t2.userId,
            t2.date
        FROM t2
        WHERE t2.row_num_2 = 1
    )
    SELECT
        t3.userId,
        DATEDIFF(t4.date, t3.date) AS days_since_joining
    FROM t3
    INNER JOIN t4
    ON t3.userId = t4.userId
    ORDER BY 1
    '''
)

days_since_joining.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+------------------+
|userId|days_since_joining|
+------+------------------+
|     2|             44736|
|     3|             27119|
|     4|             60635|
|     5|             35367|
|     6|             59317|
|     7|             50784|
|     8|             50920|
|     9|             60595|
|    10|             42437|
|    11|             53242|
+------+------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df \
    .join(
        days_since_joining, 
        on=user_churn_df.userId == days_since_joining.userId, 
        how='left_outer'
    ) \
    .drop(days_since_joining.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|
+------+-------+------------+-----------+---------------------------------------------------+------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|
|    10|      0|         795|        693|                                  1.147186147186147|             42437|
|    11|      0|         848|        665|                                  1.275187969924812|             53242|
+------+-------+------------+-----------+---------------------------------------------------+------------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 1e: Percentage of Days Online out of Total Days since Joining**

This will be a "normalized" version of Feature 1b, taking into account how much time the user has been on the application



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
percent_days_online = days_online \
    .join(
        days_since_joining, 
        on=days_online.userId == days_since_joining.userId,
        how='inner'
    ) \
    .drop(days_since_joining.userId) \
    .withColumn(
        'percent_days_online', 
        col('days_online') * 100 / col('days_since_joining')) \
    .drop('days_online').drop('days_since_joining')

percent_days_online.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------------------+
|userId|percent_days_online|
+------+-------------------+
|     2| 1.7390915593705294|
|     3| 0.8481138685054759|
|     4| 3.5260163272037603|
|     5| 0.4778465801453332|
|     6|  5.615590808705767|
|     7|0.33081285444234404|
|     8| 0.5125687352710133|
|     9|  4.521825233105042|
|    10| 1.6330089308857836|
|    11| 1.2490139363660269|
+------+-------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df \
    .join(
        percent_days_online, 
        on=user_churn_df.userId == percent_days_online.userId, 
        how='left_outer'
    ) \
    .drop(percent_days_online.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+
only showing top 10 rows

```
</div>
</div>
</div>



## 2. Measures of User Satisfaction with Songs supported on Sparkify



**Feature 2a: Average Number of Thumbs Down / Day Online**

A user that frequently thumbs down songs is more likely to churn.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
avg_thumbs_down_per_day = spark.sql(
    '''
    WITH t1 AS (
        SELECT 
            CAST(log_table.userId AS INT) AS userId, 
            DATE(TIMESTAMP(log_table.ts)) AS date,
            COUNT(*) AS count
        FROM log_table
        WHERE log_table.page = 'Thumbs Down'
        GROUP BY 1, 2
        ORDER BY 1, 2
    ),
    t2 AS (
        SELECT 
            log_table.userId,
            COUNT()
    )
    SELECT 
        t1.userId,
        AVG(t1.count) AS avg_thumbs_down_per_day
    FROM t1
    GROUP BY 1
    ORDER BY 1
    '''
)

avg_thumbs_down_per_day.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-----------------------+
|userId|avg_thumbs_down_per_day|
+------+-----------------------+
|     2|                    1.0|
|     3|                    1.0|
|     4|                    1.0|
|     6|                    1.0|
|     7|                    1.0|
|     8|                    1.0|
|     9|                    1.0|
|    10|                    1.0|
|    11|                    1.0|
|    12|                    1.0|
+------+-----------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    avg_thumbs_down_per_day, 
    on=user_churn_df.userId == avg_thumbs_down_per_day.userId, 
    how='left_outer') \
    .drop(avg_thumbs_down_per_day.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 2b: Total Number of Thumbs Down**

A user that frequently thumbs down songs is more likely to churn.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
total_thumbs_down = spark.sql(
    '''
    SELECT 
        CAST(log_table.userId AS INT) AS userId, 
        COUNT(*) AS total_thumbs_down
    FROM log_table
    WHERE log_table.page = 'Thumbs Down'
    GROUP BY CAST(log_table.userId AS INT)
    ORDER BY CAST(log_table.userId AS INT)
    '''
)

total_thumbs_down.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-----------------+
|userId|total_thumbs_down|
+------+-----------------+
|     2|                6|
|     3|                3|
|     4|               26|
|     6|               31|
|     7|                1|
|     8|                3|
|     9|               32|
|    10|                4|
|    11|                9|
|    12|                9|
+------+-----------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    total_thumbs_down, 
    on=user_churn_df.userId == total_thumbs_down.userId, 
    how='left_outer') \
    .drop(total_thumbs_down.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|total_thumbs_down|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|                6|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|                3|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|               26|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|                0|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|               31|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|                1|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|                3|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|               32|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|                4|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|                9|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 2c: Average Number of Thumbs Up / Day**

This is not exactly correlated with `Average Number of Thumbs Down / Day` as some users might prefer to give positive responses rather than negative.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
avg_thumbs_up_per_day = spark.sql(
    '''
    WITH t1 AS (
        SELECT 
            CAST(log_table.userId AS INT) AS userId, 
            DATE(TIMESTAMP(log_table.ts)) AS date,
            COUNT(*) AS count
        FROM log_table
        WHERE log_table.page = 'Thumbs Up'
        GROUP BY 1, 2
        ORDER BY 1, 2
    )
    SELECT 
        t1.userId,
        AVG(t1.count) AS avg_thumbs_up_per_day
    FROM t1
    GROUP BY 1
    ORDER BY 1
    '''
)

avg_thumbs_up_per_day.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------------------+
|userId|avg_thumbs_up_per_day|
+------+---------------------+
|     2|                  1.0|
|     3|                  1.0|
|     4|                  1.0|
|     5|                  1.0|
|     6|                  1.0|
|     7|                  1.0|
|     8|   1.0666666666666667|
|     9|                  1.0|
|    10|                  1.0|
|    11|                  1.0|
+------+---------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    avg_thumbs_up_per_day, 
    on=user_churn_df.userId == avg_thumbs_up_per_day.userId, 
    how='left_outer') \
    .drop(avg_thumbs_up_per_day.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|total_thumbs_down|avg_thumbs_up_per_day|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|                6|                  1.0|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|                3|                  1.0|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|               26|                  1.0|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|                0|                  1.0|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|               31|                  1.0|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|                1|                  1.0|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|                3|   1.0666666666666667|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|               32|                  1.0|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|                4|                  1.0|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|                9|                  1.0|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 2d: Total Number of Thumbs Up**

This is not exactly correlated with `Total Number of Thumbs Down` as some users might prefer to give positive responses rather than negative.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
total_thumbs_up = spark.sql(
    '''
    SELECT 
        CAST(log_table.userId AS INT) AS userId, 
        COUNT(*) AS total_thumbs_up
    FROM log_table
    WHERE log_table.page = 'Thumbs Up'
    GROUP BY CAST(log_table.userId AS INT)
    ORDER BY CAST(log_table.userId AS INT)
    '''
)

total_thumbs_up.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------------+
|userId|total_thumbs_up|
+------+---------------+
|     2|             29|
|     3|             14|
|     4|             95|
|     5|             11|
|     6|            165|
|     7|              7|
|     8|             16|
|     9|            118|
|    10|             37|
|    11|             40|
+------+---------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    total_thumbs_up, 
    on=user_churn_df.userId == total_thumbs_up.userId, 
    how='left_outer') \
    .drop(total_thumbs_up.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|total_thumbs_down|avg_thumbs_up_per_day|total_thumbs_up|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|                6|                  1.0|             29|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|                3|                  1.0|             14|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|               26|                  1.0|             95|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|                0|                  1.0|             11|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|               31|                  1.0|            165|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|                1|                  1.0|              7|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|                3|   1.0666666666666667|             16|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|               32|                  1.0|            118|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|                4|                  1.0|             37|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|                9|                  1.0|             40|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 2e: Total Number of Add to Playlist**

Users that curate their own playlists might be more likely to stay on Sparkify



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
total_add_to_playlist = spark.sql(
    '''
    SELECT 
        CAST(log_table.userId AS INT) AS userId, 
        COUNT(*) AS total_add_to_playlist
    FROM log_table
    WHERE log_table.page = 'Add to Playlist'
    GROUP BY CAST(log_table.userId AS INT)
    ORDER BY CAST(log_table.userId AS INT)
    '''
)

total_add_to_playlist.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+---------------------+
|userId|total_add_to_playlist|
+------+---------------------+
|     2|                   13|
|     3|                    4|
|     4|                   59|
|     5|                    8|
|     6|                   83|
|     7|                    5|
|     8|                    6|
|     9|                   77|
|    10|                    9|
|    11|                   20|
+------+---------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    total_add_to_playlist, 
    on=user_churn_df.userId == total_add_to_playlist.userId, 
    how='left_outer') \
    .drop(total_add_to_playlist.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|total_thumbs_down|avg_thumbs_up_per_day|total_thumbs_up|total_add_to_playlist|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|                6|                  1.0|             29|                   13|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|                3|                  1.0|             14|                    4|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|               26|                  1.0|             95|                   59|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|                0|                  1.0|             11|                    8|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|               31|                  1.0|            165|                   83|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|                1|                  1.0|              7|                    5|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|                3|   1.0666666666666667|             16|                    6|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|               32|                  1.0|            118|                   77|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|                4|                  1.0|             37|                    9|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|                9|                  1.0|             40|                   20|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 2f: Average Songs Played / Day Online**



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
avg_songs_per_day_online = spark.sql(
    '''
    WITH t1 AS (
        SELECT
            CAST(log_table.userId AS INT) AS userId,
            DATE_TRUNC('DD', TIMESTAMP(log_table.ts)) AS date,
            COUNT(*) AS next_songs_per_day_online
        FROM log_table
        WHERE log_table.page = 'NextSong'
        GROUP BY 1, 2
        ORDER BY 1, 2
    )
    SELECT
        t1.userId,
        AVG(t1.next_songs_per_day_online) AS avg_songs_per_day_online
    FROM t1
    GROUP BY 1
    ORDER BY 1
    '''
)

avg_songs_per_day_online.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+------------------------+
|userId|avg_songs_per_day_online|
+------+------------------------+
|     2|        1.00265604249668|
|     3|                     1.0|
|     4|      1.0014669926650366|
|     5|                     1.0|
|     6|       1.002538876547128|
|     7|      1.0067114093959733|
|     8|                   1.004|
|     9|      1.0045045045045045|
|    10|      1.0044776119402985|
|    11|      1.0046583850931676|
+------+------------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    avg_songs_per_day_online, 
    on=user_churn_df.userId == avg_songs_per_day_online.userId, 
    how='left_outer') \
    .drop(avg_songs_per_day_online.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|total_thumbs_down|avg_thumbs_up_per_day|total_thumbs_up|total_add_to_playlist|avg_songs_per_day_online|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|                6|                  1.0|             29|                   13|        1.00265604249668|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|                3|                  1.0|             14|                    4|                     1.0|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|               26|                  1.0|             95|                   59|      1.0014669926650366|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|                0|                  1.0|             11|                    8|                     1.0|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|               31|                  1.0|            165|                   83|       1.002538876547128|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|                1|                  1.0|              7|                    5|      1.0067114093959733|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|                3|   1.0666666666666667|             16|                    6|                   1.004|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|               32|                  1.0|            118|                   77|      1.0045045045045045|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|                4|                  1.0|             37|                    9|      1.0044776119402985|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|                9|                  1.0|             40|                   20|      1.0046583850931676|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+
only showing top 10 rows

```
</div>
</div>
</div>



**Feature 2g: Total Songs Played**



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
total_songs_played = spark.sql(
    '''
    SELECT
        CAST(log_table.userId AS INT) AS userId,
        COUNT(*) AS total_songs_played
    FROM log_table
    WHERE log_table.page = 'NextSong'
    GROUP BY 1
    ORDER BY 1
    '''
)

total_songs_played.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+------------------+
|userId|total_songs_played|
+------+------------------+
|     2|               755|
|     3|               214|
|     4|              2048|
|     5|               161|
|     6|              3159|
|     7|               150|
|     8|               251|
|     9|              2676|
|    10|               673|
|    11|               647|
+------+------------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    total_songs_played, 
    on=user_churn_df.userId == total_songs_played.userId, 
    how='left_outer') \
    .drop(total_songs_played.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+------------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|total_thumbs_down|avg_thumbs_up_per_day|total_thumbs_up|total_add_to_playlist|avg_songs_per_day_online|total_songs_played|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+------------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|                6|                  1.0|             29|                   13|        1.00265604249668|               755|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|                3|                  1.0|             14|                    4|                     1.0|               214|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|               26|                  1.0|             95|                   59|      1.0014669926650366|              2048|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|                0|                  1.0|             11|                    8|                     1.0|               161|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|               31|                  1.0|            165|                   83|       1.002538876547128|              3159|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|                1|                  1.0|              7|                    5|      1.0067114093959733|               150|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|                3|   1.0666666666666667|             16|                    6|                   1.004|               251|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|               32|                  1.0|            118|                   77|      1.0045045045045045|              2676|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|                4|                  1.0|             37|                    9|      1.0044776119402985|               673|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|                9|                  1.0|             40|                   20|      1.0046583850931676|               647|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+------------------+
only showing top 10 rows

```
</div>
</div>
</div>



## 3. Measure of User Satisfaction with Social Aspect of Application



**Feature 3: Total Number of Add Friend**

Users that interact more with their friends on the platform might be more likely to stay on the platform.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
total_add_friend = spark.sql(
    '''
    SELECT 
        CAST(log_table.userId AS INT) AS userId, 
        COUNT(*) AS total_add_friend
    FROM log_table
    WHERE log_table.page = 'Add Friend'
    GROUP BY CAST(log_table.userId AS INT)
    ORDER BY CAST(log_table.userId AS INT)
    '''
)

total_add_friend.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+----------------+
|userId|total_add_friend|
+------+----------------+
|     2|              20|
|     3|               1|
|     4|              46|
|     5|               3|
|     6|              41|
|     7|               1|
|     8|               5|
|     9|              40|
|    10|              12|
|    11|               6|
+------+----------------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Combine feature with full dataset
user_churn_df = user_churn_df.join(
    total_add_friend, 
    on=user_churn_df.userId == total_add_friend.userId, 
    how='left_outer') \
    .drop(total_add_friend.userId) \
    .orderBy(col('userId')) \
    .fillna(0.0)
    
user_churn_df.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+------------------+----------------+
|userId|churned|hours_online|days_online|avg_num_interactions_per_hour_online_per_day_online|days_since_joining|percent_days_online|avg_thumbs_down_per_day|total_thumbs_down|avg_thumbs_up_per_day|total_thumbs_up|total_add_to_playlist|avg_songs_per_day_online|total_songs_played|total_add_friend|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+------------------+----------------+
|     2|      0|         899|        778|                                 1.1555269922879177|             44736| 1.7390915593705294|                    1.0|                6|                  1.0|             29|                   13|        1.00265604249668|               755|              20|
|     3|      1|         254|        230|                                 1.1043478260869566|             27119| 0.8481138685054759|                    1.0|                3|                  1.0|             14|                    4|                     1.0|               214|               1|
|     4|      0|        2442|       2138|                                 1.1421889616463985|             60635| 3.5260163272037603|                    1.0|               26|                  1.0|             95|                   59|      1.0014669926650366|              2048|              46|
|     5|      0|         218|        169|                                 1.2899408284023668|             35367| 0.4778465801453332|                    0.0|                0|                  1.0|             11|                    8|                     1.0|               161|               3|
|     6|      0|        3761|       3331|                                  1.129090363254278|             59317|  5.615590808705767|                    1.0|               31|                  1.0|            165|                   83|       1.002538876547128|              3159|              41|
|     7|      0|         201|        168|                                 1.1964285714285714|             50784|0.33081285444234404|                    1.0|                1|                  1.0|              7|                    5|      1.0067114093959733|               150|               1|
|     8|      0|         334|        261|                                 1.2796934865900382|             50920| 0.5125687352710133|                    1.0|                3|   1.0666666666666667|             16|                    6|                   1.004|               251|               5|
|     9|      0|        3191|       2740|                                 1.1645985401459853|             60595|  4.521825233105042|                    1.0|               32|                  1.0|            118|                   77|      1.0045045045045045|              2676|              40|
|    10|      0|         795|        693|                                  1.147186147186147|             42437| 1.6330089308857836|                    1.0|                4|                  1.0|             37|                    9|      1.0044776119402985|               673|              12|
|    11|      0|         848|        665|                                  1.275187969924812|             53242| 1.2490139363660269|                    1.0|                9|                  1.0|             40|                   20|      1.0046583850931676|               647|               6|
+------+-------+------------+-----------+---------------------------------------------------+------------------+-------------------+-----------------------+-----------------+---------------------+---------------+---------------------+------------------------+------------------+----------------+
only showing top 10 rows

```
</div>
</div>
</div>



# Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.



### Train Test Split



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Use VectorAssembler to 
# combine features into a single
# vector
assembler = VectorAssembler(
    inputCols=[col for col in user_churn_df.columns if col not in ['userId', 'churned']], 
    outputCol='unscaled_features')

data = assembler \
    .transform(user_churn_df) \
    .select(
        col('userId'),
        col('churned').alias('label'), 
        col('unscaled_features')
    )

train, test = data.randomSplit([0.8, 0.2], seed=42)

```
</div>

</div>



### Scaling



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Scalers for the data
train_scaler = StandardScaler(
    inputCol='unscaled_features', 
    outputCol='features'
)

# Scale training data
train = train_scaler.fit(train).transform(train)

# Use the train_sclaer fitted on training data to
# scale never before seen test data
test = train_scaler.fit(test).transform(test)

```
</div>

</div>



### Logistic Regression



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Initialize Logistic Regression PySpark classifier
lr_clf = LogisticRegression()

lr_paramGrid = ParamGridBuilder() \
    .addGrid(lr_clf.regParam, [0.1, 0.01]) \
    .build()

lr_crossval = CrossValidator(
    estimator=lr_clf,
    estimatorParamMaps=lr_paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=5
)

# Run cross-validation, and choose the best set of parameters.
lr_cvModel = lr_crossval.fit(train)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Make predictions on train_scaler-scaled test data
lr_prediction = lr_cvModel.transform(test)
lr_prediction.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
|userId|label|   unscaled_features|            features|       rawPrediction|         probability|prediction|
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
|     3|    1|[254.0,230.0,1.10...|[0.17714067099705...|[478.847440188413...|[1.0,1.0944578804...|       0.0|
|     7|    0|[201.0,168.0,1.19...|[0.14017824752128...|[480.307626893803...|[1.0,2.5412537000...|       0.0|
|     9|    0|[3191.0,2740.0,1....|[2.22541685492752...|[480.275652242067...|[1.0,2.6238224205...|       0.0|
|    10|    0|[795.0,693.0,1.14...|[0.55443635213644...|[480.000418457584...|[1.0,3.4551503681...|       0.0|
|    12|    1|[1064.0,947.0,1.1...|[0.74203808638134...|[477.970396858848...|[1.0,2.6308381507...|       0.0|
|    23|    0|[782.0,689.0,1.13...|[0.54537009732163...|[478.241048269790...|[1.0,2.0070200764...|       0.0|
|    25|    0|[2279.0,1987.0,1....|[1.58938420945779...|[481.092412182016...|[1.0,1.1593647823...|       0.0|
|    32|    1|[108.0,87.0,1.241...|[0.07531965538457...|[475.734224559618...|[1.0,2.4617923695...|       0.0|
|    36|    0|[1399.0,1189.0,1....|[0.97566849891682...|[481.067478833824...|[1.0,1.1886350142...|       0.0|
|    38|    0|[1570.0,1389.0,1....|[1.09492461994240...|[481.086406166634...|[1.0,1.1663488974...|       0.0|
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
lr_cvModel.avgMetrics

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[0.8725602883706332, 0.8824869039006971]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('Accuracy: {}%'.format(
    lr_prediction.filter(lr_prediction.label == lr_prediction.prediction).count() * 100 / lr_prediction.count())
)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Accuracy: 73.91304347826087%
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Let's save the Logistic Regression model
lr_cvModel.save('./models/logistic_regression/')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Bring data back to a pandas df
lr_prediction_df = lr_prediction.select(['userId', 'label', 'probability', 'prediction']).toPandas()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
lr_user_churn_probabilities = lr_prediction_df[['userId', 'probability']]
lr_user_churn_probabilities.head()

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
      <th>userId</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>[1.0, 1.0944578804740895e-208]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[1.0, 2.54125370004055e-209]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>[1.0, 2.62382242052898e-209]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>[1.0, 3.455150368159044e-209]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>[1.0, 2.6308381507373944e-208]</td>
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
lr_y_test_pred = lr_prediction_df['prediction']
lr_y_test_true = lr_prediction_df['label']

print({
    'f1_score': f1_score(y_true=lr_y_test_true, y_pred=lr_y_test_pred),
    'precision': precision_score(y_true=lr_y_test_true, y_pred=lr_y_test_pred),
    'recall': recall_score(y_true=lr_y_test_true, y_pred=lr_y_test_pred)
})

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
{'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
```
</div>
</div>
</div>



### Random Forest Classifier



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Initialize Random Forest PySpark classifier
rf_clf = RandomForestClassifier()

rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf_clf.maxDepth, [2, 4, 6]) \
    .addGrid(rf_clf.maxBins, [20, 60]) \
    .addGrid(rf_clf.numTrees, [5, 20, 100]) \
    .build()

rf_crossval = CrossValidator(
    estimator=rf_clf,
    estimatorParamMaps=rf_paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=5
)

# Run cross-validation, and choose the best set of parameters.
rf_cvModel = rf_crossval.fit(train)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Make predictions on train_scaler-scaled test data
rf_prediction = rf_cvModel.transform(test)
rf_prediction.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
|userId|label|   unscaled_features|            features|       rawPrediction|         probability|prediction|
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
|     3|    1|[254.0,230.0,1.10...|[0.17714067099705...|[8.09691272262782...|[0.40484563613139...|       1.0|
|     7|    0|[201.0,168.0,1.19...|[0.14017824752128...|[16.9288710907704...|[0.84644355453852...|       0.0|
|     9|    0|[3191.0,2740.0,1....|[2.22541685492752...|          [13.0,7.0]|         [0.65,0.35]|       0.0|
|    10|    0|[795.0,693.0,1.14...|[0.55443635213644...|[5.57193708909955...|[0.27859685445497...|       1.0|
|    12|    1|[1064.0,947.0,1.1...|[0.74203808638134...|[1.24920634920634...|[0.06246031746031...|       1.0|
|    23|    0|[782.0,689.0,1.13...|[0.54537009732163...|[1.70973266499582...|[0.08548663324979...|       1.0|
|    25|    0|[2279.0,1987.0,1....|[1.58938420945779...|[14.9655172413793...|[0.74827586206896...|       0.0|
|    32|    1|[108.0,87.0,1.241...|[0.07531965538457...|[6.45790200138026...|[0.32289510006901...|       1.0|
|    36|    0|[1399.0,1189.0,1....|[0.97566849891682...|[13.9367315573770...|[0.69683657786885...|       0.0|
|    38|    0|[1570.0,1389.0,1....|[1.09492461994240...|[12.953125,7.046875]|[0.64765625,0.352...|       0.0|
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
rf_cvModel.avgMetrics

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[0.8398309573568195,
 0.8493680170404307,
 0.8695365094675439,
 0.860993669166083,
 0.8466845931845932,
 0.8351657231657232,
 0.8344425162959646,
 0.869359483428449,
 0.8758714944059771,
 0.8784522986591954,
 0.8629863929174273,
 0.8694869536593677,
 0.8337804724011622,
 0.892723092998955,
 0.8902449963829273,
 0.8426782642644711,
 0.8851320040113142,
 0.89159488787075]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('Accuracy: {}%'.format(
    rf_prediction.filter(rf_prediction.label == rf_prediction.prediction).count() * 100 / rf_prediction.count())
)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Accuracy: 80.43478260869566%
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Let's save the Random Forest model
rf_cvModel.save('./models/random_forest/')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Bring data back to a pandas df
rf_prediction_df = rf_prediction.select(['userId', 'label', 'probability', 'prediction']).toPandas()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
rf_user_churn_probabilities = rf_prediction_df[['userId', 'probability']]
rf_user_churn_probabilities.head()

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
      <th>userId</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>[0.4048456361313912, 0.5951543638686088]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[0.8464435545385202, 0.15355644546147978]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>[0.65, 0.35]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>[0.27859685445497795, 0.721403145545022]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>[0.06246031746031746, 0.9375396825396825]</td>
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
rf_y_test_pred = rf_prediction_df['prediction']
rf_y_test_true = rf_prediction_df['label']

print({
    'f1_score': f1_score(y_true=rf_y_test_true, y_pred=rf_y_test_pred),
    'precision': precision_score(y_true=rf_y_test_true, y_pred=rf_y_test_pred),
    'recall': recall_score(y_true=rf_y_test_true, y_pred=rf_y_test_pred)
})

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
{'f1_score': 0.7096774193548387, 'precision': 0.5789473684210527, 'recall': 0.9166666666666666}
```
</div>
</div>
</div>



### Gradient Boosted Tree Classifier



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Initialize Gradient Boosted Tree PySpark classifier
gbt_clf = GBTClassifier()

gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt_clf.maxDepth, [2, 4, 6]) \
    .addGrid(gbt_clf.maxBins, [20, 60]) \
    .build()

gbt_crossval = CrossValidator(
    estimator=gbt_clf,
    estimatorParamMaps=gbt_paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=5
)

# Run cross-validation, and choose the best set of parameters.
gbt_cvModel = gbt_crossval.fit(train)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Make predictions on train_scaler-scaled test data
gbt_prediction = gbt_cvModel.transform(test)
gbt_prediction.show(10)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
|userId|label|   unscaled_features|            features|       rawPrediction|         probability|prediction|
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
|     3|    1|[254.0,230.0,1.10...|[0.17714067099705...|[-0.4606277341328...|[0.28470215306444...|       1.0|
|     7|    0|[201.0,168.0,1.19...|[0.14017824752128...|[0.51170923857543...|[0.73563794229148...|       0.0|
|     9|    0|[3191.0,2740.0,1....|[2.22541685492752...|[0.81863931734718...|[0.83716430056844...|       0.0|
|    10|    0|[795.0,693.0,1.14...|[0.55443635213644...|[-0.0758411061857...|[0.46215198476032...|       1.0|
|    12|    1|[1064.0,947.0,1.1...|[0.74203808638134...|[-2.0808241684468...|[0.01534278360520...|       1.0|
|    23|    0|[782.0,689.0,1.13...|[0.54537009732163...|[-0.8987513393152...|[0.14215533416753...|       1.0|
|    25|    0|[2279.0,1987.0,1....|[1.58938420945779...|[0.81139611254594...|[0.83517985187636...|       0.0|
|    32|    1|[108.0,87.0,1.241...|[0.07531965538457...|[-1.1250684803252...|[0.09533765160964...|       1.0|
|    36|    0|[1399.0,1189.0,1....|[0.97566849891682...|[0.67954455696183...|[0.79561161497843...|       0.0|
|    38|    0|[1570.0,1389.0,1....|[1.09492461994240...|[0.81863931734718...|[0.83716430056844...|       0.0|
+------+-----+--------------------+--------------------+--------------------+--------------------+----------+
only showing top 10 rows

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
gbt_cvModel.avgMetrics

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[0.9137161535782226,
 0.898313640382606,
 0.8449436578746924,
 0.8721826315447004,
 0.7838053785295165,
 0.8295882527434253]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('Accuracy: {}%'.format(
    gbt_prediction.filter(gbt_prediction.label == gbt_prediction.prediction).count() * 100 / gbt_prediction.count())
)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Accuracy: 76.08695652173913%
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Let's save the Random Forest model
gbt_cvModel.save('./models/gradient_boosted_tree/')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Bring data back to a pandas df
gbt_prediction_df = gbt_prediction.select(['userId', 'label', 'probability', 'prediction']).toPandas()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
gbt_user_churn_probabilities = gbt_prediction_df[['userId', 'probability']]
gbt_user_churn_probabilities.head()

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
      <th>userId</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>[0.28470215306444185, 0.7152978469355582]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[0.7356379422914863, 0.26436205770851373]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>[0.8371643005684475, 0.16283569943155252]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>[0.46215198476032854, 0.5378480152396714]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>[0.015342783605201797, 0.9846572163947982]</td>
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
gbt_y_test_pred = gbt_prediction_df['prediction']
gbt_y_test_true = gbt_prediction_df['label']

print({
    'f1_score': f1_score(y_true=gbt_y_test_true, y_pred=gbt_y_test_pred),
    'precision': precision_score(y_true=gbt_y_test_true, y_pred=gbt_y_test_pred),
    'recall': recall_score(y_true=gbt_y_test_true, y_pred=gbt_y_test_pred)
})

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
{'f1_score': 0.6666666666666667, 'precision': 0.5238095238095238, 'recall': 0.9166666666666666}
```
</div>
</div>
</div>



### Multi-layer Perceptron Classifier



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# MultilayerPerceptronClassifier()

```
</div>

</div>



# Evaluation



# Final Steps
Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.

