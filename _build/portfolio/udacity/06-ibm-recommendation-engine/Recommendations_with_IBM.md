---
redirect_from:
  - "/portfolio/udacity/06-ibm-recommendation-engine/recommendations-with-ibm"
interact_link: content/portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM.ipynb
kernel_name: python3
has_widgets: false
title: 'Recommendation Systems for Articles with IBM'
prev_page:
  url: /portfolio/udacity/05-disaster-response/workspace/ML_Pipeline_Preparation
  title: 'Disaster Response'
next_page:
  url: /portfolio/udacity/07-datascience-capstone/Sparkify
  title: 'Predicting Churn Rates for "Sparkify"'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Recommendations with IBM

In this notebook, you will be putting your recommendation skills to use on real data from the IBM Watson Studio platform. 


You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/rubrics/2322/view).  **Please save regularly.**

By following the table of contents, you will build out a number of different methods for making recommendations that can be used for different situations. 


## Table of Contents

I. [Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
II. [Rank Based Recommendations](#Rank)<br>
III. [User-User Based Collaborative Filtering](#User-User)<br>
IV. [Content Based Recommendations (EXTRA - NOT REQUIRED)](#Content-Recs)<br>
V. [Matrix Factorization](#Matrix-Fact)<br>
VI. [Extras & Concluding](#conclusions)

At the end of the notebook, you will find directions for how to submit your work.  Let's get started by importing the necessary libraries and reading in the data.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from tqdm import tqdm
from imp import reload
import project_tests as t
reload(t);

df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

%matplotlib inline

# Show df to get an idea of the data
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
      <th>article_id</th>
      <th>title</th>
      <th>email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1430.0</td>
      <td>using pixiedust for fast, flexible, and easier...</td>
      <td>ef5f11f77ba020cd36e1105a00ab868bbdbf7fe7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1314.0</td>
      <td>healthcare python streaming application demo</td>
      <td>083cbdfa93c8444beaa4c5f5e0f5f9198e4f9e0b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1429.0</td>
      <td>use deep learning for image classification</td>
      <td>b96a4f2e92d8572034b1e9b28f9ac673765cd074</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1338.0</td>
      <td>ml optimization using cognitive assistant</td>
      <td>06485706b34a5c9bf2a0ecdac41daf7e7654ceb7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1276.0</td>
      <td>deploy your python model as a restful api</td>
      <td>f01220c46fc92c6e6b161b1849de11faacd7ccb2</td>
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
# plotting defaults
get_colors = lambda length: plt.get_cmap('Spectral')(np.linspace(0, 1.0, length))
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 10)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Show df_content to get an idea of the data
df_content.head()

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
      <th>doc_body</th>
      <th>doc_description</th>
      <th>doc_full_name</th>
      <th>doc_status</th>
      <th>article_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Skip navigation Sign in SearchLoading...\r\n\r...</td>
      <td>Detect bad readings in real time using Python ...</td>
      <td>Detect Malfunctioning IoT Sensors with Streami...</td>
      <td>Live</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No Free Hunch Navigation * kaggle.com\r\n\r\n ...</td>
      <td>See the forest, see the trees. Here lies the c...</td>
      <td>Communicating data science: A guide to present...</td>
      <td>Live</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>☰ * Login\r\n * Sign Up\r\n\r\n * Learning Pat...</td>
      <td>Here’s this week’s news in Data Science and Bi...</td>
      <td>This Week in Data Science (April 18, 2017)</td>
      <td>Live</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DATALAYER: HIGH THROUGHPUT, LOW LATENCY AT SCA...</td>
      <td>Learn how distributed DBs solve the problem of...</td>
      <td>DataLayer Conference: Boost the performance of...</td>
      <td>Live</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Skip navigation Sign in SearchLoading...\r\n\r...</td>
      <td>This video demonstrates the power of IBM DataS...</td>
      <td>Analyze NY Restaurant data using Spark in DSX</td>
      <td>Live</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



### <a class="anchor" id="Exploratory-Data-Analysis">Part I : Exploratory Data Analysis</a>

Use the dictionary and cells below to provide some insight into the descriptive statistics of the data.

`1.` What is the distribution of how many articles a user interacts with in the dataset?  Provide a visual and descriptive statistics to assist with giving a look at the number of times each user interacts with an article.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Get number of times each user interacts with an article
num_articles_per_user = df.groupby(by=['email']).count()['title'].values

# Plot a visual of the number of times each user interacts with an article
ax = plt.subplot()
ax.hist(num_articles_per_user)
plt.title('Histogram of Number of Articles Interacted per User')
plt.xlabel('Number of times each user interacts with an article')
plt.ylabel('Frequency')
for p in ax.patches:
    ax.annotate(str(round(p.get_height() * 100 / len(num_articles_per_user), 2)) + '%', 
                (p.get_x()+20, p.get_height()+10), 
                va='bottom', 
                ha='center',
                fontsize=10)
plt.grid();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../../images/portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM_5_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Fill in the median and maximum number of user_article interactios below

median_val = np.median(num_articles_per_user) # 50% of individuals interact with ____ number of articles or fewer.
max_views_by_user = num_articles_per_user.max() # The maximum number of user-article interactions by any 1 user is ______.

```
</div>

</div>



`2.` Explore and remove duplicate articles from the **df_content** dataframe.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Find and explore duplicate articles
df_content[df_content.duplicated(subset=['article_id'], keep=False)]

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
      <th>doc_body</th>
      <th>doc_description</th>
      <th>doc_full_name</th>
      <th>doc_status</th>
      <th>article_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>Follow Sign in / Sign up Home About Insight Da...</td>
      <td>Community Detection at Scale</td>
      <td>Graph-based machine learning</td>
      <td>Live</td>
      <td>50</td>
    </tr>
    <tr>
      <th>221</th>
      <td>* United States\r\n\r\nIBM® * Site map\r\n\r\n...</td>
      <td>When used to make sense of huge amounts of con...</td>
      <td>How smart catalogs can turn the big data flood...</td>
      <td>Live</td>
      <td>221</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Homepage Follow Sign in Get started Homepage *...</td>
      <td>If you are like most data scientists, you are ...</td>
      <td>Self-service data preparation with IBM Data Re...</td>
      <td>Live</td>
      <td>232</td>
    </tr>
    <tr>
      <th>365</th>
      <td>Follow Sign in / Sign up Home About Insight Da...</td>
      <td>During the seven-week Insight Data Engineering...</td>
      <td>Graph-based machine learning</td>
      <td>Live</td>
      <td>50</td>
    </tr>
    <tr>
      <th>399</th>
      <td>Homepage Follow Sign in Get started * Home\r\n...</td>
      <td>Today’s world of data science leverages data f...</td>
      <td>Using Apache Spark as a parallel processing fr...</td>
      <td>Live</td>
      <td>398</td>
    </tr>
    <tr>
      <th>578</th>
      <td>This video shows you how to construct queries ...</td>
      <td>This video shows you how to construct queries ...</td>
      <td>Use the Primary Index</td>
      <td>Live</td>
      <td>577</td>
    </tr>
    <tr>
      <th>692</th>
      <td>Homepage Follow Sign in / Sign up Homepage * H...</td>
      <td>One of the earliest documented catalogs was co...</td>
      <td>How smart catalogs can turn the big data flood...</td>
      <td>Live</td>
      <td>221</td>
    </tr>
    <tr>
      <th>761</th>
      <td>Homepage Follow Sign in Get started Homepage *...</td>
      <td>Today’s world of data science leverages data f...</td>
      <td>Using Apache Spark as a parallel processing fr...</td>
      <td>Live</td>
      <td>398</td>
    </tr>
    <tr>
      <th>970</th>
      <td>This video shows you how to construct queries ...</td>
      <td>This video shows you how to construct queries ...</td>
      <td>Use the Primary Index</td>
      <td>Live</td>
      <td>577</td>
    </tr>
    <tr>
      <th>971</th>
      <td>Homepage Follow Sign in Get started * Home\r\n...</td>
      <td>If you are like most data scientists, you are ...</td>
      <td>Self-service data preparation with IBM Data Re...</td>
      <td>Live</td>
      <td>232</td>
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
df_content.shape

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
(1056, 5)
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Remove any rows that have the same article_id - only keep the first
df_content.drop_duplicates(subset=['article_id'], keep='first', inplace=True)
df_content.shape

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
(1051, 5)
```


</div>
</div>
</div>



`3.` Use the cells below to find:

**a.** The number of unique articles that have an interaction with a user.  
**b.** The number of unique articles in the dataset (whether they have any interactions or not).<br>
**c.** The number of unique users in the dataset. (excluding null values) <br>
**d.** The number of user-article interactions in the dataset.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
unique_articles = len(set(df['article_id'].astype(int))) # The number of unique articles that have at least one interaction
total_articles = df_content['article_id'].shape[0] # The number of unique articles on the IBM platform
unique_users = len(set(df['email'].dropna())) # The number of unique users
user_article_interactions = df.shape[0] # The number of user-article interactions

```
</div>

</div>



`4.` Use the cells below to find the most viewed **article_id**, as well as how often it was viewed.  After talking to the company leaders, the `email_mapper` function was deemed a reasonable way to map users to ids.  There were a small number of null values, and it was found that all of these null values likely belonged to a single user (which is how they are stored using the function below).



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
most_viewed_article_id = str(list(df['article_id'].value_counts().items())[0][0]) # The most viewed article in the dataset as a string with one value following the decimal 
max_views = int(list(df['article_id'].value_counts().items())[0][1]) # The most viewed article in the dataset was viewed how many times?

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper(df=df):
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
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
      <th>article_id</th>
      <th>title</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1430.0</td>
      <td>using pixiedust for fast, flexible, and easier...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1314.0</td>
      <td>healthcare python streaming application demo</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1429.0</td>
      <td>use deep learning for image classification</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1338.0</td>
      <td>ml optimization using cognitive assistant</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1276.0</td>
      <td>deploy your python model as a restful api</td>
      <td>5</td>
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
## If you stored all your results in the variable names above, 
## you shouldn't need to change anything in this cell

sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_val,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': max_views_by_user,
    '`The most viewed article in the dataset was viewed _____ times.`': max_views,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}

# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
It looks like you have everything right here! Nice job!
```
</div>
</div>
</div>



### <a class="anchor" id="Rank">Part II: Rank-Based Recommendations</a>

Unlike in the earlier lessons, we don't actually have ratings for whether a user liked an article or not.  We only know that a user has interacted with an article.  In these cases, the popularity of an article can really only be based on how often an article was interacted with.

`1.` Fill in the function below to return the **n** top articles ordered with most interactions as the top. Test your function using the tests below.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Your code here
    titles, counts = zip(*df['title'].value_counts().items()) # unzips
    top_articles = titles[:n]
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Your code here
    ids, counts = zip(*df['article_id'].value_counts().items()) # unzips
    top_articles = [str(id) for id in ids[:n]]
 
    return top_articles # Return the top article ids

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print(get_top_articles(10))
print(get_top_article_ids(10))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
('use deep learning for image classification', 'insights from new york car accident reports', 'visualize car data with brunel', 'use xgboost, scikit-learn & ibm watson machine learning apis', 'predicting churn with the spss random tree algorithm', 'healthcare python streaming application demo', 'finding optimal locations of new store using decision optimization', 'apache spark lab, part 1: basic concepts', 'analyze energy consumption in buildings', 'gosales transactions for logistic regression model')
['1429.0', '1330.0', '1431.0', '1427.0', '1364.0', '1314.0', '1293.0', '1170.0', '1162.0', '1304.0']
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

# Test each of your three lists from above
t.sol_2_test(get_top_articles)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Your top_5 looks like the solution list! Nice job.
Your top_10 looks like the solution list! Nice job.
Your top_20 looks like the solution list! Nice job.
```
</div>
</div>
</div>



### <a class="anchor" id="User-User">Part III: User-User Based Collaborative Filtering</a>


`1.` Use the function below to reformat the **df** dataframe to be shaped with users as the rows and articles as the columns.  

* Each **user** should only appear in each **row** once.


* Each **article** should only show up in one **column**.  


* **If a user has interacted with an article, then place a 1 where the user-row meets for that article-column**.  It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.  


* **If a user has not interacted with an item, then place a zero where the user-row meets for that article-column**. 

Use the tests to make sure the basic structure of your matrix matches what is expected by the solution.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Fill in the function here
    df_unique = df[['article_id', 'user_id']].drop_duplicates()
    user_item = pd.concat([
        df_unique, 
        pd.Series(
            data=np.ones(df_unique.shape[0]), 
            name='interacted', 
            index=df_unique.index)
    ], axis=1).pivot_table(
        index='user_id', 
        columns='article_id', 
        values='interacted').fillna(0.0)
    
    return user_item # return the user_item matrix 

user_item = create_user_item_matrix(df)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
## Tests: You should just need to run this cell.  Don't change the code.
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
print("You have passed our quick tests!  Please proceed!")

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
You have passed our quick tests!  Please proceed!
```
</div>
</div>
</div>



`2.` Complete the function below which should take a user_id and provide an ordered list of the most similar users to that user (from most similar to least similar).  The returned result should not contain the provided user_id, as we know that each user is similar to him/herself. Because the results for each user here are binary, it (perhaps) makes sense to compute similarity as the dot product of two users. 

Use the tests to test your function.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    # Tuple that contains the dot product similarity and user_id
    Similarity = collections.namedtuple(typename='Similarity',
                                        field_names=('neighbor_id', 'val'))

    # compute similarity of each user to the provided user
    # remove the own user's id
    neighbor_similarity = [Similarity(neighbor_id=user_row[0],
                                      val=np.dot(user_row[1], user_item.loc[user_id])) for user_row in user_item.drop(labels=user_id, axis=0).iterrows()]
    
    # sort by similarity
    # create list of just the ids
    most_similar_users = [sim.neighbor_id for sim in sorted(neighbor_similarity, key=lambda sim: sim.val, reverse=True)]
    
    return most_similar_users # return a list of the users in order from most to least similar
        

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Do a spot check of your function
print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))
print("The 5 most similar users to user 3933 are: {}".format(find_similar_users(3933)[:5]))
print("The 3 most similar users to user 46 are: {}".format(find_similar_users(46)[:3]))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
The 10 most similar users to user 1 are: [3933, 23, 3782, 203, 4459, 131, 3870, 46, 4201, 49]
The 5 most similar users to user 3933 are: [1, 23, 3782, 203, 4459]
The 3 most similar users to user 46 are: [4201, 23, 3782]
```
</div>
</div>
</div>



`3.` Now that you have a function that provides the most similar users to each user, you will want to use these users to find articles you can recommend.  Complete the functions below to return the articles you would recommend to each user. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    # Your code here
    article_names = list(df[df['article_id'].astype(str).isin(article_ids)]['title'].drop_duplicates())
    
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # Your code here
    article_ids = [str(user_item.columns[idx]) for idx, val in enumerate(user_item.loc[user_id]) if val]
    article_names = get_article_names(article_ids)
    
    return article_ids, article_names # return the ids and names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    # Your code here
    most_similar_users = find_similar_users(user_id)
    recs = []
    for sim_user_id in most_similar_users:
        sim_user = np.array(user_item.loc[sim_user_id].astype(int))
        curr_user = np.array(user_item.loc[user_id].astype(int))
        xor_mask = sim_user ^ curr_user
        for article_id in [user_item.columns[idx] for idx, val in enumerate(sim_user & xor_mask) if val]:
            if len(recs) == m:
                return recs
            else:
                recs.append(str(article_id))
    
    return recs # return your recommendations for this user_id    

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Check Results
get_article_names(user_user_recs(1, 10)) # Return 10 recommendations for user 1

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['got zip code data? prep it for analytics. – ibm watson data lab – medium',
 'timeseries data analysis of iot events by using jupyter notebook',
 'graph-based machine learning',
 'using brunel in ipython/jupyter notebooks',
 'experience iot with coursera',
 'the 3 kinds of context: machine learning and the art of the frame',
 'deep forest: towards an alternative to deep neural networks',
 'this week in data science (april 18, 2017)',
 'higher-order logistic regression for large datasets',
 'using machine learning to predict parking difficulty']
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Test your functions here - No need to change this code - just run this cell
assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_article_names(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures','self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
print("If this is all you see, you passed all of our tests!  Nice job!")

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
If this is all you see, you passed all of our tests!  Nice job!
```
</div>
</div>
</div>



`4.` Now we are going to improve the consistency of the **user_user_recs** function from above.  

* Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.


* Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be  what would be obtained from the **top_articles** function you wrote earlier.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # Your code here
    # Tuple that contains the dot product similarity and user_id
    Neighbor = collections.namedtuple(typename='Neighbor',
                                      field_names=('neighbor_id', 'similarity', 'num_interactions'))

    # compute similarity of each user to the provided user 
    # and the number of article interactions
    # remove the own user's id
    neighbors = [Neighbor(
        neighbor_id=user_row[0],
        similarity=np.dot(user_row[1], user_item.loc[user_id]),
        num_interactions=df[df['user_id'] == user_row[0]].shape[0]
    ) for user_row in user_item.drop(labels=user_id, axis=0).iterrows()]
    
    # sort by similarity, then by number of article interactions
    neighbors_df = pd.DataFrame(sorted(neighbors, key=lambda neighbor: (neighbor.similarity, neighbor.num_interactions), reverse=True))
    
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    # Your code here
    neighbors_df = get_top_sorted_users(user_id)
    
    # For easy lookup on what articles are more interacted than others for sorting
    most_interaction_articles_dict = dict(df['article_id'].value_counts().items())
    
    # List of recommendations we'll give
    recs = []
    for neighbor_row in neighbors_df.iterrows():
        sim_user = np.array(user_item.loc[neighbor_row[1].neighbor_id].astype(int))
        curr_user = np.array(user_item.loc[user_id].astype(int))
        xor_mask = sim_user ^ curr_user
        for article_id in sorted([user_item.columns[idx] for idx, val in enumerate(sim_user & xor_mask) if val], key=lambda article_id: most_interaction_articles_dict[int(article_id)], reverse=True):
            if len(recs) == m:
                rec_names = get_article_names(recs)
                return recs, rec_names
            else:
                recs.append(str(article_id))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Quick spot check - don't change this code - just use it to test your functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
The top 10 recommendations for user 20 are the following article ids:
['1330.0', '1427.0', '1364.0', '1170.0', '1162.0', '1304.0', '1351.0', '1160.0', '1354.0', '1368.0']

The top 10 recommendations for user 20 are the following article names:
['apache spark lab, part 1: basic concepts', 'predicting churn with the spss random tree algorithm', 'analyze energy consumption in buildings', 'use xgboost, scikit-learn & ibm watson machine learning apis', 'putting a human face on machine learning', 'gosales transactions for logistic regression model', 'insights from new york car accident reports', 'model bike sharing data with spss', 'analyze accident reports on amazon emr spark', 'movie recommender system with spark machine learning']
```
</div>
</div>
</div>



`5.` Use your functions from above to correctly fill in the solutions to the dictionary below.  Then test your dictionary against the solution.  Provide the code you need to answer each following the comments below.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
### Tests with a dictionary of results

user1_most_sim = get_top_sorted_users(1).loc[0].neighbor_id # Find the user that is most similar to user 1 
user131_10th_sim = get_top_sorted_users(131).loc[9].neighbor_id # Find the 10th most similar user to user 131

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
## Dictionary Test Here
sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim, 
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}

t.sol_5_test(sol_5_dict)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
This all looks good!  Nice job!
```
</div>
</div>
</div>



`6.` If we were given a new user, which of the above functions would you be able to use to make recommendations?  Explain.  Can you think of a better way we might make recommendations?  Use the cell below to explain a better method for new users.



**Provide your response here.**
> We would use the `get_top_articles` or `get_top_article_ids` function to generically recommend the most popular articles since we don't yet have any information about the new user.



`7.` Using your existing functions, provide the top 10 recommended articles you would provide for the a new user below.  You can test your function against our thoughts to make sure we are all on the same page with how we might make a recommendation.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
new_user = '0.0'

# What would your recommendations be for this new user '0.0'?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to 
new_user_recs = get_top_article_ids(n=10) # Your recommendations here



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
assert set(new_user_recs) == set(['1314.0','1429.0','1293.0','1427.0','1162.0','1364.0','1304.0','1170.0','1431.0','1330.0']), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."

print("That's right!  Nice job!")

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
That's right!  Nice job!
```
</div>
</div>
</div>



### <a class="anchor" id="Content-Recs">Part IV: Content Based Recommendations (EXTRA - NOT REQUIRED)</a>

Another method we might use to make recommendations is to perform a ranking of the highest ranked articles associated with some term.  You might consider content to be the **doc_body**, **doc_description**, or **doc_full_name**.  There isn't one way to create a content based recommendation, especially considering that each of these columns hold content related information.  

`1.` Use the function body below to create a content based recommender.  Since there isn't one right answer for this recommendation tactic, no test functions are provided.  Feel free to change the function inputs if you decide you want to try a method that requires more input values.  The input values are currently set with one idea in mind that you may use to make content based recommendations.  One additional idea is that you might want to choose the most popular recommendations that meet your 'content criteria', but again, there is a lot of flexibility in how you might make these recommendations.

### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def make_content_recs():
    '''
    INPUT:
    
    OUTPUT:
    
    '''

```
</div>

</div>



`2.` Now that you have put together your content-based recommendation system, use the cell below to write a summary explaining how your content based recommender works.  Do you see any possible improvements that could be made to your function?  Is there anything novel about your content based recommender?

### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.



**Write an explanation of your content based recommendation system here.**



`3.` Use your content-recommendation system to make recommendations for the below scenarios based on the comments.  Again no tests are provided here, because there isn't one right answer that could be used to find these content based recommendations.

### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# make recommendations for a brand new user


# make a recommendations for a user who only has interacted with article id '1427.0'



```
</div>

</div>



### <a class="anchor" id="Matrix-Fact">Part V: Matrix Factorization</a>

In this part of the notebook, you will build use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.

`1.` You should have already created a **user_item** matrix above in **question 1** of **Part III** above.  This first question here will just require that you run the cells to get things set up for the rest of **Part V** of the notebook. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Load the matrix here
user_item_matrix = pd.read_pickle('user_item_matrix.p')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# quick look at the matrix
user_item_matrix.head()

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
      <th>article_id</th>
      <th>0.0</th>
      <th>100.0</th>
      <th>1000.0</th>
      <th>1004.0</th>
      <th>1006.0</th>
      <th>1008.0</th>
      <th>101.0</th>
      <th>1014.0</th>
      <th>1015.0</th>
      <th>1016.0</th>
      <th>...</th>
      <th>977.0</th>
      <th>98.0</th>
      <th>981.0</th>
      <th>984.0</th>
      <th>985.0</th>
      <th>986.0</th>
      <th>990.0</th>
      <th>993.0</th>
      <th>996.0</th>
      <th>997.0</th>
    </tr>
    <tr>
      <th>user_id</th>
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
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 714 columns</p>
</div>
</div>


</div>
</div>
</div>



`2.` In this situation, you can use Singular Value Decomposition from [numpy](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html) on the user-item matrix.  Use the cell to perform SVD, and explain why this is different than in the lesson.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Perform SVD on the User-Item Matrix Here

u, s, vt = np.linalg.svd(user_item_matrix) # use the built in to get the three matrices

```
</div>

</div>



**Provide your response here.**
> Regular SVD requires your `user_item` matrix to not contain any missing values. Prior to FunkSVD, all missing values are imputed with 0, which introduces significant bias to the model. Our current `user_item` matrix does not contain any missing values because we're merely finding out whether a user will / will not interact with an article. FunkSVD uses gradient descent to get the $U$ and $V^\top$ matrix by using the values we have and ignoring the missing values, leaving out the $S$ matrix that consists of Singular values on the diagonal while the original SVD builds all 3 of these matrices. In the lesson, some users have not seen the movies and thus, we can't explicitly impute an accurate rating for them to use the original SVD.



`3.` Now for the tricky part, how do we choose the number of latent features to use?  Running the below cell, you can see that as the number of latent features increases, we obtain a lower error rate on making predictions for the 1 and 0 values in the user-item matrix.  Run the cell below to get an idea of how the accuracy improves as we increase the number of latent features.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
num_latent_feats = np.arange(10,700+10,20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)
    
    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)
    
    
plt.plot(num_latent_feats, 1 - np.array(sum_errs)/df.shape[0]);
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../../images/portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM_55_0.png)

</div>
</div>
</div>



`4.` From the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations.  Instead, we might split our dataset into a training and test set of data, as shown in the cell below.  

Use the code from question 3 to understand the impact on accuracy of the training and test sets of data with different numbers of latent features. Using the split below: 

* How many users can we make predictions for in the test set?  
* How many users are we not able to make predictions for because of the cold start problem?
* How many articles can we make predictions for in the test set?  
* How many articles are we not able to make predictions for because of the cold start problem?



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
df_train = df.head(40000)
df_test = df.tail(5993)

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    # Your code here
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    test_idx = user_item_test.index
    test_arts = user_item_test.columns
    
    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# How many users can we make predictions for in the test set?
len(set(user_item_train.index).intersection(set(test_idx)))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
20
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# How many users in the test set are we not able to make 
# predictions for because of the cold start problem?
len(test_idx) - len(set(user_item_train.index).intersection(set(test_idx)))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
662
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# How many articles can we make predictions for in the test set?
len(set(user_item_train.columns).intersection(set(test_arts)))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
574
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# How many articles in the test set are we not able to 
# make predictions for because of the cold start problem?
len(test_arts) - len(set(user_item_train.columns).intersection(set(test_arts)))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Replace the values in the dictionary below
a = 662 
b = 574 
c = 20 
d = 0 


sol_4_dict = {
    'How many users can we make predictions for in the test set?': c, # letter here, 
    'How many users in the test set are we not able to make predictions for because of the cold start problem?': a, # letter here, 
    'How many articles can we make predictions for in the test set?': b, # letter here,
    'How many articles in the test set are we not able to make predictions for because of the cold start problem?': d # letter here
}

t.sol_4_test(sol_4_dict)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Awesome job!  That's right!  All of the test articles are in the training data, but there are only 20 test users that were also in the training set.  All of the other users that are in the test set we have no data on.  Therefore, we cannot make predictions for these users using SVD.
```
</div>
</div>
</div>



`5.` Now use the **user_item_train** dataset from above to find U, S, and V transpose using SVD. Then find the subset of rows in the **user_item_test** dataset that you can predict using this matrix decomposition with different numbers of latent features to see how many features makes sense to keep based on the accuracy on the test data. This will require combining what was done in questions `2` - `4`.

Use the cells below to explore how well SVD works towards making predictions for recommendations on the test data.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# fit SVD on the user_item_train matrix
u_train, s_train, vt_train = np.linalg.svd(user_item_train) # fit svd similar to above then use the cells below

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Use these cells to see how well you can use the training 
# decomposition to predict on test data
num_latent_feats = np.arange(10,700+10,20)
total_train_accuracy, total_test_accuracy = [], []

for k in tqdm(num_latent_feats):
    
    # restructure with k latent features
    u_new, s_new, vt_new = u_train[:, :k], np.diag(s_train[:k]), vt_train[:k, :]
    
    # users we can make predictions for in the test set because they also 
    # exist in the training set
    users_testable = user_item_train.index.intersection(user_item_test.index)
    users_testable_idx = [list(user_item_train.index).index(user_id) for user_id in users_testable]
    u_test = u_train[users_testable_idx, :k]
    
    # articles we can make predictions for in the test set because they also 
    # exist in the training set
    articles_testable = user_item_train.columns.intersection(user_item_test.columns)
    articles_testable_idx = [list(user_item_train.columns).index(article_id) for article_id in articles_testable]
    vt_test = vt_train[:k, articles_testable_idx]
    
    # Get the predicted user item matrix for training data
    # with k latent features
    user_item_train_pred = np.around(
        np.dot(
            u_new,
            np.dot(
                s_new,
                vt_new
            )
        )
    )
    
    # Get the predicted user item matrix for testing data
    # with k latent features
    user_item_test_pred = np.around(
        np.dot(
            u_test,
            np.dot(
                s_new,
                vt_test
            )
        )
    )
    
    # Get sum of all the entries where predicted
    # matrix is different from actual matrix
    train_errors = np.sum(np.sum(np.abs(np.subtract(user_item_train, user_item_train_pred))))
    test_errors = np.sum(np.sum(np.abs(np.subtract(user_item_test.loc[users_testable, articles_testable], user_item_test_pred))))
    
    total_train_accuracy.append(1 - train_errors / (user_item_train.shape[0] * user_item_train.shape[1]))
    total_test_accuracy.append(1 - test_errors / (len(users_testable) * len(articles_testable)))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.plot(num_latent_feats, total_train_accuracy, label='Train Error')
plt.plot(num_latent_feats, total_test_accuracy, label='Test Error')
plt.xlabel('Number of Latent Features')
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features')
plt.legend()
plt.grid();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../../images/portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM_66_0.png)

</div>
</div>
</div>



`6.` Use the cell below to comment on the results you found in the previous question. Given the circumstances of your results, discuss what you might do to determine if the recommendations you make with any of the above recommendation systems are an improvement to how users currently find articles? 



**Your response here.**
> Seems like with a few number of latent features, our model was much more generalizable, and thus having a higher test accuracy. As we increase the number of latent features, our train accuracy increases. However, the test accuracy decreases. In order to determine whether our recommendations have practical significance on improving how users find articles, we could conduct an A/B test to first see how statistically significant our new recommendations are in increasing certain evaluation metrics such as the following:
1. Number of unique article interactions per user
2. Average reading time (how long a user stays on the article page) per article per user



<a id='conclusions'></a>
### Extras
Using your workbook, you could now save your recommendations for each user, develop a class to make new predictions and update your results, and make a flask app to deploy your results.  These tasks are beyond what is required for this project.  However, from what you learned in the lessons, you certainly capable of taking these tasks on to improve upon your work here!


## Conclusion

> Congratulations!  You have reached the end of the Recommendations with IBM project! 

> **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the [rubric](https://review.udacity.com/#!/rubrics/2322/view). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.


## Directions to Submit

> Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).

> Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.

> Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations! 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Recommendations_with_IBM.ipynb'])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0
```


</div>
</div>
</div>

