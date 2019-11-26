---
redirect_from:
  - "/portfolio/udacity/05-disaster-response/workspace/ml-pipeline-preparation"
interact_link: content/portfolio/udacity/05-disaster-response/workspace/ML_Pipeline_Preparation.ipynb
kernel_name: python3
has_widgets: false
title: 'Disaster Response'
prev_page:
  url: /portfolio/udacity/04-exploring-condos-sg/exploring-house-prices-singapore-part-3-crispdm.html
  title: 'Exploring Condominiums in Singapore'
next_page:
  url: /portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM.html
  title: 'Recommendation Systems for Articles with IBM'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# ML Pipeline Preparation
Follow the instructions below to help you create your ML pipeline.
### 1. Import libraries and load data from database.
- Import Python libraries
- Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
- Define feature and target variables X and Y



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
import string
import timeit
import sys
from collections import Counter

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# Function to print bold text
bold = lambda text: color.BOLD + text + color.END

# Plotly
from plotly.graph_objs import Bar, Pie, Histogram

# Punkt sentence tokenizer models 
# that help to detect sentence boundaries
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Library of stopwords from nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Model for Stemming and Lemmatization
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

# For POS Tagging
nltk.download(['averaged_perceptron_tagger'])

# import sklearn libraries
# from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.externals import joblib

# To handle imbalanced data
from imblearn.over_sampling import SMOTE, RandomOverSampler

# To help us stack models
from mlxtend.classifier import StackingClassifier

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def load_data(database_filepath):
    """Creates sql engine from database_filepath,\
       reads it in as a pandas dataframe
                 
    Args:
    -----------
    database_filepath: A str type, file path to sql database .db file
    
    Returns:
    --------
    X: Pandas dataframe of features (message text)
    y: Pandas dataframe of labels of the message
    y.columns: list of categories for the labels of message
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name=database_filepath.split('/')[-1].split('.')[0], con=engine)
    
    X = df['message']
    y = df.loc[:, 'related':]
    return X, y, y.columns

X, y, category_names = load_data('./DisasterResponse.db')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0    Weather update - a cold front from Cuba that c...
1              Is the Hurricane over or is it not over
2                      Looking for someone but no name
3    UN reports Leogane 80-90 destroyed. Only Hospi...
4    says: west side of Haiti, rest of the country ...
Name: message, dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
y.head()

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
      <th>related</th>
      <th>request</th>
      <th>offer</th>
      <th>aid_related</th>
      <th>medical_help</th>
      <th>medical_products</th>
      <th>search_and_rescue</th>
      <th>security</th>
      <th>military</th>
      <th>child_alone</th>
      <th>...</th>
      <th>aid_centers</th>
      <th>other_infrastructure</th>
      <th>weather_related</th>
      <th>floods</th>
      <th>storm</th>
      <th>fire</th>
      <th>earthquake</th>
      <th>cold</th>
      <th>other_weather</th>
      <th>direct_report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>
</div>


</div>
</div>
</div>



### 2. Write a tokenization function to process your text data



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def tokenize(text: str):
    """
    Function: 
    ---------
    Processes the text string into word tokens for an NLP model,
    with stop words removed, lemmatized, and everything is lowercase
    
    Parameters:
    -----------
    text: A str type, the sentence you want to tokenize
    
    Returns:
    --------
    A list of word tokens
    """
    return [WordNetLemmatizer().lemmatize(token.strip(), pos='v') for token in word_tokenize(text.lower()) if token not in stopwords.words("english") and token not in list(string.punctuation)]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
tokenize('The dispute has hurt the global economy, crimped U.S. exports, damaged American manufacturers and rattled corporate executives and small-business owners alike.')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['dispute',
 'hurt',
 'global',
 'economy',
 'crimp',
 'u.s.',
 'export',
 'damage',
 'american',
 'manufacturers',
 'rattle',
 'corporate',
 'executives',
 'small-business',
 'owners',
 'alike']
```


</div>
</div>
</div>



### 3. Build a machine learning pipeline
This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('=============================')  
print('Building Model:')
print('-----------------------------')

# Aggregate an ensemble of RandomForest classifier chains and feed them
# to the meta classifier
print('Creating ClassifierChains...')
chains = [ClassifierChain(base_estimator=RandomForestClassifier(n_estimators=50), order='random', random_state=42) for _ in range(10)]

# Meta Classifier that will take the predictions
# of each output of the classifier chains and figure out
# the weight of each classifier in predicting labels
print('Adding Meta Classifier...')
meta_clf = MultiOutputClassifier(AdaBoostClassifier())

# Stack the base learners 
print('Stacking Meta Classifier on top of ClassifierChains...')
sclf = StackingClassifier(classifiers=chains,
                          meta_classifier=meta_clf)

# Resample dataset to be balanced
# print('Initializing SMOTE to balance dataset...')
# sm = SMOTE(random_state=42)

# Final Pipeline
print('Building Pipeline...')
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('tfidf_vect', TfidfVectorizer(tokenizer=tokenize)),
        ]))
    ])),
    ('sclf', sclf)
])

parameters = {
    'features__text_pipeline__tfidf_vect__ngram_range': ((1, 3), (1, 5))
}

print('Initializing GridSearchCV...')
model = GridSearchCV(pipeline, param_grid=parameters, cv=2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
=============================
Building Model:
-----------------------------
Creating ClassifierChains...
Adding Meta Classifier...
Stacking Meta Classifier on top of ClassifierChains...
Building Pipeline...
Initializing GridSearchCV...
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# train test split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

```
</div>

</div>



### 4. Train pipeline
- Split data into train and test sets
- Train pipeline



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
model.get_params().keys()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
dict_keys(['cv', 'error_score', 'estimator__memory', 'estimator__steps', 'estimator__verbose', 'estimator__features', 'estimator__sclf', 'estimator__features__n_jobs', 'estimator__features__transformer_list', 'estimator__features__transformer_weights', 'estimator__features__verbose', 'estimator__features__text_pipeline', 'estimator__features__text_pipeline__memory', 'estimator__features__text_pipeline__steps', 'estimator__features__text_pipeline__verbose', 'estimator__features__text_pipeline__tfidf_vect', 'estimator__features__text_pipeline__tfidf_vect__analyzer', 'estimator__features__text_pipeline__tfidf_vect__binary', 'estimator__features__text_pipeline__tfidf_vect__decode_error', 'estimator__features__text_pipeline__tfidf_vect__dtype', 'estimator__features__text_pipeline__tfidf_vect__encoding', 'estimator__features__text_pipeline__tfidf_vect__input', 'estimator__features__text_pipeline__tfidf_vect__lowercase', 'estimator__features__text_pipeline__tfidf_vect__max_df', 'estimator__features__text_pipeline__tfidf_vect__max_features', 'estimator__features__text_pipeline__tfidf_vect__min_df', 'estimator__features__text_pipeline__tfidf_vect__ngram_range', 'estimator__features__text_pipeline__tfidf_vect__norm', 'estimator__features__text_pipeline__tfidf_vect__preprocessor', 'estimator__features__text_pipeline__tfidf_vect__smooth_idf', 'estimator__features__text_pipeline__tfidf_vect__stop_words', 'estimator__features__text_pipeline__tfidf_vect__strip_accents', 'estimator__features__text_pipeline__tfidf_vect__sublinear_tf', 'estimator__features__text_pipeline__tfidf_vect__token_pattern', 'estimator__features__text_pipeline__tfidf_vect__tokenizer', 'estimator__features__text_pipeline__tfidf_vect__use_idf', 'estimator__features__text_pipeline__tfidf_vect__vocabulary', 'estimator__sclf__average_probas', 'estimator__sclf__classifiers', 'estimator__sclf__drop_last_proba', 'estimator__sclf__meta_classifier__estimator__algorithm', 'estimator__sclf__meta_classifier__estimator__base_estimator', 'estimator__sclf__meta_classifier__estimator__learning_rate', 'estimator__sclf__meta_classifier__estimator__n_estimators', 'estimator__sclf__meta_classifier__estimator__random_state', 'estimator__sclf__meta_classifier__estimator', 'estimator__sclf__meta_classifier__n_jobs', 'estimator__sclf__meta_classifier', 'estimator__sclf__store_train_meta_features', 'estimator__sclf__use_clones', 'estimator__sclf__use_features_in_secondary', 'estimator__sclf__use_probas', 'estimator__sclf__verbose', 'estimator__sclf__classifierchain-1', 'estimator__sclf__classifierchain-2', 'estimator__sclf__classifierchain-3', 'estimator__sclf__classifierchain-4', 'estimator__sclf__classifierchain-5', 'estimator__sclf__classifierchain-6', 'estimator__sclf__classifierchain-7', 'estimator__sclf__classifierchain-8', 'estimator__sclf__classifierchain-9', 'estimator__sclf__classifierchain-10', 'estimator__sclf__classifierchain-1__base_estimator__bootstrap', 'estimator__sclf__classifierchain-1__base_estimator__class_weight', 'estimator__sclf__classifierchain-1__base_estimator__criterion', 'estimator__sclf__classifierchain-1__base_estimator__max_depth', 'estimator__sclf__classifierchain-1__base_estimator__max_features', 'estimator__sclf__classifierchain-1__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-1__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-1__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-1__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-1__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-1__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-1__base_estimator__n_estimators', 'estimator__sclf__classifierchain-1__base_estimator__n_jobs', 'estimator__sclf__classifierchain-1__base_estimator__oob_score', 'estimator__sclf__classifierchain-1__base_estimator__random_state', 'estimator__sclf__classifierchain-1__base_estimator__verbose', 'estimator__sclf__classifierchain-1__base_estimator__warm_start', 'estimator__sclf__classifierchain-1__base_estimator', 'estimator__sclf__classifierchain-1__cv', 'estimator__sclf__classifierchain-1__order', 'estimator__sclf__classifierchain-1__random_state', 'estimator__sclf__classifierchain-2__base_estimator__bootstrap', 'estimator__sclf__classifierchain-2__base_estimator__class_weight', 'estimator__sclf__classifierchain-2__base_estimator__criterion', 'estimator__sclf__classifierchain-2__base_estimator__max_depth', 'estimator__sclf__classifierchain-2__base_estimator__max_features', 'estimator__sclf__classifierchain-2__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-2__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-2__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-2__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-2__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-2__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-2__base_estimator__n_estimators', 'estimator__sclf__classifierchain-2__base_estimator__n_jobs', 'estimator__sclf__classifierchain-2__base_estimator__oob_score', 'estimator__sclf__classifierchain-2__base_estimator__random_state', 'estimator__sclf__classifierchain-2__base_estimator__verbose', 'estimator__sclf__classifierchain-2__base_estimator__warm_start', 'estimator__sclf__classifierchain-2__base_estimator', 'estimator__sclf__classifierchain-2__cv', 'estimator__sclf__classifierchain-2__order', 'estimator__sclf__classifierchain-2__random_state', 'estimator__sclf__classifierchain-3__base_estimator__bootstrap', 'estimator__sclf__classifierchain-3__base_estimator__class_weight', 'estimator__sclf__classifierchain-3__base_estimator__criterion', 'estimator__sclf__classifierchain-3__base_estimator__max_depth', 'estimator__sclf__classifierchain-3__base_estimator__max_features', 'estimator__sclf__classifierchain-3__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-3__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-3__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-3__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-3__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-3__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-3__base_estimator__n_estimators', 'estimator__sclf__classifierchain-3__base_estimator__n_jobs', 'estimator__sclf__classifierchain-3__base_estimator__oob_score', 'estimator__sclf__classifierchain-3__base_estimator__random_state', 'estimator__sclf__classifierchain-3__base_estimator__verbose', 'estimator__sclf__classifierchain-3__base_estimator__warm_start', 'estimator__sclf__classifierchain-3__base_estimator', 'estimator__sclf__classifierchain-3__cv', 'estimator__sclf__classifierchain-3__order', 'estimator__sclf__classifierchain-3__random_state', 'estimator__sclf__classifierchain-4__base_estimator__bootstrap', 'estimator__sclf__classifierchain-4__base_estimator__class_weight', 'estimator__sclf__classifierchain-4__base_estimator__criterion', 'estimator__sclf__classifierchain-4__base_estimator__max_depth', 'estimator__sclf__classifierchain-4__base_estimator__max_features', 'estimator__sclf__classifierchain-4__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-4__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-4__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-4__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-4__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-4__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-4__base_estimator__n_estimators', 'estimator__sclf__classifierchain-4__base_estimator__n_jobs', 'estimator__sclf__classifierchain-4__base_estimator__oob_score', 'estimator__sclf__classifierchain-4__base_estimator__random_state', 'estimator__sclf__classifierchain-4__base_estimator__verbose', 'estimator__sclf__classifierchain-4__base_estimator__warm_start', 'estimator__sclf__classifierchain-4__base_estimator', 'estimator__sclf__classifierchain-4__cv', 'estimator__sclf__classifierchain-4__order', 'estimator__sclf__classifierchain-4__random_state', 'estimator__sclf__classifierchain-5__base_estimator__bootstrap', 'estimator__sclf__classifierchain-5__base_estimator__class_weight', 'estimator__sclf__classifierchain-5__base_estimator__criterion', 'estimator__sclf__classifierchain-5__base_estimator__max_depth', 'estimator__sclf__classifierchain-5__base_estimator__max_features', 'estimator__sclf__classifierchain-5__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-5__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-5__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-5__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-5__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-5__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-5__base_estimator__n_estimators', 'estimator__sclf__classifierchain-5__base_estimator__n_jobs', 'estimator__sclf__classifierchain-5__base_estimator__oob_score', 'estimator__sclf__classifierchain-5__base_estimator__random_state', 'estimator__sclf__classifierchain-5__base_estimator__verbose', 'estimator__sclf__classifierchain-5__base_estimator__warm_start', 'estimator__sclf__classifierchain-5__base_estimator', 'estimator__sclf__classifierchain-5__cv', 'estimator__sclf__classifierchain-5__order', 'estimator__sclf__classifierchain-5__random_state', 'estimator__sclf__classifierchain-6__base_estimator__bootstrap', 'estimator__sclf__classifierchain-6__base_estimator__class_weight', 'estimator__sclf__classifierchain-6__base_estimator__criterion', 'estimator__sclf__classifierchain-6__base_estimator__max_depth', 'estimator__sclf__classifierchain-6__base_estimator__max_features', 'estimator__sclf__classifierchain-6__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-6__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-6__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-6__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-6__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-6__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-6__base_estimator__n_estimators', 'estimator__sclf__classifierchain-6__base_estimator__n_jobs', 'estimator__sclf__classifierchain-6__base_estimator__oob_score', 'estimator__sclf__classifierchain-6__base_estimator__random_state', 'estimator__sclf__classifierchain-6__base_estimator__verbose', 'estimator__sclf__classifierchain-6__base_estimator__warm_start', 'estimator__sclf__classifierchain-6__base_estimator', 'estimator__sclf__classifierchain-6__cv', 'estimator__sclf__classifierchain-6__order', 'estimator__sclf__classifierchain-6__random_state', 'estimator__sclf__classifierchain-7__base_estimator__bootstrap', 'estimator__sclf__classifierchain-7__base_estimator__class_weight', 'estimator__sclf__classifierchain-7__base_estimator__criterion', 'estimator__sclf__classifierchain-7__base_estimator__max_depth', 'estimator__sclf__classifierchain-7__base_estimator__max_features', 'estimator__sclf__classifierchain-7__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-7__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-7__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-7__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-7__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-7__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-7__base_estimator__n_estimators', 'estimator__sclf__classifierchain-7__base_estimator__n_jobs', 'estimator__sclf__classifierchain-7__base_estimator__oob_score', 'estimator__sclf__classifierchain-7__base_estimator__random_state', 'estimator__sclf__classifierchain-7__base_estimator__verbose', 'estimator__sclf__classifierchain-7__base_estimator__warm_start', 'estimator__sclf__classifierchain-7__base_estimator', 'estimator__sclf__classifierchain-7__cv', 'estimator__sclf__classifierchain-7__order', 'estimator__sclf__classifierchain-7__random_state', 'estimator__sclf__classifierchain-8__base_estimator__bootstrap', 'estimator__sclf__classifierchain-8__base_estimator__class_weight', 'estimator__sclf__classifierchain-8__base_estimator__criterion', 'estimator__sclf__classifierchain-8__base_estimator__max_depth', 'estimator__sclf__classifierchain-8__base_estimator__max_features', 'estimator__sclf__classifierchain-8__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-8__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-8__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-8__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-8__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-8__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-8__base_estimator__n_estimators', 'estimator__sclf__classifierchain-8__base_estimator__n_jobs', 'estimator__sclf__classifierchain-8__base_estimator__oob_score', 'estimator__sclf__classifierchain-8__base_estimator__random_state', 'estimator__sclf__classifierchain-8__base_estimator__verbose', 'estimator__sclf__classifierchain-8__base_estimator__warm_start', 'estimator__sclf__classifierchain-8__base_estimator', 'estimator__sclf__classifierchain-8__cv', 'estimator__sclf__classifierchain-8__order', 'estimator__sclf__classifierchain-8__random_state', 'estimator__sclf__classifierchain-9__base_estimator__bootstrap', 'estimator__sclf__classifierchain-9__base_estimator__class_weight', 'estimator__sclf__classifierchain-9__base_estimator__criterion', 'estimator__sclf__classifierchain-9__base_estimator__max_depth', 'estimator__sclf__classifierchain-9__base_estimator__max_features', 'estimator__sclf__classifierchain-9__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-9__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-9__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-9__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-9__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-9__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-9__base_estimator__n_estimators', 'estimator__sclf__classifierchain-9__base_estimator__n_jobs', 'estimator__sclf__classifierchain-9__base_estimator__oob_score', 'estimator__sclf__classifierchain-9__base_estimator__random_state', 'estimator__sclf__classifierchain-9__base_estimator__verbose', 'estimator__sclf__classifierchain-9__base_estimator__warm_start', 'estimator__sclf__classifierchain-9__base_estimator', 'estimator__sclf__classifierchain-9__cv', 'estimator__sclf__classifierchain-9__order', 'estimator__sclf__classifierchain-9__random_state', 'estimator__sclf__classifierchain-10__base_estimator__bootstrap', 'estimator__sclf__classifierchain-10__base_estimator__class_weight', 'estimator__sclf__classifierchain-10__base_estimator__criterion', 'estimator__sclf__classifierchain-10__base_estimator__max_depth', 'estimator__sclf__classifierchain-10__base_estimator__max_features', 'estimator__sclf__classifierchain-10__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-10__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-10__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-10__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-10__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-10__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-10__base_estimator__n_estimators', 'estimator__sclf__classifierchain-10__base_estimator__n_jobs', 'estimator__sclf__classifierchain-10__base_estimator__oob_score', 'estimator__sclf__classifierchain-10__base_estimator__random_state', 'estimator__sclf__classifierchain-10__base_estimator__verbose', 'estimator__sclf__classifierchain-10__base_estimator__warm_start', 'estimator__sclf__classifierchain-10__base_estimator', 'estimator__sclf__classifierchain-10__cv', 'estimator__sclf__classifierchain-10__order', 'estimator__sclf__classifierchain-10__random_state', 'estimator', 'iid', 'n_jobs', 'param_grid', 'pre_dispatch', 'refit', 'return_train_score', 'scoring', 'verbose'])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('=============================')  
print('Training Model:')
print('-----------------------------')
model.fit(X_train, y_train)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
=============================
Training Model:
-----------------------------
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_traceback_line}
```

    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-10-a155e9e43e98> in <module>()
          2 print('Training Model:')
          3 print('-----------------------------')
    ----> 4 model.fit(X_train, y_train)
    

    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
        685                 return results
        686 
    --> 687             self._run_search(evaluate_candidates)
        688 
        689         # For multi-metric evaluation, store the best_index_, best_params_ and


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
       1146     def _run_search(self, evaluate_candidates):
       1147         """Search all candidates in param_grid"""
    -> 1148         evaluate_candidates(ParameterGrid(self.param_grid))
       1149 
       1150 


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params)
        664                                for parameters, (train, test)
        665                                in product(candidate_params,
    --> 666                                           cv.split(X, y, groups)))
        667 
        668                 if len(out) < 1:


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in __call__(self, iterable)
        922                 self._iterating = self._original_iterator is not None
        923 
    --> 924             while self.dispatch_one_batch(iterator):
        925                 pass
        926 


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in dispatch_one_batch(self, iterator)
        757                 return False
        758             else:
    --> 759                 self._dispatch(tasks)
        760                 return True
        761 


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in _dispatch(self, batch)
        714         with self._lock:
        715             job_idx = len(self._jobs)
    --> 716             job = self._backend.apply_async(batch, callback=cb)
        717             # A job can complete so quickly than its callback is
        718             # called before we get here, causing self._jobs to


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/_parallel_backends.py in apply_async(self, func, callback)
        180     def apply_async(self, func, callback=None):
        181         """Schedule a func to be run"""
    --> 182         result = ImmediateResult(func)
        183         if callback:
        184             callback(result)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/_parallel_backends.py in __init__(self, batch)
        547         # Don't delay the application, to avoid keeping the input
        548         # arguments in memory
    --> 549         self.results = batch()
        550 
        551     def get(self):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in __call__(self)
        223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        224             return [func(*args, **kwargs)
    --> 225                     for func, args, kwargs in self.items]
        226 
        227     def __len__(self):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in <listcomp>(.0)
        223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        224             return [func(*args, **kwargs)
    --> 225                     for func, args, kwargs in self.items]
        226 
        227     def __len__(self):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/model_selection/_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, error_score)
        512             estimator.fit(X_train, **fit_params)
        513         else:
    --> 514             estimator.fit(X_train, y_train, **fit_params)
        515 
        516     except Exception as e:


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/imblearn/pipeline.py in fit(self, X, y, **fit_params)
        238         Xt, yt, fit_params = self._fit(X, y, **fit_params)
        239         if self._final_estimator != 'passthrough':
    --> 240             self._final_estimator.fit(Xt, yt, **fit_params)
        241         return self
        242 


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/mlxtend/classifier/stacking_classification.py in fit(self, X, y, sample_weight)
        159                 print(_name_estimators((clf,))[0][1])
        160             if sample_weight is None:
    --> 161                 clf.fit(X, y)
        162             else:
        163                 clf.fit(X, y, sample_weight=sample_weight)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/multioutput.py in fit(self, X, Y)
        582         self : object
        583         """
    --> 584         super().fit(X, Y)
        585         self.classes_ = [estimator.classes_
        586                          for chain_idx, estimator


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/multioutput.py in fit(self, X, Y)
        445         for chain_idx, estimator in enumerate(self.estimators_):
        446             y = Y[:, self.order_[chain_idx]]
    --> 447             estimator.fit(X_aug[:, :(X.shape[1] + chain_idx)], y)
        448             if self.cv is not None and chain_idx < len(self.estimators_) - 1:
        449                 col_idx = X.shape[1] + chain_idx


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/ensemble/forest.py in fit(self, X, y, sample_weight)
        328                     t, self, X, y, sample_weight, i, len(trees),
        329                     verbose=self.verbose, class_weight=self.class_weight)
    --> 330                 for i, t in enumerate(trees))
        331 
        332             # Collect newly grown trees


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in __call__(self, iterable)
        922                 self._iterating = self._original_iterator is not None
        923 
    --> 924             while self.dispatch_one_batch(iterator):
        925                 pass
        926 


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in dispatch_one_batch(self, iterator)
        757                 return False
        758             else:
    --> 759                 self._dispatch(tasks)
        760                 return True
        761 


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in _dispatch(self, batch)
        714         with self._lock:
        715             job_idx = len(self._jobs)
    --> 716             job = self._backend.apply_async(batch, callback=cb)
        717             # A job can complete so quickly than its callback is
        718             # called before we get here, causing self._jobs to


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/_parallel_backends.py in apply_async(self, func, callback)
        180     def apply_async(self, func, callback=None):
        181         """Schedule a func to be run"""
    --> 182         result = ImmediateResult(func)
        183         if callback:
        184             callback(result)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/_parallel_backends.py in __init__(self, batch)
        547         # Don't delay the application, to avoid keeping the input
        548         # arguments in memory
    --> 549         self.results = batch()
        550 
        551     def get(self):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in __call__(self)
        223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        224             return [func(*args, **kwargs)
    --> 225                     for func, args, kwargs in self.items]
        226 
        227     def __len__(self):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in <listcomp>(.0)
        223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        224             return [func(*args, **kwargs)
    --> 225                     for func, args, kwargs in self.items]
        226 
        227     def __len__(self):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/ensemble/forest.py in _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees, verbose, class_weight)
        116             curr_sample_weight *= compute_sample_weight('balanced', y, indices)
        117 
    --> 118         tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
        119     else:
        120         tree.fit(X, y, sample_weight=sample_weight, check_input=False)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/tree/tree.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        814             sample_weight=sample_weight,
        815             check_input=check_input,
    --> 816             X_idx_sorted=X_idx_sorted)
        817         return self
        818 


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/tree/tree.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        378                                            min_impurity_split)
        379 
    --> 380         builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)
        381 
        382         if self.n_outputs_ == 1:


    KeyboardInterrupt: 


```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(model.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (model.cv_results_[cv_keys[0]][r],
             model.cv_results_[cv_keys[1]][r] / 2.0,
             model.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % model.best_params_)
print('Training Accuracy: %.2f' % model.best_score_)

```
</div>

</div>



### 5. Test your model
Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
y_pred = model.predict(X_test)

for idx, label in enumerate(category_names):
    print("Classification Report for {}:".format(bold(label)))
    print(
        classification_report(
            y_true=np.array(y_test)[:, idx], 
            y_pred=y_pred[:, idx]
        )
    )

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
for average_type in ['micro', 'macro']:
    print('{} F1-score: {}'.format(average_type, f1_score(y_test, y_pred, average=average_type)))
    print('{} Precision: {}'.format(average_type, precision_score(y_test, y_pred, average=average_type)))
    print('{} Recall: {}'.format(average_type, recall_score(y_test, y_pred, average=average_type)))

accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)

```
</div>

</div>



### 6. Improve your model
Use grid search to find better parameters. 



### 7. Test your model
Show the accuracy, precision, and recall of the tuned model.  

Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!



### 8. Try improving your model further. Here are a few ideas:
* try other machine learning algorithms
* add other features besides the TF-IDF



### 9. Export your model as a pickle file



### 10. Use this notebook to complete `train.py`
Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.



---
## Resources

- [sklearn Classifier chain example](https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html#sphx-glr-auto-examples-multioutput-plot-classifier-chain-yeast-py)
- [A Deep Dive Into Sklearn Pipelines](https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines)
- [Pipeline for Multi-Label Classification with One-vs-Rest Meta Classifier](http://queirozf.com/entries/scikit-learn-pipeline-examples#pipeline-for-multi-label-classification-with-one-vs-rest-meta-classifier)
- [Stacked Classification and GridSearch](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/)

