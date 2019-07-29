---
redirect_from:
  - "/portfolio/udacity/05-disaster-response/workspace/ml-pipeline-preparation"
interact_link: content/portfolio/udacity/05-disaster-response/workspace/ML_Pipeline_Preparation.ipynb
kernel_name: python3
has_widgets: false
title: 'Disaster Response'
prev_page:
  url: /portfolio/udacity/04-exploring-condos-sg/exploring-house-prices-singapore-part-3-crispdm
  title: 'Exploring Condominiums in Singapore'
next_page:
  url: /portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM
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
chains = [ClassifierChain(base_estimator=RandomForestClassifier(), order='random', random_state=42) for _ in range(10)]

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
print('Initializing Random Over Sampler to balance dataset...')
ros = RandomOverSampler(random_state=42)

# Final Pipeline
print('Building Pipeline...')
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('tfidf_vect', TfidfVectorizer(tokenizer=tokenize)),
        ]))#,
#         ('starting_verb', StartingVerbExtractor())
    ])),
    ('ros', ros),
    ('sclf', sclf)
])

parameters = {
    'features__text_pipeline__tfidf_vect__ngram_range': ((1, 1), (1, 2)),
    'sclf__classifierchain-1__base_estimator__n_estimators': [2, 3],
    'sclf__classifierchain-2__base_estimator__n_estimators': [2, 3]
#     'features__text_pipeline__tfidf_vect__max_df': (0.5, 0.75, 1.0),
#     'features__text_pipeline__tfidf_vect__max_features': (None, 5000, 10000),
#     'features__text_pipeline__tfidf_vect__use_idf': (True, False),
#     'features__transformer_weights': (
#         {'text_pipeline': 1, 'starting_verb': 0.5},
#         {'text_pipeline': 0.5, 'starting_verb': 1},
#         {'text_pipeline': 0.8, 'starting_verb': 1},
#     )
}

model = GridSearchCV(pipeline, param_grid=parameters, cv=3)

# # Hyperparameters for tuning in GridSearch
# params = {
#     'features__text_pipeline__tfidf_vect__ngram_range': ((1, 1), (1, 2)),
#     'features__text_pipeline__tfidf_vect__max_df': (0.5, 0.75, 1.0),
#     'features__text_pipeline__tfidf_vect__max_features': (None, 5000, 10000),
#     'features__text_pipeline__tfidf_vect__use_idf': (True, False),
#     'sclf__classifiers__classifierchain__base_estimator__randomforestclassifier__n_estimators': [100, 200],
#     'sclf__classifiers__classifierchain__base_estimator__randomforestclassifier__min_samples_split': [2, 5, 10],
#     'sclf__classifiers__classifierchain__base_estimator__adaboostclassifier__n_estimators': [100, 200],
#     'sclf__classifiers__classifierchain__base_estimator__adaboostclassifier__learning_rate': [0.01, 0.1, 1.0],
#     'sclf__classifiers__classifierchain__base_estimator__gradientboostingclassifier__n_estimators': [100, 200],
#     'sclf__classifiers__classifierchain__base_estimator__gradientboostingclassifier__learning_rate': [0.01, 0.1],
#     'sclf__classifiers__classifierchain__base_estimator__gradientboostingclassifier__min_samples_split': [2, 5, 10],
#     'sclf__classifiers__classifierchain__base_estimator__extratreesclassifier__n_estimators': [100, 200],
#     'sclf__classifiers__classifierchain__base_estimator__extratreesclassifier__min_samples_split': [2, 5, 10],
#     'sclf__classifiers__classifierchain__base_estimator__svc__C': [0.01, 0.1, 1.0],
#     'sclf__classifiers__classifierchain__base_estimator__svc__kernel': ['rbf', 'sigmoid'],
#     'sclf__classifiers__classifierchain__base_estimator__kneighborsclassifier__n_neighbors': [1, 5],
#     'sclf__classifiers__classifierchain__base_estimator__kneighborsclassifier__n_neighbors': [1, 5],
#     'sclf__meta_classifier__base_estimator__C': [0.1, 10.0],
#     'features__transformer_weights': (
#         {'text_pipeline': 1, 'starting_verb': 0.5},
#         {'text_pipeline': 0.5, 'starting_verb': 1},
#         {'text_pipeline': 0.8, 'starting_verb': 1},
#     )
# }

# print('Initializing GridSearchCV...')
# model = GridSearchCV(estimator=pipeline, 
#                      param_grid=params,
#                      cv=5,
#                      refit=True)

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
Initializing Random Over Sampler to balance dataset...
Building Pipeline...
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
dict_keys(['cv', 'error_score', 'estimator__memory', 'estimator__steps', 'estimator__verbose', 'estimator__features', 'estimator__ros', 'estimator__sclf', 'estimator__features__n_jobs', 'estimator__features__transformer_list', 'estimator__features__transformer_weights', 'estimator__features__verbose', 'estimator__features__text_pipeline', 'estimator__features__text_pipeline__memory', 'estimator__features__text_pipeline__steps', 'estimator__features__text_pipeline__verbose', 'estimator__features__text_pipeline__tfidf_vect', 'estimator__features__text_pipeline__tfidf_vect__analyzer', 'estimator__features__text_pipeline__tfidf_vect__binary', 'estimator__features__text_pipeline__tfidf_vect__decode_error', 'estimator__features__text_pipeline__tfidf_vect__dtype', 'estimator__features__text_pipeline__tfidf_vect__encoding', 'estimator__features__text_pipeline__tfidf_vect__input', 'estimator__features__text_pipeline__tfidf_vect__lowercase', 'estimator__features__text_pipeline__tfidf_vect__max_df', 'estimator__features__text_pipeline__tfidf_vect__max_features', 'estimator__features__text_pipeline__tfidf_vect__min_df', 'estimator__features__text_pipeline__tfidf_vect__ngram_range', 'estimator__features__text_pipeline__tfidf_vect__norm', 'estimator__features__text_pipeline__tfidf_vect__preprocessor', 'estimator__features__text_pipeline__tfidf_vect__smooth_idf', 'estimator__features__text_pipeline__tfidf_vect__stop_words', 'estimator__features__text_pipeline__tfidf_vect__strip_accents', 'estimator__features__text_pipeline__tfidf_vect__sublinear_tf', 'estimator__features__text_pipeline__tfidf_vect__token_pattern', 'estimator__features__text_pipeline__tfidf_vect__tokenizer', 'estimator__features__text_pipeline__tfidf_vect__use_idf', 'estimator__features__text_pipeline__tfidf_vect__vocabulary', 'estimator__ros__random_state', 'estimator__ros__ratio', 'estimator__ros__return_indices', 'estimator__ros__sampling_strategy', 'estimator__sclf__average_probas', 'estimator__sclf__classifiers', 'estimator__sclf__drop_last_proba', 'estimator__sclf__meta_classifier__estimator__algorithm', 'estimator__sclf__meta_classifier__estimator__base_estimator', 'estimator__sclf__meta_classifier__estimator__learning_rate', 'estimator__sclf__meta_classifier__estimator__n_estimators', 'estimator__sclf__meta_classifier__estimator__random_state', 'estimator__sclf__meta_classifier__estimator', 'estimator__sclf__meta_classifier__n_jobs', 'estimator__sclf__meta_classifier', 'estimator__sclf__store_train_meta_features', 'estimator__sclf__use_clones', 'estimator__sclf__use_features_in_secondary', 'estimator__sclf__use_probas', 'estimator__sclf__verbose', 'estimator__sclf__classifierchain-1', 'estimator__sclf__classifierchain-2', 'estimator__sclf__classifierchain-3', 'estimator__sclf__classifierchain-4', 'estimator__sclf__classifierchain-5', 'estimator__sclf__classifierchain-6', 'estimator__sclf__classifierchain-7', 'estimator__sclf__classifierchain-8', 'estimator__sclf__classifierchain-9', 'estimator__sclf__classifierchain-10', 'estimator__sclf__classifierchain-1__base_estimator__bootstrap', 'estimator__sclf__classifierchain-1__base_estimator__class_weight', 'estimator__sclf__classifierchain-1__base_estimator__criterion', 'estimator__sclf__classifierchain-1__base_estimator__max_depth', 'estimator__sclf__classifierchain-1__base_estimator__max_features', 'estimator__sclf__classifierchain-1__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-1__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-1__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-1__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-1__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-1__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-1__base_estimator__n_estimators', 'estimator__sclf__classifierchain-1__base_estimator__n_jobs', 'estimator__sclf__classifierchain-1__base_estimator__oob_score', 'estimator__sclf__classifierchain-1__base_estimator__random_state', 'estimator__sclf__classifierchain-1__base_estimator__verbose', 'estimator__sclf__classifierchain-1__base_estimator__warm_start', 'estimator__sclf__classifierchain-1__base_estimator', 'estimator__sclf__classifierchain-1__cv', 'estimator__sclf__classifierchain-1__order', 'estimator__sclf__classifierchain-1__random_state', 'estimator__sclf__classifierchain-2__base_estimator__bootstrap', 'estimator__sclf__classifierchain-2__base_estimator__class_weight', 'estimator__sclf__classifierchain-2__base_estimator__criterion', 'estimator__sclf__classifierchain-2__base_estimator__max_depth', 'estimator__sclf__classifierchain-2__base_estimator__max_features', 'estimator__sclf__classifierchain-2__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-2__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-2__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-2__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-2__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-2__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-2__base_estimator__n_estimators', 'estimator__sclf__classifierchain-2__base_estimator__n_jobs', 'estimator__sclf__classifierchain-2__base_estimator__oob_score', 'estimator__sclf__classifierchain-2__base_estimator__random_state', 'estimator__sclf__classifierchain-2__base_estimator__verbose', 'estimator__sclf__classifierchain-2__base_estimator__warm_start', 'estimator__sclf__classifierchain-2__base_estimator', 'estimator__sclf__classifierchain-2__cv', 'estimator__sclf__classifierchain-2__order', 'estimator__sclf__classifierchain-2__random_state', 'estimator__sclf__classifierchain-3__base_estimator__bootstrap', 'estimator__sclf__classifierchain-3__base_estimator__class_weight', 'estimator__sclf__classifierchain-3__base_estimator__criterion', 'estimator__sclf__classifierchain-3__base_estimator__max_depth', 'estimator__sclf__classifierchain-3__base_estimator__max_features', 'estimator__sclf__classifierchain-3__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-3__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-3__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-3__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-3__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-3__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-3__base_estimator__n_estimators', 'estimator__sclf__classifierchain-3__base_estimator__n_jobs', 'estimator__sclf__classifierchain-3__base_estimator__oob_score', 'estimator__sclf__classifierchain-3__base_estimator__random_state', 'estimator__sclf__classifierchain-3__base_estimator__verbose', 'estimator__sclf__classifierchain-3__base_estimator__warm_start', 'estimator__sclf__classifierchain-3__base_estimator', 'estimator__sclf__classifierchain-3__cv', 'estimator__sclf__classifierchain-3__order', 'estimator__sclf__classifierchain-3__random_state', 'estimator__sclf__classifierchain-4__base_estimator__bootstrap', 'estimator__sclf__classifierchain-4__base_estimator__class_weight', 'estimator__sclf__classifierchain-4__base_estimator__criterion', 'estimator__sclf__classifierchain-4__base_estimator__max_depth', 'estimator__sclf__classifierchain-4__base_estimator__max_features', 'estimator__sclf__classifierchain-4__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-4__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-4__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-4__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-4__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-4__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-4__base_estimator__n_estimators', 'estimator__sclf__classifierchain-4__base_estimator__n_jobs', 'estimator__sclf__classifierchain-4__base_estimator__oob_score', 'estimator__sclf__classifierchain-4__base_estimator__random_state', 'estimator__sclf__classifierchain-4__base_estimator__verbose', 'estimator__sclf__classifierchain-4__base_estimator__warm_start', 'estimator__sclf__classifierchain-4__base_estimator', 'estimator__sclf__classifierchain-4__cv', 'estimator__sclf__classifierchain-4__order', 'estimator__sclf__classifierchain-4__random_state', 'estimator__sclf__classifierchain-5__base_estimator__bootstrap', 'estimator__sclf__classifierchain-5__base_estimator__class_weight', 'estimator__sclf__classifierchain-5__base_estimator__criterion', 'estimator__sclf__classifierchain-5__base_estimator__max_depth', 'estimator__sclf__classifierchain-5__base_estimator__max_features', 'estimator__sclf__classifierchain-5__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-5__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-5__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-5__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-5__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-5__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-5__base_estimator__n_estimators', 'estimator__sclf__classifierchain-5__base_estimator__n_jobs', 'estimator__sclf__classifierchain-5__base_estimator__oob_score', 'estimator__sclf__classifierchain-5__base_estimator__random_state', 'estimator__sclf__classifierchain-5__base_estimator__verbose', 'estimator__sclf__classifierchain-5__base_estimator__warm_start', 'estimator__sclf__classifierchain-5__base_estimator', 'estimator__sclf__classifierchain-5__cv', 'estimator__sclf__classifierchain-5__order', 'estimator__sclf__classifierchain-5__random_state', 'estimator__sclf__classifierchain-6__base_estimator__bootstrap', 'estimator__sclf__classifierchain-6__base_estimator__class_weight', 'estimator__sclf__classifierchain-6__base_estimator__criterion', 'estimator__sclf__classifierchain-6__base_estimator__max_depth', 'estimator__sclf__classifierchain-6__base_estimator__max_features', 'estimator__sclf__classifierchain-6__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-6__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-6__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-6__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-6__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-6__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-6__base_estimator__n_estimators', 'estimator__sclf__classifierchain-6__base_estimator__n_jobs', 'estimator__sclf__classifierchain-6__base_estimator__oob_score', 'estimator__sclf__classifierchain-6__base_estimator__random_state', 'estimator__sclf__classifierchain-6__base_estimator__verbose', 'estimator__sclf__classifierchain-6__base_estimator__warm_start', 'estimator__sclf__classifierchain-6__base_estimator', 'estimator__sclf__classifierchain-6__cv', 'estimator__sclf__classifierchain-6__order', 'estimator__sclf__classifierchain-6__random_state', 'estimator__sclf__classifierchain-7__base_estimator__bootstrap', 'estimator__sclf__classifierchain-7__base_estimator__class_weight', 'estimator__sclf__classifierchain-7__base_estimator__criterion', 'estimator__sclf__classifierchain-7__base_estimator__max_depth', 'estimator__sclf__classifierchain-7__base_estimator__max_features', 'estimator__sclf__classifierchain-7__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-7__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-7__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-7__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-7__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-7__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-7__base_estimator__n_estimators', 'estimator__sclf__classifierchain-7__base_estimator__n_jobs', 'estimator__sclf__classifierchain-7__base_estimator__oob_score', 'estimator__sclf__classifierchain-7__base_estimator__random_state', 'estimator__sclf__classifierchain-7__base_estimator__verbose', 'estimator__sclf__classifierchain-7__base_estimator__warm_start', 'estimator__sclf__classifierchain-7__base_estimator', 'estimator__sclf__classifierchain-7__cv', 'estimator__sclf__classifierchain-7__order', 'estimator__sclf__classifierchain-7__random_state', 'estimator__sclf__classifierchain-8__base_estimator__bootstrap', 'estimator__sclf__classifierchain-8__base_estimator__class_weight', 'estimator__sclf__classifierchain-8__base_estimator__criterion', 'estimator__sclf__classifierchain-8__base_estimator__max_depth', 'estimator__sclf__classifierchain-8__base_estimator__max_features', 'estimator__sclf__classifierchain-8__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-8__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-8__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-8__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-8__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-8__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-8__base_estimator__n_estimators', 'estimator__sclf__classifierchain-8__base_estimator__n_jobs', 'estimator__sclf__classifierchain-8__base_estimator__oob_score', 'estimator__sclf__classifierchain-8__base_estimator__random_state', 'estimator__sclf__classifierchain-8__base_estimator__verbose', 'estimator__sclf__classifierchain-8__base_estimator__warm_start', 'estimator__sclf__classifierchain-8__base_estimator', 'estimator__sclf__classifierchain-8__cv', 'estimator__sclf__classifierchain-8__order', 'estimator__sclf__classifierchain-8__random_state', 'estimator__sclf__classifierchain-9__base_estimator__bootstrap', 'estimator__sclf__classifierchain-9__base_estimator__class_weight', 'estimator__sclf__classifierchain-9__base_estimator__criterion', 'estimator__sclf__classifierchain-9__base_estimator__max_depth', 'estimator__sclf__classifierchain-9__base_estimator__max_features', 'estimator__sclf__classifierchain-9__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-9__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-9__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-9__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-9__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-9__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-9__base_estimator__n_estimators', 'estimator__sclf__classifierchain-9__base_estimator__n_jobs', 'estimator__sclf__classifierchain-9__base_estimator__oob_score', 'estimator__sclf__classifierchain-9__base_estimator__random_state', 'estimator__sclf__classifierchain-9__base_estimator__verbose', 'estimator__sclf__classifierchain-9__base_estimator__warm_start', 'estimator__sclf__classifierchain-9__base_estimator', 'estimator__sclf__classifierchain-9__cv', 'estimator__sclf__classifierchain-9__order', 'estimator__sclf__classifierchain-9__random_state', 'estimator__sclf__classifierchain-10__base_estimator__bootstrap', 'estimator__sclf__classifierchain-10__base_estimator__class_weight', 'estimator__sclf__classifierchain-10__base_estimator__criterion', 'estimator__sclf__classifierchain-10__base_estimator__max_depth', 'estimator__sclf__classifierchain-10__base_estimator__max_features', 'estimator__sclf__classifierchain-10__base_estimator__max_leaf_nodes', 'estimator__sclf__classifierchain-10__base_estimator__min_impurity_decrease', 'estimator__sclf__classifierchain-10__base_estimator__min_impurity_split', 'estimator__sclf__classifierchain-10__base_estimator__min_samples_leaf', 'estimator__sclf__classifierchain-10__base_estimator__min_samples_split', 'estimator__sclf__classifierchain-10__base_estimator__min_weight_fraction_leaf', 'estimator__sclf__classifierchain-10__base_estimator__n_estimators', 'estimator__sclf__classifierchain-10__base_estimator__n_jobs', 'estimator__sclf__classifierchain-10__base_estimator__oob_score', 'estimator__sclf__classifierchain-10__base_estimator__random_state', 'estimator__sclf__classifierchain-10__base_estimator__verbose', 'estimator__sclf__classifierchain-10__base_estimator__warm_start', 'estimator__sclf__classifierchain-10__base_estimator', 'estimator__sclf__classifierchain-10__cv', 'estimator__sclf__classifierchain-10__order', 'estimator__sclf__classifierchain-10__random_state', 'estimator', 'iid', 'n_jobs', 'param_grid', 'pre_dispatch', 'refit', 'return_train_score', 'scoring', 'verbose'])
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

    <ipython-input-38-a155e9e43e98> in <module>()
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
        919             # remaining jobs.
        920             self._iterating = False
    --> 921             if self.dispatch_one_batch(iterator):
        922                 self._iterating = self._original_iterator is not None
        923 


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
        236 
        237         """
    --> 238         Xt, yt, fit_params = self._fit(X, y, **fit_params)
        239         if self._final_estimator != 'passthrough':
        240             self._final_estimator.fit(Xt, yt, **fit_params)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/imblearn/pipeline.py in _fit(self, X, y, **fit_params)
        196                 X, fitted_transformer = fit_transform_one_cached(
        197                     cloned_transformer, None, X, y,
    --> 198                     **fit_params_steps[name])
        199             elif hasattr(cloned_transformer, "fit_resample"):
        200                 X, y, fitted_transformer = fit_resample_one_cached(


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/memory.py in __call__(self, *args, **kwargs)
        353 
        354     def __call__(self, *args, **kwargs):
    --> 355         return self.func(*args, **kwargs)
        356 
        357     def call_and_shelve(self, *args, **kwargs):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/imblearn/pipeline.py in _fit_transform_one(transformer, weight, X, y, **fit_params)
        570 def _fit_transform_one(transformer, weight, X, y, **fit_params):
        571     if hasattr(transformer, 'fit_transform'):
    --> 572         res = transformer.fit_transform(X, y, **fit_params)
        573     else:
        574         res = transformer.fit(X, y, **fit_params).transform(X)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
        910             sum of n_components (output dimension) over transformers.
        911         """
    --> 912         results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        913         if not results:
        914             # All transformers are None


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/pipeline.py in _parallel_func(self, X, y, fit_params, func)
        940             message=self._log_message(name, idx, len(transformers)),
        941             **fit_params) for idx, (name, transformer,
    --> 942                                     weight) in enumerate(transformers, 1))
        943 
        944     def transform(self, X):


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/joblib/parallel.py in __call__(self, iterable)
        919             # remaining jobs.
        920             self._iterating = False
    --> 921             if self.dispatch_one_batch(iterator):
        922                 self._iterating = self._original_iterator is not None
        923 


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


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
        714     with _print_elapsed_time(message_clsname, message):
        715         if hasattr(transformer, 'fit_transform'):
    --> 716             res = transformer.fit_transform(X, y, **fit_params)
        717         else:
        718             res = transformer.fit(X, y, **fit_params).transform(X)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/imblearn/pipeline.py in fit_transform(self, X, y, **fit_params)
        274             return Xt
        275         elif hasattr(last_step, 'fit_transform'):
    --> 276             return last_step.fit_transform(Xt, yt, **fit_params)
        277         else:
        278             return last_step.fit(Xt, yt, **fit_params).transform(Xt)


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
       1650         """
       1651         self._check_params()
    -> 1652         X = super().fit_transform(raw_documents)
       1653         self._tfidf.fit(X)
       1654         # X is already a transformed view of raw_documents so


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
       1056 
       1057         vocabulary, X = self._count_vocab(raw_documents,
    -> 1058                                           self.fixed_vocabulary_)
       1059 
       1060         if self.binary:


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in _count_vocab(self, raw_documents, fixed_vocab)
        968         for doc in raw_documents:
        969             feature_counter = {}
    --> 970             for feature in analyze(doc):
        971                 try:
        972                     feature_idx = vocabulary[feature]


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in <lambda>(doc)
        350                                                tokenize)
        351             return lambda doc: self._word_ngrams(
    --> 352                 tokenize(preprocess(self.decode(doc))), stop_words)
        353 
        354         else:


    <ipython-input-33-9625f3c81258> in tokenize(text)
         14     A list of word tokens
         15     """
    ---> 16     return [WordNetLemmatizer().lemmatize(token.strip(), pos='v') for token in word_tokenize(text.lower()) if token not in stopwords.words("english") and token not in list(string.punctuation)]
    

    <ipython-input-33-9625f3c81258> in <listcomp>(.0)
         14     A list of word tokens
         15     """
    ---> 16     return [WordNetLemmatizer().lemmatize(token.strip(), pos='v') for token in word_tokenize(text.lower()) if token not in stopwords.words("english") and token not in list(string.punctuation)]
    

    /anaconda3/envs/geopandas/lib/python3.7/site-packages/nltk/corpus/reader/wordlist.py in words(self, fileids, ignore_lines_startswith)
         23         return [
         24             line
    ---> 25             for line in line_tokenize(self.raw(fileids))
         26             if not line.startswith(ignore_lines_startswith)
         27         ]


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/nltk/tokenize/simple.py in line_tokenize(text, blanklines)
        138 
        139 def line_tokenize(text, blanklines='discard'):
    --> 140     return LineTokenizer(blanklines).tokenize(text)
    

    /anaconda3/envs/geopandas/lib/python3.7/site-packages/nltk/tokenize/simple.py in tokenize(self, s)
        115         # If requested, strip off blank lines.
        116         if self._blanklines == 'discard':
    --> 117             lines = [l for l in lines if l.rstrip()]
        118         elif self._blanklines == 'discard-eof':
        119             if lines and not lines[-1].strip():


    /anaconda3/envs/geopandas/lib/python3.7/site-packages/nltk/tokenize/simple.py in <listcomp>(.0)
        115         # If requested, strip off blank lines.
        116         if self._blanklines == 'discard':
    --> 117             lines = [l for l in lines if l.rstrip()]
        118         elif self._blanklines == 'discard-eof':
        119             if lines and not lines[-1].strip():


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

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
0.256 +/- 0.00 {'features__text_pipeline__tfidf_vect__ngram_range': (1, 1), 'sclf__classifierchain-1__base_estimator__n_estimators': 2}
0.269 +/- 0.00 {'features__text_pipeline__tfidf_vect__ngram_range': (1, 1), 'sclf__classifierchain-1__base_estimator__n_estimators': 3}
0.252 +/- 0.00 {'features__text_pipeline__tfidf_vect__ngram_range': (1, 2), 'sclf__classifierchain-1__base_estimator__n_estimators': 2}
0.268 +/- 0.00 {'features__text_pipeline__tfidf_vect__ngram_range': (1, 2), 'sclf__classifierchain-1__base_estimator__n_estimators': 3}
Best parameters: {'features__text_pipeline__tfidf_vect__ngram_range': (1, 1), 'sclf__classifierchain-1__base_estimator__n_estimators': 3}
Training Accuracy: 0.27
```
</div>
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

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Classification Report for [1mrelated[0m:
              precision    recall  f1-score   support

           0       0.60      0.45      0.51      1245
           1       0.84      0.91      0.87      3998

    accuracy                           0.80      5243
   macro avg       0.72      0.68      0.69      5243
weighted avg       0.78      0.80      0.79      5243

Classification Report for [1mrequest[0m:
              precision    recall  f1-score   support

           0       0.90      0.97      0.93      4352
           1       0.75      0.46      0.57       891

    accuracy                           0.88      5243
   macro avg       0.82      0.72      0.75      5243
weighted avg       0.87      0.88      0.87      5243

Classification Report for [1moffer[0m:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5219
           1       0.00      0.00      0.00        24

    accuracy                           1.00      5243
   macro avg       0.50      0.50      0.50      5243
weighted avg       0.99      1.00      0.99      5243

Classification Report for [1maid_related[0m:
              precision    recall  f1-score   support

           0       0.73      0.87      0.79      3079
           1       0.74      0.54      0.62      2164

    accuracy                           0.73      5243
   macro avg       0.73      0.70      0.71      5243
weighted avg       0.73      0.73      0.72      5243

Classification Report for [1mmedical_help[0m:
              precision    recall  f1-score   support

           0       0.94      0.98      0.96      4808
           1       0.52      0.27      0.36       435

    accuracy                           0.92      5243
   macro avg       0.73      0.62      0.66      5243
weighted avg       0.90      0.92      0.91      5243

Classification Report for [1mmedical_products[0m:
              precision    recall  f1-score   support

           0       0.95      0.99      0.97      4964
           1       0.57      0.16      0.26       279

    accuracy                           0.95      5243
   macro avg       0.76      0.58      0.61      5243
weighted avg       0.93      0.95      0.94      5243

Classification Report for [1msearch_and_rescue[0m:
              precision    recall  f1-score   support

           0       0.97      1.00      0.99      5107
           1       0.27      0.03      0.05       136

    accuracy                           0.97      5243
   macro avg       0.62      0.51      0.52      5243
weighted avg       0.96      0.97      0.96      5243

Classification Report for [1msecurity[0m:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5147
           1       0.11      0.01      0.02        96

    accuracy                           0.98      5243
   macro avg       0.55      0.50      0.50      5243
weighted avg       0.97      0.98      0.97      5243

Classification Report for [1mmilitary[0m:
              precision    recall  f1-score   support

           0       0.98      0.99      0.98      5085
           1       0.41      0.20      0.26       158

    accuracy                           0.97      5243
   macro avg       0.69      0.59      0.62      5243
weighted avg       0.96      0.97      0.96      5243

Classification Report for [1mchild_alone[0m:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5243

    accuracy                           1.00      5243
   macro avg       1.00      1.00      1.00      5243
weighted avg       1.00      1.00      1.00      5243

Classification Report for [1mwater[0m:
              precision    recall  f1-score   support

           0       0.96      0.99      0.97      4908
           1       0.74      0.37      0.50       335

    accuracy                           0.95      5243
   macro avg       0.85      0.68      0.74      5243
weighted avg       0.94      0.95      0.94      5243

Classification Report for [1mfood[0m:
              precision    recall  f1-score   support

           0       0.95      0.98      0.96      4659
           1       0.79      0.58      0.67       584

    accuracy                           0.94      5243
   macro avg       0.87      0.78      0.82      5243
weighted avg       0.93      0.94      0.93      5243

Classification Report for [1mshelter[0m:
              precision    recall  f1-score   support

           0       0.94      0.98      0.96      4775
           1       0.68      0.41      0.51       468

    accuracy                           0.93      5243
   macro avg       0.81      0.69      0.74      5243
weighted avg       0.92      0.93      0.92      5243

Classification Report for [1mclothing[0m:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5173
           1       0.52      0.16      0.24        70

    accuracy                           0.99      5243
   macro avg       0.76      0.58      0.62      5243
weighted avg       0.98      0.99      0.98      5243

Classification Report for [1mmoney[0m:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5131
           1       0.51      0.20      0.28       112

    accuracy                           0.98      5243
   macro avg       0.75      0.60      0.64      5243
weighted avg       0.97      0.98      0.97      5243

Classification Report for [1mmissing_people[0m:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5180
           1       0.22      0.03      0.06        63

    accuracy                           0.99      5243
   macro avg       0.61      0.52      0.52      5243
weighted avg       0.98      0.99      0.98      5243

Classification Report for [1mrefugees[0m:
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      5073
           1       0.34      0.14      0.20       170

    accuracy                           0.96      5243
   macro avg       0.66      0.57      0.59      5243
weighted avg       0.95      0.96      0.96      5243

Classification Report for [1mdeath[0m:
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      4996
           1       0.66      0.38      0.49       247

    accuracy                           0.96      5243
   macro avg       0.82      0.69      0.73      5243
weighted avg       0.96      0.96      0.96      5243

Classification Report for [1mother_aid[0m:
              precision    recall  f1-score   support

           0       0.89      0.96      0.92      4551
           1       0.40      0.19      0.26       692

    accuracy                           0.86      5243
   macro avg       0.64      0.57      0.59      5243
weighted avg       0.82      0.86      0.83      5243

Classification Report for [1minfrastructure_related[0m:
              precision    recall  f1-score   support

           0       0.94      0.99      0.96      4907
           1       0.20      0.03      0.05       336

    accuracy                           0.93      5243
   macro avg       0.57      0.51      0.51      5243
weighted avg       0.89      0.93      0.91      5243

Classification Report for [1mtransport[0m:
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      5008
           1       0.52      0.10      0.16       235

    accuracy                           0.96      5243
   macro avg       0.74      0.55      0.57      5243
weighted avg       0.94      0.96      0.94      5243

Classification Report for [1mbuildings[0m:
              precision    recall  f1-score   support

           0       0.96      0.99      0.98      4974
           1       0.59      0.19      0.29       269

    accuracy                           0.95      5243
   macro avg       0.77      0.59      0.63      5243
weighted avg       0.94      0.95      0.94      5243

Classification Report for [1melectricity[0m:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5128
           1       0.58      0.13      0.21       115

    accuracy                           0.98      5243
   macro avg       0.78      0.56      0.60      5243
weighted avg       0.97      0.98      0.97      5243

Classification Report for [1mtools[0m:
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      5208
           1       0.50      0.03      0.05        35

    accuracy                           0.99      5243
   macro avg       0.75      0.51      0.53      5243
weighted avg       0.99      0.99      0.99      5243

Classification Report for [1mhospitals[0m:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5191
           1       0.00      0.00      0.00        52

    accuracy                           0.99      5243
   macro avg       0.50      0.50      0.50      5243
weighted avg       0.98      0.99      0.98      5243

Classification Report for [1mshops[0m:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5218
           1       0.50      0.04      0.07        25

    accuracy                           1.00      5243
   macro avg       0.75      0.52      0.54      5243
weighted avg       0.99      1.00      0.99      5243

Classification Report for [1maid_centers[0m:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5179
           1       0.08      0.02      0.03        64

    accuracy                           0.99      5243
   macro avg       0.53      0.51      0.51      5243
weighted avg       0.98      0.99      0.98      5243

Classification Report for [1mother_infrastructure[0m:
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      5018
           1       0.18      0.02      0.04       225

    accuracy                           0.95      5243
   macro avg       0.57      0.51      0.51      5243
weighted avg       0.92      0.95      0.94      5243

Classification Report for [1mweather_related[0m:
              precision    recall  f1-score   support

           0       0.86      0.95      0.90      3771
           1       0.82      0.61      0.70      1472

    accuracy                           0.85      5243
   macro avg       0.84      0.78      0.80      5243
weighted avg       0.85      0.85      0.85      5243

Classification Report for [1mfloods[0m:
              precision    recall  f1-score   support

           0       0.95      0.99      0.97      4812
           1       0.83      0.42      0.56       431

    accuracy                           0.95      5243
   macro avg       0.89      0.71      0.76      5243
weighted avg       0.94      0.95      0.94      5243

Classification Report for [1mstorm[0m:
              precision    recall  f1-score   support

           0       0.94      0.98      0.96      4764
           1       0.66      0.40      0.50       479

    accuracy                           0.93      5243
   macro avg       0.80      0.69      0.73      5243
weighted avg       0.92      0.93      0.92      5243

Classification Report for [1mfire[0m:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5190
           1       0.00      0.00      0.00        53

    accuracy                           0.99      5243
   macro avg       0.49      0.50      0.50      5243
weighted avg       0.98      0.99      0.98      5243

Classification Report for [1mearthquake[0m:
              precision    recall  f1-score   support

           0       0.96      0.99      0.98      4728
           1       0.87      0.64      0.74       515

    accuracy                           0.96      5243
   macro avg       0.92      0.82      0.86      5243
weighted avg       0.95      0.96      0.95      5243

Classification Report for [1mcold[0m:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5139
           1       0.69      0.19      0.30       104

    accuracy                           0.98      5243
   macro avg       0.84      0.60      0.65      5243
weighted avg       0.98      0.98      0.98      5243

Classification Report for [1mother_weather[0m:
              precision    recall  f1-score   support

           0       0.95      0.99      0.97      4976
           1       0.38      0.13      0.20       267

    accuracy                           0.95      5243
   macro avg       0.67      0.56      0.58      5243
weighted avg       0.93      0.95      0.93      5243

Classification Report for [1mdirect_report[0m:
              precision    recall  f1-score   support

           0       0.87      0.92      0.90      4233
           1       0.58      0.44      0.50      1010

    accuracy                           0.83      5243
   macro avg       0.73      0.68      0.70      5243
weighted avg       0.82      0.83      0.82      5243

```
</div>
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

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
micro F1-score: 0.6088721162062091
micro Precision: 0.7449255161599443
micro Recall: 0.5148413510747185
macro F1-score: 0.2949373610199553
macro Precision: 0.47367953797005047
macro Recall: 0.23426002104518323
Accuracy: 0.9417954097526862
```
</div>
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

