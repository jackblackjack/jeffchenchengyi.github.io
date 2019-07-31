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
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.externals import joblib

# To handle imbalanced data
from imblearn.over_sampling import SMOTE

# To help us stack models
from mlxtend.classifier import StackingClassifier

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

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


def build_model():
    """
    Function:
    ---------
    Uses SMOTE Oversampling technique from imblearn
    to balance our dataset, builds model with a variety of base learners,
    uses ClassifierChains and ensembles them by stacking a
    meta learner on top, with grid search implemented
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    Model ready to be fitted
    """
    print('=============================')  
    print('Building Model:')
    print('-----------------------------')

    # Aggregate an ensemble of RandomForest classifier chains and feed them
    # to the meta classifier
    print('Creating ClassifierChains...')
    chains = [ClassifierChain(base_estimator=RandomForestClassifier(n_estimators=100), order='random', random_state=42) for _ in range(5)]

    # Meta Classifier that will take the predictions
    # of each output of the classifier chains and figure out
    # the weight of each classifier in predicting labels
    print('Adding Meta Classifier...')
    meta_clf = MultiOutputClassifier(AdaBoostClassifier())

    # Stack the base learners 
    print('Stacking Meta Classifier on top of ClassifierChains...')
    sclf = StackingClassifier(classifiers=chains,
                              meta_classifier=meta_clf)

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
        'features__text_pipeline__tfidf_vect__ngram_range': ((1, 2), (1, 10))
    }

    print('Initializing GridSearchCV...')
    model = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    return model


def multilabel_process_train(X_train, y_train):
    """
    Function:
    ---------
    Trains the multilabel disaster response model
    with a custom Pipeline that includes the following:
        - Ensemble of Classifier chains of varying classification 
          models to account for label dependencies
        - Stacking classifier chains by using a meta classifier to 
          find the best weight for each classifier chain
        - Grid Search with CV for best model
           
    Parameters:
    -----------
    X_train: Pandas Dataframe of the messages from the disaster
             response dataset
       
    y_train: Pandas Dataframe of the labels for the 
             messages
    
    Returns:
    --------
    Final trained model 
    """
    model = build_model()
    
    print('=============================')  
    print('Training Model:')
    print('-----------------------------')
    model.fit(X_train, y_train)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(model.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (model.cv_results_[cv_keys[0]][r],
                 model.cv_results_[cv_keys[1]][r] / 2.0,
                 model.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % model.best_params_)
    print('Training Accuracy: %.2f' % model.best_score_)
    return model
    
    
def evaluate_model(model, X_test, y_test, category_names):
    """
    Function:
    ---------
    Prints individual classification reports for each
    class in the disaster response categories and an overall
    summary of the Macro and micro f1, recall, and precision scores
           
    Parameters:
    -----------
    model: Model to be used for prediction
    
    X_test: Pandas Dataframe of the messages from the disaster 
            response dataset  
            
    y_test: Pandas Dataframe of the labels for the 
            messages
            
    category_names: List of the disaster response classes so that
                    we can print a classification report for each
    
    Returns:
    --------
    Final trained model
    """
    print('=============================')  
    print('Evaluating Model:')
    print('-----------------------------')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')  
    print('Classification Scores:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    y_pred = model.predict(X_test)

    for idx, label in enumerate(category_names):
        print("Classification Report for {}:".format(bold(label)))
        print(
            classification_report(
                y_true=np.array(y_test)[:, idx], 
                y_pred=y_pred[:, idx]
            )
        )
        
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Aggregate Summary Statistics:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for average_type in ['micro', 'macro']:
        print('{} F1-score: {}'.format(average_type, f1_score(y_test, y_pred, average=average_type)))
        print('{} Precision: {}'.format(average_type, precision_score(y_test, y_pred, average=average_type)))
        print('{} Recall: {}'.format(average_type, recall_score(y_test, y_pred, average=average_type)))
    
    accuracy = (y_pred == y_test).mean()
    print("Test Accuracy:", accuracy)


def save_model(model, model_filepath):
    print('=============================')  
    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    print('-----------------------------')
    # Save to file in the current working directory  
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Build and Train Model
        model = multilabel_process_train(X_train, y_train)
        try:
            # Evaluate Model
            evaluate_model(model, X_test, y_test, category_names)
        finally:
            # Save Model
            save_model(model, model_filepath)
            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()