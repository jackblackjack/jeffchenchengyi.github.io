import json
import plotly
import pandas as pd
import numpy as np
from itertools import combinations

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
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

X, y, category_names = load_data('../data/DisasterResponse.db')

# load model
model = joblib.load("../models/classifier.pkl")

# Creating graphs for visualization
cat_counts = pd.DataFrame(y.sum(axis=0).sort_values(), columns=['count'])

# Get the pairs of categories with 
# the highest cosine similarity
cat1, cat2, cosine_sim = list(zip(*sorted([(cat1, cat2, np.dot(y[cat1], y[cat2]) / (np.linalg.norm(y[cat1]) * np.linalg.norm(y[cat2]))) \
        if np.linalg.norm(y[cat1]) * np.linalg.norm(y[cat2]) > 0
        else (cat1, cat2, 0)
    for cat1, cat2 in combinations(y.columns, 2) \
], key=lambda val: val[2], reverse=True)))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_counts.index,
                    y=cat_counts['count']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=cat_counts.index, 
                    values=cat_counts['count']
                )
            ],

            'layout': {
                'title': 'Pie Chart of Category Breakdown'
            }
        },
        {
            'data': [
                Bar(
                    x=[x1 + ' and ' + x2 for x1, x2 in zip(cat1, cat2)][:20],
                    y=cosine_sim[:20],
                    orientation = 'v'
                )
            ],

            'layout': {
                'title': 'Top 50 Most Similiar Categories (Based on Cosine Similarity)',
                'yaxis': {
                    'title': "Cosine Similarity"
                },
                'xaxis': {
                    'title': "Category Pairs"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(y.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()