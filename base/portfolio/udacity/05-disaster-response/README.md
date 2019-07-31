# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. You can install all required packages to run this project through `pip install -r requirements.txt`.

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

For this project, we will be working with disaster data from Figure Eight to build a model for an API that classifies disaster messages and incorporate this model in a web application using Flask.

## File Descriptions <a name="files"></a>

In our `workspace/` folder, we have 2 notebooks:
    1. `ETL_Pipeline_Preparation.ipynb` - Contains the ETL operations to get a cleaned dataset to be used for the NLP task.
    2. `ML_Pipeline_Preparation.ipynb` - The ML pipeline used for classifying the text into one of the disaster response categories.

Our `app/` folder contains the files that are required to launch the flask dashboard.

In our `data/` folder, we have `process_data.py` to perform the ETL operations to prepare the data for the Machine learning model

In our `models/` folder, we have:
    1. `train_classifier.py` - Used to train a classifier on the data from `data/` to predict the disaster response categories given a message
        - Classifier Details: Cross Validated set of 5 Random Forest Classifier Chains and an Adaboost meta classifier for the final multilabel prediction
    2. `classifier.pkl` - The trained model

## Results<a name="results"></a>

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This project is part of Udacity's Data Science Nano Degree Term 2 program, and the disaster response data used is all provided by Udacity and Figure Eight.
