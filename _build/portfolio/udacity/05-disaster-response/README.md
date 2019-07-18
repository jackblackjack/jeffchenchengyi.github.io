---
redirect_from:
  - "/portfolio/udacity/05-disaster-response/readme"
title: 'Disaster Response'
prev_page:
  url: /portfolio/udacity/04-exploring-condos-sg/exploring-house-prices-singapore-part-3-crispdm
  title: 'Exploring Condominiums in Singapore'
next_page:
  url: /portfolio/udacity/06-ibm-recommendation-engine/Recommendations_with_IBM
  title: 'Recommendation Systems for Articles with IBM'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---
# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
