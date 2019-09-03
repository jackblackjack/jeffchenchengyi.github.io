---
redirect_from:
  - "/portfolio/dengai/readme"
title: 'DengAI'
prev_page:
  url: /portfolio/udacity/07-datascience-capstone/Sparkify
  title: 'Predicting Churn Rates for "Sparkify"'
next_page:
  url: /portfolio/dengai/00-dengai
  title: '00 - Data Preprocessing'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---
# DengAI: Predicting Disease Spread
HOSTED BY DRIVENDATA

<a href='https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/'><img src='https://media.giphy.com/media/TXOj4fWZR83XW/giphy.gif' style='border: 5px solid black; border-radius: 5px;'/></a>

### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [Results](#results)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

Your goal is to predict the `total_cases` label for each (`city`, `year`, `weekofyear`) in the test set. There are two cities, San Juan and Iquitos, with test data for each city spanning 5 and 3 years respectively. You will make one submission that contains predictions for both cities. The data for each city have been concatenated along with a `city` column indicating the source: `sj` for San Juan and `iq` for Iquitos. The test set is a pure future hold-out, meaning the test data are sequential and non-overlapping with any of the training data. Throughout, missing values have been filled as `NaNs`. You can find the list of features [here](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/82/#features_list).

## File Descriptions <a name="files"></a>

There are multiple notebooks available here to showcase work related to the competition:
1. [00 - Data Preprocessing](https://jeffchenchengyi.github.io/portfolio/dengai/00-dengai.html)
2. [01 - Naive Regressors](https://jeffchenchengyi.github.io/portfolio/dengai/01-dengai.html)
3. [02 - Gauss-Markov Assumptions (OLS)](https://jeffchenchengyi.github.io/portfolio/dengai/02-dengai.html)
4. [03a - Feature Scaling and PCA](https://jeffchenchengyi.github.io/portfolio/dengai/03a-dengai.html)
5. [03b - Reducing number of PCA components](https://jeffchenchengyi.github.io/portfolio/dengai/03b-dengai.html)
6. [04 - Looking at the Benchmark](https://jeffchenchengyi.github.io/portfolio/dengai/04-dengai.html)
7. [05 - TPOT](https://jeffchenchengyi.github.io/portfolio/dengai/05-dengai.html)
8. [06a - Time Series Analysis](https://jeffchenchengyi.github.io/portfolio/dengai/06a-dengai.html)
9. [06b - Time Series Analysis (Continued)](https://jeffchenchengyi.github.io/portfolio/dengai/06b-dengai.html)
10. [07 - Eliminating Seasonality \[UNDER CONSTRUCTION\]](https://jeffchenchengyi.github.io/portfolio/dengai/07-dengai.html)

## Results<a name="results"></a>

So far, the best results we've got is an **MAE: 26.057** through training a Support Vector Regressor on a subset of de-seasonalized features.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to DRIVENDATA for the dengue data of San Juan and Iquitos.
