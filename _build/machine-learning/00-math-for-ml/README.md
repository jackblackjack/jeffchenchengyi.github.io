---
redirect_from:
  - "/machine-learning/00-math-for-ml/readme"
title: 'Math for Machine Learning'
prev_page:
  url: /machine-learning/08-genetic-algorithms/saga-fpga
  title: 'Evolutionary Algorithms on FPGAs'
next_page:
  url: /machine-learning/00-math-for-ml/calculus
  title: 'Calculus'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---
# Exploring Housing Prices in Singapore

<img src="https://thesmartlocal.com/wp-content/uploads/2014/09/images_easyblog_images_2088_Beautiful-Homes_Hillside-House-1.jpg" />

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

You can install all required packages to run this project through `pip install -r requirements.txt`.

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in scraping property listing data from www.99.co to better understand:

1. The Distribution of Property listings by districts and whether the clusters we form by clustering the property listings by their features such as `sqft` and `number of bus stops nearby` will correspond to the 28 designated districts in Singapore.
2. What are the most important features for predicting the price of a property?
3. What are the properties with the largest land plots in the District 9, 10, 11 (Central) area of Singapore?

## File Descriptions <a name="files"></a>

There are 4 notebooks available here to showcase work related to the above questions:
1. `exploring-house-prices-singapore-part-1-extract-transform.ipynb` - Part 1 of the ETL operations (Webscraping www.99.co)
2. `exploring-house-prices-singapore-part-2-transform-load.ipynb` - Part 2 of the ETL operations (Transforming and Saving the files to `.csv` for later use)
3. `exploring-house-prices-singapore-part-3-crispdm.ipynb` - The main analysis with all the code
4. `exploring-house-prices-singapore-part-3-crispdm-non-technical` - Analysis without the code

Along with the notebooks, we also have a few scripts such as `clean_99co.py`, `scrape_99co.py`, and `webscraper_99co.py` that contain the functions required for scraping, cleaning, and transforming the data from 99.co for use in our machine learning models and analysis.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://jeffchenchengyi.github.io/portfolio/udacity/04-exploring-condos-sg/exploring-house-prices-singapore-part-3-crispdm-non-technical.html).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to www.99.co for the data.
