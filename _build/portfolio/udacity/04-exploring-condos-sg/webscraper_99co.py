# General Utilities for Web Scraping
import sys
import os
from os.path import join
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import date
import csv
import json

# Machine Learning Utitilies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Url
base_url = 'https://www.99.co/singapore/sale'

# Helper function to create a new folder
def mkdir(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
        else:
            print("(%s) already exists" % (path))

def get_property_links(url):
    """
    Purpose:
    --------
    Scrapes the for-sale listings on 99.co to get all the unique property weblinks on 99.co
    
    Parameters:
    -----------
    url: (str) Base Url for us to start scraping
    
    Returns:
    --------
    property_ids: (list) List of all the web links to the individual residential properties for sale on 99.co 
    """
    property_links = []
    for page_idx in tqdm(range(1, 401)):
        query = '?page_num={}'.format(page_idx)
        response = requests.get(url+query)
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_listings = soup.findAll(name='a', attrs={'data-click-id':'listing-item'})
            for list_idx in range(len(page_listings)):
                property_links.append(join(url, 'property', page_listings[list_idx].attrs['href'][page_listings[list_idx].attrs['href'].rfind('/')+1:]))
                
    return property_links

def get_cluster_id(url):
    """
    Purpose:
    --------
    Retrieve cluster_id required to access 99.co API from the web link of the house
    
    Parameters:
    -----------
    url: (str) The url for the specific house
    
    Returns:
    --------
    cluster_id, e.g. detQxccQq5tXJQj5jfELdTq8
    """
    # Try getting the cluster id for 50 times before we give up
    for _ in range(50):
        response = requests.get(url)
        if response.ok:
            if 'cluster_id' in response.text:
                cluster_pos = response.text.find('cluster_id')
                return response.text[cluster_pos:cluster_pos+50].split("\"")[2]
            elif 'clusterId' in response.text:
                cluster_pos = response.text.find('clusterId')
                return response.text[cluster_pos:cluster_pos+50].split("\"")[2]
            else:
                print('Cluster Id not present ...')
                return None
    print('Max retries exceeded ...')
    return None

def get_transact_history(cluster_id, transaction_type='sale'):
    """
    Purpose:
    --------
    To get the full transaction history of a house given the 
    specific cluster_id for the house
    
    Parameters:
    -----------
    cluster_id: (str) A specific cluster_id for the house created by 99.co, e.g. detQxccQq5tXJQj5jfELdTq8
    transaction_type: (str) Either 'sale' or 'rent' transaction history
    
    Returns:
    --------
    Dataframe containing the transaction histroy with 5 columns: 
    Date, Block, Unit, Area, Price (psf) if 'sale'
    Date, Area, Price (psf) if 'rent'
    """
    # Some error checking
    valid_transaction_type = {'sale', 'rent'}
    if transaction_type not in valid_transaction_type:
        raise ValueError("results: valid_transaction_type must be one of %r." % valid_transaction_type)
    
    # Retrieve Transaction table using API
    api_url = 'https://www.99.co/api/v1/web/clusters/{0}/transactions/table/history?transaction_type={1}&page_size={2}'.format(cluster_id, transaction_type, 10000)
    response = requests.get(api_url)
    if response.ok:
        transactions_data = json.loads(response.text)['data']
        headers = [header['title'] for header in transactions_data['headers']]
        rows = [[datum['title'] for datum in row] for row in transactions_data['rows']]
        return pd.DataFrame(data=rows, columns=headers)
    
def get_commute_and_nearby_data(cluster_id):
    """
    Purpose:
    --------
    To retrieve the json data of the commute and nearby information
    of the properties through the 99.co internal API
    
    Parameters:
    -----------
    cluster_id: (str) A specific cluster_id for the house created by 99.co, e.g. detQxccQq5tXJQj5jfELdTq8
    
    Returns:
    --------
    Dictionary of the commute and nearby information of the cluster specified by id
    """
    # Commute Data
    commute_url = 'https://www.99.co/mobile/v3/ios/clusters/{0}/sections/commute'.format(cluster_id)
    commute_response = requests.get(commute_url)
    if commute_response.ok and len(json.loads(commute_response.text)['data']):
        commute_data = json.loads(commute_response.text)['data']               
    else:
        commute_data = None
    
    # Nearby Data
    nearby_url = 'https://www.99.co/api/v1/web/clusters/{0}/nearby'.format(cluster_id)
    nearby_response = requests.get(nearby_url)
    if nearby_response.ok and len(json.loads(nearby_response.text)['data']):
        if 'categories' in json.loads(nearby_response.text)['data'].keys():
            nearby_data = json.loads(nearby_response.text)['data']['categories'] 
        else:
            nearby_data = None
    else:
        nearby_data = None
    
    # Aggregate of both
    categories = {category['key']: category['data'] for category in nearby_data}
    categories['commute'] = commute_data
    return categories

def get_property_feats(url):
    """
    Purpose:
    --------
    To get all the key property details from the webpage
    
    Parameters:
    -----------
    url: (str) Url for the link to the property webpage
    
    Returns:
    --------
    Dictionary with all the key information
    
    """
    # Dictionary to store all the property features found
    feats = {}
    
    # Get Webpage
    response = requests.get(url)
    if response.ok:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Property type
        breadcrumbs = soup.findAll('li', {'class': 'Breadcrumbs__listItem__ENKNt'})
        feats['breadcrumbs'] = [breadcrumb.text.lower() for breadcrumb in breadcrumbs]
        
        # Property Name
        left_column_summary, left_column_content = soup.findAll(name='div', attrs={'class': 'Listing__leftColumn__3k7xe'})
        right_column_summary = soup.findAll(name='div', attrs={'class': 'Listing__rightColumn__1YqJU Listing__summaryRight__2wU__'})[0]

        property_name_raw = left_column_summary.findChildren(name='h1')
        if len(property_name_raw):
            feats['property_name'] = property_name_raw[0].text
        else:
            feats['property_name'] = None

        # Nearest MRT to property
        nearest_mrt_raw = left_column_summary.findChildren(name='p', attrs={'class': 'Text__text__x0JSc NearestMrt__title__iVNyM'})
        if len(nearest_mrt_raw):
            feats['nearest_mrt'] = nearest_mrt_raw[0].text
        else:
            feats['nearest_mrt'] = None

        # Location
        location_raw = left_column_summary.findChildren(name='p', attrs={'class': 'Text__text__x0JSc'})
        if len(location_raw):
            feats['location'] = location_raw[0].text
        else:
            feats['location'] = None

        # Summary Features
        summary_feats_raw = left_column_summary.findChildren(name='p', attrs={'class': 'Text__text__x0JSc Listing__summaryText__1QR5z'})
        if len(summary_feats_raw):
            feats['summary_feats'] = {['num_beds', 'num_baths', 'sqft', 'psf'][idx]: summary_feats_raw[idx].text for idx in range(len(summary_feats_raw))}
        else:
            feats['summary_feats'] = None

        # Price of property
        price_raw = right_column_summary.findChildren(name='h3', attrs={'class': 'Heading__heading__2ncUp'})
        if len(price_raw):
            feats['price'] = price_raw[0].text
        else:
            feats['price'] = None

        # Key Details
        key_details_name = left_column_content.findChildren(name='div', attrs={'class': 'Tag__tag__1ngVj'})
        key_details_data = left_column_content.findChildren(name='p', attrs={'class': 'Text__text__x0JSc'})
        feats['key_details'] = {key.text: value.text for key, value in zip(key_details_name, key_details_data)}

        # Ammenities
        feats['ammenities'] = [data.text for data in left_column_content.findChildren(name='p', attrs={'class': 'Text__text__x0JSc Listing__amenityLabel__CQblY'})]

        # Project Overview
        project_overview_raw = soup.findAll(name='div', attrs={'ProjectOverviewCard__container__3SC53'})
        if len(project_overview_raw):
            project_overview = {data.text.split(':')[0].lower(): data.text.split(':')[1] for data in project_overview_raw[0].findChildren(name='p', attrs={'class': 'Text__text__x0JSc'})}
            attrs = project_overview_raw[0].findChildren(name='a', attrs={'class': 'Link__link__2aXf0'})[0].attrs
            if 'href' in attrs.keys():
                project_overview['link'] = base_url[:base_url.find('singapore')-1] + attrs['href']
                feats['project_overview'] = project_overview
            else:
                feats['project_overview'] = None
        else:
            feats['project_overview'] = None
        
    return feats

def get_price_trends(cluster_id):
    """
    Purpose:
    --------
    To get the price trends for each property from the 99.co internal API
    
    Parameters:
    -----------
    cluster_id: (str) A specific cluster_id for the house created by 99.co, e.g. detQxccQq5tXJQj5jfELdTq8
    
    Returns:
    --------
    Dictionary with price trend information about the property specified by cluster id
    """
    # Dictionary for price trends
    price_trends = {}
    
    # Options for query string
    transaction_type_options = ['sale', 'rent']
    time_frame_options = ['1m', '3m', '6m', '1y', '5y', '10y', 'all']
    bedroom_options = ['0', '1', '2', '3', '4', '5%2C6%2C7%2C8%2C9%2C10'] # 0: Studio apartment, 5%2C6%2C7%2C8%2C9%2C10: 5+ bedrooms
    
    # Key Statistics
    key_stats = {}
    for transaction_type in transaction_type_options:
        for time_frame in time_frame_options:
            response = requests.get('https://www.99.co/api/v1/web/clusters/{0}/transactions/key-stats?transaction_type={1}&time_frame={2}'.format(cluster_id, transaction_type, time_frame))
            if response.ok and len(json.loads(response.text)['data']):
                key_stats['{}_{}'.format(transaction_type, time_frame)] = json.loads(response.text)['data']
            else:
                key_stats['{}_{}'.format(transaction_type, time_frame)] = None
    price_trends['key_stats'] = key_stats
    
    # Transaction Volumes
    transaction_vol = {}
    for transaction_type in transaction_type_options:
        for time_frame in time_frame_options:
            for bedrooms in bedroom_options:
                response = requests.get('https://www.99.co/api/v1/web/clusters/{0}/transactions/chart/bar?transaction_type={1}&value=volume&time_frame={2}&bedrooms={3}'.format(cluster_id, transaction_type, time_frame, bedrooms))
                if response.ok and len(json.loads(response.text)['data']):
                    transaction_vol['{}_{}_{}'.format(transaction_type, time_frame, bedrooms[0])] = json.loads(response.text)['data']
                else:
                    transaction_vol['{}_{}_{}'.format(transaction_type, time_frame, bedrooms[0])] = None
    price_trends['transaction_vol'] = transaction_vol
    
    # Transaction Prices
    transaction_price = {}
    for transaction_type in transaction_type_options:
        for time_frame in time_frame_options:
            for bedrooms in bedroom_options:
                response = requests.get('https://www.99.co/api/v1/web/clusters/{0}/transactions/chart/line?transaction_type={1}&value=avg_price&time_frame={2}&bedrooms={3}'.format(cluster_id, transaction_type, time_frame, bedrooms))
                if response.ok and len(json.loads(response.text)['data']):
                    transaction_price['{}_{}_{}'.format(transaction_type, time_frame, bedrooms[0])] = json.loads(response.text)['data']
                else:
                    transaction_price['{}_{}_{}'.format(transaction_type, time_frame, bedrooms[0])] = None
    price_trends['transaction_price'] = transaction_price
    
    return price_trends
            
def get_residential_property_info(url, path='./data/99.co/data/{}'.format(date.today().strftime("%Y_%m_%d"))):
    """
    Purpose:
    --------
    Given the Web URL of the property, retrieve all the 
    information available about it from 99.co and save it into csv and json
    to be used later
    
    Parameters:
    -----------
    url: (str) Url of the property
    path: (str) Directory to save the data collected to
    
    Returns:
    --------
    Nothing, saves everything to either csv or json
    """
    # Unique key to the property listing
    key = url[url.rfind('/')+1:]
    
    # Create directory specific to property listing
    path = join(path, key)
    mkdir(path)
    
    # Cluster ID to use 99.co API
    cluster_id = get_cluster_id(url=url)
    
    # Property Information
    if cluster_id != None:
#         transaction_history_rent = get_transact_history(cluster_id=cluster_id, transaction_type='rent')
#         transaction_history_rent.to_csv(join(path, 'transaction_history_rent.csv'))

#         transaction_history_sale = get_transact_history(cluster_id=cluster_id, transaction_type='sale')
#         transaction_history_sale.to_csv(join(path, 'transaction_history_sale.csv'))

#         price_trends = get_price_trends(cluster_id=cluster_id)
#         with open(join(path, 'price_trends.json'), 'w') as json_file:
#             json.dump(price_trends, json_file)

        property_feats = get_property_feats(url=url)
        with open(join(path, 'property_feats.json'), 'w') as json_file:
            json.dump(property_feats, json_file)

        commute_and_nearby_data = get_commute_and_nearby_data(cluster_id=cluster_id)
        with open(join(path, 'commute_and_nearby_data.json'), 'w') as json_file:
            json.dump(commute_and_nearby_data, json_file)
    else:
        print('cluster_id for {} cannot be found.'.format(key))
        
def get_all_property_listing_data(df):
    """
    Purpose:
    --------
    Get's all the property listing information for 
    today
    
    Parameters:
    -----------
    df: (Dataframe) Pandas Dataframe of all the property listing urls
    
    Returns:
    --------
    Nothing. Makes a function call to get_residential_property_info for each url scraped
    """
    for url in tqdm(df['url']):
        get_residential_property_info(url)