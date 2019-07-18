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

from webscraper_99co import get_property_links, get_all_property_listing_data

# Read in the property links csv
property_df = pd.read_csv('./data/99.co/property_links.csv', index_col=[0])
get_all_property_listing_data(property_df)