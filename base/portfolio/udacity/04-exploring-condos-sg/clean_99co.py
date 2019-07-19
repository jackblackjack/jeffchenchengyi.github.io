# Utilities
from datetime import date
from collections import *
import re
import numpy as np

# Dictionary of School Categories in Singapore
# and their associated terms
SCHOOL_CATEGORIES = {
    'primary': {'primary', 'school'}, 
    'secondary': {'secondary', 'school'}, 
    'polytechnic': {'polytechnic'},
    'pre-tertiary': {'junior', 'college'}, 
    'tertiary': {'university'}
}

def get_schools(commute_and_nearby_data):
    """
    Purpose:
    --------
    Get the count of the school types from commute_and_nearby_data (json)
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of the count of each type of school
    
    """
    def get_school_type(school_name_list):
        """
        Purpose:
        --------
        Gets the school type from the school_cat dictionary 
        
        Parameters:
        -----------
        school_name_list: (list) A list of the words inside the school name
        
        Returns:
        --------
        school_type: (str) The school type
        """
        for school_type, school_terms in SCHOOL_CATEGORIES.items():
            if len(school_terms.intersection(school_name_list)) == len(school_terms):
                return school_type
        return 'others'
    
    # If there are no schools nearby return None
    if len((commute_and_nearby_data.get('school', {}) if commute_and_nearby_data.get('school', {}) != None else {}).get('places', {})) == 0:
        return {}

    # Apply the get_school_type function to each
    # school in our list of schools nearby and use 
    # Counter to count them
    schools = Counter(
        list(
            map(
                lambda school_name_list: get_school_type(school_name_list), 
                [school_data['name'].lower().split() for school_data in commute_and_nearby_data['school']['places']]
            )
        )
    )
    
    return dict(schools)

def get_subway_stations(commute_and_nearby_data):
    """
    Purpose:
    --------
    Get the number of subways nearby and also the average duration to the nearest subway stations
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of number of subways nearby and also the average duration to the nearest subway stations
    """
    return {'num_subways_nearby': 0, 'average_transit_duration_to_subway': None} if len((commute_and_nearby_data.get('subway_station', {}) if commute_and_nearby_data.get('subway_station', {}) != None else {}).get('places', {})) == 0 else {
        'num_subways_nearby': len(commute_and_nearby_data['subway_station']['places']),
        'average_transit_duration_to_subway': np.mean([float(mrt_data['modes']['transit']['duration'].split('min')[0].strip()) for mrt_data in commute_and_nearby_data['subway_station']['places'] if 'transit' in mrt_data['modes'].keys()])
    }

def get_post_boxes(commute_and_nearby_data):
    """
    Purpose:
    --------
    Checks whether property has post boxes nearby
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of {'post_box_nearby': 0} or {'post_box_nearby': 1}
    
    """
    return {'post_box_nearby': int(len((commute_and_nearby_data.get('post_box', {}) if commute_and_nearby_data.get('post_box', {}) != None else {}).get('places', {})) > 0)}

def get_atms(commute_and_nearby_data):
    """
    Purpose:
    --------
    Checks whether property has atms nearby
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of {'atm_nearby': 0} or {'atm_nearby': 1}
    
    """
    return {'atm_nearby': int(len((commute_and_nearby_data.get('atm', {}) if commute_and_nearby_data.get('atm', {}) != None else {}).get('places', {})) > 0)}

def get_parks(commute_and_nearby_data):
    """
    Purpose:
    --------
    Checks whether property has parks nearby
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of {'park_nearby': 0} or {'park_nearby': 1}
    
    """
    return {'park_nearby': int(len((commute_and_nearby_data.get('park', {}) if commute_and_nearby_data.get('park', {}) != None else {}).get('places', {})) > 0)}

def get_supermarkets(commute_and_nearby_data):
    """
    Purpose:
    --------
    Get the number of supermarkets nearby and also the average duration to the nearest supermarkets
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of number of supermarkets nearby and also the average duration to the nearest supermarkets
    """
    return {'num_supermarkets_nearby': 0, 'average_transit_duration_to_supermarket': None} if len((commute_and_nearby_data.get('supermarket', {}) if commute_and_nearby_data.get('supermarket', {}) != None else {}).get('places', {})) == 0 else {
        'num_supermarkets_nearby': len(commute_and_nearby_data['supermarket']['places']),
        'average_transit_duration_to_supermarket': np.mean([float(supermarket_data['modes']['transit']['duration'].split('min')[0].strip()) for supermarket_data in commute_and_nearby_data['supermarket']['places'] if 'transit' in supermarket_data['modes'].keys()])
    }

def get_clinics(commute_and_nearby_data):
    """
    Purpose:
    --------
    Get the number of clinics nearby and also the average duration to the nearest clinics
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of number of clinics nearby and also the average duration to the nearest clinics
    """
    return {'num_clinics_nearby': 0, 'average_walking_duration_to_clinic': None} if len((commute_and_nearby_data.get('clinic', {}) if commute_and_nearby_data.get('clinic', {}) != None else {}).get('places', {})) == 0 else {
        'num_clinics_nearby': len(commute_and_nearby_data['clinic']['places']),
        'average_walking_duration_to_clinic': np.mean([float(clinic_data['modes']['walk']['duration'].split('min')[0].strip()) for clinic_data in commute_and_nearby_data['clinic']['places'] if 'walk' in clinic_data['modes'].keys()])
    }

def get_bus_stations(commute_and_nearby_data):
    """
    Purpose:
    --------
    Get the number of bus stations nearby and also the average duration to the nearest bus stations
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of number of bus stations nearby and also the average duration to the nearest bus stations
    """
    return {'num_bus_stations_nearby': 0, 'average_walking_duration_to_bus_station': None} if len((commute_and_nearby_data.get('bus_station', {}) if commute_and_nearby_data.get('bus_station', {}) != None else {}).get('places', {})) == 0 else {
        'num_bus_stations_nearby': len(commute_and_nearby_data['bus_station']['places']),
        'average_walking_duration_to_bus_station': np.mean([float(bus_station_data['modes']['walk']['duration'].split('min')[0].strip()) for bus_station_data in commute_and_nearby_data['bus_station']['places'] if 'walk' in bus_station_data['modes'].keys()])
    }

def get_post_offices(commute_and_nearby_data):
    """
    Purpose:
    --------
    Checks whether property has post offices nearby
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of {'post_office_nearby': 0} or {'post_office_nearby': 1}
    
    """
    return {'post_office_nearby': int(len((commute_and_nearby_data.get('post_office', {}) if commute_and_nearby_data.get('post_office', {}) != None else {}).get('places', {})) > 0)}

def get_banks(commute_and_nearby_data):
    """
    Purpose:
    --------
    Get the number of banks nearby and also the average duration to the nearest banks
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of number of banks nearby and also the average duration to the nearest banks
    """
    return {'num_banks_nearby': 0, 'average_transit_duration_to_bank': None} if len((commute_and_nearby_data.get('bank', {}) if commute_and_nearby_data.get('bank', {}) != None else {}).get('places', {})) == 0 else {
        'num_banks_nearby': len(commute_and_nearby_data['bank']['places']),
        'average_transit_duration_to_bank': np.mean([float(bank_data['modes']['transit']['duration'].split('min')[0].strip()) for bank_data in commute_and_nearby_data['bank']['places'] if 'transit' in bank_data['modes'].keys()])
    }

def get_commute(commute_and_nearby_data):
    """
    Purpose:
    --------
    Get the transit duration from property to each high traffic area
    
    Parameters:
    -----------
    commute_and_nearby_data: Dictionary of the commute and nearby data scraped from 99.co
    
    Returns:
    --------
    Dictionary of transit duration from property to each high traffic area
    """
    return {} if len((commute_and_nearby_data.get('commute', {}) if commute_and_nearby_data.get('commute', {}) != None else {}).get('places', {})) == 0 else {'transit_duration_to_{}'.format('_'.join(high_traffic_area['name'].lower().split())): float(high_traffic_area['modes']['transit']['duration'].split('min')[0].strip()) for high_traffic_area in commute_and_nearby_data['commute']['places'] if 'transit' in high_traffic_area['modes'].keys()}


def get_property_type(breadcrumbs):
    """
    Purpose:
    --------
    Get the property type from the breadcrumbs for cluster analysis
    
    Parameters:
    -----------
    property_feats: List of breadcrumbs on website
    
    Returns:
    --------
    Property type
    """
    for breadcrumb in breadcrumbs:
        if 'hdb' in breadcrumb:
            return 'hdb'
        elif 'land' in breadcrumb:
            return 'landed'
        elif 'condo' in breadcrumb:
            return 'condo'
    
    return 'others'


def get_cleaned_features(property_feats):
    """
    Purpose:
    --------
    Get the essential property features required for analysis
    
    Parameters:
    -----------
    property_feats: Dictionary of the raw property features scraped from 99.co
    
    Returns:
    --------
    Dictionary of essential property features
    """
    return {
        **{'type': get_property_type(property_feats.get('breadcrumbs', {}))},
        **{key: float(re.sub('[^0-9\.]', '', name)) for key, name in (property_feats.get('summary_feats', {}) if property_feats.get('summary_feats', {}) != None else {}).items() if re.sub('[^0-9\.]', '', name).isnumeric()}, 
        **{'price': float(re.sub('[^0-9\.]', '', property_feats.get('price', ''))) if re.sub('[^0-9\.]', '', property_feats.get('price', '')) else None}, 
        **(property_feats.get('key_details', {}) if property_feats.get('key_details', {}) != None else {}),
        **{'ammenities': len((property_feats.get('ammenities', {}) if property_feats.get('ammenities', {}) != None else {}))},
        **{
            'total_units': int(re.sub('[^0-9\.]', '', (property_feats.get('project_overview', {}) if property_feats.get('project_overview', {}) != None else {}).get('total units', ''))) if re.sub('[^0-9\.]', '', (property_feats.get('project_overview', {}) if property_feats.get('project_overview', {}) != None else {}).get('total units', '')) else None, 
            'year_since_completion': int(date.today().year) - int(re.sub('[^0-9\.]', '', (property_feats.get('project_overview', {}) if property_feats.get('project_overview', {}) != None else {}).get('year of completion', ''))) if re.sub('[^0-9\.]', '', (property_feats.get('project_overview', {}) if property_feats.get('project_overview', {}) != None else {}).get('year of completion', '')) else None,
            'tenure': (property_feats.get('project_overview', {}) if property_feats.get('project_overview', {}) != None else {}).get('tenure', '').strip(),
            'link': (property_feats.get('project_overview', {}) if property_feats.get('project_overview', {}) != None else {}).get('link', None)
        }
    }