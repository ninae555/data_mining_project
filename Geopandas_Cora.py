# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%

import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from geopy.distance import distance

# Load building data and create GeoDataFrame
building = pd.read_csv('BpData_after_cleaning_without_Nan.csv')

# Get address from user and geocode it
gdf = gpd.GeoDataFrame(building, geometry=gpd.points_from_xy(building.LONGITUDE, building.LATITUDE))
geolocator = Nominatim(user_agent='my_app')
address = input ("Enter the address: ")
location = geolocator.geocode(address)
input_latitude = location.latitude
input_longitude = location.longitude
input_point =(input_longitude, input_latitude)

# Define the permit noise levels dictionary
permit_noise_levels = {'DEMOLITION': 4, 'EXCAVATION ONLY': 4, 'SHEETING AND SHORING': 4, 'RAZE': 4,
                       'NEW BUILDING': 4, 'BUILDING': 4, 'SPECIAL BUILDING': 4, 'CIVIL PLANS': 4,
                       'ELEVATOR - REPAIR': 3, 'ELEVATOR - NEW': 3, 'ELEVATOR - ALTERATION': 3,
                       'MECHANICAL': 3, 'DECK': 3, 'GARAGE': 3, 'SWIMMING POOL': 3,
                       'ELECTRICAL - HEAVY UP': 2, 'BOILER': 2, 'PLUMBING AND GAS': 2,
                       'FOUNDATION ONLY': 2, 'ADDITION ALTERATION REPAIR': 2, 'ELECTRICAL': 2,
                       'ELECTRICAL - GENERAL': 2, 'SOLAR SYSTEM': 2, 'GAS FITTING': 2,
                       'TENANT LAYOUT': 2, 'FENCE': 2, 'AWNING': 1, 'RETAINING WALL': 1,
                       'SIGN': 1, 'SHED': 1}

# Define the permit_count function to count permits of a given type within a radius of a point
def permit_count(radius, permit_type, input_point, gdf):
    """
    Count of a given permit_type within a circle of a given radius from the input_point.
    """
    distances = gdf['geometry'].apply(lambda x: distance(input_point, (x.y, x.x)).meters)
    nearby_gdf = gdf[distances <= radius]
    permit_counts = nearby_gdf['PERMIT_SUBTYPE_NAME'].value_counts()
    if permit_type in permit_counts:
        return permit_counts[permit_type]
    else:
        return 0




# %%
