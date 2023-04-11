# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%

import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.distance import distance

df = pd.read_csv('Building_Permits_in_2023.csv')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE))
geolocator = Nominatim(user_agent="my_app")
address = input("Enter the address (street, city, state, zip): ")
location = geolocator.geocode(address)
input_latitude = location.latitude
input_longitude = location.longitude
input_point = (input_latitude, input_longitude)

radius = 300  #meters

distances = gdf['geometry'].apply(lambda x: distance(input_point, (x.y, x.x)).meters)
nearby_gdf = gdf[distances <= radius]

construction_nearby = len(nearby_gdf)
print (construction_nearby)

# %%
