# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.distance import distance
import folium

df = pd.read_csv('/Users/coramartin/Documents/GitHub/data_mining_project/Data/BpData_after_cleaning_with_Nan.csv')
df['PERMIT_SUBTYPE_NAME'] = df['PERMIT_SUBTYPE_NAME'].replace(['MISCELLANEOUS', 'nan', 'PERMIT'], 'MISCELLANEOUS')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE))
geolocator = Nominatim (user_agent='my app')
address = input("Enter the address: ")
location = geolocator.geocode(address)
input_latitude = location.latitude
input_longitude = location.longitude
input_point = (input_latitude, input_longitude)

#%%
# See Annex A for a categorization of construction based off of noise
# NOISE SCORE
distances = gdf['geometry'].apply(lambda x: distance(input_point, (x.y, x.x)).feet)
nearby_gdf = gdf[distances <= 200]
permit_subtype_counts = nearby_gdf["PERMIT_SUBTYPE_NAME"].value_counts()
print("The permits within a radius of 200 feet from your property include: \n", permit_subtype_counts.to_string(header=False))

## NOISE SCORE SECTION

# Green marker indicates the inputted address
# Red markers indicate the permits within the given radius
center = (input_point[0], input_point[1])
map = folium.Map(location=center, zoom_start=16)
folium.Marker(location=center, icon=folium.Icon(color='green')).add_to(map)
radius_m = 200 * 0.3048
circle = folium.Circle(location=center, radius=radius_m, color='blue', fill_opacity=0.2)
circle.add_to(map)
for _, row in nearby_gdf.iterrows():
    folium.Marker(location=[row['LATITUDE'], row['LONGITUDE']], icon=folium.Icon(color='red')).add_to(map)
map

#%%
# NEIGHBORHOOD SCORE

distances = gdf['geometry'].apply(lambda x: distance(input_point, (x.y, x.x)).feet)
nearby_gdf = gdf[distances <= 800]
permit_counts = nearby_gdf["PERMIT_SUBTYPE_NAME"].value_counts()
print("The permits within a radius of 500 feet from your property include: \n", permit_counts.to_string(header=False))

## NOISE SCORE SECTION

# Green marker indicates the inputted address
# Red markers indicate the permits within the given radius
center = (input_point[0], input_point[1])
map = folium.Map(location=center, zoom_start=16)
folium.Marker(location=center, icon=folium.Icon(color='green')).add_to(map)
radius_m = 800 * 0.3048
circle = folium.Circle(location=center, radius=radius_m, color='blue', fill_opacity=0.2)
circle.add_to(map)
for _, row in nearby_gdf.iterrows():
    folium.Marker(location=[row['LATITUDE'], row['LONGITUDE']], icon=folium.Icon(color='red')).add_to(map)
map

# %%
### ANNEX A
## VERY LOUD 
# BUILDING
# DEMOLITION
# GARAGE
# SPECIAL BUILDING
# EXCAVATION ONLY
# NEW BUILDING
# SHEETING AND SHORING
# FOUNDATION ONLY
# RAZE
# CIVIL PLANS
#
## LOUD
# PLUMBING AND GAS
# TENANT LAYOUT
# ALTERATION AND REPAIR
# ELEVATOR - ALTERATION
# ELEVATOR - NEW
# ELEVATOR - REPAIR
# ADDITION ALTERATION REPAIR
# DECK
# RETAINING WALL
# PLUMBING
# SWIMMING POOL
#
## SOME NOISE
#
# ELECTRICAL
# MECHANICAL
# ELECTRICAL - GENERAL
# EXPEDITED
# FENCE
# BOILER
# GAS FITTING
# SOLAR SYSTEM
# ELECTRICAL - HEAVY UP
# PLUMBING
# VARIANCE
# SHED
# MISCELLANEOUS
#
## NO NOISE
# SIGN
# AWNING


# %%
