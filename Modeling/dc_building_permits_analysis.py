#!/usr/bin/env python
# coding: utf-8

# %%

import os

import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from folium.plugins import FloatImage, HeatMap
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# %%

# Get the path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# Construct the file paths to the CSV files
csv_file_path_2022 = os.path.join(script_dir, "..", "data", "Building_Permits_in_2022.csv")
csv_file_path_2023 = os.path.join(script_dir, "..", "data", "Building_Permits_in_2023.csv")

# Read the CSV files into DataFrames
building_permits_2022 = pd.read_csv(csv_file_path_2022)
building_permits_2023 = pd.read_csv(csv_file_path_2023)

# Concatenate the DataFrames
building_permits = pd.concat([building_permits_2022, building_permits_2023], axis=0)

# %%

building_permits.shape

# %%

building_permits.columns

# %%

print(building_permits.info())

# %%

status_counts = building_permits["APPLICATION_STATUS_NAME"].value_counts()

# Create a bar chart with custom colors
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
palette = sns.color_palette("husl", len(status_counts))
sns.barplot(x=status_counts.index, y=status_counts.values, palette=palette)

# Customize the chart
plt.xlabel("Application Status")
plt.ylabel("Count")
plt.title("Distribution of Application Status")
plt.xticks(rotation=45)

plt.show()

# %%

building_permits["APPLICATION_STATUS_NAME"].value_counts()

# %%

building_permits = building_permits[building_permits["APPLICATION_STATUS_NAME"] == "PERMIT ISSUED"]

# %%

building_permits.shape

# %%
building_permits.isnull().sum()

# %%
missing_values = building_permits.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=True)

# Define the colors and colormaps
colors_high = ["#ff5a5f", "#c81d25"]
colors_low = ["#2196f3", "#bbdefb"]
cmap_low = mpl.colors.LinearSegmentedColormap.from_list("low_map", colors_low, N=256)
cmap_high = mpl.colors.LinearSegmentedColormap.from_list("high_map", colors_high, N=256)
norm_low = mpl.colors.Normalize(missing_values.min(), missing_values.mean())
norm_high = mpl.colors.Normalize(missing_values.mean(), missing_values.max())

# Create a horizontal bar chart
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
ax = plt.gca()

# Plot bars with different colormaps
below_average = missing_values[missing_values <= missing_values.mean()]
above_average = missing_values[missing_values > missing_values.mean()]
bar1 = ax.barh(below_average.index, below_average.values, color=cmap_low(norm_low(below_average.values)), zorder=2)
bar2 = ax.barh(above_average.index, above_average.values, color=cmap_high(norm_high(above_average.values)), zorder=2)

# Customize the chart
plt.xlabel("Count of missing values")
plt.ylabel("Columns with missing values")
plt.title("Missing values in construction dataset")

# Annotate the average line
xmin, xmax = ax.get_xlim()
x_pos = missing_values.mean() / xmax + 0.03
plt.show()

# %%
# dropping entirely null columns
building_permits.dropna(axis=1, how="all", inplace=True)

# %%
# Remove unnecessary columns
columns_to_drop = [
    # "X",
    # "Y",
    # "XCOORD",
    # "YCOORD",
    "OBJECTID",  # Not Useful in the analysis
    "GLOBALID",  # Not Useful in the analysis
    "CREATED_USER",  # Not Useful in the analysis
    "LAST_EDITED_USER",  # Not Useful in the analysis
    "APPLICATION_STATUS_NAME",  # As we are keeping obeservations APPLICATION_STATUS_NAME == 'Permit Issued'
    "BUSINESSIMPROVEMENTDISTRICT",  # mostly nulls
    "CREATED_DATE",  # it is the csv file generation date which is not useful
    "LAST_EDITED_DATE",  # it is the csv file generation date which is not useful
    "MARADDRESSREPOSITORYID",  # removed as it's an internal address repository ID that does not provide useful information for predicting construction duration.
    "ANC",  # removed because it represents Advisory Neighborhood Commissions, which are smaller than wards and provide similar location information. We decided to use wards for simplicity.
    "SMD",  # removed because it represents Single Member Districts, which are smaller than wards and provide similar location information. We decided to use wards for simplicity.
    "PSA",  # removed because it represents Police Service Areas, which are unrelated to construction duration
    "FULL_ADDRESS",  # removed because it is the exact address where the construction is happpening not using
    "NEIGHBORHOODCLUSTER",  # as they are smaller than wards and provide similar location information. We decided to use wards for simplicity.
]
building_permits.drop(columns=columns_to_drop, inplace=True)

# %%
building_permits.columns

# %%
building_permits["ISSUE_DATE"] = pd.to_datetime(building_permits["ISSUE_DATE"])
building_permits["LASTMODIFIEDDATE"] = pd.to_datetime(building_permits["LASTMODIFIEDDATE"])

# %%
building_permits.isnull().sum()

# %%
building_permits.info()

# %%

duplicate_rows = building_permits.duplicated()
print(f"Number of duplicate rows: {duplicate_rows.sum()}")

# %%


def get_duplicate_column_pairs(df):
    duplicate_column_pairs = {}
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                duplicate_column_pairs[df.columns[j]] = df.columns[i]
    return duplicate_column_pairs


# %%

duplicate_column_pairs = get_duplicate_column_pairs(building_permits)
print(f"Duplicate column pairs: {duplicate_column_pairs}")

# %%

building_permits.shape

# %%

building_permits.to_csv("../Data/building_permits.csv")

# %%

building_permits.info()

# %%

df_folium = building_permits[["LATITUDE", "LONGITUDE"]]


def generateBaseMap(loc, zoom=12, tiles="OpenStreetMap"):
    return folium.Map(location=loc, control_scale=True, zoom_start=zoom, tiles=tiles)


base_map = generateBaseMap([38.8951100, -77.0363700])

map_values = df_folium[["LATITUDE", "LONGITUDE"]].values.tolist()

hm = HeatMap(
    map_values,
    gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.7: "orange", 0.8: "red"},
    min_opacity=0.05,
    max_opacity=0.9,
    radius=25,
    blur=15,
    use_local_extrema=False,
)

base_map.add_child(hm)

# %%

coordinates = building_permits[["LATITUDE", "LONGITUDE"]]

# %%

scaler = StandardScaler()
scaled_coordinates = scaler.fit_transform(coordinates)

# %%

inertia_values = []
silhouette_scores = []
k_range = range(4, 15)  # Test K values from 4 to 14

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(scaled_coordinates)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_coordinates, kmeans.labels_))

# %%

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

ax1.plot(k_range, inertia_values, marker="o", linewidth=2)
ax1.set_xlabel("Number of Clusters", fontsize=12, fontfamily="serif")
ax1.set_ylabel("Inertia", fontsize=12, fontfamily="serif")
ax1.set_title("Elbow Method", fontsize=14, fontweight="bold", fontfamily="serif")
ax1.grid(linestyle="--", alpha=0.7)

ax2.plot(k_range, silhouette_scores, marker="o", linewidth=2)
ax2.set_xlabel("Number of Clusters", fontsize=12, fontfamily="serif")
ax2.set_ylabel("Silhouette Score", fontsize=12, fontfamily="serif")
ax2.set_title("Silhouette Scores", fontsize=14, fontweight="bold", fontfamily="serif")
ax2.grid(linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%

optimal_k = 7  # the optimal K based on the Elbow method and Silhouette score
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
kmeans.fit(scaled_coordinates)

# %%
building_permits["cluster"] = kmeans.labels_


# %%

building_permits.cluster.unique()

# %%

# Calculate the density of each cluster normalized
cluster_density = building_permits["cluster"].value_counts(normalize=True)
print(cluster_density)

# %%

# Calculate the center of the map based on the average latitude and longitude
map_center = [building_permits["LATITUDE"].mean(), building_permits["LONGITUDE"].mean()]

# Create a Folium map with a tile layer that displays location names
m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB Positron")

# colors to use for the clusters on map
colors = ["#E6194B", "#3CB44B", "#0082C8", "#F58231", "#911EB4", "#46F0F0", "#FABEBE"]

# Add the data points to the map
for index, row in building_permits.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=1,
        popup=f"Cluster: {row['cluster']}",
        tooltip=f"Cluster: {row['cluster']}",
        color=colors[row["cluster"]],
        fill=False,
        fill_opacity=0.1,
        opacity=0.1,
    ).add_to(m)

# Create a custom legend image using PIL
legend_img = Image.new("RGBA", (300, 200), (255, 255, 255, 192))
draw = ImageDraw.Draw(legend_img)
draw.text((10, 10), "Clusters and their densities:", (0, 0, 0))

for i, color in enumerate(colors):
    draw.rectangle([(10, 30 + i * 20), (30, 50 + i * 20)], fill=color)
    draw.text((40, 30 + i * 20), f"Cluster {i}: {cluster_density[i] * 100:.1f}%", (0, 0, 0))

# Save the legend image to a file
legend_img.save("legend.png", "PNG")

# Add the legend image to the map
FloatImage("legend.png", bottom=10, left=10).add_to(m)

# Save the map as an HTML file
m.save("map.html")
