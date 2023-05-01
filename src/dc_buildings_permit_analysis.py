# %%
# Importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import folium
from folium.plugins import HeatMap
from scipy import stats
from scipy.stats import ttest_ind
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
from shapely.geometry import Polygon
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.model_selection import cross_val_score
from shapely.geometry import Point


# %%

# Get the path to the directory where the script is located
script_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

# Construct the file paths to the CSV files
csv_file_path_2022 = os.path.join(
    script_dir, "..", "data", "Building_Permits_in_2022.csv"
)
csv_file_path_2023 = os.path.join(
    script_dir, "..", "data", "Building_Permits_in_2023.csv"
)

# Read the CSV files into DataFrames
building_permits_2022 = pd.read_csv(csv_file_path_2022)
building_permits_2023 = pd.read_csv(csv_file_path_2023)

# Concatenate the DataFrames
building_permits = pd.concat(
    [building_permits_2022, building_permits_2023], axis=0
)

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

building_permits = building_permits[
    building_permits["APPLICATION_STATUS_NAME"] == "PERMIT ISSUED"
]

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
cmap_low = mpl.colors.LinearSegmentedColormap.from_list(
    "low_map", colors_low, N=256
)
cmap_high = mpl.colors.LinearSegmentedColormap.from_list(
    "high_map", colors_high, N=256
)
norm_low = mpl.colors.Normalize(missing_values.min(), missing_values.mean())
norm_high = mpl.colors.Normalize(missing_values.mean(), missing_values.max())

# Create a horizontal bar chart
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
ax = plt.gca()

# Plot bars with different colormaps
below_average = missing_values[missing_values <= missing_values.mean()]
above_average = missing_values[missing_values > missing_values.mean()]
bar1 = ax.barh(
    below_average.index,
    below_average.values,
    color=cmap_low(norm_low(below_average.values)),
    zorder=2,
)
bar2 = ax.barh(
    above_average.index,
    above_average.values,
    color=cmap_high(norm_high(above_average.values)),
    zorder=2,
)

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
]
building_permits.drop(columns=columns_to_drop, inplace=True)

# %%
building_permits.columns

# %%
building_permits["ISSUE_DATE"] = pd.to_datetime(building_permits["ISSUE_DATE"])
building_permits["LASTMODIFIEDDATE"] = pd.to_datetime(
    building_permits["LASTMODIFIEDDATE"]
)

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


##1) load the data
df = pd.read_csv('../Data/building_permits.csv')
df_2022 = pd.read_csv('../Data/Building_Permits_in_2022.csv')
df_2023 = pd.read_csv('../Data/BpData_after_cleaning_with_Nan.csv')


# %%


df_construction =  copy.deepcopy(df) #pd.DataFrame({})
df_construction = df_construction.drop(df_construction[df_construction['PERMIT_TYPE_NAME'] != 'CONSTRUCTION'].index)
# print(df_construction.head())


# %%


##2) we first observeing the data through histogram. 

## 2.1 ISSUE_DATE, by day, the purpose is to see if there are some trends in the issue date, may be the issue of constructions are decreasing, or maybe there are some other patterns.
df_issuedate = copy.deepcopy(df['ISSUE_DATE'])
dt_issue, dt_issue_month,dt_issue_monthsliced, count = [], [], {}, 0
def yearcheck(dt_issue, count = 0, name = 'issue'): #check year see if in 2022 or 2023
    for i in dt_issue:
        if i[0:4] != '2023' and i[0:4] != '2022':
            count += 1
    print('Num of year out of 2022 and 2023:',count,f'\t(0 means the {name} time period is not out of year 2023)')    # 0 means the time period is not out of year 2023

def datetreat(df_issuedate, dt_issue=[], dt_issue_month=[], dt_issue_monthsliced={}):   #put date processes in a function for easy treat with other date data.
    for i in df_issuedate:  # accurate to date
        dt_issue.append(i[0:10])
    for i in range(len(df_issuedate)):  #accurate to month
        dt_issue_month.append(df_issuedate[i][2:7])
    dt_issue.sort()
    dt_issue_month.sort()
    # print(dt_issue[0:5],'\n',dt_issue_month[0:5])     #checkline
    for i in ['01','02','03','04','05','06','07','08','09','10','11','12']: #initialize the dt_issue_monthsliced
        dt_issue_monthsliced['-'.join(['22',i])] = []
        if i == '01' or i == '02' or i == '03' or i =='04':
            dt_issue_monthsliced['-'.join(['23',i])] = []
        # print(dt_issue_monthsliced)    #chekline
# datetreat(df_issuedate, dt_issue=[], dt_issue_month=[], dt_issue_monthsliced)  #checkline
    for i in range(len(dt_issue)):  #now get the 2022 and 2023 month slice
        dt_issue_monthsliced[dt_issue_month[i]].append(dt_issue[i][8:10])
    # print(dt_issue,'\n', dt_issue_month,'\n',dt_issue_monthsliced)
    return dt_issue, dt_issue_month, dt_issue_monthsliced

dt_issue, dt_issue_month, dt_issue_monthsliced = datetreat(df_issuedate, dt_issue, dt_issue_month, dt_issue_monthsliced)
yearcheck(dt_issue, count, name = 'issue')

def month_show(dt_issue_month, name='issue'): #easy function for month_show histgram
    sns.histplot(data = dt_issue_month)
    plt.title("Month hist for {0}".format(name)) 
    plt.xticks(rotation=-45)
    plt.show()# we found that there's only three month's data, and the number of construction-site may be decreasing?(the date data is apprently not enough)
month_show(dt_issue_month, name = 'issue')
print("It seems that the month distribution is equal.")
for key in dt_issue_monthsliced:
    sns.histplot(data = dt_issue_monthsliced[key])
    plt.title(f'month {key} issue num')
    plt.xticks(rotation=-45)
    plt.show()
    # print(key)
# print(dt_issue_monthsliced['22-01'])
print('Comparing the pictures we finally find that all days with few issue numbers seems all weekends, so that there may be fewer application than weekdays.')



# %%


## 2.4 we now observe the PERMIT_TYPE_NAME data.
df_PERMIT_TYPE_NAME = copy.deepcopy(df['PERMIT_TYPE_NAME'])
print('PERMIT_TYPE_NAME classification:',df_PERMIT_TYPE_NAME.unique())
value_counts = df_PERMIT_TYPE_NAME.value_counts()
# print(value_counts)
plt.pie(value_counts,labels = value_counts.index, autopct='%1.1f%%')
plt.title('Pie chart for composition of PERMIT_TYPE_NAME')
plt.show()

df_sorted = value_counts.sort_values(ascending=True)
# print(value_counts)
fig, ax = plt.subplots()
df_sorted.plot(kind = 'barh')
# ax.barh(df_sorted.index, df_sorted)
for i, v in enumerate(df_sorted):
    ax.text(v + 0.1, i, str(v), ha='center', va='center')
plt.title('Number of PERMIT_TYPE_NAME')
plt.show()
print('The pie chart shows that 35.5% of building-permits are construction which is the number 17158')



df_PERMIT_TYPE_NAME = copy.deepcopy(df_2022['PERMIT_TYPE_NAME'])
print('PERMIT_TYPE_NAME classification:',df_PERMIT_TYPE_NAME.unique())
value_counts = df_PERMIT_TYPE_NAME.value_counts()
# print(value_counts)
plt.pie(value_counts,labels = value_counts.index, autopct='%1.1f%%')
plt.title('Pie chart for composition of PERMIT_TYPE_NAME')
plt.show()

df_sorted = value_counts.sort_values(ascending=True)
# print(value_counts)
fig, ax = plt.subplots()
df_sorted.plot(kind = 'barh')
# ax.barh(df_sorted.index, df_sorted)
for i, v in enumerate(df_sorted):
    ax.text(v + 0.1, i, str(v), ha='center', va='center')
plt.title('Number of PERMIT_TYPE_NAME')
plt.show()
print('The pie chart shows that 33.8% of building-permits are construction in 2022, which is the number 16209')


df_PERMIT_TYPE_NAME = copy.deepcopy(df_2023['PERMIT_TYPE_NAME'])
print('PERMIT_TYPE_NAME classification:',df_PERMIT_TYPE_NAME.unique())
value_counts = df_PERMIT_TYPE_NAME.value_counts()
# print(value_counts)
plt.pie(value_counts,labels = value_counts.index, autopct='%1.1f%%')
plt.title('Pie chart for composition of PERMIT_TYPE_NAME')
plt.show()

df_sorted = value_counts.sort_values(ascending=True)
# print(value_counts)
fig, ax = plt.subplots()
df_sorted.plot(kind = 'barh')
# ax.barh(df_sorted.index, df_sorted)
for i, v in enumerate(df_sorted):
    ax.text(v + 0.1, i, str(v), ha='center', va='center')
plt.title('Number of PERMIT_TYPE_NAME')
plt.show()
print('The pie chart shows that 31.4% of building-permits are construction in 2023, which is the number 3201')
print('It seem that the proportion of construction is decreasing, however the data in 2023 is not complete.')


# %%


## 2.5 we now observe the PERMIT_SUBTYPE_NAME data.
df_PERMIT_SUBTYPE_NAME = {}
print(type(df))
for index, row in df.iterrows():  # initialize
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME] = []
for index, row in df.iterrows():    
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME].append(row.PERMIT_SUBTYPE_NAME)

value_counts,count = {},0
for i in df_PERMIT_SUBTYPE_NAME.keys():
    df_PERMIT_SUBTYPE_NAME_series = pd.Series(df_PERMIT_SUBTYPE_NAME[i])
    value_counts[i] = df_PERMIT_SUBTYPE_NAME_series.value_counts()
    count = sum(value_counts[i])
    plt.pie(value_counts[i],labels = value_counts[i].index, autopct='%1.1f%%')
    plt.title(f'Pie chart for {i} in PERMIT_TYPE_NAME')
    plt.annotate(f"Num of total is {count}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(1.1, 0.5))
    plt.show()
print('we redo for construction.')


# %%


## for construction in  the pie charts, we redo it.
value_counts_construction = copy.deepcopy(value_counts['CONSTRUCTION'])
value_counts_construction.sort_values()
count = sum(value_counts_construction)
# print(value_counts_construction)
others, index = copy.deepcopy(value_counts_construction.index[10:]), copy.deepcopy(value_counts_construction.index[0:10])
# print(sum(value_counts_construction[others]),value_counts_construction[index])
sum_others = pd.Series([sum(value_counts_construction[others])])
sum_others.index = ['others']
new_value_counts_construction = value_counts_construction[index].append(sum_others)
plt.pie(new_value_counts_construction,labels = new_value_counts_construction.index, autopct='%1.1f%%')
plt.annotate(f"Num of total is {count}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(-1.5, 1))
plt.title(f'Pie chart for construction in PERMIT_TYPE_NAME')
plt.show()
print('The propotions of items in others are less than 1.4%.')
print('others:',others.values[:])

df_sorted = value_counts_construction.sort_values(ascending=True)
# print(value_counts)
fig, ax = plt.subplots()
fig.set_size_inches(6, 10)
df_sorted.plot(kind = 'barh')
# ax.barh(df_sorted.index, df_sorted)
for i, v in enumerate(df_sorted):
    ax.text(v - 1.5, i, str(v))
plt.title('Number of construction in PERMIT_TYPE_NAME')
plt.show()


# %%


##do the redo for construction for 2022
df_PERMIT_SUBTYPE_NAME = {}
print(type(df))
for index, row in df_2022.iterrows():  # initialize
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME] = []
for index, row in df_2022.iterrows():    
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME].append(row.PERMIT_SUBTYPE_NAME)
value_counts = {}
for i in df_PERMIT_SUBTYPE_NAME.keys():
    df_PERMIT_SUBTYPE_NAME_series = pd.Series(df_PERMIT_SUBTYPE_NAME[i])
    value_counts[i] = df_PERMIT_SUBTYPE_NAME_series.value_counts()
    count = sum(value_counts[i])
value_counts_construction = copy.deepcopy(value_counts['CONSTRUCTION'])
value_counts_construction.sort_values()
others, index = copy.deepcopy(value_counts_construction.index[10:]), copy.deepcopy(value_counts_construction.index[0:10])
# print(sum(value_counts_construction[others]),value_counts_construction[index])
sum_others = pd.Series([sum(value_counts_construction[others])])
sum_others.index = ['others']
new_value_counts_construction = value_counts_construction[index].append(sum_others)
plt.pie(new_value_counts_construction,labels = new_value_counts_construction.index, autopct='%1.1f%%')
plt.annotate(f"Num of total is {count}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(-1.5, 1))
plt.title(f'Pie chart for construction in PERMIT_TYPE_NAME')
plt.show()
print('The propotions of items in others are less than 1.6%.')
print('others:',others.values[:])


# %%


##do the redo for construction for 2023
df_PERMIT_SUBTYPE_NAME = {}
print(type(df))
for index, row in df_2023.iterrows():  # initialize
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME] = []
for index, row in df_2023.iterrows():    
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME].append(row.PERMIT_SUBTYPE_NAME)
value_counts = {}
for i in df_PERMIT_SUBTYPE_NAME.keys():
    df_PERMIT_SUBTYPE_NAME_series = pd.Series(df_PERMIT_SUBTYPE_NAME[i])
    value_counts[i] = df_PERMIT_SUBTYPE_NAME_series.value_counts()
    count = sum(value_counts[i])
value_counts_construction = copy.deepcopy(value_counts['CONSTRUCTION'])
value_counts_construction.sort_values()
others, index = copy.deepcopy(value_counts_construction.index[10:]), copy.deepcopy(value_counts_construction.index[0:10])
# print(sum(value_counts_construction[others]),value_counts_construction[index])
sum_others = pd.Series([sum(value_counts_construction[others])])
sum_others.index = ['others']
new_value_counts_construction = value_counts_construction[index].append(sum_others)
plt.pie(new_value_counts_construction,labels = new_value_counts_construction.index, autopct='%1.1f%%')
plt.annotate(f"Num of total is {count}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(-1.5, 1))
plt.title(f'Pie chart for construction in PERMIT_TYPE_NAME')
plt.show()
print('The propotions of items in others are less than 1.6%.')
print('others:',others.values[:])
print('There seems little change for proportions in construction.')


# %%


## 2.6 PERMIT_CATEGORY_NAME
value_counts_CATEGORY_NAME = df.PERMIT_CATEGORY_NAME.value_counts()
count = 0
# print(np.isnan(df.PERMIT_CATEGORY_NAME[0]))
others, index = copy.deepcopy(value_counts_CATEGORY_NAME.index[8:]), copy.deepcopy(value_counts_CATEGORY_NAME.index[0:8])
sum_others = pd.Series([sum(value_counts_CATEGORY_NAME[others])])
sum_others.index = ['others']
new_value_counts_CATEGORY_NAME = value_counts_CATEGORY_NAME[index].append(sum_others)
count1 = sum(value_counts_CATEGORY_NAME)
for i in df.PERMIT_CATEGORY_NAME:
    if pd.isnull(i) == True:
        count += 1
# fig = plt.figure(figsize=(5, 5))
plt.pie(new_value_counts_CATEGORY_NAME,labels = new_value_counts_CATEGORY_NAME.index, autopct='%1.1f%%')
plt.title(f'Pie chart for PERMIT_CATEGORY_NAME')
plt.annotate(f"Num of NaN is {count}\nNum of total is {count1}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(1.3, 1))
plt.annotate(f'others:{others.values[:]}', fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(-1.5, -1.3))
plt.show()
print('others:',others.values[:])

df_sorted = value_counts_CATEGORY_NAME.sort_values(ascending=True)
# print(value_counts)
fig, ax = plt.subplots()
fig.set_size_inches(6, 10)
df_sorted.plot(kind = 'barh')
# ax.barh(df_sorted.index, df_sorted)
for i, v in enumerate(df_sorted):
    ax.text(v - 1.5, i, str(v))
plt.title('Number of value_counts_CATEGORY_NAME')
plt.show()


# %%


## 2.8 radius plot with position
## Heat map for building_permits.

dc_map = folium.Map(location=[38.9072, -77.0369], zoom_start=12)

HeatMap(data=df[['LATITUDE', 'LONGITUDE']].groupby(['LATITUDE', 'LONGITUDE']).sum().reset_index().values.tolist(), radius=8, max_zoom=13,gradient={0.1:'blue',0.3:'lime',0.5:'yellow',0.7:'orange',0.9:'red'}).add_to(dc_map)

dc_map


# %%


## Heat map for construction.
HeatMap(data=df_construction[['LATITUDE', 'LONGITUDE']].groupby(['LATITUDE', 'LONGITUDE']).sum().reset_index().values.tolist(), radius=8, max_zoom=13,gradient={0.1:'blue',0.3:'lime',0.5:'yellow',0.7:'orange',0.9:'red'}).add_to(dc_map)

dc_map


# %%


## 2.10 PERMIT_APPLICANT
count = 0
for i in df.PERMIT_APPLICANT:
    if pd.isnull(i) == False:
        count += 1
print('There are {:.2f}% of the building-permits have PERMIT_APPLICANT in table, others are empty.'.format(count/len(df)*100))

##for construction
count = 0
for i in df_construction.PERMIT_APPLICANT:
    if pd.isnull(i) == False:
        count += 1
print('There are {:.2f}% of the construction have PERMIT_APPLICANT in table, others are empty.'.format(count/len(df)*100))


# %%


## 2.11 FEES_PAID
count1,count2 = 0,0
for i in df.FEES_PAID:
    if pd.isnull(i) == True:
        count1 += 1
    if i == 0:
        count2 += 1
print(f"There are {count1} of NaN data in FEES_PAID, and {count2} of them haven't paid yet for building-permits.")

##for construction
count1,count2 = 0,0
for i in df_construction.FEES_PAID:
    if pd.isnull(i) == True:
        count1 += 1
    if i == 0:
        count2 += 1
print(f"There are {count1} of NaN data in FEES_PAID, and {count2} of them haven't paid yet for construction.")


# %%


## 2.12 OWNER_NAME
count = 0
for i in df.OWNER_NAME:
    if pd.isnull(i) == False:
        count += 1
print('There are {:.2f}% of the building-permits have a owner or more in table, others are empty.'.format(count/len(df)*100))

##for construction
count = 0
for i in df_construction.OWNER_NAME:
    if pd.isnull(i) == False:
        count += 1
print('There are {:.2f}% of the construction have a owner or more in table, others are empty.'.format(count/len(df)*100))


# %%


## 2.13 DISTRICT
value_DISTRICT = df.DISTRICT.value_counts()
count1, count2 = 0, 0
count2 = sum(value_DISTRICT)
for i in df.DISTRICT:
    if pd.isnull(i) == True:
        count1 += 1
plt.pie(value_DISTRICT,labels = value_DISTRICT.index, autopct='%1.1f%%')
plt.title(f'Pie chart for building-permits by DISTRICT')
plt.annotate(f"Num of NaN is {count1}\nNum of total is {count2}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(0.7, 1))
plt.show()
print('It shows that there are the first, second and fourth district has more building-permit than others.')

##for construction
value_DISTRICT = df_construction.DISTRICT.value_counts()
count1, count2 = 0, 0
count2 = sum(value_DISTRICT)
for i in df.DISTRICT:
    if pd.isnull(i) == True:
        count1 += 1
plt.pie(value_DISTRICT,labels = value_DISTRICT.index, autopct='%1.1f%%')
plt.title(f'Pie chart for constructions by DISTRICT')
plt.annotate(f"Num of NaN is {count1}\nNum of total is {count2}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(0.7, 1))
plt.show()
print('It shows that there are the first, second and fourth district has more construction than others. The proportions for building-permits and constructions are similar.')


# %%


## 2.14 NEIGHBORHOODCLUSTER
value_counts_NEIGHBORHOODCLUSTER = df.NEIGHBORHOODCLUSTER.value_counts()
count = 0
for i in df.NEIGHBORHOODCLUSTER:
    if pd.isnull(i) == True:
        count += 1
# print(f"Num of NaN is {count}")
others, index = copy.deepcopy(value_counts_NEIGHBORHOODCLUSTER.index[16:]), copy.deepcopy(value_counts_NEIGHBORHOODCLUSTER.index[0:16])
# print('others:',others)
sum_others = pd.Series([sum(value_counts_NEIGHBORHOODCLUSTER[others])])
sum_others.index = ['others']
new_value_counts_NEIGHBORHOODCLUSTER = value_counts_NEIGHBORHOODCLUSTER[index].append(sum_others)
plt.pie(new_value_counts_NEIGHBORHOODCLUSTER,labels = new_value_counts_NEIGHBORHOODCLUSTER.index, autopct='%1.1f%%')
plt.title(f'Pie chart for building-permits by APPLICATION_STATUS_NAME')
plt.annotate(f"Others: {others.values}", fontsize=8, color='black', xy=(0,1),xytext=(-1.6, -1.5))
plt.show()

##for construction
value_counts_NEIGHBORHOODCLUSTER = df_construction.NEIGHBORHOODCLUSTER.value_counts()
count = 0
for i in df_construction.NEIGHBORHOODCLUSTER:
    if pd.isnull(i) == True:
        count += 1
# print(f"Num of NaN is {count}")
others, index = copy.deepcopy(value_counts_NEIGHBORHOODCLUSTER.index[16:]), copy.deepcopy(value_counts_NEIGHBORHOODCLUSTER.index[0:16])
# print('others:',others)
sum_others = pd.Series([sum(value_counts_NEIGHBORHOODCLUSTER[others])])
sum_others.index = ['others']
new_value_counts_NEIGHBORHOODCLUSTER = value_counts_NEIGHBORHOODCLUSTER[index].append(sum_others)
plt.pie(new_value_counts_NEIGHBORHOODCLUSTER,labels = new_value_counts_NEIGHBORHOODCLUSTER.index, autopct='%1.1f%%')
plt.title(f'Pie chart for construction by APPLICATION_STATUS_NAME')
plt.annotate(f"Others: {others.values}", fontsize=8, color='black', xy=(0,1),xytext=(-1.6, -1.5))
plt.show()
print('The ranking in building-permits and construction changes after ranking 11th. The cluster 26,18,8 has the most of the constructions')


# %%


##3) other plot and multi-variables analyze.

## 3.1 position varaibles X_Y and LONGITUDE_LATITUDE analyze.
X = np.array(df['X'])
Y = np.array(df['Y'])
LONGITUDE = np.array(df['LONGITUDE'])
LATITUDE = np.array(df['LATITUDE'])
X_Y = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
LONGITUDE_LATITUDE = np.concatenate((LONGITUDE.reshape(-1, 1), LATITUDE.reshape(-1, 1)), axis=1)
# print(X_Y[:5], LONGITUDE_LATITUDE[:5])

stat_X_Y, p_X_Y = stats.shapiro(X_Y)
stat_LONGITUDE_LATITUDE, p_LONGITUDE_LATITUDE = stats.shapiro(LONGITUDE_LATITUDE)
stat_X, p_X = stats.shapiro(X)
stat_LONGITUDE, p_LONGITUDE = stats.shapiro(LONGITUDE)
stat_Y, p_Y = stats.shapiro(Y)
stat_LATITUDE, p_LATITUDE = stats.shapiro(LATITUDE)
if p_X_Y > 0.05 and p_LONGITUDE_LATITUDE > 0.05:
    print("We find that the two two-elemnet-stats X_Y and  LONGITUDE_LATITUDE obey normal distribution.")
else:
    print("We find that the two two-elemnet-stats X_Y and  LONGITUDE_LATITUDE do not obey normal distribution,but because the amount of data is large enough, we assume that it is normally distrubuted.")
print('p-value for X:',p_X,',p-value for LONGITUDE',p_LONGITUDE,',p-value for Y',p_Y,',p-value for LATITUDE',p_LATITUDE, 'Every one of them do not obey normal distribution, as well.')

t_statistic, p_value = ttest_ind(X, LONGITUDE)
print(f"Because p-value:{p_value}>0.05, we believe that X and LONGITUDE have no difference.")
print(f"Because the maximum distance of X and LONGITUDE is {max(abs(X-LONGITUDE))*111000} meter(as 1degree is 111km), we believe there are no difference between them.")
t_statistic, p_value = ttest_ind(Y, LATITUDE)
print(f"Because p-value:{p_value}>0.05, we believe that Y and LATITUDE have no difference.")
print(f"Because the maximum distance of X and LONGITUDE is {max(abs(Y-LATITUDE))*111000} meter(as 1degree is 111km), we believe there are no difference between them.")

#consider yeojohnson transformation? How to do 2-demensional t-test?!!


# %%


##4) analyze the influnce by DISTRICT.
df11 = copy.deepcopy(df_2022)
for i in range(len(df11)):
    df11['ISSUE_DATE'][i] = df11['ISSUE_DATE'][i][5:7]
df21 = df11[['ISSUE_DATE','DISTRICT']]
data = df21.value_counts()
# print(data.index[0])
data_t = {'ISSUE_DATE':[],'DISTRICT':[],'Num':[]}
for i in range(len(data)):
    data_t['ISSUE_DATE'].append(data.index[i][0])
    data_t['DISTRICT'].append(data.index[i][1])
    data_t['Num'].append(data[i])
# print(data_t)
data_t1 = pd.DataFrame(data_t)
data_t1_sorted = data_t1.sort_values(by = ["ISSUE_DATE", "DISTRICT"])
# print(data_t1_sorted)
##picture
sns.lineplot(data = data_t1_sorted, x = 'ISSUE_DATE', hue = 'DISTRICT', y = 'Num')
plt.legend(loc = (1.05,0.5))
plt.title('Issue_date num by District in 2022.')
plt.show()


# %%


df11 = copy.deepcopy(df_2022)
for i in range(len(df11)):
    df11['ISSUE_DATE'][i] = df11['ISSUE_DATE'][i][5:7]
df21 = df11[['ISSUE_DATE','NEIGHBORHOODCLUSTER']]
data = df21.value_counts()
# print(data.index[0])
data_t = {'ISSUE_DATE':[],'NEIGHBORHOODCLUSTER':[],'Num':[]}
for i in range(len(data)):
    data_t['ISSUE_DATE'].append(data.index[i][0])
    data_t['NEIGHBORHOODCLUSTER'].append(data.index[i][1])
    data_t['Num'].append(data[i])
# print(data_t)
data_t1 = pd.DataFrame(data_t)
data_t1_sorted = data_t1.sort_values(by = ["ISSUE_DATE", "NEIGHBORHOODCLUSTER"])
# print(data_t1_sorted)


# %%


sns.lineplot(data = data_t1_sorted, x = 'ISSUE_DATE', hue = 'NEIGHBORHOODCLUSTER', y = 'Num')
plt.legend(loc = (1.05,-1.5))
plt.title('By NEIGHBORHOODCLUSTER in 2022.')
plt.show()


# %%


data_t1_sorted_26 = data_t1_sorted[data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 26']
data_t1_sorted_18 = data_t1_sorted[data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 18']
data_t1_sorted_8 = data_t1_sorted[data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 8']
data_t1_sorted_6 = data_t1_sorted[data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 6']
data_t1_sorted_25 = data_t1_sorted[data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 25']
data_t2 = pd.concat([data_t1_sorted_26,data_t1_sorted_18,data_t1_sorted_8,data_t1_sorted_6,data_t1_sorted_25],axis = 0)
print(data_t2.head())
# data_t2 = data_t1_sorted[data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 26' or data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 18' or data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 8' or data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 6' or data_t1_sorted['NEIGHBORHOODCLUSTER'] == 'Cluster 25']
sns.lineplot(data = data_t2, x = 'ISSUE_DATE', hue = 'NEIGHBORHOODCLUSTER', y = 'Num')
plt.legend(loc = (1.05,0.5))
plt.title('Top 5 NEIGHBORHOODCLUSTER in 2022.')
plt.show()


# %%


##5) Correlation  
df1 = copy.deepcopy(df)
df2 = df1.corr()
print(df2)
strong_corr = []
for i in range(1,len(df2)):
    for j in range(i+1,len(df2)):
        # print(df2.index[i])
        if abs(df2[df2.index[i]][df2.index[j]]) > 0.8 :
            strong_corr.append(f'{df2.index[i]}-{df2.index[j]}')
print('\n',f'There is a strong correlation between {strong_corr}')





# %%

df_folium = building_permits[["LATITUDE", "LONGITUDE"]]


def generateBaseMap(loc, zoom=12, tiles="OpenStreetMap"):
    return folium.Map(
        location=loc, control_scale=True, zoom_start=zoom, tiles=tiles
    )


base_map = generateBaseMap([38.8951100, -77.0363700])

map_values = df_folium[["LATITUDE", "LONGITUDE"]].values.tolist()

hm = HeatMap(
    map_values,
    gradient={
        0.2: "blue",
        0.4: "lime",
        0.6: "yellow",
        0.7: "orange",
        0.8: "red",
    },
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
    silhouette_scores.append(
        silhouette_score(scaled_coordinates, kmeans.labels_)
    )

# %%

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

ax1.plot(k_range, inertia_values, marker="o", linewidth=2)
ax1.set_xlabel("Number of Clusters", fontsize=12, fontfamily="serif")
ax1.set_ylabel("Inertia", fontsize=12, fontfamily="serif")
ax1.set_title(
    "Elbow Method", fontsize=14, fontweight="bold", fontfamily="serif"
)
ax1.grid(linestyle="--", alpha=0.7)

ax2.plot(k_range, silhouette_scores, marker="o", linewidth=2)
ax2.set_xlabel("Number of Clusters", fontsize=12, fontfamily="serif")
ax2.set_ylabel("Silhouette Score", fontsize=12, fontfamily="serif")
ax2.set_title(
    "Silhouette Scores", fontsize=14, fontweight="bold", fontfamily="serif"
)
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
map_center = [
    building_permits["LATITUDE"].mean(),
    building_permits["LONGITUDE"].mean(),
]

# Create a Folium map with a tile layer that displays location names
m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB Positron")

# colors to use for the clusters on map
colors = [
    "#E6194B",
    "#3CB44B",
    "#0082C8",
    "#F58231",
    "#911EB4",
    "#46F0F0",
    "#FABEBE",
]

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
    draw.text(
        (40, 30 + i * 20),
        f"Cluster {i}: {cluster_density[i] * 100:.1f}%",
        (0, 0, 0),
    )

# Save the legend image to a file
legend_img.save("legend.png", "PNG")

# Add the legend image to the map
FloatImage("legend.png", bottom=10, left=10).add_to(m)

# Save the map as an HTML file
m.save("map.html")

# %%
# Convert the DataFrame to a GeoDataFrame
geometry = [Point(xy) for xy in zip(building_permits.LONGITUDE, building_permits.LATITUDE)]
crs = {'init': 'epsg:4326'}
geo_df = gpd.GeoDataFrame(building_permits, crs=crs, geometry=geometry)

# Buffer the points by 0.5 miles (about 804.5 meters)
buffered_geo_df = geo_df.to_crs({'init': 'epsg:3857'})
buffered_geo_df['geometry'] = buffered_geo_df['geometry'].buffer(804.5)
buffered_geo_df = buffered_geo_df.to_crs({'init': 'epsg:4326'})

# Calculate the bounding box of the study area
bounds = buffered_geo_df.to_crs({"init": "epsg:3857"}).bounds

min_x = bounds.minx.min()
min_y = bounds.miny.min()
max_x = bounds.maxx.max()
max_y = bounds.maxy.max()
# %%

# Set the size of the cells (0.5 miles in meters)
cell_size = 804.5

# Generate the grid cells
x_coords = np.arange(min_x, max_x, cell_size)
y_coords = np.arange(min_y, max_y, cell_size)
polygons = []
# %%

# Set the size of the cells (0.5 miles in meters)
cell_size = 804.5

# Generate the grid cells
x_coords = np.arange(min_x, max_x, cell_size)
y_coords = np.arange(min_y, max_y, cell_size)
polygons = []

# %%

for x in x_coords:
    for y in y_coords:
        polygons.append(
            Polygon(
                [
                    (x, y),
                    (x + cell_size, y),
                    (x + cell_size, y + cell_size),
                    (x, y + cell_size),
                ]
            )
        )

grid = gpd.GeoDataFrame({"geometry": polygons}, crs={"init": "epsg:3857"})
grid = grid.to_crs({"init": "epsg:4326"})

# %%

# Spatial join between buffered_geo_df and grid
buffered_geo_df = buffered_geo_df.to_crs({"init": "epsg:3857"})
point_grid_join = gpd.sjoin(
    buffered_geo_df, grid, how="left", op="within", rsuffix="_grid"
)

# %%
# Perform the spatial join
point_grid_join = gpd.sjoin(
    grid.reset_index().rename(columns={"index": "index_grid"}),
    geo_df,
    how="inner",
    op="intersects",
)

# Calculate the construction count per grid cell
construction_count_per_cell = point_grid_join.groupby("index_grid").size()
grid["construction_count"] = construction_count_per_cell
grid = grid.to_crs({"init": "epsg:4326"})

# %%

threshold = grid["construction_count"].quantile(0.75)
grid["high_construction"] = (grid["construction_count"] > threshold).astype(
    int
)
threshold = grid['construction_count'].quantile(0.75)
grid['high_prob'] = grid['construction_count'] >= threshold


# %%
# Split the dataset into training and testing sets
X = grid.drop(columns=['construction_count', 'high_construction', 'geometry'])
y = grid['high_construction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)


# %%
# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", square=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# %%

# Create a GeoDataFrame with the test data and the predicted values
test_gdf = grid.loc[X_test.index].copy()
test_gdf["predicted_high_construction"] = y_pred

# Plot the true and predicted high construction areas
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

test_gdf.plot(
    column="high_construction", cmap="coolwarm", legend=True, ax=ax[0]
)
ax[0].set_title("True High Construction Areas")

test_gdf.plot(
    column="predicted_high_construction",
    cmap="coolwarm",
    legend=True,
    ax=ax[1],
)
ax[1].set_title("Predicted High Construction Areas")

plt.show()


# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5, scoring="f1")

# Print the mean and standard deviation of the F1 scores
print(f"Mean F1 Score: {scores.mean():.2f}")
print(f"Standard Deviation: {scores.std():.2f}")

# %%

# Create a GeoDataFrame with the test data and the predicted values
test_gdf = grid.loc[X_test.index].copy()
test_gdf['predicted_high_construction'] = y_pred

# Plot the true and predicted high construction areas
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

test_gdf.plot(column='high_construction', cmap='coolwarm', legend=True, ax=ax[0])
ax[0].set_title('True High Construction Areas')

test_gdf.plot(column='predicted_high_construction', cmap='coolwarm', legend=True, ax=ax[1])
ax[1].set_title('Predicted High Construction Areas')

plt.show()
# %%
