#%%
##a cell for import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import datetime

#%%
##1) load the data
df = pd.read_csv('BpData_after_cleaning_with_Nan.csv')

#%%
##2) we first observeing the data through histogram.   ## try add a map with location later!!

## 2.1 ISSUE_DATE, by day, the purpose is to see if there are some trends in the issue date, may be the issue of constructions are decreasing, or maybe there are some other patterns.
df_issuedate = copy.deepcopy(df['ISSUE_DATE'])
dt_issue, dt_issue_month,dt_issue_monthsliced, count = [], [], {}, 0
def yearcheck(dt_issue, count = 0, name = 'issue'): #check year
    for i in dt_issue:
        if i[0:4] != '2023':
            count += 1
    print(count,f'\t(0 means the {name} time period is not out of year 2023)')    # 0 means the time period is not out of year 2023
def datetreat(df_issuedate, dt_issue=[], dt_issue_month=[], dt_issue_monthsliced={}):   #put them in a function for easy treat with other date data.
    for i in df_issuedate:  # accurate to date
        dt_issue.append(i[0:10])
    for i in range(len(dt_issue)):  # date without year.
        dt_issue[i] = dt_issue[i][5:]
    for i in range(len(dt_issue)):  #accurate to month
        dt_issue_month.append(dt_issue[i][0:2])
    dt_issue.sort()
    dt_issue_month.sort()
    for i in ['01','02','03','04','05','06','07','08','09','10','11','12']: #initialize the dt_issue_monthsliced
        dt_issue_monthsliced[i] = []
    for i in range(len(dt_issue)):#now get the month slice
        dt_issue_monthsliced[dt_issue_month[i]].append(dt_issue[i][3:5])
    # print(dt_issue,'\n', dt_issue_month,'\n',dt_issue_monthsliced)
    return dt_issue, dt_issue_month, dt_issue_monthsliced
yearcheck(dt_issue, count)
dt_issue, dt_issue_month, dt_issue_monthsliced = datetreat(df_issuedate, dt_issue, dt_issue_month, dt_issue_monthsliced)
def month_show(dt_issue_month, name='issue'): #easy function for month_show
    sns.histplot(data = dt_issue_month)
    plt.title("Month hist for {0}".format(name)) 
    plt.show()# we found that there's only three month's data, and the number of construction-site may be decreasing?(the date data is apprently not enough)
month_show(dt_issue_month, name = 'issue')
print("we found that there's only three months' data, and the number of construction-site may be decreasing?(the date data may be not enough.")     #check a bit later!!
sns.histplot(data = dt_issue_monthsliced['01'])
plt.title('month 01 issue num')
plt.show()
sns.histplot(data = dt_issue_monthsliced['02']) 
plt.title('month 02 issue num')
plt.show()
sns.histplot(data = dt_issue_monthsliced['03']) 
plt.title('month 03 issue num')
plt.show()
print('There seems that there are a fixed few days in each month with no issue date infomation, but we do not know why. It seems that there are no other anticipated obvious patterns.')

#%%
## 2.2 The same as before, we treat CREATED_DATE, LAST_EDITED_DATE
df_CREATED_DATE = copy.deepcopy(df['CREATED_DATE'])
yearcheck(df_CREATED_DATE,0,'created')
dt_create, dt_create_month, dt_create_monthsliced = datetreat(df_CREATED_DATE, [], [], {})
month_show(dt_create_month, 'created')
print("we found that there's only one month's data.")

df_LAST_EDITED_DATE = copy.deepcopy(df['LAST_EDITED_DATE'])
yearcheck(df_LAST_EDITED_DATE,0,'last_edited')
dt_LAST_EDITED, dt_LAST_EDITED_month, dt_LAST_EDITED_monthsliced = datetreat(df_LAST_EDITED_DATE, [], [], {})
month_show(dt_LAST_EDITED_month, 'last_edited')
print("we found that there's only one month's data.")

count1, count2 = 0, 0
for i in df_LAST_EDITED_DATE:
    if i != df_LAST_EDITED_DATE[0]:
        count1 += 1
for i in df_LAST_EDITED_DATE:
    if i != df_LAST_EDITED_DATE[0]:
        count2 += 1
if count1 == 0 and count2 ==0:
    print(f'The CREATED_DATE and LAST_EDITED_DATE are all in one time {df_LAST_EDITED_DATE[0]}.')

#%%
## 2.3 The same as before, we treat LASTMODIFIEDDATE
df_LASTMODIFIEDDATE = copy.deepcopy(df['LASTMODIFIEDDATE'])
yearcheck(df_LASTMODIFIEDDATE,0,'LASTMODIFIEDDATE')
dt_LASTMODIFIEDDATE, dt_LASTMODIFIEDDATE_month, dt_LASTMODIFIEDDATE_monthsliced = datetreat(df_LASTMODIFIEDDATE, [], [], {})
month_show(dt_LASTMODIFIEDDATE_month, 'LASTMODIFIEDDATE')
print("we found that the last modifieddate data is apparently increasing in three month.")
sns.histplot(data = dt_LASTMODIFIEDDATE_monthsliced['01'])
plt.title('month 01 LASTMODIFIED num')
plt.show()
sns.histplot(data = dt_LASTMODIFIEDDATE_monthsliced['02']) 
plt.title('month 02 LASTMODIFIED num')
plt.show()
sns.histplot(data = dt_LASTMODIFIEDDATE_monthsliced['03']) 
plt.title('month 03 LASTMODIFIED num')
plt.show()
print('I am a bit confused with the pattern, is it really possible? Maybe there are some activity we do not know, such as some bidding activities?')

#%%
## 2.4 we now observe the PERMIT_TYPE_NAME data.
df_PERMIT_TYPE_NAME = copy.deepcopy(df['PERMIT_TYPE_NAME'])
print('PERMIT_TYPE_NAME classification:',df_PERMIT_TYPE_NAME.unique())
value_counts = df_PERMIT_TYPE_NAME.value_counts()
print(value_counts)
plt.pie(value_counts,labels = value_counts.index, autopct='%1.1f%%')
plt.title('Pie chart for PERMIT_TYPE_NAME')
plt.show()


#%%
## 2.5 we now observe the PERMIT_SUBTYPE_NAME data.
df_PERMIT_SUBTYPE_NAME = {}
print(type(df))
for index, row in df.iterrows():  # initialize
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME] = []
for index, row in df.iterrows():    # initialize
    df_PERMIT_SUBTYPE_NAME[row.PERMIT_TYPE_NAME].append(row.PERMIT_SUBTYPE_NAME)

value_counts = {}
for i in df_PERMIT_SUBTYPE_NAME.keys():
    df_PERMIT_SUBTYPE_NAME_series = pd.Series(df_PERMIT_SUBTYPE_NAME[i])
    value_counts[i] = df_PERMIT_SUBTYPE_NAME_series.value_counts()
    plt.pie(value_counts[i],labels = value_counts[i].index, autopct='%1.1f%%')
    plt.title(f'Pie chart for {i} in PERMIT_TYPE_NAME')
    plt.show()


## for construction in  the pie charts, we redo it.
value_counts_construction = copy.deepcopy(value_counts['CONSTRUCTION'])
value_counts_construction.sort_values()
# print(value_counts_construction)
others, index = copy.deepcopy(value_counts_construction.index[9:]), copy.deepcopy(value_counts_construction.index[0:9])
print('others:',others)
# print(sum(value_counts_construction[others]),value_counts_construction[index])
sum_others = pd.Series([sum(value_counts_construction[others])])
sum_others.index = ['others']
new_value_counts_construction = value_counts_construction[index].append(sum_others)
plt.pie(new_value_counts_construction,labels = new_value_counts_construction.index, autopct='%1.1f%%')
plt.title(f'Pie chart for counstruction in PERMIT_TYPE_NAME')
plt.show()
print('The propotions of items in others are less than 2.2%.')
# check if we can joint two pie chats in python!!

#%%
## 2.6 PERMIT_CATEGORY_NAME
value_counts_CATEGORY_NAME = df.PERMIT_CATEGORY_NAME.value_counts()
count = 0
# print(np.isnan(df.PERMIT_CATEGORY_NAME[0]))
for i in df.PERMIT_CATEGORY_NAME:
    if pd.isnull(i) == True:
        count += 1
# fig = plt.figure(figsize=(5, 5))
plt.pie(value_counts_CATEGORY_NAME,labels = value_counts_CATEGORY_NAME.index, autopct='%1.1f%%')
plt.title(f'Pie chart for PERMIT_CATEGORY_NAME')
plt.annotate(f"Num of NaN is {count}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(1.3, 1))
plt.show()

#%%
## 2.7 APPLICATION_STATUS_NAME
value_counts_APPLICATION_STATUS = df.APPLICATION_STATUS_NAME.value_counts()
count = 0
for i in df.APPLICATION_STATUS_NAME:
    if pd.isnull(i) == True:
        count += 1
# fig = plt.figure(figsize=(5, 5))
others, index = copy.deepcopy(value_counts_APPLICATION_STATUS.index[2:]), copy.deepcopy(value_counts_APPLICATION_STATUS.index[0:2])
print('others:',others)
# print(sum(value_counts_construction[others]),value_counts_construction[index])
sum_others = pd.Series([sum(value_counts_APPLICATION_STATUS[others])])
sum_others.index = ['others']
new_value_counts_APPLICATION_STATUS = value_counts_APPLICATION_STATUS[index].append(sum_others)
plt.pie(new_value_counts_APPLICATION_STATUS,labels = new_value_counts_APPLICATION_STATUS.index, autopct='%1.1f%%')
plt.title(f'Pie chart for APPLICATION_STATUS_NAME')
plt.annotate(f"Num of NaN is {count}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(0.8, 1))
plt.annotate(f"Others: {others.values}", fontsize=8, color='black', xy=(0,1),xytext=(-1.6, -1.3))
plt.show()

#%%
## 2.8 radius plot with position
## undone, should be done!!!

#%%
## 2.9 NLP for DESC_OF_WORK?
## undone(considering)

#%%
## 2.10 PERMIT_APPLICANT
count = 0
for i in df.PERMIT_APPLICANT:
    if pd.isnull(i) == False:
        count += 1
print('There are {:.2f}% of the constructions have PERMIT_APPLICANT in table, others are empty.'.format(count/len(df)*100))

#%%
## 2.11 FEES_PAID
count1,count2 = 0,0
for i in df.FEES_PAID:
    if pd.isnull(i) == True:
        count1 += 1
    if i == 0:
        count2 += 1
print(f"There are {count1} of NaN data in FEES_PAID, and {count2} of them haven't paid yet.")
# consider if need more analyze?!!

#%% 
## 2.12 OWNER_NAME
count = 0
for i in df.OWNER_NAME:
    if pd.isnull(i) == False:
        count += 1
print('There are {:.2f}% of the constructions have a owner or more in table, others are empty.'.format(count/len(df)*100))

#%%
## 2.13 DISTRICT
value_DISTRICT = df.DISTRICT.value_counts()
count = 0
for i in df.DISTRICT:
    if pd.isnull(i) == True:
        count += 1
plt.pie(value_DISTRICT,labels = value_DISTRICT.index, autopct='%1.1f%%')
plt.title(f'Pie chart for DISTRICT')
plt.annotate(f"Num of NaN is {count}", fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'),xy=(0,1),xytext=(0.7, 1.1))
plt.show()

#%%
## 2.14 NEIGHBORHOODCLUSTER
value_counts_NEIGHBORHOODCLUSTER = df.NEIGHBORHOODCLUSTER.value_counts()
count = 0
for i in df.NEIGHBORHOODCLUSTER:
    if pd.isnull(i) == True:
        count += 1
print(f"Num of NaN is {count}")
others, index = copy.deepcopy(value_counts_NEIGHBORHOODCLUSTER.index[16:]), copy.deepcopy(value_counts_NEIGHBORHOODCLUSTER.index[0:16])
print('others:',others)
sum_others = pd.Series([sum(value_counts_NEIGHBORHOODCLUSTER[others])])
sum_others.index = ['others']
new_value_counts_NEIGHBORHOODCLUSTER = value_counts_NEIGHBORHOODCLUSTER[index].append(sum_others)
plt.pie(new_value_counts_NEIGHBORHOODCLUSTER,labels = new_value_counts_NEIGHBORHOODCLUSTER.index, autopct='%1.1f%%')
plt.title(f'Pie chart for APPLICATION_STATUS_NAME')
plt.annotate(f"Others: {others.values}", fontsize=8, color='black', xy=(0,1),xytext=(-1.6, -1.5))
plt.show()

#%%
## 2.15 BUSINESSIMPROVEMENTDISTRICT
value_counts_BUSINESSIMPROVEMENTDISTRICT = df.BUSINESSIMPROVEMENTDISTRICT.value_counts()
count = 0
for i in df.BUSINESSIMPROVEMENTDISTRICT:
    if pd.isnull(i) == True:
        count += 1
print(f"Num of NaN is {count}")
others, index = copy.deepcopy(value_counts_BUSINESSIMPROVEMENTDISTRICT.index[9:]), copy.deepcopy(value_counts_BUSINESSIMPROVEMENTDISTRICT.index[0:9])
print('others:',others)
sum_others = pd.Series([sum(value_counts_BUSINESSIMPROVEMENTDISTRICT[others])])
sum_others.index = ['others']
new_value_counts_BUSINESSIMPROVEMENTDISTRICT = value_counts_BUSINESSIMPROVEMENTDISTRICT[index].append(sum_others)
plt.pie(new_value_counts_BUSINESSIMPROVEMENTDISTRICT,labels = new_value_counts_BUSINESSIMPROVEMENTDISTRICT.index, autopct='%1.1f%%')
plt.title(f'Pie chart for APPLICATION_STATUS_NAME')
plt.annotate(f"Others: {others.values}", fontsize=8, color='black', xy=(0,1),xytext=(-1.5, -1.2))
plt.show()

#%%
## 2.16 Created and Last edited user
df1 = df.CREATED_USER.value_counts()
df2 = df.LAST_EDITED_USER.value_counts()
# print(df1,'\n\n',df2,'\n\n',len(df),'\n')
print("Created and Last edited user are all OCTO_ETL, may be the meaning of auto extract-transform-load by computer.")

#%%
##3) other plot and multi-variables analyze.

## 3.1 position varaibles X_Y and LONGITUDE_LATITUDE analyze.
X = np.array(df['X'])
Y = np.array(df['Y'])
LONGITUDE = np.array(df['LONGITUDE'])
LATITUDE = np.array(df['LATITUDE'])
X_Y = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
LONGITUDE_LATITUDE = np.concatenate((LONGITUDE.reshape(-1, 1), LATITUDE.reshape(-1, 1)), axis=1)
# print(X_Y[:5], LONGITUDE_LATITUDE[:5])

from scipy.stats import normaltest
from scipy import stats
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
print(p_X,p_LONGITUDE,p_Y,p_LATITUDE)

from scipy.stats import ttest_ind
t_statistic, p_value = ttest_ind(X, LONGITUDE)
print(f"Because p-value:{p_value}>0.05, we believe that X and LONGITUDE have no difference.")
print(f"Because the maximum distance of X and LONGITUDE is {max(abs(X-LONGITUDE))*111000} meter(as 1degree is 111km), we believe there are no difference between them.")
t_statistic, p_value = ttest_ind(Y, LATITUDE)
print(f"Because p-value:{p_value}>0.05, we believe that Y and LATITUDE have no difference.")
print(f"Because the maximum distance of X and LONGITUDE is {max(abs(Y-LATITUDE))*111000} meter(as 1degree is 111km), we believe there are no difference between them.")

#consider yeojohnson transformation? How to do 2-demensional t-test?!!
#%%
##4) analyze the influnce by DISTRICT.
df1 = copy.deepcopy(df)
for i in range(len(df1)):
    df1['ISSUE_DATE'][i] = df1['ISSUE_DATE'][i][5:7]
df2 = df1[['ISSUE_DATE','DISTRICT']]
data = df2.value_counts()
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
sns.lineplot(data = data_t1_sorted, x = 'ISSUE_DATE', hue = 'DISTRICT', y = 'Num')
plt.show()

#undone

#%%
##5) Correlation    #Try heat map later!!
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


#unfinished part for heatmap:

# ISSUE_DATE1 = pd.get_dummies(df1['ISSUE_DATE'])
# PERMIT_TYPE_NAME1 = pd.get_dummies(df1['PERMIT_TYPE_NAME'])
# PERMIT_SUBTYPE_NAME1 = pd.get_dummies(df1['PERMIT_SUBTYPE_NAME'])
# PERMIT_CATEGORY_NAME1 = pd.get_dummies(df1['PERMIT_CATEGORY_NAME'])
# LASTMODIFIEDDATE1 = pd.get_dummies(df1['LASTMODIFIEDDATE'])
# DISTRICT1 = pd.get_dummies(df1['DISTRICT'])
# NEIGHBORHOODCLUSTER1 = pd.get_dummies(df1['NEIGHBORHOODCLUSTER'])
# # print(ISSUE_DATE)

# df2 = df1[['X','Y','ISSUE_DATE','PERMIT_TYPE_NAME','PERMIT_SUBTYPE_NAME','PERMIT_CATEGORY_NAME','FEES_PAID','LASTMODIFIEDDATE','LATITUDE','LONGITUDE','XCOORD','YCOORD','MARADDRESSREPOSITORYID','WARD','DISTRICT','PSA','NEIGHBORHOODCLUSTER']]
# # print(df2)
# # df2['ISSUE_DATE'] = ISSUE_DATE1
# print(len(df2['ISSUE_DATE']),'\n',len(ISSUE_DATE))
# # df2['PERMIT_TYPE_NAME'] = PERMIT_TYPE_NAME
# # df2['PERMIT_SUBTYPE_NAME'] = PERMIT_SUBTYPE_NAME
# # df2['PERMIT_CATEGORY_NAME'] = PERMIT_CATEGORY_NAME
# # df2['LASTMODIFIEDDATE'] = LASTMODIFIEDDATE
# # df2['DISTRICT'] = DISTRICT
# # df2['NEIGHBORHOODCLUSTER'] = NEIGHBORHOODCLUSTER

# # sns.heatmap(df2, annot=True, cmap='YlGnBu')
# # plt.title('Heatmap')
# # plt.show()


#%%
## a cell for test
# df3 = pd.DataFrame({'color':['red', 'blue', 'green', 'red', 'green']})
# one_hot = pd.get_dummies(df3['color'])
# df3 = pd.concat([df3, one_hot], axis=1)
# print(df3)

