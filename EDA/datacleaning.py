#%%
##a cell for import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
##1) load the data
df = pd.read_csv('Building_Permits_in_2023.csv')

#%%
##2) observe the data
##2.1   df.head()
print(df.head())
#%%
##2.2   info()
df.info()
#%%
##2.3   through observation we find x,y and LATITUDE, LONGITUDE maybe duplicated, but actually they are not.
print('#2.3\t',sum(df.X==df.LONGITUDE), sum(df.Y == df.LATITUDE))    #We will do statistic test in EDA.

##2.4   through observation we find PERMIT_CATEGORY_NAME may be empty,but it's not.
df_temp = df.PERMIT_CATEGORY_NAME.dropna()
print('\n#2.4\t',df_temp)

##2.5   through observation, check if city and state are all NAN.
print('\n#2.5\t',sum(pd.notnull(df.CITY)), sum(pd.notnull(df.STATE)))   # We can drop the two columns.

##2.6   same about ZIPCODE, DCSTATADDRESSKEY, DCSTATLOCATIONKEY, HOTSPOT2006NAME, HOTSPOT2005NAME, HOTSPOT2004NAME, BUSINESSIMPROVEMENTDISTRICT.
print('\n#2.6\t',sum(pd.notnull(df.ZIPCODE)),'\t',sum(pd.notnull(df.DCSTATADDRESSKEY)),'\t',sum(pd.notnull(df.DCSTATLOCATIONKEY)),'\t',sum(pd.notnull(df.HOTSPOT2006NAME)),'\n',sum(pd.notnull(df.HOTSPOT2005NAME)),'\t',sum(pd.notnull(df.HOTSPOT2004NAME)),'\t',sum(pd.notnull(df.BUSINESSIMPROVEMENTDISTRICT)))  # We can drop the columns.

## 2.7  check again if have columns that are empty.
empty_columns = []
for i in  df.columns:
    if sum(pd.notnull(df[i])) == 0:
        empty_columns.append(i)
print('\n#2.7\t',empty_columns,'\n\t',len(empty_columns))


#%%
##3) do data cleaning
##first get the columns and from info() we know it's a (10186 entries)*(44 columns) dataframe, and get a copy of dataframe
df_cols = df.columns
print(len(df_cols))

import copy
df1 = copy.deepcopy(df)

##3.1    do data cleaning for step 2.
for i in empty_columns:
    del df1[i]
print(len(df1.columns)) #colmun length has been delete from 44 to 36, so it's correct. 
df1.info()

#%%
##3.2    Clean data of wrong format
col_type = {}
for i in df1.columns:
    col_type[i] = df1[i].dtype
# print(col_type)

##long int to str
for i in range(len(df1)):
    df1['OBJECTID'][i] = "{:.0f}".format(df1['OBJECTID'][i])
    df1['DCRAINTERNALNUMBER'][i] = "{:.0f}".format(df1['DCRAINTERNALNUMBER'][i])

## Date:ISSUE_DATE/ LASTMODIFIEDDATE/CREATED_DATE/LAST_EDITED_DATE  
df1['ISSUE_DATE'] = pd.to_datetime(df1['ISSUE_DATE'])
df1['LASTMODIFIEDDATE'] = pd.to_datetime(df1['LASTMODIFIEDDATE'])
df1['CREATED_DATE'] = pd.to_datetime(df1['CREATED_DATE'])
df1['LAST_EDITED_DATE'] = pd.to_datetime(df1['LAST_EDITED_DATE']) 

## check the length of other dtype('int') data
for j in df1.columns:
    if df1[j].dtype == np.int64:
        max = 0
        for i in df1[j]:
            if len(str(i)) > max:
                max = len(str(i))
        print(j,' length:', max)

## we can convert them to string if needed.
# for j in df1.columns:
#     if col_type[j] == np.int64:
#         df1[j] = str(df1[j])

## test
# print(df1.to_string())
# col_type = {}
# for i in df1.columns:
#     col_type[i] = df1[i].dtype
# print(col_type)

#%%
##3.3   drop NaN in certain columns.
## check how many NaN in each columns.
for i in df1.columns:
    if sum(pd.isna(df1[i])) != 0:
        print(i,sum(pd.isna(df1[i])))
print('\nnum of observations:',len(df1))


df2 = copy.deepcopy(df1)
## we try drop NaN in small amount, which we difine as <1000; and drop the column in large amount.
df2.dropna(subset=['PERMIT_SUBTYPE_NAME'], inplace = True)
df2.dropna(subset=['SSL'], inplace = True)
df2.dropna(subset=['ZONING'], inplace = True)
df2.dropna(subset=['OWNER_NAME'], inplace = True)
del df2['PERMIT_CATEGORY_NAME']
del df2['DESC_OF_WORK']
del df2['PERMIT_APPLICANT']
del df2['FEE_TYPE']
del df2['BUSINESSIMPROVEMENTDISTRICT']

## test
# print('\nnum of observations:',len(df2))
# for i in df2.columns:
#     if sum(pd.isna(df2[i])) != 0:
#         print(i,sum(pd.isna(df2[i])),)


##3.4    Check/remove duplicates
print('\n',sum(df1.duplicated()))

duplicate_columns = []
for i in df1.columns:
    if sum(df1[i].duplicated()) != 0:
        duplicate_columns.append(i)
print(duplicate_columns)
## because we don't have duplicated entire row, I do not do the dropduplicate at this time.


#%%
##4) output the data
df1.to_csv("BpData_after_cleaning_with_Nan.csv")
df2.to_csv("BpData_after_cleaning_without_Nan.csv")

#%%
df1.info()