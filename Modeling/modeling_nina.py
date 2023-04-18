#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#%%
# Loading the dataset
df = pd.read_csv('/Users/admin/Documents/data_mining_project/Data/building_permits.csv')
#%%
# Handling missing values
# Drop columns with too many missing values
df.drop(columns=['PERMIT_CATEGORY_NAME'], inplace=True)
# Filling missing values in other columns
df['PERMIT_SUBTYPE_NAME'].fillna('Unknown', inplace=True)
df['ZONING'].fillna('Unknown', inplace=True)
df['PERMIT_APPLICANT'].fillna('Unknown', inplace=True)
df['FEE_TYPE'].fillna('Unknown', inplace=True)
df['OWNER_NAME'].fillna('Unknown', inplace=True)
#%%
# Converting datetime columns
df['ISSUE_DATE'] = pd.to_datetime(df['ISSUE_DATE'])
df['LASTMODIFIEDDATE'] = pd.to_datetime(df['LASTMODIFIEDDATE'])
df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'])
df['LAST_EDITED_DATE'] = pd.to_datetime(df['LAST_EDITED_DATE'])
#%%
# Splitting the data into features and target variable
X = df.drop(columns=['NEW_CONSTRUCTION_RISK'])
y = df['NEW_CONSTRUCTION_RISK']
#%%
# Converting categorical variables using one-hot encoding or label encoding
# One-hot encode categorical columns
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(X[['PERMIT_TYPE_NAME', 'PERMIT_SUBTYPE_NAME', 'ZONING']])
X_ohe_df = pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(['PERMIT_TYPE_NAME', 'PERMIT_SUBTYPE_NAME', 'ZONING']))
X = pd.concat([X, X_ohe_df], axis=1)
X.drop(columns=['PERMIT_TYPE_NAME', 'PERMIT_SUBTYPE_NAME', 'ZONING'], inplace=True)
#%%
# Labeling encode other categorical columns
le = LabelEncoder()
X['PERMIT_ID'] = le.fit_transform(X['PERMIT_ID'])
X['FULL_ADDRESS'] = le.fit_transform(X['FULL_ADDRESS'])
X['SSL'] = le.fit_transform(X['SSL'])
X['PERMIT_APPLICANT'] = le.fit_transform(X['PERMIT_APPLICANT'])
X['FEE_TYPE'] = le.fit_transform(X['FEE_TYPE'])
X['OWNER_NAME'] = le.fit_transform(X['OWNER_NAME'])
#%%
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further pre-processing or feature engineering can be done on X_train and X_test as needed
