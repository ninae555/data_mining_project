#%%
##a cell for import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

#%%
##1) load the data
df = pd.read_csv('building_permits.csv')
df_fee_ward = df[['FEES_PAID','WARD']]
df_fee_ward.dropna()
df_fee_ward = df_fee_ward[df_fee_ward['FEES_PAID']!=0]

df_fee_ward_1 = df_fee_ward[df_fee_ward['WARD']==1]
df_fee_ward_2 = df_fee_ward[df_fee_ward['WARD']==2]
df_fee_ward_3 = df_fee_ward[df_fee_ward['WARD']==3]
df_fee_ward_4 = df_fee_ward[df_fee_ward['WARD']==4]
df_fee_ward_5 = df_fee_ward[df_fee_ward['WARD']==5]
df_fee_ward_6 = df_fee_ward[df_fee_ward['WARD']==6]
df_fee_ward_7 = df_fee_ward[df_fee_ward['WARD']==7]
df_fee_ward_8 = df_fee_ward[df_fee_ward['WARD']==8]

# %%
# print(df_fee_ward)

fig, ax = plt.subplots()
sns.boxplot(df_fee_ward,x = 'WARD', y = 'FEES_PAID')
# plt.show()
# q1 = np.percentile(df_fee_ward_1, 5)
# q3 = np.percentile(df_fee_ward_1, 95)
# sns.boxplot(df_fee_ward_1,y = 'FEES_PAID',showfliers=False,whis=(0, 99))
ax.set_ylim(0, 600)
plt.show()

print(df_fee_ward.describe()['FEES_PAID'])

#%%
# print(df_fee_ward_1.describe()['FEES_PAID']['mean'])
# print(df_fee_ward_2.describe()['FEES_PAID']['mean'])
# print(df_fee_ward_3.describe()['FEES_PAID']['mean'])
# print(df_fee_ward_4.describe()['FEES_PAID']['mean'])
# print(df_fee_ward_5.describe()['FEES_PAID']['mean'])
# print(df_fee_ward_6.describe()['FEES_PAID']['mean'])
# print(df_fee_ward_7.describe()['FEES_PAID']['mean'])
# print(df_fee_ward_8.describe()['FEES_PAID']['mean'])

#%%
from scipy.stats import f_oneway

f_statistic, p_value = f_oneway(df_fee_ward_1['FEES_PAID'],df_fee_ward_2['FEES_PAID'], df_fee_ward_3['FEES_PAID'],df_fee_ward_4['FEES_PAID'],df_fee_ward_5['FEES_PAID'],df_fee_ward_6['FEES_PAID'],df_fee_ward_7['FEES_PAID'],df_fee_ward_8['FEES_PAID'])
print("stat", f_statistic)
print("p_value", p_value)

print('\nAlthough the distribution of median look similarly, but they are significantly different by mean as p_value less than 0.05.')

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X = df_fee_ward['FEES_PAID']
y = df_fee_ward['WARD']
X_array = X.to_numpy()
X_reshaped = np.reshape(X_array, (-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", knn.score(X_test, y_test))

#%%
fig, ax = plt.subplots()
plt.scatter(X_test[:, 0], X_test[:, 0], c=y_pred)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
plt.show()
print(np.unique(y_pred))


#%%
# import tensorflow as tf
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# n_trees = 100
# max_depth = 5
# learning_rate = 0.1

# loss = tf.keras.losses.mean_squared_error

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# metrics = ['mse']

# model = tf.keras.estimator.BoostedTreesRegressor(
#     n_batches_per_layer=1,
#     n_trees=n_trees,
#     max_depth=max_depth,
#     loss_reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
#     learning_rate=learning_rate,
#     optimizer=optimizer,
#     metrics=metrics
# )

# train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x=X_train,
#     y=y_train,
#     batch_size=32,
#     num_epochs=None,
#     shuffle=True
# )

# val_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x=X_val,
#     y=y_val,
#     batch_size=32,
#     num_epochs=1,
#     shuffle=False
# )

# model.train(input_fn=train_input_fn, steps=None)

# model.evaluate(input_fn=val_input_fn)

# test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x=X_test,
#     y=y_test,
#     batch_size=32,
#     num_epochs=1,
#     shuffle=False
# )

# model.predict(input_fn=test_input_fn)



#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X = df_fee_ward['FEES_PAID']
y = df_fee_ward['WARD']
X_array = X.to_numpy()
X_reshaped = np.reshape(X_array, (-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train, y_train)

score = rf.score(X_test, y_test)
print(f"Accuracy: {score}")