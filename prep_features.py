import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import random
random.seed(123)
import os

"""
input files: train.csv, test.csv, store_states.csv
output files: feature_train_data.csv, feature_test_data.csv

RUN: python prep_features.py
"""

# data directory
data_dir = 'data/'

# load data: 'train.csv', or 'test.csv'
data_type = 'train'

df = pd.read_csv(os.path.join(data_dir, data_type+'.csv'))
df_state = pd.read_csv(os.path.join(data_dir, 'store_states.csv'))

print("loaded: ", df.shape)
print(df[:5])

print("state loaded: ", df_state.shape )
print(df_state[:5])

# only keep the data when: store is open, and sales is not zero
# yw_notes: there are records when store is open and sales is 0, outliers?
if data_type == 'train':
    to_keep = (df['Open'] == 1) & (df['Sales'] > 0)
elif data_type == 'test':
    to_keep = (df['Open'] == 1)
df = df[to_keep]
print("Only keep 'Open' stores with non-zero 'Sales': ", df.shape)

print("check NaNs")
print(df.isna().sum())
#df_store.isna().sum()

"""
# yw_notes: if using pd.merge to add 'State' column, the trained embedding has worse val error, why? 
# yw_notes: if using pd.merge to add 'State' column, then df is ordered by Store, not good for train/test?
df = pd.merge(df, df_state, on='Store')
"""
print("Add 'State' column")
get_state = df_state.set_index('Store')['State'] # a Series, usage: get_state[store_id] 
df['State'] = list(map(lambda x: get_state[x], df['Store']))
print(df[:5])

"""
# convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, cache=True)

# create 'Year', 'Month', 'Day', 'State' columns
print("Create Year, Month, Day, State columns")
def get_ymd_state(date, store_id):
    return date.year, date.month, date.day, get_state[store_id]
df['Year'], df['Month'], df['Day'], df['State'] = \
zip(*list(map(get_ymd_state, df['Date'], df['Store'])))
"""

print("prepare features")

# convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, cache=True)

# set Date to index
df.set_index(keys='Date', inplace=True)

# create 'Year', 'Month', 'Day'
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day

# all features
features = ['Open', 'Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'State']
# target
if data_type == 'train':
    target = ['Sales']
elif data_type == 'test':
    target = []
df = df[features + target]

print("features and target")
print(df[:5])

print("Label encoding...")
le_path = os.path.join(data_dir, 'les.pickle')
if data_type == 'train':
    les = []
    # apply label encoding to all features
    for ff in features:
        le = LabelEncoder()
        le.fit(df[ff])
        les.append(le) # for saving
        df[ff] = le.transform(df[ff])
    print(df[:5])
    print("Save label encoders to: {}".format(le_path))
    # must save the label encoders (labling might change)
    with open(le_path, 'wb') as f:
        les_dict = dict(zip(features, les)) # usage: les_dict['Store']
        pickle.dump(les_dict, f, -1)   
        
elif data_type == 'test':
    print("Load label encoders from: {}".format(le_path))
    with open(le_path, 'rb') as f:
        les_dict = pickle.load(f)
    for ff, le in les_dict.items():
        df[ff] = le.transform(df[ff])
    print(df[:5])


# save features: 'feature_train_data.csv', or 'feature_test_data.csv'
feature_path = os.path.join(data_dir, 'feature_'+data_type+'_data.csv')
print("Save features to: {}".format(feature_path))
df.to_csv(feature_path)


"""
# load store
store_data_path = os.path.join(data_dir, 'store.csv')
df_store = pd.read_csv(store_data_path)


# add the 'State' column to df_store (needed?)
df_store['State'] = list(map(lambda x: get_state[x], df_store['Store']))

# save a few file: 'store_new.csv', store with 'State' column (needed?)
store_new_data_path = os.path.join(data_dir, 'store_new.csv')

# save the new store file (needed?)
df_store.to_csv(store_new_data_path, index=False)

"""
