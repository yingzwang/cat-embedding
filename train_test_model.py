# train and test a model for predicting sales.
# input to the prediction model: 1) trained embeddings, or 2) one-hot encoded

import pickle
import numpy as np
np.random.seed(123)
from models import embed_features, load_embeddings, get_train_val, LinearModel, RF
from sklearn.preprocessing import OneHotEncoder
import sys
sys.setrecursionlimit(10000)
import os
from time import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pandas as pd

# directory of data
data_dir = 'data/'

# load embeddings
em_name = 'embeddings_auto.pickle' # auto calculated (half of the original dim, at most 50)
#em_name = 'embeddings_ref.pickle' # paper ver

model_str = "RF" # 'LinearModel', 'RF', ref: "models.py"
#encode_types = ["NN-embedding", "one-hot", "label-encode"] 
encode_types = ["NN-embedding", "label-encode"] 

def encode_features(X_train, X_val, encode_type):
    assert(encode_type in ["NN-embedding", "one-hot", "label-encode"])
    
    if encode_type == "NN-embedding":
        print("******* use NN embedding features ...")
        embeddings_path = os.path.join(data_dir, em_name)
        features_em, embeddings_dict, em_size = load_embeddings(embeddings_path)
        X_train = embed_features(X_train, embeddings_dict, features, features_em) # X shape (num_records, 42)
        X_val = embed_features(X_val, embeddings_dict, features, features_em) # X shape (num_records, 42)
        print("training X: ", X_train.shape)
        print("validation X: ", X_val.shape)
        
    elif encode_type == "one-hot":
        print("******* use one-hot encoded features ...")
        enc = OneHotEncoder(sparse=False)
        enc.fit(X_train)
        X_train = enc.transform(X_train) # X shape (num_records, 1182)
        X_val = enc.transform(X_val) # X shape (num_records, 1182)
        print("training X: ", X_train.shape)
        print("validation X: ", X_val.shape)

    elif encode_type == "label-encode":
        # do nothing, features are already label-encoded
        print("******* use label encoded features ...")
        print("training X: ", X_train.shape)
        print("validation X: ", X_val.shape)
    
    return X_train, X_val
    

def eval_performance(X_train, y_train, X_val, y_val, encode_type, model_str):
    model_strs = ["LinearModel", "RF"]
    assert(model_str in model_strs)
    
    X_train, X_val = encode_features(X_train, X_val, encode_type)

    # start training
    start = time() # get starting time
    print("Fitting {} ...".format(model_str))
    eval_str = model_str + "(X_train, y_train, X_val, y_val)"
    model = eval(eval_str)
    end = time() # get ending time
    train_time = (end - start) / 60 # time used, minutes
    train_err = model.evaluate(X_train, y_train)
    val_err = model.evaluate(X_val, y_val)
    print("model: ", model.__class__.__name__)
    print("encode: ", encode_type)
    print("training error MAPE: {:.4f}".format(train_err))
    print("validation error MAPE: {:.4f}".format(val_err))
    print("training time: {:.4f} minutes".format(train_time))
    
    # results['feature_encoding'] = encode_type 
    # results['model'] = model.__class__.__name__
    model_name = model.__class__.__name__
    results = {}
    results['feature_dimension'] = X_train.shape[1]
    results['train_time'] = train_time
    results['val_error'] = val_err
    rs = pd.Series(results, name=encode_type)
    return rs, model_name


val_ratio = 0.1      # don't change this. Must be the same as in train_embedding.py
shuffle_data = False # don't change this. Must be the same as in train_embedding.py

#one_hot_as_input = False
#embeddings_as_input = True

# load training data: features X, target y
df_train = pd.read_csv(os.path.join(data_dir, 'feature_train_data.csv'))

# features and targets for training
target = ['Sales']
# features used
features = ['Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'State']
X_train, X_val, y_train, y_val = get_train_val(df_train, features, target, val_ratio, shuffle_data)

# evaluate performances of a prediction model (for all encoding types)
model_res = []
for encode_type in encode_types:
    rs, model_name = eval_performance(X_train, y_train, X_val, y_val, encode_type, model_str)
    model_res.append(rs)
    
# convert to DataFrame
model_res = pd.concat(model_res, axis=1).T
print(model_res)

# for RF results, add the 'one-hot' row
"""
model_res.loc['one-hot'] = [1182, np.NaN, np.NaN] # rows: 'embed', 'label', 'one-hot'
model_res = model_res.iloc[['0', '2', '1']]  # rows: 'embed', 'one-hot', 'label'
model_res.index = ["embed", "one-hot", "label"]
"""

# save performance results
res_dir = 'results/'
res_path = os.path.join(res_dir, "res_"+model_name+".csv")
print("Save performance results to: {}".format(res_path))
model_res.to_csv(res_path)

# scikit learn does not support GPU
"""
# read performance results
res_dir = 'results/'
model_name = 'LinearModel'
#model_name = 'RF'
res_path = os.path.join(res_dir, "res_"+model_name+".csv")
print("Load performance results from: {}".format(res_path))
model_res = pd.read_csv(res_path, index_col=0)

"""

# plot performance results
#model_res.index = ["embed", "one-hot", "label"]
axes = model_res.plot.bar(subplots=True, figsize=(10, 4), layout=(1, 3), legend=None, rot=30)
#axes[0, 0].set_ylim(0, 1200)
#axes[0, 1].set_ylim(0, 10)
#axes[0, 2].set_ylim(0, 0.31)
plt.tight_layout()
fig_path = os.path.join(res_dir, "res_"+model_name+".pdf")
print("Save performance figure to: {}".format(fig_path))
plt.savefig(fig_path, bbox_inches='tight')

# save embedding size
embeddings_path = os.path.join(data_dir, em_name)
_, _, em_size = load_embeddings(embeddings_path)
res_path = os.path.join(res_dir, "em_size.csv")
print("Save embedding size to: {}".format(res_path))
em_size.to_csv(res_path)



"""
# load testing data: features X, target y
df_test = pd.read_csv(os.path.join(data_dir, 'feature_test_data.csv'))
features = ['Open', 'Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'State']
X = df_train[features]
X = np.asarray(X)
if embeddings_as_input:
    print("Using learned embeddings as input")
    X = embed_features(X, embeddings_path) # X shape (num_records, 42)
    print("feature shape: ", X.shape)


# load original, created by 'prepare_features.py'
feature_path = os.path.join(data_dir, 'feature_train_data.pickle')
f = open(feature_path, 'rb')
(X, y) = pickle.load(f)


print("Fitting Neural Network...")
model = NN(X_train, y_train, X_val, y_val)

print("Fitting Random Forest...") # embedding: 4.0553 mins, 0.1258; le: 0.169
model = RF(X_train, y_train, X_val, y_val)

print("Fitting Linear Regression...") # embedding: 0.0149 min, 0.1454
model = LinearModel(X_train, y_train, X_val, y_val)

print("Fitting KNN...") # embedding: 25.78 min, 0.135
model = KNN(X_train, y_train, X_val, y_val)

 print("Fitting XGBoost...")
model = XGBoost(X_train, y_train, X_val, y_val)
"""
