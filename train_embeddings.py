# only train NN model to obtain the embedding, no test.
# output file: embeddings.pickle

import pickle
import numpy as np
import pandas as pd
np.random.seed(123)
from models import NN_with_EntityEmbedding, calc_embedding_size, save_embeddings, get_train_val
import sys
sys.setrecursionlimit(10000)
import os
from time import time


# directory of data 
data_dir = 'data/'


# features to be embedded
features_em = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'State']
# specify embedding size
#N_em = [20, 4, 2, 6, 15, 7]; em_name = 'embeddings.pickle'
#N_em = [10, 6, 2, 6, 10, 6]; em_name = 'embeddings_ref.pickle' # ref: same as paper, embeddings_ref.pickle
N_em = None; em_name = 'embeddings_auto.pickle'

    
# load training data: features X, target y
df_train = pd.read_csv(os.path.join(data_dir, 'feature_train_data.csv'))

# calculate embedding size
em_size = calc_embedding_size(df_train, features_em, N_em)
print("embedding size:")
print(em_size)

# features and targets for training
target = ['Sales']
# features used
features = ['Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'State']
val_ratio = 0.1 
X_train, X_val, y_train, y_val = get_train_val(df_train, features, target, val_ratio)


start = time() # get starting time
print("Fitting NN_with_EntityEmbedding...")
epochs = 10
model_nne = NN_with_EntityEmbedding(X_train, y_train, X_val, y_val, epochs, features, features_em, em_size)
end = time() # get ending time
train_time = (end - start) / 60 # time used, minutes
val_err = model_nne.evaluate(X_val, y_val)
print("validation error MAPE: {:.4f}".format(val_err))
print("training time: {:.4f} minutes".format(train_time))

results = {}
results['feature_dimension'] = X_train.shape[1]
results['train_time'] = train_time
results['val_error'] = val_err
rs = pd.Series(results, name="NN_with_embedding")
    
# save performance results
res_dir = 'results/'
res_path = os.path.join(res_dir, "res_NN_with_embedding.csv")
print("Save performance results to: {}".format(res_path))
rs.to_csv(res_path)

# save the trained embeddings
embeddings_path = os.path.join(data_dir, em_name)
save_embeddings(model_nne.model, embeddings_path, features_em)

#models = []
#print("Fitting NN_with_EntityEmbedding...")
#models.append(NN_with_EntityEmbedding(X_train, y_train, X_val, y_val, 10)) # NN model is trained for 10 epochs

## file path for saving the embeddings
#embeddings_path = os.path.join(data_dir, 'embeddings.pickle')
#print("Saving trained embeddings to {}".format(embeddings_path))
#model = models[0].model
#save_embeddings(model, embeddings_path)

"""
f = open(os.path.join(data_dir, 'feature_train_data.pickle'), 'rb')
(X, y) = pickle.load(f)
"""
