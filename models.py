import pickle
import numpy as np
import pandas as pd
np.random.seed(123) 
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
#import xgboost as xgb
from sklearn import neighbors
from sklearn.preprocessing import Normalizer

from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint


def sample(X, y, n):
    '''
    take n random samples from the original data X, y
    X: numpy array
    '''
    num_row = X.shape[0]
    indices = np.random.randint(num_row, size=n)
    return X[indices], y[indices]

def get_train_val(df_train, features, target, val_ratio=0.1, shuffle_data=False):
    """ 
    Use the latest 10% records for validation.
    From the remaining 90% records, randomly select 200k for training.
    """
    print("Check: 'Date' should be in descending order:")
    print(df_train[:5])
    print(df_train[-5:])

    X = df_train[features]
    y = df_train[target]
    print("features used:")
    print(pd.concat((X[:5], y[:5]), axis=1))

    # convert to numpy array, so that shuffle, sample can index
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if shuffle_data:
        print("Use shuffled data")
        sh = np.arange(X.shape[0])
        np.random.shuffle(sh)
        X = X[sh]
        y = y[sh]
    else:
        print("Use un-shuffled data")

    num_records = X.shape[0]
    val_size = int(val_ratio * num_records)
    
    # latest 10% for validation,  84433 records
    X_val = X[:val_size] 
    y_val = y[:val_size]
    
    # remaining 90%, 759905 records
    X_train = X[val_size:] 
    y_train = y[val_size:]
    
    # randomly select 200k records for training (for data sparsity)
    X_train, y_train = sample(X_train, y_train, 200000)
    print("training X: ", X_train.shape)
    print("validation X: ", X_val.shape)
    return X_train, X_val, y_train, y_val

def calc_embedding_size(df, features_em, N_em):
    """ 
    df: pandas DataFrame
    features_em: features to be embedded, e.g., ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'State']
    N_em: embedding dim, None (automatically derived), or specified, e.g., 
    # original dim: [1115, 7, 3, 12, 31, 12]
    # embedding dim:[  10, 6, 2,  6, 10, 6]
    """
    # original dimension
    N_cat = df[features_em].nunique()
    if N_em is None:
        # calculated embedding dim, half of the original dim, at most 50
        N_em = N_cat.apply(lambda x: int(min(np.ceil(x/2), 50))) 
    else:
        # specified embedding dim
        N_em = pd.Series(N_em, index=N_cat.index) 
    em_size = pd.DataFrame([N_cat, N_em], index=['orig_dim', 'em_dim'])
    return em_size


def save_embeddings(model, embeddings_path, features_em):
    embeddings = []
    for fname in features_em:
        # embedding layer name: see models.py, class NN_with_EntityEmbedding
        em = model.get_layer(fname+'_em').get_weights()[0]
        embeddings.append(em)
    embeddings_dict = dict(zip(features_em, embeddings)) # usage: embeddings_dict['Store']
    with open(embeddings_path, 'wb') as f:
        print("Save trained embeddings to: {}".format(embeddings_path))
        pickle.dump(embeddings_dict, f, -1)


def load_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        print("Load embeddings from: {}".format(embeddings_path))
        embeddings_dict = pickle.load(f) # usage: embeddings_dict['Store']
    features_em = list(embeddings_dict.keys())
    em_size = pd.DataFrame(index=['orig_dim', 'em_dim'], columns=features_em)
    for ff, em_vec in embeddings_dict.items():
        em_size[ff] = em_vec.shape
    print("embedding size:")
    print(em_size)
    return features_em, embeddings_dict, em_size


def embed_features(X, embeddings_dict, features, features_em):
    """
    convert categorical features to embeddings
    X: numpy array
    X_em: numpy array
    """
    (num_records, num_features) = X.shape
    assert(num_features == len(features))
    
    X_em = np.array([]).reshape((num_records,0))
    for ii, fname in enumerate(features):
        if fname not in features_em:
            feat_em = X[:, ii] # original value
        else:
            feat_em = list(map(lambda x: embeddings_dict[fname][x], X[:, ii])) # embedded value
            
        feat_em = np.array(feat_em).reshape(num_records, -1)
        X_em = np.concatenate((X_em, feat_em), axis=1)
    return X_em


class Model(object):

    def evaluate(self, X_val, y_val):
        """ 
        Calculate MAPE performance of the model
        """
        assert(min(y_val) > 0)
        y_val = y_val.ravel()             # shape (84434,)
        y_hat = self.guess(X_val).ravel() # shape (84434,)
        assert(y_val.shape == y_hat.shape)

        # calculate MAPE
        err_ape = np.abs((y_val - y_hat) / y_val)# APE: absolute percentage error
        err_mape = np.mean(err_ape) # MAPE: mean absolute percentage error
        return err_mape
    

class LinearModel(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.clf = linear_model.LinearRegression()
        self.clf.fit(X_train, np.log(y_train))
        
        # print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return np.exp(self.clf.predict(feature))


class RF(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.clf = RandomForestRegressor(n_estimators=200, verbose=1, max_depth=35, min_samples_split=2,
                                         min_samples_leaf=1, n_jobs=-1)
        self.clf.fit(X_train, np.log(y_train)) # verbose=1: print training progress
        
        # print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return np.exp(self.clf.predict(feature))


class SVM(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.__normalize_data()
        self.clf = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001,
                       C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

        self.clf.fit(self.X_train, np.log(self.y_train))
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def __normalize_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

    def guess(self, feature):
        return np.exp(self.clf.predict(feature))

"""
class XGBoost(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        dtrain = xgb.DMatrix(X_train, label=np.log(y_train))
        evallist = [(dtrain, 'train')]
        param = {'nthread': -1,
                 'max_depth': 7,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'reg:linear',
                 'colsample_bytree': 0.7,
                 'subsample': 0.7}
        num_round = 3000
        self.bst = xgb.train(param, dtrain, num_round, evallist)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return np.exp(self.bst.predict(dtest))
"""

class HistoricalMedian(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.history = {}
        self.feature_index = [1, 2, 3, 4]
        for x, y in zip(X_train, y_train):
            key = tuple(x[self.feature_index])
            self.history.setdefault(key, []).append(y)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = np.array(features)
        features = features[:, self.feature_index]
        guessed_sales = [np.median(self.history[tuple(feature)]) for feature in features]
        return np.array(guessed_sales)


class KNN(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.normalizer = Normalizer()
        self.normalizer.fit(X_train)
        self.clf = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance', p=1)
        self.clf.fit(self.normalizer.transform(X_train), np.log(y_train))
        print("Result on validation data: ", self.evaluate(self.normalizer.transform(X_val), y_val))

    def guess(self, feature):
        return np.exp(self.clf.predict(self.normalizer.transform(feature)))


class NN_with_EntityEmbedding(Model):

    def __init__(self, X_train, y_train, X_val, y_val, epochs, features, features_em, em_size):
        super().__init__()
        self.epochs = epochs
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val))) # save ymax for scaling
        self.__build_keras_model(features, features_em, em_size)
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        # X: numpy array
        num_features = X.shape[1] # num features
        X_list = [X[:, [ff]] for ff in range(num_features)] # a list, each X_list[i] has shape (num_records, 1)
        return X_list
    
    def __build_keras_model(self, features, features_em, em_size):    
        input_model = []
        output_embeddings = []
        for fname in features:
            if fname not in features_em:
                # features we do not embed, e.g., numerical features
                input_feature = Input(shape=(1,))
                output_feature = Dense(1)(input_feature) 
                # option: try other encoding, e.g., (cos, sin)
            else:
                # features we want to embed
                input_feature = Input(shape=(1,))
                orig_dim, em_dim = em_size[fname]
                output_feature = Embedding(orig_dim, em_dim, name=fname+'_em')(input_feature)
                output_feature = Reshape(target_shape=(em_dim,))(output_feature)
            input_model.append(input_feature)
            output_embeddings.append(output_feature)
            
        output_model = Concatenate()(output_embeddings)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)
        # build the model
        self.model = KerasModel(inputs=input_model, outputs=output_model)
        self.model.compile(loss='mean_absolute_error', optimizer='adam')
    
    
    def _val_for_fit(self, val):
        val = np.log(val) / self.max_log_y # for y, apply log and then scale to [0, 1]
        return val

    def _val_for_pred(self, val):
        return np.exp(val * self.max_log_y) # for y_hat, scale back and apply exp

    def fit(self, X_train, y_train, X_val, y_val):
        # didn't use callbacks, so validation set is not used for training embeddings. for fair comparison.
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        # print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)


class NN(Model):

    def __init__(self, X_train, y_train, X_val, y_val, epochs=10):
        super().__init__()
        self.epochs = epochs
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def __build_keras_model(self):
        self.model = Sequential()
        self.model.add(Dense(1000, kernel_initializer="uniform", input_dim=1183))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, kernel_initializer="uniform"))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = np.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return np.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, self._val_for_fit(y_train),
                       validation_data=(X_val, self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
