# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:28:31 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:03:24 2019

@author: kevin
"""

# add correlation coeffients into features 
import pandas as pd
import pyarrow.parquet as pq
import os
import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm 
from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import Model
from tensorflow.contrib.rnn import *
from sklearn.model_selection import train_test_split 
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting

filepath = '../'
os.listdir(filepath)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#subset_train = pq.read_pandas(filepath+'/train.parquet',
#                              columns=[str(i) for i in range(5)]).to_pandas()
#subset_train.info()
#subset_train.head()
#plt.plot(subset_train)
#subset_train.to_csv('subset_train.csv')

# load train labels 
y_train = pd.read_csv(filepath+'/metadata_train.csv')
y_train = y_train.set_index(['id_measurement','phase'])
y_train.head()
y_train.info()

sample_size = 800000
N_Splits = 3 

def Matt_corcoe(y_true,y_pred):
    y_pred_pos = K.round(K.clip(y_pred,0,1))
    y_pred_neg = 1-y_pred_pos
    
    y_pos = K.round(K.clip(y_true,0,1))
    y_neg = 1-y_pos
    
    tp = K.sum(y_pos*y_pred_pos)
    tn = K.sum(y_neg*y_pred_neg)
    
    fp = K.sum(y_neg*y_pred_pos)
    fn = K.sum(y_pos*y_pred_neg)
    
    numerator = (tp * tn - fp * fn)
    denmintor = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator/(denmintor + K.epsilon())

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

min_num = -128 
max_num = 127
# This function standardize the data from (-128 to 127) to (-1 to 1)
def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    ts_std = (ts - min_data) / (max_data - min_data)   
    return ts_std * (range_needed[-1] - range_needed[0] ) + range_needed[0]    
    
    
# extract features
def transform_ts(ts, n_dim=400):
    # convert int8 into int 64
    ts = ts.astype('int64')
    ts_standardized = min_max_transf(ts, min_data=min_num, max_data=max_num)
    #print(ts_standardized[0:10])
    bucket_size = int(sample_size / n_dim)
    ts_new = []
    for i in range(0, sample_size, bucket_size):
        ts_rerange = ts_standardized[i: i+bucket_size]
        labels,counts = np.unique(ts_rerange,return_counts=True)
        ShanEntropy = stats.entropy(counts,base=10)
        mean = ts_rerange.mean()
        std = ts_rerange.std()
        max_val = mean + std 
        min_val = mean - std 
        mad = ts_rerange.mad()
        mode_series = stats.mode(ts_rerange)[0][0]
        mode = float(mode_series)
        skew = ts_rerange.skew()
        kurtosis = ts_rerange.kurt() - 3
        percentil_calc = np.percentile(ts_rerange,[10,25,50,75,90])
        perdis = percentil_calc[-1]-percentil_calc[0]
        ts_new.append(np.concatenate([np.asarray([max_val,min_val,mean,std,mad,mode,perdis,skew,ShanEntropy,kurtosis]),
                                      percentil_calc]))
    return np.asarray(ts_new)

def phases_corr(ts_three,n_dim=400):
    bucket_size = int(sample_size / n_dim)
    corr_phase =[]
    for i in range(0, sample_size, bucket_size):
        ts_rerange = ts_three[i: i+bucket_size,:]
        temp = np.corrcoef(ts_rerange)
        corr_phase.append(np.asarray(temp[0,1],temp[0,2],temp[1,2]))
    return np.asarray(corr_phase)
        
# transform raw data 
def prep_data(start, end):
    praq_train = pq.read_pandas(filepath+'/train.parquet', 
                                columns=[str(i) for i in range(start, end)]).to_pandas()
    X = []
    y = []
    # using tdqm to evaluate processing time
    for id_measurement in tqdm(y_train.index.levels[0].unique()[int(start/3):int(end/3)]):
        phase_correff = []
        # for each phase of the signal
        for phase in [0,1,2]:
            X_signal = []
            signal_id, target = y_train.loc[id_measurement].loc[phase]
            y.append(target)
            if phase==0:
                idx = [signal_id, signal_id+1, signal_id+2]
                phase_correff.append(phases_corr(praq_train[str(idx)]))
            X_signal.append(np.concatenate(transform_ts(praq_train[str(signal_id)]),
                                           phase_correff))
            X_signal = np.concatenate(X_signal, axis=1)
            X.append(X_signal)
        # concatenate all the 3 phases in one matrix
        # X_signal = np.concatenate(X_signal, axis=1)
        # add the data to X
        # X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y 

# load all data 
X = []
y = []
def load_all():
    total_size = len(y_train)
    for ini, end in [(0, int(total_size/2)), (int(total_size/2), total_size)]:
        X_temp, y_temp = prep_data(ini, end)
        X.append(X_temp)
        y.append(y_temp)

# load all data 
load_all()
X = np.concatenate(X)
y = np.concatenate(y)
print(X.shape, y.shape)
# save data into file, a numpy specific format
np.save("X.npy",X)
np.save("y.npy",y)
# load training data 
# X = np.load('X.npy')
# y = np.load('y.npy')

# This is NN LSTM Model creation
def model_lstm(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    # This is the LSTM layer
    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = CuDNNLSTM(32, return_sequences=True)(inp)
    # The second LSTM layer
    # x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)
    x = CuDNNLSTM(32, return_sequences=True)(x)
    # attention layer 
    x = Attention(input_shape[1])(x)
    # A intermediate full connected (Dense) can help to deal with nonlinears outputs
    x = Dense(64, activation="relu")(x)
    # x = Dense(32, activation="relu")(x)
    # A binnary classification as this must finish with shape (1,)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    # Pay attention in the addition of Matt_corcoe metric in the compilation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Matt_corcoe])
    
    return model

# train model 
splits = list(StratifiedKFold(n_splits=N_Splits, shuffle=True, random_state=2019).split(X, y))
preds_val = []
y_val = []
# Then, iteract with each fold
# If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]
for idx, (train_idx, val_idx) in enumerate(splits):
    K.clear_session() 
    print("Beginning fold {}".format(idx+1))
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    model = model_lstm(train_X.shape)
    # This checkpoint helps to avoid overfitting. 
    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True,
                           save_weights_only=True, verbose=1, monitor='val_Matt_corcoe', mode='max')
    model.fit(train_X, train_y, batch_size=128, epochs=80, 
              validation_data=[val_X, val_y], callbacks=[ckpt])
    # loads the best weights saved by the checkpoint
    model.load_weights('weights_{}.h5'.format(idx))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict(val_X, batch_size=512))
    # and the values of true y
    y_val.append(val_y)

# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[:,0]
y_val = np.concatenate(y_val)
preds_val.shape, y_val.shape

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        y_true_val = tf.convert_to_tensor(y_true.astype(np.float64))
        y_pred=tf.convert_to_tensor((y_proba > threshold).astype(np.float64))
        score = K.eval(Matt_corcoe(y_true_val, y_pred))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result

best_threshold = threshold_search(y_val, preds_val)['threshold']

# load test data 
meta_test = pd.read_csv(filepath+'/metadata_test.csv')
meta_test = meta_test.set_index(['signal_id'])
meta_test.head()
# divide test data into several parts
first_sig = meta_test.index[0]
n_parts = 9
max_line = len(meta_test)
part_size = int(max_line / n_parts)
last_part = max_line % n_parts
print(first_sig, n_parts, max_line, part_size, last_part, n_parts * part_size + last_part)
start_end = [[x, x+part_size] for x in range(first_sig, max_line + first_sig, part_size)]
start_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]
print(start_end)
X_test = []
phase_corr = []
for start, end in start_end:
    subset_test = pq.read_pandas(filepath+'/test.parquet', 
                                 columns=[str(i) for i in range(start, end)]).to_pandas()
    for k in range(0,subset_test.shape[1],3):
        phase_corr.append(np.corrcoef(subset_test[str(start+k)],subset_test[str(start+k+1)])[0,1])
        phase_corr.append(np.corrcoef(subset_test[str(start+k)],subset_test[str(start+k+2)])[0,1])
        phase_corr.append(np.corrcoef(subset_test[str(start+k+1)],subset_test[str(start+k+2)])[0,1])
    for i in tqdm(subset_test.columns):
        id_measurement, phase = meta_test.loc[int(i)]
        subset_test_col = subset_test[i]
        subset_trans = transform_ts(subset_test_col)
        X_test.append([i, id_measurement, phase, subset_trans])

# X_test_input = np.asarray([np.concatenate([X_test[i][3],X_test[i+1][3], 
#                                X_test[i+2][3]], axis=1) for i in range(0,len(X_test), 3)])
X_test_input = np.asarray([np.concatenate([X_test[i][3]],axis=1) for i in range(0,len(X_test))])
np.save("X_test.npy",X_test_input)
X_test_input.shape

# load submission sample 
submission = pd.read_csv(filepath+'/sample_submission.csv')
print('the length of submission is ', len(submission))
submission.head()

# predicted values         
preds_test = []
for i in range(N_Splits):
    model.load_weights('weights_{}.h5'.format(i))
    pred = model.predict(X_test_input, batch_size=300, verbose=1)
    preds_test.append(pred)
 
preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)
preds_test.shape
print('the predicted positive samples are ', preds_test.sum())   
preds_test_Data = pd.DataFrame(data=preds_test)
preds_test_Data.to_csv('preds_test.csv') 

#correct the predicted values based on phase-correction 
phase_corr_data = pd.DataFrame(data = phase_corr) 
phase_corr_data.to_csv('phase_corr.csv')
phase_corr_abs = phase_corr_data.abs()  
mean_corr = phase_corr_abs.mean()
std_corr = phase_corr_abs.std()
pred_pos = preds_test_Data[preds_test_Data[0] ==1]
predpos_index = pred_pos.index 
preds_test_new = preds_test
for i in range(len(predpos_index)):
    if predpos_index[i]%3 ==0 :
        idx = [predpos_index[i],predpos_index[i]+1,predpos_index[i]+2]
        three_phase_corr = phase_corr_abs.loc[idx]
        if (np.abs(three_phase_corr.mean())>np.abs(mean_corr.values-std_corr.values)).bool():
            preds_test_new[idx]=1
    elif predpos_index[i]%3 ==1 :
        idx = [predpos_index[i]-1, predpos_index[i], predpos_index[i]+1]
        three_phase_corr = phase_corr_abs.loc[idx]
        if (np.abs(three_phase_corr.mean())>np.abs(mean_corr.values-std_corr.values)).bool():
            preds_test_new[idx]=1
    else:
        idx = [predpos_index[i]-2, predpos_index[i]-1, predpos_index[i]]
        three_phase_corr = phase_corr_abs.loc[idx]
        if (np.abs(three_phase_corr.mean())>np.abs(mean_corr.values-std_corr.values)).bool():
            preds_test_new[idx]=1

pos_samples = sum(preds_test_new) 
print('the corrected positive samples are ', pos_samples)   

# submission['target'] = preds_test
submission['target'] = preds_test
submission.to_csv('submission.csv', index=False)
submission.head()



    




        
