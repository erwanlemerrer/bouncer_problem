''' Code for the experiment proposed in Section 4.2 of the paper "The Bouncer Problem: Challenges to Remote Explainability".
Code author: Erwan Le Merrer (elemerre@acm.org)

The code leverages the German Credit Dataset, available at https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data).

Rationale: the code is composed of 3 parts:
1) Train a model to predict credit default based on all features, including discriminative ones (i.e., age, sex, employment,foreigner)
2) Train a model on all but the discriminative features
3) This part simply computes the fraction of label changes (ie, IPs) between the prediction of the original and the model without discriminative features
   (the four discriminatory features of each of 50 test profiles are sequentially replaced by the ones of the 49 remaining profiles; each resulting test profile is fed to the model for prediction)
   The results are used as a basis for Figure 4, after averaging 30 trials.
'''

from __future__ import division

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers

import pandas as pd
import numpy as np

import random
import sys
import os.path

'''Params
'''
test_size = 50 # for final comparison of the two models with real inputs from the dataset
epochs=100

# reading formating data
train = pd.read_csv('./german.data-numeric', sep="\s+", header=None)
full = train.iloc[:, :-1].values.astype('float32')
labels = train.iloc[:,-1].values.astype('int32')
labels -= 1 # instead of 1 or 2 representation

test_indices = random.sample(range(0, len(labels)), test_size)

X_test = np.take(full, test_indices, axis=0)
y_test = np.take(labels, test_indices, axis=0)

X_train = np.delete(full, test_indices, axis=0)
y_train = np.delete(labels, test_indices, axis=0)


# pre-processing
scale_X_train = np.max(X_train)
X_train /= scale_X_train
scale_X_test = np.max(X_test)
X_test /= scale_X_test

''' 1) Original neural network with no attack
'''
model = Sequential()
model.add(Dense(23, input_dim=X_train.shape[1]))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = optimizers.adam(lr=0.1)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
hist = model.fit(X_train, y_train, epochs=epochs, validation_split=0.25, callbacks=[] ,verbose=2)

print("avg %f / std %f" % (np.average(hist.history['val_acc']),np.std(hist.history['val_acc'])))

''' 2) Tweaked neural network without discriminative features
'''
x_ = [6,7,10,15] # column indices of discriminatory features: 
                 # respectively employment, sex/status, age, foreigner (/!\ not same number in non numeric csv!
# fully delete rows:
X_train_legit = np.delete(X_train, x_, axis=1)
X_test_legit  = np.delete(X_test,x_, axis=1)

assert X_train_legit.shape[1]+len(x_) == X_train.shape[1]

# y_train same, no need for modifications
X_train_legit /= np.max(X_train_legit)
X_test_legit /= np.max(X_test_legit)

model_proxy = Sequential()
model_proxy.add(Dense(23, input_dim=X_train_legit.shape[1]))
model_proxy.add(Activation('sigmoid'))
model_proxy.add(Dense(1))
model_proxy.add(Activation('sigmoid'))

model_proxy.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
hist_proxy = model_proxy.fit(X_train_legit, y_train, epochs=epochs, validation_split=0.25, callbacks=[] ,verbose=2)
print("avg %f / std %f" % (np.average(hist_proxy.history['val_acc']),np.std(hist_proxy.history['val_acc'])))



''' 3) Analysis
'''
preds = model.predict_classes(X_test, verbose=0)
preds_legit = model_proxy.predict_classes(X_test_legit, verbose=0)
#for l in range(len(preds)): print("%d %d" % (preds[l],preds_legit[l]))
print("Disagreement_ratio=%f" % (np.count_nonzero(np.not_equal(preds,preds_legit))/test_size))


'all2all user swapping discriminative features'
print("*** Percentage of incoherent pairs (IPs) ***")
c = 0
ct = 0
for i in range(len(X_test)):
    X_test_ = np.copy(X_test)
    for j in range(len(X_test)):
        if i == j: continue
        
        X_test_[i][6] = X_test[j][6]
        X_test_[i][7] = X_test[j][7]
        X_test_[i][10] = X_test[j][10]
        X_test_[i][15] = X_test[j][15]

        if model.predict_classes(np.reshape(X_test[i], (1, 24)), verbose=0) != \
            model.predict_classes(np.reshape(X_test_[i], (1, 24)), verbose=0):
            c+=1
        ct+=1
print("All four: %f" % (c/ct))
fall = c/ct

c = 0
ct = 0
for i in range(len(X_test)):
    X_test_ = np.copy(X_test)
    for j in range(len(X_test)):
        if i == j: continue
        
        X_test_[i][6] = X_test[j][6]

        if model.predict_classes(np.reshape(X_test[i], (1, 24)), verbose=0) != \
            model.predict_classes(np.reshape(X_test_[i], (1, 24)), verbose=0):
            c+=1
        ct+=1
print("Employment:  %f" % (c/ct))
f6 = c/ct

c = 0
ct = 0
for i in range(len(X_test)):
    X_test_ = np.copy(X_test)
    for j in range(len(X_test)):
        if i == j: continue
        
        X_test_[i][7] = X_test[j][7]

        if model.predict_classes(np.reshape(X_test[i], (1, 24)), verbose=0) != \
            model.predict_classes(np.reshape(X_test_[i], (1, 24)), verbose=0):
            c+=1
        ct+=1
print("Sex/status: %f" % (c/ct))
f7 = c/ct

c = 0
ct = 0
for i in range(len(X_test)):
    X_test_ = np.copy(X_test)
    for j in range(len(X_test)):
        if i == j: continue

        X_test_[i][10] = X_test[j][10]

        if model.predict_classes(np.reshape(X_test[i], (1, 24)), verbose=0) != \
            model.predict_classes(np.reshape(X_test_[i], (1, 24)), verbose=0):
            c+=1
        ct+=1
print("Age: %f" % (c/ct))
f10 = c/ct

c = 0
ct = 0
for i in range(len(X_test)):
    X_test_ = np.copy(X_test)
    for j in range(len(X_test)):
        if i == j: continue

        X_test_[i][15] = X_test[j][15]

        if model.predict_classes(np.reshape(X_test[i], (1, 24)), verbose=0) != \
            model.predict_classes(np.reshape(X_test_[i], (1, 24)), verbose=0):
            c+=1
        ct+=1
print("Foreigner: %f" % (c/ct))
f15 = c/ct

