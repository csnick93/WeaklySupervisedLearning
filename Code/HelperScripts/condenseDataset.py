# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:32:41 2017

@author: nickv
"""

"""
Script to condense embeddedMNIST dataset
"""

import os
import numpy as np

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                         'Datasets','EmbeddedMNIST','embeddedMNIST_gray.npz')

perc = 0.01

saveTo_path = data_path.rstrip('.npz')+'_'+str(perc)+'.npz'

data = np.load(data_path)

X_train = data['X_train']
y_train = data['y_train']

X_valid = data['X_valid']
y_valid = data['y_valid']


X_test = data['X_test']
y_test = data['y_test']

if ('y_train_seg' in data and 'y_valid_seg' in data and 'y_test_seg' in data):
    y_train_seg = data['y_train_seg']
    y_valid_seg = data['y_valid_seg']
    y_test_seg = data['y_test_seg']


if ('y_train_seg' in data and 'y_valid_seg' in data and 'y_test_seg' in data):
    np.savez(saveTo_path,X_train=X_train[0:int(X_train.shape[0]*perc)], 
             y_train = y_train[0:int(y_train.shape[0]*perc)], 
             y_train_seg = y_train_seg[0:int(y_train_seg.shape[0]*perc)],
             X_valid=X_valid[0:int(X_valid.shape[0]*perc)], 
             y_valid = y_valid[0:int(y_valid.shape[0]*perc)], 
             y_valid_seg = y_valid_seg[0:int(y_valid_seg.shape[0]*perc)],
             X_test=X_test[0:int(X_test.shape[0]*perc)], 
             y_test = y_test[0:int(y_test.shape[0]*perc)], 
             y_test_seg = y_test_seg[0:int(y_test_seg.shape[0]*perc)])
    
else:
    np.savez(saveTo_path,X_train=X_train[0:int(X_train.shape[0]*perc)], 
             y_train = y_train[0:int(y_train.shape[0]*perc)], 
             X_valid=X_valid[0:int(X_valid.shape[0]*perc)], 
             y_valid = y_valid[0:int(y_valid.shape[0]*perc)],
             X_test=X_test[0:int(X_test.shape[0]*perc)], 
             y_test = y_test[0:int(y_test.shape[0]*perc)])