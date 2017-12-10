# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 19:18:53 2017

@author: nickv
"""
import os
import numpy as np

datasets = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),'Datasets')


files = [os.path.join('EmbeddedMNIST','embeddedMNIST.npz'),
         os.path.join('EmbeddedMNIST','embeddedMNIST_0.1.npz'),
         os.path.join('GRAZ','graz.npz'),
         os.path.join('MNIST','mnist.npz'),
         os.path.join('VOC','voc_1000.npz'),
         os.path.join('ClutteredMNIST','cMNIST.npz'),
         os.path.join('CIFAR10','cifar10.npz')]

for f in files:
    print(f)
    data = np.load(os.path.join(datasets,f))
    
    X_train = data['X_train']
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
    
    X_val = data['X_valid']
    X_val = np.reshape(X_val,(X_val.shape[0],X_val.shape[1],X_val.shape[2],1))
    
    X_test = data['X_test']
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
    
    if 'y_train_seg' in data:
        np.savez(os.path.join(datasets,f), X_train = X_train, y_train = data['y_train'], y_train_seg = data['y_train_seg'],
                 X_valid = X_val, y_valid = data['y_valid'], y_valid_seg=data['y_valid_seg'], 
                 X_test = X_test, y_test = data['y_test'], y_test_seg = data['y_test_seg'])
        
    else:
        np.savez(os.path.join(datasets,f), X_train = X_train, y_train = data['y_train'], 
                 X_valid = X_val, y_valid = data['y_valid'], 
                 X_test = X_test, y_test = data['y_test'])