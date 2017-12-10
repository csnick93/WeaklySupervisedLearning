# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:49:47 2017

@author: nickv
"""

import numpy as np
import os
from keras.datasets import cifar10

def prepare_cifar10(target_path, color = True):
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    I = np.identity(10) 
    y_train_one_hot = I[y_train[:,0]]
    y_test_one_hot = I[y_test[:,0]]
    x_val, x_test = x_test[:5000], x_test[5000:]
    y_valid_one_hot,y_test_one_hot = y_test_one_hot[:5000],y_test_one_hot[5000:]
    
    def normalize(x):
        x_norm = np.zeros(x.shape)
        for i,img in enumerate(x):
            img = img-np.min(img)
            img = img/np.max(img)
            x_norm[i] = img
        return x_norm
    
    if not color:
        x_train_gray = np.average(x_train,axis=3)
        x_val_gray = np.average(x_val,axis=3)
        x_test_gray = np.average(x_test,axis=3)
        
        x_train_gray = np.reshape(x_train_gray, (x_train_gray.shape[0],x_train_gray.shape[1],x_train_gray.shape[2],1))
        x_val_gray = np.reshape(x_val_gray, (x_val_gray.shape[0],x_val_gray.shape[1],x_val_gray.shape[2],1))
        x_test_gray = np.reshape(x_test_gray, (x_test_gray.shape[0],x_test_gray.shape[1],x_test_gray.shape[2],1))
        
        x_train_gray = normalize(x_train_gray)
        x_val_gray = normalize(x_val_gray)
        x_test_gray = normalize(x_test_gray)
        
        np.savez(os.path.join(target_path,'cifar10_gray.npz'), X_train = x_train_gray, y_train = y_train_one_hot,
                 X_valid = x_val_gray, y_valid = y_valid_one_hot, X_test = x_test_gray, 
                 y_test = y_test_one_hot)
        
    else:
        x_train = normalize(x_train)
        x_val = normalize(x_val)
        x_test = normalize(x_test)
        np.savez(os.path.join(target_path,'cifar10.npz'), X_train = x_train, y_train = y_train_one_hot,
                 X_valid = x_val, y_valid = y_valid_one_hot, X_test = x_test, 
             y_test = y_test_one_hot)
    

if __name__=='__main__':
    color = True
    prepare_cifar10(r'D:\MasterProjekt\WeaklySupervisedLearning\Datasets\CIFAR10', color)
    