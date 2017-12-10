# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:58:32 2017

@author: nickv
"""
import os
import numpy as np

def colorize_cmnist(path):
    data = np.load(os.path.join(path,'cMNIST.npz'))
    
    X_train = np.zeros((10000,40,40,3))
    X_valid = np.zeros((1000,40,40,3))
    X_test = np.zeros((1000,40,40,3))
    
    for i,img in enumerate(data['X_train']):
        X_train[i,:,:,0] = img[:,:,0]
        X_train[i,:,:,1] = img[:,:,0]
        X_train[i,:,:,2] = img[:,:,0]
               
    for i,img in enumerate(data['X_valid']):
        X_valid[i,:,:,0] = img[:,:,0]
        X_valid[i,:,:,1] = img[:,:,0]
        X_valid[i,:,:,2] = img[:,:,0]
               
    for i,img in enumerate(data['X_test']):
        X_test[i,:,:,0] = img[:,:,0]
        X_test[i,:,:,1] = img[:,:,0]
        X_test[i,:,:,2] = img[:,:,0]
              
              
    np.savez(os.path.join(path,'cMNIST_color.npz'),
             X_train = X_train, y_train = data['y_train'], X_valid = X_valid, 
            y_valid = data['y_valid'], X_test = X_test, y_test=data['y_test'])
    
    
if __name__=='__main__':
    #
    p = os.path.dirname(os.path.dirname(os.getcwd()))
    path = os.path.join(p,'Datasets','ClutteredMNIST')
    print(path)
    #colorize_cmnist(p)