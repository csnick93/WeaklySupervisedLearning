# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 19:56:43 2017

@author: nickv
"""

import os
import numpy as np

def repair_graz(color = True):
    path = os.path.dirname(os.path.dirname(os.getcwd()))
    path = os.path.join(path,'Datasets','GRAZ')
    if color:
        path = os.path.join(path,'graz_color.npz')
    else:
        path = os.path.join(path,'graz.npz')
        
    data = np.load(path)
    
    X_train = data['X_train']
    X_val = data['X_valid']
    X_test = data['X_test']
    y_train_seg = data['y_train_seg']
    y_val_seg = data['y_val_seg']
    y_test_seg = data['y_test_seg']
    
    
    # repair labels
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']
    
    def one_hot(labels):
        identity = np.identity(len(set(labels)))
        one_hot = identity[labels]
        return one_hot
    
    y_train_one_hot = one_hot(y_train)
    y_valid_one_hot = one_hot(y_valid)
    y_test_one_hot = one_hot(y_test)
    
    print('Saving to file')
    np.savez(path, X_train = X_train, y_train = y_train_one_hot, y_train_seg = y_train_seg,
             X_valid = X_val, y_valid = y_valid_one_hot, y_valid_seg=y_val_seg, 
             X_test = X_test, y_test = y_test_one_hot, y_test_seg = y_test_seg)
    
    
    
    
if __name__=='__main__':
    color = True
    repair_graz(color)