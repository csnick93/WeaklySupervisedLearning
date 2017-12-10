# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 22:10:01 2017

@author: nickv
"""

import numpy as np
import os

def concatenate_datasets(d, target_name):
    """
    Method concatenates datasets. Necessary for VOC preparation
    due to dataset size.
    
    Args:
        d: path to directory where datasets lie
        target_name: name of concatenated dataset
    """    
    
    for i,dataset in enumerate(os.listdir(d)):
        print(i)
        data = np.load(os.path.join(d,dataset))
        
        if i == 0:
            X_train = data['X_train']
            y_train = data['y_train']
            y_train_seg = data['y_train_seg']
            
            X_val = data['X_valid']
            y_val = data['y_valid']
            y_val_seg = data['y_valid_seg']
            
            X_test = data['X_test']
            y_test = data['y_test']
            y_test_seg = data['y_test_seg']
        else:
            X_train = np.concatenate((X_train,data['X_train']))
            y_train = np.concatenate((y_train,data['y_train']))
            y_train_seg = np.concatenate((y_train_seg,data['y_train_seg']))
            
            X_val = np.concatenate((X_val,data['X_valid']))
            y_val = np.concatenate((y_val,data['y_valid']))
            y_val_seg = np.concatenate((y_val_seg,data['y_valid_seg']))
            
            X_test = np.concatenate((X_test,data['X_test']))
            y_test = np.concatenate((y_test,data['y_test']))
            y_test_seg = np.concatenate((y_test_seg,data['y_test_seg']))
        
    print('Saving to file')
    np.savez(os.path.join(d,target_name), X_train = X_train, y_train = y_train, y_train_seg = y_train_seg,
             X_valid = X_val, y_valid = y_val, y_valid_seg = y_val_seg,
             X_test = X_test, y_test = y_test, y_test_seg = y_test_seg)
    
if __name__=='__main__':
    concatenate_datasets(r'D:\MasterProjekt\WeaklySupervisedLearning\Datasets\trial_voc', 'voc.npz')