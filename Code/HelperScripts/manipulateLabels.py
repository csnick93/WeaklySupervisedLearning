# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:55:22 2017

@author: nickv
"""

"""
Small script that manipulates .npz files regarding the saved format
    - e.g. manipulate labels to convert them to a one-hot-vector
"""

import numpy as np
import os


def reduceDataAmount(data_path,perc):
    """
    reduce the dataset to a percentile of its size
    
    Parameters:
        perc: the percentage we want the data to be reduced to
    """
    data = np.load(data_path)
    
    # extract data
    X_train = data['X_train']
    y_train = data['y_train']
    y_train_seg = data['y_train_seg']
    X_val = data['X_valid']
    y_val = data['y_valid']
    y_val_seg = data['y_valid_seg']
    X_test = data['X_test']
    y_test = data['y_test']
    y_test_seg = data['y_test_seg']

    #save under changed name
    target_path = data_path.replace('.npz','_'+str(perc)+'.npz')
    np.savez(target_path,X_train=X_train[:int(perc*X_train.shape[0])], 
            y_train = y_train[:int(perc*y_train.shape[0])], 
            y_train_seg = y_train_seg[:int(perc*y_train_seg.shape[0])],
            X_valid = X_val[:int(perc*X_val.shape[0])], 
            y_valid = y_val[:int(perc*y_val.shape[0])], 
            y_valid_seg = y_val_seg[:int(perc*y_val_seg.shape[0])],
            X_test = X_test[:int(perc*X_test.shape[0])], 
            y_test = y_test[:int(perc*y_test.shape[0])], 
            y_test_seg = y_test_seg[:int(perc*y_test_seg.shape[0])])
    
def renameLabels(data_path):
    data = np.load(data_path)
    
    # extract data
    X_train = data['X_train']
    y_train = data['y_train_label']
    y_train_seg = data['y_train_seg']
    X_val = data['X_valid']
    y_val = data['y_valid_label']
    y_val_seg = data['y_val_seg']
    X_test = data['X_test']
    y_test = data['y_test_label']
    y_test_seg = data['y_test_seg']
    
    #save under same name
    np.savez(data_path,X_train=X_train, y_train = y_train, y_train_seg = y_train_seg,
            X_valid = X_val, y_valid = y_val, y_valid_seg = y_val_seg,
            X_test = X_test, y_test = y_test, y_test_seg = y_test_seg)

def convertToOneHotLabels(data_path, t_path):
    """
    Parameters
        data_path: path where the data lies
        t_path: path where manipulated dataset shall be saved
    """
    # load dataset
    data = np.load(data_path)
    
    # extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_valid']
    y_val = data['y_valid']
    X_test = data['X_test']
    y_test = data['y_test']
    
    max_label = np.max([np.max(y_train),np.max(y_val),np.max(y_test)])
    
    # manipulate labels
    y = [y_train,y_val,y_test]
    y_man = []
    
    for i in range(len(y)):
        y_man.append(np.eye(max_label+1)[y[i].flatten()])
    
        
    y_train_man, y_val_man, y_test_man= y_man
    
    
    # save labels
    np.savez(t_path,X_train=X_train, y_train = y_train_man, 
             X_valid = X_val, y_valid = y_val_man, 
             X_test = X_test, y_test = y_test_man)
    
    
    # check saved file
    data = np.load(t_path)
    print(data.files)
    
def reshapeImages(data_path, t_path):
    """
    Parameters
        data_path: path where the data lies
        t_path: path where manipulated dataset shall be saved
    """
    # load dataset
    data = np.load(data_path)
    
    # extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_valid']
    y_val = data['y_valid']
    X_test = data['X_test']
    y_test = data['y_test']
    
    X_train = X_train.reshape((X_train.shape[0],40,40))
    X_val = X_val.reshape((X_val.shape[0],40,40))
    X_test = X_test.reshape((X_test.shape[0],40,40))
    
    # save labels
    np.savez(t_path,X_train=X_train, y_train = y_train, 
             X_valid = X_val, y_valid = y_val, 
             X_test = X_test, y_test = y_test)
    
    
if __name__ == '__main__':
    reshapeImages(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'Datasets',
                 'ClutteredMNIST', 'mnist_sequence1_sample_5distortions5x5_one_hot.npz'),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'Datasets',
                 'ClutteredMNIST', 'mnist_sequence1_sample_5distortions5x5_one_hot_reshaped.npz'))
    reduceDataAmount(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'Datasets',
                 'EmbeddedMNIST', 'embeddedMNIST.npz'), 0.1)
    