# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 09:43:37 2017

@author: nickv
"""
import numpy as np
import os

def prepare_mnist(dest_path,color):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
    X_train = mnist.train.images.reshape((55000,28,28,1))
    y_train = mnist.train.labels
    X_val = mnist.validation.images.reshape((5000,28,28,1))
    y_val = mnist.validation.labels
    X_test = mnist.test.images.reshape((10000,28,28,1))
    y_test = mnist.test.labels
    
    def normalize(x):
        x_norm = np.zeros(x.shape)
        for i,img in enumerate(x):
            img = img-np.min(img)
            img = img/np.max(img)
            x_norm[i] = img
        return x_norm
    
    if color:
        X_train = np.tile(X_train,reps = [1,1,1,3])
        X_val = np.tile(X_val,reps = [1,1,1,3])
        X_test = np.tile(X_test,reps = [1,1,1,3])
    
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)
    
    f = 'mnist_color.npz' if color else 'mnist.npz'
    np.savez(os.path.join(dest_path,f), X_train = X_train, y_train = y_train,
             X_valid = X_val, y_valid = y_val, X_test = X_test, y_test = y_test)
    
    
    
if __name__=='__main__':
    color = True
    prepare_mnist(r'D:\MasterProjekt\WeaklySupervisedLearning\Datasets\MNIST',color)