# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:13:12 2017

@author: nickv
"""

"""
Class Dataloader

This class is responsible for loading the different datasets for the weakly
supervised learning task. The default datasets include clutteredMNIST,...
It is also possible to load datasets from a local drive by providing
the respective path to the folder.

NOTE:
    For now it is required that the datasets are in a .npz format containing:
        X_train, y_train, X_valid, y_valid, X_test,y_test
    There will later be a script that brings data into this required format.
"""

import os
import numpy as np
from sys import path
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'HelperClasses'))
from data import Data
#import urllib



class DataLoader:
    
    ####################
    # Static Variables #
    ####################
    
    # list of default datasets with respective link to server where dataset lies (for now: just use local path)
    default_datasets = {'MNIST': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.realpath(__file__)))),
                                    'Datasets','MNIST','mnist.npz')),
                        'MNIST_color': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.realpath(__file__)))),
                                    'Datasets','MNIST','mnist_color.npz')),
                        'cMNIST': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.realpath(__file__)))),
                                    'Datasets','ClutteredMNIST','cMNIST.npz')),
                        'cMNIST_color': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.realpath(__file__)))),
                                    'Datasets','ClutteredMNIST','cMNIST_color.npz')),
                        'red_cMNIST': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                        os.path.dirname(os.path.realpath(__file__)))),
                                                        'Datasets','ClutteredMNIST','cMNIST_0.01.npz')),
                        'embMNIST_gray': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.realpath(__file__)))),
                                    'Datasets','EmbeddedMNIST','embeddedMNIST_gray.npz')),
                        'embMNIST': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.realpath(__file__)))),
                                    'Datasets','EmbeddedMNIST','embeddedMNIST.npz')),
                        'cifar_gray': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                                os.path.dirname(os.path.realpath(__file__)))),
                                                                'Datasets','CIFAR10','cifar10_gray.npz')),
                        'cifar': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                                os.path.dirname(os.path.realpath(__file__)))),
                                                                'Datasets','CIFAR10','cifar10.npz')),
                        'graz_gray': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                                os.path.dirname(os.path.realpath(__file__)))),
                                                                'Datasets','GRAZ','graz_gray.npz')),
                        'graz': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                                os.path.dirname(os.path.realpath(__file__)))),
                                                                'Datasets','GRAZ','graz.npz')),
                        'graz_red_gray': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                                os.path.dirname(os.path.realpath(__file__)))),
                                                                'Datasets','GRAZ','graz_gray_0.01.npz')),
                        'graz_red': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                                os.path.dirname(os.path.realpath(__file__)))),
                                                                'Datasets','GRAZ','graz_0.01.npz')),
                        'voc': os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                                                                os.path.dirname(os.path.realpath(__file__)))),
                                                                'Datasets','VOC','voc.npz'))}     
    
    
   
    def __init__(self,dataset):
        """
        Args:
            dataset: The dataset to be loaded 
                        - either a default dataset: cMNIST,...
                        - or a local dataset: provide path
        """
        self.dataset = dataset        
        
    
    ################ 
    # Load dataset #
    ################
    
    def load(self):
        """
        Loading the dataset (potentially from a server address)
        """
        print("Loading dataset...")
        
        if self.dataset in DataLoader.default_datasets:
            # loading default dataset from somewhere on server
            #dataFolder,_ = urllib.request.urlretrieve(DataLoader.default_datasets(self.dataset), 
            #                          os.path.join(self.targetFolder,self.dataset))
            # for now just use the local path
            data = np.load(DataLoader.default_datasets[self.dataset],encoding='bytes')
        else:
            # loading local dataset
            data = np.load(self.dataset)
        
        data_ = Data(data)
        
        return data_
        
if __name__ == "__main__":
    pass