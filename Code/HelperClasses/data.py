#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:24:57 2017

@author: nick
"""
import numpy as np

class Data:
    """
    Helper class Data
    Encapsulates all the training, validation and test data. This typically
    includes the images, labels and segmentation maps.
    """
    def __init__(self,data):
        """
        Args:
            data: - data loaded from a .npz file
            
        Attributes:
            X_train,X_val, X_test: input samples
            y_train, y_val, y_test: labels
            y_train_seg, y_val_seg, y_test_seg: segmentation masks if available
            y_train_bb, y_val_bb, y_test_bb: bounding box coordinates (xmin,xmax,ymin,ymax if available)
            img_height, img_width: original dimensions of input images
        """
        self.X_train = data['X_train'].astype(np.float32)
        self.X_val = data['X_valid'].astype(np.float32)
        self.X_test = data['X_test'].astype(np.float32)
        self.y_train = data['y_train'].astype(np.float32)
        self.y_val = data['y_valid'].astype(np.float32)
        self.y_test = data['y_test'].astype(np.float32)
        if 'y_train_seg' in data:
            self.y_train_seg = data['y_train_seg']
        if 'y_valid_seg' in data:
            self.y_val_seg = data['y_valid_seg']
        if 'y_test_seg' in data:
            self.y_test_seg = data['y_test_seg']
        ##
        if 'y_train_bb' in data:
            self.y_train_bb = data['y_train_bb']
        if 'y_valid_bb' in data:
            self.y_val_bb = data['y_valid_bb']
        if 'y_test_bb' in data:
            self.y_test_bb = data['y_test_bb']
        
            
        # check dimensionality of data
        if self.X_train.ndim == 3:
            self.img_height, self.img_width = self.X_train[0].shape
            self.color_channels = 1
        elif self.X_train.ndim == 4:
            self.img_height, self.img_width, self.color_channels = self.X_train[0].shape
        else:
            raise ValueError('Need 2D or 3D input images!')
            
        
    
    def getData(self,dataset,batchSize = None, indices = None, segMap = False, bb =False):
        """
        Method returning a set of training samples
        
        Args:
            - dataset: the set from which data should come, i.e. 'train', 'val', 'test'
            - batchSize: the number of samples to be returned. If None, return
                            whole dataset
            - indices: specify concrete datasamples to be returned. If not 
                        specified, get first batchSize samples
            - segMap: boolean, that indicates whether segmentation Map should be 
                        returned as well
            - bb: boolean that indicates whether bounding box coords should be 
                        returned as well
        """
        
        dic = {'train': ('X_train','y_train','y_train_seg','y_train_bb'), 
               'val': ('X_val','y_val', 'y_val_seg', 'y_val_bb'), 
               'test': ('X_test','y_test','y_test_seg','y_test_bb')}
        
        dataset_ = dic[dataset]
        # check that whether segMap criterion can be fulfilled
        if (segMap and not np.any(self.__dict__[dataset_[2]])):
            raise RuntimeError('Desired segmentation mask data is not available!')
        
        # check whether bounding box data is available
        if (bb and not np.any(self.__dict__[dataset_[3]])):
            raise RuntimeError('Desired bounding box data not availabel!')
                    
        if not batchSize and not np.any(indices):
            # return whole dataset
            if segMap and bb:
                return (self.__dict__[dataset_[0]],self.__dict__[dataset_[1]], 
                        self.__dict__[dataset_[2]],self.__dict__[dataset_[3]])
            elif segMap:
                return (self.__dict__[dataset_[0]],self.__dict__[dataset_[1]], 
                        self.__dict__[dataset_[2]])
            elif bb:
                return (self.__dict__[dataset_[0]],self.__dict__[dataset_[1]], 
                        self.__dict__[dataset_[3]])
            else:
                return (self.__dict__[dataset_[0]],self.__dict__[dataset_[1]])
        elif not np.any(indices):
            # return first batch size samples
            if segMap and bb:
                return (self.__dict__[dataset_[0]][:batchSize],self.__dict__[dataset_[1]][:batchSize],
                        self.__dict__[dataset_[2]][:batchSize],self.__dict__[dataset_[3]][:batchSize])
            elif segMap:
                return (self.__dict__[dataset_[0]][:batchSize],self.__dict__[dataset_[1]][:batchSize],
                        self.__dict__[dataset_[2]][:batchSize])
            elif bb:
                return (self.__dict__[dataset_[0]][:batchSize],self.__dict__[dataset_[1]][:batchSize],
                        self.__dict__[dataset_[3]][:batchSize])
            else:
                return (self.__dict__[dataset_[0]][:batchSize],self.__dict__[dataset_[1]][:batchSize])
        else:
            if segMap and bb:
                return (self.__dict__[dataset_[0]][indices],self.__dict__[dataset_[1]][indices], 
                        self.__dict__[dataset_[2]][indices],self.__dict__[dataset_[3]][indices])  
            elif segMap:
                return (self.__dict__[dataset_[0]][indices],self.__dict__[dataset_[1]][indices], 
                        self.__dict__[dataset_[2]][indices])  
            elif bb:
                return (self.__dict__[dataset_[0]][indices],self.__dict__[dataset_[1]][indices], 
                        self.__dict__[dataset_[3]][indices]) 
            else:
                return (self.__dict__[dataset_[0]][indices],self.__dict__[dataset_[1]][indices])
        
    def hasData(self, dataset, segMap = False, bb = False):
        """
        Method that checks whether type of dataset is present in data
        
        Args:
            dataset: the set from which data should come, i.e. 'train', 'val', 'test'
            segMap: boolean, that indicates whether we should look for segmentation maps
        """
        dic = {'train': ('X_train','y_train','y_train_seg','y_train_bb'), 
               'val': ('X_val','y_val', 'y_val_seg', 'y_val_bb'), 
               'test': ('X_test','y_test','y_test_seg','y_test_bb')}
        
        dataset_ = dic[dataset]
        
        return (dataset_[0] in self.__dict__.keys() and dataset_[1] in self.__dict__.keys() and 
                (not segMap or dataset_[2] in self.__dict__.keys()) and 
                (not bb or dataset_[3] in self.__dict__.keys()))
        
    def getSize(self, dataset):
        """
        Method that returns size of the dataset
        
        Args:
            dataset: - the set of interest, i.e. 'train', 'val', 'test'
        """
        dic = {'train': 'X_train', 'val': 'X_val', 'test': 'X_test'}
        dataset_= dic[dataset]
        
        if dataset_ in self.__dict__.keys():
            return self.__dict__[dataset_].shape[0]
        else:
            return 0
        
    def get_dimensions(self):
        """
        return current dimensionality of the input images
        """
        return self.X_train[0].shape
    
    def get_label_dimensions(self):
        """
        return dimensionality of labels (typically one-hot vectors)
        """
        return self.y_train[0].shape[0]
    
    def getImgHeight(self):
        return self.img_height
    
    def getImgWidth(self):
        return self.img_width
    
    def get_num_color_channels(self):
        return self.color_channels
    
    def flatten(self):
        """
        - Flatten input 2D images to 1D
        - Flattened input necessary for networks
        - Only flattens data if that did not already happen
        """
        dic = {'train': 'X_train', 'val': 'X_val', 'test': 'X_test'}
        
        for d in dic:
            if self.__dict__[dic[d]].ndim == 3:
                self.__dict__[dic[d]] = self.__dict__[dic[d]].reshape((self.__dict__[dic[d]].shape[0],
                                     self.__dict__[dic[d]].shape[1]*self.__dict__[dic[d]].shape[2]))
            if self.__dict__[dic[d]].ndim == 4:
                self.__dict__[dic[d]] = self.__dict__[dic[d]].reshape((self.__dict__[dic[d]].shape[0],
                                     self.__dict__[dic[d]].shape[1]*self.__dict__[dic[d]].shape[2]*\
                                    self.__dict__[dic[d]].shape[3]))
            
    
    def reshape(self):
        """
        - bring input data back to 2D shape
        - only reshapes the data that needs reshaping
        """
        dic = {'train': 'X_train', 'val': 'X_val', 'test': 'X_test'}
        
        for d in dic:
            if self.__dict__[dic[d]].ndim == 2:
                self.__dict__[dic[d]] = self.__dict__[dic[d]].reshape((self.__dict__[dic[d]].shape[0],
                                     self.img_height,self.img_width))
            elif self.__dict__[dic[d]].ndim == 3:
                self.__dict__[dic[d]] = self.__dict__[dic[d]].reshape((self.__dict__[dic[d]].shape[0],
                                     self.img_height,self.img_width, self.color_channels))
