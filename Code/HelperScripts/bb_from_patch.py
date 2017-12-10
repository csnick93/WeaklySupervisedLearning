#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:17:35 2017

@author: nick
"""

"""
Script that attempts to compute bounding box 
given the parameters gx,gy,delta,sigma,attention_n
"""

import numpy as np
import os

class DRAD_Params:
    """
    Encapsulation class for parameters
    """
    def __init__(self,gx,gy,delta,sigma,attention_n):
        self.gx= gx
        self.gy = gy
        self.delta = delta
        self.sigma = sigma
        self.attention_n = attention_n
        
class BoundingBox:
    """
    Datastructure to encapsule bounding box coordinates
    """
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise ValueError('Invalid arguments for bounding box object')
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
    def area(self):
        """
        compute area of bounding box
        """
        return (self.x_max-self.x_min)*(self.y_max-self.y_min)
    
    def cout(self):
        import matplotlib.pyplot as plt
        plt.plot([self.y_min, self.y_max,self.y_max,self.y_min, self.y_min],
                 [self.x_min,self.x_min,self.x_max,self.x_max,self.x_min])
        plt.show()
        plt.close()


def computeBBfromParams(params):
    """
    Compute the bounding box given the params
    """
    x_min = np.int32(params.gx - (params.delta*(params.attention_n-1)+1)/2)# - params.sigma)
    x_max = np.int32(params.gx + (params.delta*(params.attention_n-1)+1)/2)# + params.sigma)
    y_min = np.int32(params.gy - (params.delta*(params.attention_n-1)+1)/2)# - params.sigma)
    y_max = np.int32(params.gy + (params.delta*(params.attention_n-1)+1)/2)# + params.sigma)

    return BoundingBox(x_min,x_max,y_min,y_max)
    

def computeBBfromSM(segMask):
        """
        Method that computes bounding box from a binary segmentation mask with
        only a single object present
        
        Arguments:
            segMask: 2D binary image
        """
        if len(segMask.shape)!= 2:
            raise ValueError('Cannot compute bounding box from non-2D image')
            
        x_coords,y_coords = np.where(segMask > 0)
        
        return BoundingBox(np.min(y_coords), np.max(y_coords), np.min(x_coords), np.max(x_coords))

def computePatch(params,img):
    """
    Compute the image patch given
    
    Arguments:
        params: attention parameters
        img: image of interest
    """
    # compute attention windows
    img_height = img.shape[0]
    img_width = img.shape[1]
    grid = np.arange(params.attention_n,dtype = np.float32).reshape((
                        params.attention_n,1))
    #print('Grid Shape: %s' %(grid.shape,))
    mu_x = params.gx + (grid - params.attention_n/2+0.5)*params.delta
    #print('Mu_x Shape: %s' %(mu_x.shape,))
    mu_y = params.gy + (grid - params.attention_n/2+0.5)*params.delta    
    #print('Mu_y Shape: %s' %(mu_y.shape,))
    img_x = np.tile(np.arange(img_width, dtype=np.float32), params.attention_n).reshape(
                [params.attention_n, img_width])
    #print('Img_x Shape: %s' %(img_x.shape,))
    img_y = np.tile(np.arange(img_height, dtype=np.float32), params.attention_n).reshape(
                [params.attention_n, img_height])
    #print('Img_y Shape: %s' %(img_y.shape,))
    Fx = np.exp(-np.square((img_x-mu_x)/(2*params.sigma)))
    Fy = np.exp(-np.square((img_y-mu_y)/(2*params.sigma)))
    
    # normalize
    Fx = np.transpose(np.divide(np.transpose(Fx), np.sum(Fx,axis=1)))
    Fy = np.transpose(np.divide(np.transpose(Fy), np.sum(Fx,axis=1)))
    
    return np.dot(Fy,np.dot(img,np.transpose(Fx)))
    
def loadMNISTImage():
    """
    return a MNIST image    
    """
    p = os.path.join(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                           'Datasets','ClutteredMNIST', 
                           'cMNIST_0.01.npz')
    data = np.load(p)
    return data['X_train'][0]


def loadEmbMNISTImage():
    """
    return a embMNIST image    
    """
    p = os.path.join(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                           'Datasets','EmbeddedMNIST', 
                           'embeddedMNIST_0.1.npz')
    data = np.load(p)
    return data['X_train'][0], data['y_train_seg'][0] 
    

def getParams():
    """
    return fictional parameters
    """
    gx = 18     # in width
    gy = 38     # in height 
    delta = 1
    sigma = 1
    attention_n = 22
    
    return DRAD_Params(gx,gy,delta,sigma,attention_n)
    
    
    
    

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    img, seg = loadEmbMNISTImage()
    params = getParams()
    
    plt.figure()
    
    plt.subplot(141)
    plt.imshow(img,cmap='gray')
    
    patch = computePatch(params,img)
    
    
    plt.subplot(142)
    plt.imshow(patch,cmap='gray')
    
    
    
    plt.subplot(143)
    plt.imshow(seg, cmap='gray')
    
    
    bb_true = computeBBfromSM(seg)
    bb = computeBBfromParams(params)
    rect = patches.Rectangle((bb.x_min,bb.y_min),bb.x_max - bb.x_min,
                                         bb.y_max - bb.y_min, linewidth=1,edgecolor='b',facecolor='none')
    rect_true = patches.Rectangle((bb_true.x_min,bb_true.y_min),bb_true.x_max - bb_true.x_min,
                                         bb_true.y_max - bb_true.y_min, linewidth=1,edgecolor='r',facecolor='none')
    
    ax = plt.subplot(144)
    ax.imshow(img,cmap='gray')
    ax.add_patch(rect)
    ax.add_patch(rect_true)
    
    
    plt.show()
    
    
    