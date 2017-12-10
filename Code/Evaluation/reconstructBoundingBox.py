# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 19:50:18 2017

@author: nickv
"""


import numpy as np
import os
from sys import path
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'HelperClasses'))
from BoundingBox import BoundingBox

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
    
def computeBBfromBB(bb_coords):
    """
    Method that converts np array of coords (xmin,xmax,ymin,ymax) into
    a bounding box.
    
    Args:
        bb_coords: the bounding box coordinates
    """
    return BoundingBox(bb_coords[0],bb_coords[1],bb_coords[2],bb_coords[3])
    
def computeBBfromParamsDRAD(params):
    """
    Compute the bounding box given the params
    """    
    x_min = np.int32(params.gx - (params.delta*(params.attention_n-1)+1)/2)# - params.sigma)
    x_max = np.int32(params.gx + (params.delta*(params.attention_n-1)+1)/2)# + params.sigma)
    y_min = np.int32(params.gy - (params.delta*(params.attention_n-1)+1)/2)# - params.sigma)
    y_max = np.int32(params.gy + (params.delta*(params.attention_n-1)+1)/2)# + params.sigma)

    return BoundingBox(x_min,x_max,y_min,y_max)
    
def computeDRADPatch(params,img):
    """
    Compute the image patch given attention parameters and image of interest
    
    Arguments:
        params: attention parameters
        img: image of interest
    """
    # compute attention windows
    img_height = img.shape[0]
    img_width = img.shape[1]
    grid = np.arange(params.attention_n,dtype = np.float32).reshape((
                        params.attention_n,1))
    mu_x = params.gx + (grid - params.attention_n/2+0.5)*params.delta
    mu_y = params.gy + (grid - params.attention_n/2+0.5)*params.delta    
    img_x = np.tile(np.arange(img_width, dtype=np.float32), params.attention_n).reshape(
                [params.attention_n, img_width])
    img_y = np.tile(np.arange(img_height, dtype=np.float32), params.attention_n).reshape(
                [params.attention_n, img_height])
    Fx = np.exp(-np.square((img_x-mu_x)/(2*params.sigma)))
    Fy = np.exp(-np.square((img_y-mu_y)/(2*params.sigma)))
    
    # normalize
    Fx = np.transpose(np.divide(np.transpose(Fx), np.sum(Fx,axis=1)))
    Fy = np.transpose(np.divide(np.transpose(Fy), np.sum(Fx,axis=1)))
    
    return np.dot(Fy,np.dot(img,np.transpose(Fx)))