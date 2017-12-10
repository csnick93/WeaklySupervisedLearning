# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 21:51:03 2017

@author: nickv
"""

from mnistConv import mnistConv

def cnn(params, name):
    """
    Function that returns appropriate cnn result wrt name
    
    Arguments:
        - params: dictionary containing parameters to call function
        - name: name of the dataset being used in drad graph (e.g. cMNIST, embMNIST)
    """
    if name=='MNIST':
        return mnistConv(**params)
    
    else:
        raise RuntimeError('No pretrained CNN for ' + name + ' present.')
        