# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:21:46 2017

@author: nickv
"""

"""
Class that is used as interface to build default graphs (such as DRAD)
"""
import os
from sys import path
path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'DRAD'))
from drad_graph import Drad
path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'SelfTransfer'))
from self_transfer import Self_Transfer

class GraphBuilder:
    """
    Class responsible for building the desired model. Serves as intermediate
    class between Network Manager and the actual Network class.
    """
    
    def __init__(self):
        pass
    
    def build_graph(self,name,params):
        """
        Build the specified graph with specified parameters
            name: - name of graph (e.g. drad)
            params: - parameters necessary for construction as dictionary
        """
        if name == 'drad':
            drad = Drad(**params)
            drad.build_graph()
        elif name=='self_transfer':
            self_transfer = Self_Transfer(**params)
            self_transfer.build_graph()
        else:
            raise RuntimeError("Network " + name + " not available. Aborting...")
        
    
