# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:44:36 2017

@author: nickv
"""

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
        