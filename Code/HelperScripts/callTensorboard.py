# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:15:20 2017

@author: nickv
"""

"""
Small script to call tensorboard
"""
import os

def callTensorboard(logdirs,names):
    command ='tensorboard --logdir='
    for i, l in enumerate(logdirs):
        command+= names[i]+':'+l+','
    command = command.rstrip(',')
    os.system(command)
    
    
if __name__ == "__main__":  
    logdirs = [r'D:\MasterProjekt\WeaklySupervisedLearning\Networks\SelfTransfer\EmbMNIST\model1']
              
    names = ['model'+str(i) for i in range(len(logdirs))]
    callTensorboard(logdirs,names)