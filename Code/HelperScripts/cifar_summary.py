# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 10:35:55 2017

@author: nickv
"""

import os
import csv

def cifar_results_csv(root = r'D:\MasterProjekt\WeaklySupervisedLearning\Networks\pretrainedCNN\cifar'):
    """
    Root: root directory for trained cifar models
    """
    # extract infos
    header = ['Folder', 'Number of epochs', 'Batch size', 'Dropout', 'l2_reg',
              'Optimizer', 'Optimizer Parameters', 'CNN Parameters', 'Data']
    infos = []
    for d in os.listdir(root):
        data = []
        if '.' in d:
            continue
        if os.path.exists(os.path.join(root,d,'info.txt')):
            data.append(d)
            with open(os.path.join(root,d,'info.txt'),'r') as f:
                for line in f:
                    if 'Parameters' in line:
                        data.append(line.split('Parameters: ')[1].replace(',','').rstrip('\n'))
                    else:
                        data.append(line.split(': ')[1].replace(',','').rstrip('\n'))
            infos.append(data)
            
    # write to csv file
    with open(os.path.join(root,'summary.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar ='|', quoting = 
                            csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for info in infos:
            writer.writerow(info)
    
if __name__=='__main__':
    cifar_results_csv()
        
        