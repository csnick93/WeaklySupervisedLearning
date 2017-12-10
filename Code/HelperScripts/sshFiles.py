#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:43:56 2017

@author: nick
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:09:35 2016

@author: nick

send zip files via ssh
"""

import os


def receiveFiles(source_folder,dest_folder,cluster = r"na02zijy@cluster.i5.informatik.uni-erlangen.de:"):
    """
    all zip files lying in source_folder at cluster are being sent via ssh to dest_folder
    """
    os.system("scp -r "+cluster+source_folder +" "+dest_folder)
    
def sendFiles(source_folder,dest_folder):
    os.system("scp -r " + source_folder +" na02zijy@cluster.i5.informatik.uni-erlangen.de:"+dest_folder)
        

            
if __name__ == "__main__":
    cluster = r'na02zijy@faui00a.informatik.uni-erlangen.de:'
    source = r'/proj/ciptmp/na02zijy/WeaklySupervisedLearning/Datasets/EmbeddedMNIST/embeddedMNIST.npz'
    dest = r'/home/nick/Desktop/MasterProjekt/WeaklySupervisedLearning/Datasets/EmbeddedMNIST'
    
    receiveFiles(source,dest)

