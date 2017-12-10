# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 23:08:44 2017

@author: nickv
"""

import numpy as np
import os

def test_4d_matmul_numpy():
    np.random.seed(0)
    batch_size = 10
    height = 40
    width = 40
    att_N = 20
    channels = 3
    delta = 1
    gx = 25
    gy = 15
    sigma2 = 1
    
    batch_size = 10
    """
    get input images
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Datasets','ClutteredMNIST')
    data = np.load(os.path.join(path,'cMNIST_color.npz'))

    x_inp = data['X_train'][:batch_size].astype(np.float32)
    
    
    
    ################################
    """
    compute reasonable filters
    """
    grid = np.arange(att_N)
    
    mu_x = gx + (grid-att_N/2 + 0.5)*delta
    #mu_x = np.tile(mu_x,batch_size).reshape((batch_size,att_N))
    mu_y = gy + (grid-att_N/2 + 0.5)*delta
    #mu_y = np.tile(mu_y,batch_size).reshape((batch_size,att_N))
    
    Fx = np.empty((att_N,width))
    Fy = np.empty((att_N,height))
    
    for i in range(Fx.shape[0]):
        for a in range(Fx.shape[1]):
            Fx[i,a] = np.exp(-(a-mu_x[i])**2/(2*sigma2))
    Fx_norm = np.transpose(np.true_divide(np.transpose(Fx),np.sum(Fx,axis=1)))
     
    for j in range(Fy.shape[0]):
       for b in range(Fy.shape[1]):
           Fy[j,b] = np.exp(-(b-mu_y[j])**2/(2*sigma2))
    Fy_norm = np.transpose(np.true_divide(np.transpose(Fy),np.sum(Fy,axis=1)))
    ##############################################
    # multiple filters with image
    ##
    
    import matplotlib.pyplot as plt
    
    #glimpse = np.zeros((batch_size,att_N,att_N,3))
    tmp = np.zeros((batch_size,height,att_N,3))
    
    for b in range(batch_size):
        x = x_inp[b]        
        
        for i in range(3):
            #glimpse[b,:,:,i] = np.dot(Fy_norm,np.dot(x[:,:,i],np.transpose(Fx_norm)))
            tmp[b,:,:,i] = np.dot(x[:,:,i],np.transpose(Fx_norm))
            
    print(tmp[0])       

            
            
        
        
        
        
#####################################################################
#####################################################################       
        
def test_4d_matmul_tf():
    import tensorflow as tf
    sess = tf.InteractiveSession()
    
    np.random.seed(0)
    batch_size = 10
    height = 40
    width = 40
    att_N = 20
    channels = 3
    delta = 1
    gx = 25
    gy = 15
    sigma2 = 1
    
    batch_size = 10
    """
    get input images
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Datasets','ClutteredMNIST')
    data = np.load(os.path.join(path,'cMNIST_color.npz'))

    x_inp = data['X_train'][:batch_size].astype(np.float32)
    
    
    
    ################################
    """
    compute reasonable filters
    """
    grid = np.arange(att_N)
    
    mu_x = gx + (grid-att_N/2 + 0.5)*delta
    #mu_x = np.tile(mu_x,batch_size).reshape((batch_size,att_N))
    mu_y = gy + (grid-att_N/2 + 0.5)*delta
    #mu_y = np.tile(mu_y,batch_size).reshape((batch_size,att_N))
    
    Fx = np.empty((att_N,width))
    Fy = np.empty((att_N,height))
    
    for i in range(Fx.shape[0]):
        for a in range(Fx.shape[1]):
            Fx[i,a] = np.exp(-(a-mu_x[i])**2/(2*sigma2))
    Fx_norm = np.transpose(np.true_divide(np.transpose(Fx),np.sum(Fx,axis=1)))
    Fx_tile = np.tile(Fx_norm,[batch_size,1,1])
    
    if np.any(Fx_tile[0] != Fx_norm):
        raise RuntimeError('Tiling failed')
     
    for j in range(Fy.shape[0]):
       for b in range(Fy.shape[1]):
           Fy[j,b] = np.exp(-(b-mu_y[j])**2/(2*sigma2))
    Fy_norm = np.transpose(np.true_divide(np.transpose(Fy),np.sum(Fy,axis=1)))
    Fy_tile = np.tile(Fy_norm,[batch_size,1,1])
    
    ##############################################
    # multiple filters with image
    ##
    def filter_3d(inp):
        x = inp[0]
        Fxt = inp[1]
        Fy = inp[2]
        
        #x = tf.transpose(x,[0,1,3,2])
        x = tf.transpose(x,[0,2,1])
        #x = tf.reshape(x,(batch_size,height*channels,width))
        x = tf.reshape(x,(height*channels,width))
        
        tmp = tf.matmul(x,Fxt)
        
        #tmp_reshape = tf.reshape(tmp,(batch_size,height,channels,att_N))
        tmp_reshape = tf.reshape(tmp,(height,channels,att_N))
        
        #tmp_transpose = tf.transpose(tmp_reshape,perm=[0,1,3,2])
        tmp_transpose = tf.transpose(tmp_reshape,perm=[0,2,1])
        
        #tmp_reshape_again = tf.reshape(tmp_transpose,(batch_size,height,att_N*channels))
        tmp_reshape_again = tf.reshape(tmp_transpose,(height,att_N*channels))
        
        #print(tmp_transpose.eval())
        
        
        result = tf.matmul(Fy,tmp_reshape_again)
        
        glimpse = tf.reshape(result,(att_N,att_N,channels))
        
        return glimpse, Fxt,Fy
    
    
    
    x = tf.constant(x_inp,dtype=tf.float32)
    Fx = tf.constant(Fx_tile,dtype=tf.float32)
    Fxt = tf.cast(tf.transpose(Fx,perm=[0,2,1]),dtype=tf.float32)
    Fy = tf.constant(Fy_tile,dtype=tf.float32)
    glimpse,_,_ = tf.map_fn(filter_3d, (x,Fxt,Fy),
                                dtype=(tf.float32,tf.float32,tf.float32))
    
    glimpses = glimpse.eval()
    
    ##################################
    import matplotlib.pyplot as plt
    
    for i,g in enumerate(glimpses):
        plt.imshow(x_inp[i])
        plt.show()
        plt.close()
        
        plt.imshow(g)
        plt.show()
        plt.close()
    
    
    
    
    
    
if __name__=='__main__':
    np.set_printoptions(threshold=np.nan)
    test_4d_matmul_tf()
    #test_4d_matmul_numpy()