# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:50:58 2017

@author: nickv
"""

# -*- coding: utf-8 -*-
###
# Start interactive session
###

import tensorflow as tf
import numpy as np
import os
from sys import path
path.insert(0,os.path.join(os.path.dirname(os.getcwd()), 'HelperConvNets'))
from CNN import weight_variable, bias_variable, conv2d, max_pool
#############################################################################
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
'''
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                               grad,
                                               op.outputs[1],
                                               op.get_attr("ksize"),
                                               op.get_attr("strides"),
                                               padding=op.get_attr("padding"))
'''
#############################################################################
    
class Self_Transfer:
    def __init__(self, save_model_to, img_size, label_size,
                 kernel_sizes = [3,3], feature_maps=[8,16], pooling = [2,2], 
                 padding_type = 'VALID', dense_dims = [512], kernel_size_loc = 3,
                 batch_normalization=False, local_response_normalization = True):
        """
        Args:
            save_model_to: folder path where model shall be saved
            img_size: image dimensions (height,width,channels)
            label_size: size of label vector
            kernel_sizes,feature_maps,pooling,dense_dims_cnn: cnn parameters
            padding_type: padding type for convolutions {'VALID', 'SAME'}
            dense_dims: dimensionality of dense layers in classification part
            kernel_size_loc: kernel size of conv layer in location part
        """
        self.save_model_to = save_model_to
        self.img_size = img_size
        self.label_size = label_size
        # joint CNN parameters
        self.batch_normalization = batch_normalization
        self.local_response_normalization = local_response_normalization
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        self.pooling = pooling
        self.padding_type = padding_type 
        # fully connected parameters
        self.dense_dims = dense_dims
        # location conv layer
        self.kernel_size_loc = kernel_size_loc
        
        # create info.txt in folder with the parameter configuration
        with open(os.path.join(self.save_model_to,'info.txt'),'w') as info:
            info.write("Configuration of Self Transfer graph:\n")
            for k in self.__dict__.keys():
                info.write('\t'+k +": "+ str(self.__dict__[k])+"\n")
    
    def build_graph(self):
        """
        Method setting up the self-transfer graph
        """
        with tf.Graph().as_default():
            
            x = tf.placeholder(tf.float32, shape = ([None,self.img_size[0],self.img_size[1],self.img_size[2]]))
            keep_prob = tf.placeholder(tf.float32)
            batch_norm_training = tf.placeholder(tf.bool)

            if(len(self.kernel_sizes)!=len(self.feature_maps) or 
               len(self.feature_maps)!=len(self.pooling)):
                raise RuntimeError('Input parameters are not consistent. The kernel_size \
                           parameter, the feature_maps parameter and the pooling \
                           parameters have varying \
                           length')

            # save argmax tensors, weight tensors, and conv tensors in a list
            argmax_pooling = []
            weight_tensors = []
            bias_tensors = []
            conv_tensors  = [x]

            #############
            # JOINT CNN #
            #############
            for l in range(len(self.kernel_sizes)):
                with tf.name_scope('Conv_'+str(l)):
                    if l == 0:
                        ##
                        # first layer
                        ##
                        W_conv = weight_variable([self.kernel_sizes[0],self.kernel_sizes[0],
                                                  self.img_size[-1],self.feature_maps[0]], 
                                                  name='Weights_0')
                        b_conv = bias_variable([self.feature_maps[0]], 
                                                name='Bias_0')
                        weight_tensors.append(W_conv)
                        bias_tensors.append(b_conv)
                        
                        # reshape input to 2D to enable convolutional operation
                        x_image = tf.reshape(x,[-1,self.img_size[0],self.img_size[1],self.img_size[2]],
                                             name='x_img_reshape')
                        
                        tmp_out = conv2d(x_image,W_conv, self.padding_type)+b_conv
                                        
                        # apply batch normalization
                        if self.batch_normalization:
                            tmp_out = tf.layers.batch_normalization(tmp_out, axis=1, training=batch_norm_training)
                        
                        #apply relu activation AFTER batch normalization
                        tmp_out = tf.nn.relu(tmp_out)
                        
                    else:
                        W_conv = weight_variable([self.kernel_sizes[l],self.kernel_sizes[l],
                                                  self.feature_maps[l-1],self.feature_maps[l]], 
                                                  name='Weights_'+str(l))
                        b_conv = bias_variable([self.feature_maps[l]],
                                                name='Bias_'+str(l))
                        weight_tensors.append(W_conv)
                        bias_tensors.append(b_conv)
                        
                        tmp_out = conv2d(tmp_out,W_conv, self.padding_type)+b_conv
                        
                        # apply batch normalization
                        if self.batch_normalization:
                            tmp_out = tf.layers.batch_normalization(tmp_out, axis=1, training=batch_norm_training)
                        
                        #apply conv layer with relu activation function
                        tmp_out = tf.nn.relu(tmp_out)
                        
                    # do pooling if appropriate
                    if self.pooling[l] > 1:
                        tmp_out, argmax_indices = tf.nn.max_pool_with_argmax(tmp_out,
                                                                         ksize = [1,self.pooling[l], self.pooling[l],1],
                                                                         strides = [1,self.pooling[l], self.pooling[l],1],
                                                                         padding = 'VALID',
                                                                         )
                        argmax_pooling.append(argmax_indices)
                        
                    if self.local_response_normalization:
                        tmp_out = tf.nn.lrn(tmp_out, 5, bias = 1.0, alpha = 1e-4,
                                            beta = 0.75)
                    
                    conv_tensors.append(tmp_out)
            
            joint_cnn_out = tmp_out
            
            ########################
            # DENSE CLASSIFICATION # 
            ########################
            # reshape the cnn output
            output_shape_flat = np.prod(np.array(joint_cnn_out.get_shape().as_list()[1:],dtype=np.int32),dtype=np.int32)
            tmp_out = tf.reshape(joint_cnn_out,[-1,output_shape_flat])
            tmp_out = tf.nn.dropout(tmp_out, keep_prob)
            #first dense layer
            W_dense = weight_variable([output_shape_flat, self.dense_dims[0]],
                                                  name='dense_weights_'+str(0))
            b_dense = bias_variable([self.dense_dims[0]],
                                    name = 'dense_bias_'+str(0))
            tmp_out = tf.matmul(tmp_out,W_dense)+ b_dense
            if self.batch_normalization:
                tmp_out = tf.layers.batch_normalization(tmp_out, training=batch_norm_training)              
            tmp_out = tf.nn.relu(tmp_out)
            tmp_out = tf.nn.dropout(tmp_out, keep_prob)
            # iterate through fully connected layers
            for d in range(1,len(self.dense_dims)):
                with tf.name_scope('Dense_'+str(d)):
                    W_dense = weight_variable([self.dense_dims[d-1], self.dense_dims[d]],
                                                  name='dense_weights_'+str(d))
                    b_dense = bias_variable([self.dense_dims[d]],
                                            name = 'dense_bias_'+str(d))
                    tmp_out = tf.matmul(tmp_out,W_dense)+ b_dense
                    if self.batch_normalization:
                        tmp_out = tf.layers.batch_normalization(tmp_out, training=batch_norm_training)              
                    tmp_out = tf.nn.relu(tmp_out)
                    tmp_out = tf.nn.dropout(tmp_out, keep_prob)
            # project to label size output
            W_dense = weight_variable([tmp_out.get_shape().as_list()[-1], self.label_size],
                                                  name='dense_weights_'+str(len(self.dense_dims)))
            b_dense = bias_variable([self.label_size],
                                    name = 'dense_bias_'+str(len(self.dense_dims)))
            tmp_out = tf.matmul(tmp_out,W_dense)+ b_dense
            if self.batch_normalization:
                tmp_out = tf.layers.batch_normalization(tmp_out, training=batch_norm_training)              
            tmp_out = tf.nn.relu(tmp_out)
            classification_out = tf.nn.dropout(tmp_out, keep_prob)
            
            
            #####################
            # LOCALIZATION MAPS #
            #####################
            # push through label_size x image conv layer 
            W_conv = weight_variable([self.kernel_size_loc,self.kernel_size_loc,self.feature_maps[-1],self.label_size], 
                                      name='Weights_loc')
            b_conv = bias_variable([self.label_size], name='Bias_loc')
            weight_tensors.append(W_conv)
            bias_tensors.append(b_conv)
            tmp_out = conv2d(joint_cnn_out,W_conv, self.padding_type)+b_conv
            if self.batch_normalization:
                tmp_out = tf.layers.batch_normalization(tmp_out, axis=1, training=batch_norm_training)
            tmp_out = tf.nn.relu(tmp_out)
            # global max pooling
            pool_size = tmp_out.get_shape().as_list()[1:-1]
            location_out, argmax_indices = tf.nn.max_pool_with_argmax(tmp_out, 
                                                                      ksize = [1,pool_size[0],pool_size[1],1], 
                                                                      strides = [1,1,1,1],
                                                                      padding = 'VALID')
            argmax_pooling.append(argmax_indices)
            location_out_shape = location_out.get_shape().as_list()
            #_ , guessed_class = tf.nn.max_pool_with_argmax(location_out, ksize=(1,1,1,location_out_shape[-1]),
            #                                               strides = (1,1,1,location_out_shape[-1]), padding='VALID')
            location_out = tf.reshape(location_out,(-1,location_out_shape[-1]))
            # find out guessed class by looking at highest activation in location out tensor
            
            

            ##
            # add necessary parameters to graph collection
            ##
            tf.add_to_collection('x',x)
            tf.add_to_collection('classification_out',classification_out)
            tf.add_to_collection('location_out', location_out)
            #tf.add_to_collection('guessed_class', guessed_class)
            tf.add_to_collection('keep_prob',keep_prob)
            tf.add_to_collection('batch_norm_training', batch_norm_training)
            ##
            # save weight tensors, bias tensors as well as argmax tensors to be 
            # able to do backtracking
            tf.add_to_collection('num_argmax_pools',len(argmax_pooling))
            for i,p in enumerate(argmax_pooling):
                tf.add_to_collection('argmax_pooling_'+str(i),p)
            tf.add_to_collection('num_weights',len(weight_tensors))
            for i,w in enumerate(weight_tensors):
                tf.add_to_collection('weights_'+str(i),w)
            for i,b in enumerate(bias_tensors):
                tf.add_to_collection('bias_'+str(i),b)
            for i,c in enumerate(conv_tensors):
                tf.add_to_collection('conv_tensor_'+str(i),c)
            
            
            ##
            # save the graph
            ##
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver= tf.train.Saver()
                saver.save(sess,os.path.join(self.save_model_to,"model"))
                saver.export_meta_graph(os.path.join(self.save_model_to,"model.meta"))
                
            
            
            
