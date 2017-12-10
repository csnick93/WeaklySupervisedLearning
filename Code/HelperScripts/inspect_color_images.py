# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:22:56 2017

@author: nickv
"""

import tensorflow as tf
import numpy as np
import os

path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Datasets','ClutteredMNIST')
data = np.load(os.path.join(path,'cMNIST_color.npz'))

x_inp = data['X_train'][:10].astype(np.float32)
y_inp = data['y_train'][:10].astype(np.float32)

print(x_inp.dtype)

log_dir = r'D:\MasterProjekt\TestFolder'


with tf.Graph().as_default():
    with tf.Session() as sess:

        x = tf.placeholder(tf.float32,[None,40,40,3], name='x')
        y = tf.placeholder(tf.float32,[None,10], name = 'y')
        
        #tf.summary.image('image',x,10)
        
        x_reshape = tf.reshape(x,[-1,40*40*3])
        
        out = tf.layers.dense(x_reshape,10, activation = tf.sigmoid)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=out))
        
        train_step = tf.train.AdamOptimizer().minimize(loss)
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')
        tf.global_variables_initializer().run()
        
        for i in range(10):
            summary,_ = sess.run([merged,train_step], feed_dict={x:x_inp, y:y_inp})
            train_writer.add_summary(summary,i)
            
        train_writer.close()