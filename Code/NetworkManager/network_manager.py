# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:48:50 2017

@author: nickv
"""

"""
Class Network

This class is responsible for loading a tensorflow graph from a .meta 
file. It will hold different attributes to configure the graph, such as
the type of optimizer, the number of epochs, the objective function etc.
"""

import tensorflow as tf
import numpy as np
import os

class Network:
    """
    Class responsible for loading the created model and training it, 
    writing out a summary and saving the trained network again.
    """
    def __init__(self, model_name, model_path, opt, opt_params, num_epochs, 
                 batch_size, data, summary_folder,summary_intervals, 
                 complete_set, keep_prob, l2_reg, clip_gradient, clip_value):
        """
        Args:
            model_name: name of model being trained
            model_path: path to where the tensorflow graph is saved
                        (e.g. ..\\Networks\\DRAD\\model1)
            opt: type of optimizer {adam, adagrad,...}
            opt_params: - parameters for optimizer saved as a tuple
                        - the parameters are inserted in order of constructor
                        - e.g. for Adam: (learning_rate, beta1,beta2,epsilon, use_locking, name)
            num_epochs : - the number of epochs for training 
                         - true epochs, so complete dataset is being looked
                            at for num_epochs
            epoch_factor : - multiplicator for num_epochs to get true number of
                                epochs with respect to batch_size
            batch_size: - batch size used for training
            data: - data for training 
                  - object of Data class
            summary_folder: - path to folder where tensorboard summaries
                                shall be saved
            summary_intervals: - number of epochs after which validation error
                               shall be calculated and summaries be updated
            complete_set: list of flags deciding if whole dataset is used for evaluation in summary
                                [complete_set_train,complete_set_val, complete_set_test]
            keep_prob: 1-dropout for the model to be trained
            l2_reg: l2 regularization rate for the network weights
            clip_gradient: flag indicating whether gradient should be clipped to prevent exploding gradient
            clip_value: threshold value where gradient clipping should happen
        """
        self.model_name = model_name
        self.model_path = model_path
        self.opt = opt
        self.opt_params = opt_params
        self.num_epochs = num_epochs
        self.epoch_factor = np.int32(data.getSize('train')/ batch_size)
        self.batch_size = batch_size
        self.data = data
        self.summary_folder = summary_folder
        # create summary subfolders
        if not os.path.exists(os.path.join(self.summary_folder,'train')):
            os.makedirs(os.path.join(self.summary_folder,'train'))
            os.mkdir(os.path.join(self.summary_folder,'val'))
            os.mkdir(os.path.join(self.summary_folder,'test'))
        self.summary_intervals = summary_intervals
        self.complete_set = complete_set
        self.keep_prob = keep_prob
        self.l2_reg = l2_reg
        self.clip_gradient = clip_gradient
        self.clip_value = clip_value
        
    
    def load_and_train(self):
        """
        1) Load the graph from meta file
        2) Train according to specified configuration
        """
        print("Loading and training the network...")
        with tf.Graph().as_default():
            def next_batch(dataset, complete_set = False):
                """
                Return batch of input and output samples from data drawn randomly
                    dataset: indicates from which set to draw samples {"train", "val", "test"}
                    complete_set: flag only relevant if dataset is 'train', 
                                    if true return the complete training set
                Workflow in case of train:
                    1) Get number of samples available in train/val/test set
                    2) Draw batch_size many random indices
                    3) Return batch of samples at those indices
                Workflow in case of test/val:
                    1) Return whole data set
                """
                if complete_set:
                    x_batch,y_batch = self.data.getData(dataset)
                else:
                    no_samples = self.data.getSize(dataset) 
                    indices = np.random.randint(0,no_samples, self.batch_size)
                    x_batch,y_batch = self.data.getData(dataset,self.batch_size, indices)
                    
                return (x_batch,y_batch)
        
        
            with tf.Session() as sess:
                # load the model from model path and load necessary variables
                loader = tf.train.import_meta_graph(os.path.join(self.model_path,
                                                                     "model.meta"))
                loader.restore(sess, os.path.join(self.model_path,'model'))
                x = tf.get_collection('x')[0]
                keep_prob = tf.get_collection('keep_prob')[0]
                batch_norm_training = tf.get_collection('batch_norm_training')[0]
                                        
                ##
                # specify loss
                ##
                regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
                with tf.name_scope('loss'):
                    if self.model_name == 'drad':
                        y = tf.get_collection("y")[0]
                        y_ = tf.placeholder(tf.float32, shape = [None,y.shape[1]], name = 'label_placeholder')     
                        softmax = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
                        cross_entropy = tf.reduce_mean(softmax,
                                                       name = "cross_entropy_loss")
                        reg_vars = [tf_var for tf_var in tf.trainable_variables()
                                        if not ('noreg' in tf_var.name.lower() or 'bias' in tf_var.name.lower())]
                        reg_term = tf.contrib.layers.apply_regularization(regularizer, 
                                                                      reg_vars)
                        loss = cross_entropy+reg_term
                    elif self.model_name == 'self_transfer':
                        y = tf.get_collection('classification_out')[0]
                        location_out = tf.get_collection('location_out')[0]
                        y_ = tf.placeholder(tf.float32, shape = [None,location_out.shape[1]], name = 'label_placeholder')
                        alpha = tf.placeholder(tf.float32,name='alpha')
                        tf.add_to_collection('alpha',alpha)
                        tf.summary.scalar('alpha',alpha)
                        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
                        tf.summary.scalar('Classification_Loss' , classification_loss)
                        location_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = location_out))
                        tf.summary.scalar('Location_Loss', location_loss)
                        reg_vars = [tf_var for tf_var in tf.trainable_variables()
                                        if not ('noreg' in tf_var.name.lower() or 'bias' in tf_var.name.lower())]
                        reg_term = tf.contrib.layers.apply_regularization(regularizer, 
                                                                      reg_vars)
                        loss = (1-alpha)*classification_loss+alpha*location_loss+reg_term
                        
                    tf.summary.scalar('loss' , loss)
                
                ###
                # training step
                ###
                if self.opt == "adam":
                    with tf.name_scope('train'):
                        if not self.clip_gradient:
                            train_step = tf.train.AdamOptimizer(**self.opt_params).minimize(loss)
                        else:
                            optimizer = tf.train.AdamOptimizer(**self.opt_params)
                            gradients , variables = zip(*optimizer.compute_gradients(loss))
                            gradients,_ = tf.clip_by_global_norm(gradients, self.clip_value)
                            train_step = optimizer.apply_gradients(zip(gradients,variables))
                elif self.opt == 'sgd':
                    with tf.name_scope('train'):
                        if not self.clip_gradient:
                            train_step = tf.train.GradientDescentOptimizer(**self.opt_params).minimize(loss)
                        else:
                            optimizer = tf.train.GradientDescentOptimizer(**self.opt_params)
                            gradients , variables = zip(*optimizer.compute_gradients(loss))
                            gradients,_ = tf.clip_by_global_norm(gradients, self.clip_value)
                            train_step = optimizer.apply_gradients(zip(gradients,variables))
                elif self.opt == 'momentum':
                    with tf.name_scope('train'):
                        if not self.clip_gradient:
                            train_step = tf.train.MomentumOptimizer(**self.opt_params).minimize(loss)
                        else:
                            optimizer = tf.train.MomentumOptimizer(**self.opt_params)
                            gradients , variables = zip(*optimizer.compute_gradients(loss))
                            gradients,_ = tf.clip_by_global_norm(gradients, self.clip_value)
                            train_step = optimizer.apply_gradients(zip(gradients,variables))
                else:
                    raise ValueError('No valid optimizer specified!\nAborting...')
                
                ###
                # measuring performance
                ###
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
                    # monitor location output classification accuracy as well for self transfer
                    if self.model_name == 'self_transfer':
                        loc_correct_prediction = tf.equal(tf.argmax(location_out,1), tf.argmax(y_,1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name = "accuracy")
                    tf.summary.scalar('accuracy',accuracy)
                    if self.model_name == 'self_transfer':
                        loc_accuracy = tf.reduce_mean(tf.cast(loc_correct_prediction,tf.float32), name = "accuracy")
                        tf.summary.scalar('location_class_accuracy',loc_accuracy)
                # add accuracy function to collection so it is available for
                # potential later evaluation
                saver = tf.train.Saver()
                self.load_model(sess, saver)
                tf.add_to_collection("y_",y_)
                tf.add_to_collection("accuracy",accuracy)
                tf.add_to_collection("loss", loss)
                
                # Merge all the summaries and write them out 
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(self.summary_folder,'train'), sess.graph)
                validation_writer = tf.summary.FileWriter(os.path.join(self.summary_folder,'val'))
                test_writer = tf.summary.FileWriter(os.path.join(self.summary_folder,'test'))
                
                sess.run(tf.global_variables_initializer())
                saver.save(sess,os.path.join(self.model_path,"model"))
                saver.export_meta_graph(os.path.join(self.model_path,"model.meta"))
                
                ###
                # Training loop
                ###
                for i in range(self.num_epochs):
                    if self.model_name=='self_transfer':
                        alpha_value = 0 if i < self.num_epochs*0.2 else np.float32(i)/self.num_epochs             
                        additional_feeds = {alpha:alpha_value}
                    else:
                        additional_feeds = {}
                    # one epoch consists of epoch_factor many batch training iterations
                    for j in range(self.epoch_factor):
                        batch = next_batch("train")
                        feed_dict = {x: batch[0], y_: batch[1], 
                                 keep_prob: self.keep_prob, batch_norm_training: True}
                        for k in additional_feeds:
                            feed_dict[k] = additional_feeds[k]
                        sess.run([train_step], feed_dict = feed_dict)
                    # determine validation error every summary_intervals
                    if i% self.summary_intervals == (self.summary_intervals-1):
                        print("=====================")
                        print("Completed Epoch %d/%d"%(i+1,self.num_epochs))
                        # Compute training performance
                        batch = next_batch('train', complete_set = self.complete_set[0])
                        feed_dict = {x: batch[0], y_: batch[1], 
                                 keep_prob: self.keep_prob, batch_norm_training: True}
                        for k in additional_feeds:
                            feed_dict[k] = additional_feeds[k]
                        if self.model_name=='self_transfer':
                            summary, acc, _,_, _,_= sess.run([merged, accuracy, 
                                                           loc_accuracy, loss, location_loss, 
                                                           classification_loss], 
                                                   feed_dict = feed_dict)
                        else:
                            summary, acc, _ = sess.run([merged, accuracy, loss], 
                                                       feed_dict = feed_dict)
                        train_writer.add_summary(summary,i)
                        print("Training Accuracy : %3.2f"%(acc))
                        #Compute validation performance
                        batch = next_batch("val", complete_set = self.complete_set[1])
                        feed_dict = {x: batch[0], y_: batch[1], 
                                 keep_prob: self.keep_prob, batch_norm_training: True}
                        for k in additional_feeds:
                            feed_dict[k] = additional_feeds[k]
                        if self.model_name=='self_transfer':
                            summary, acc, _,_, _,_= sess.run([merged, accuracy, 
                                                           loc_accuracy, loss, location_loss, 
                                                           classification_loss], 
                                                   feed_dict = feed_dict)
                        else:
                            summary, acc, _ = sess.run([merged, accuracy, loss], 
                                                       feed_dict = feed_dict)    
                        validation_writer.add_summary(summary, i)
                        print("Validation Accuracy : %3.2f"%(acc))
                        self.save_model(sess,saver,acc)                
                        
                
                        
                ###
                # Testing trained network
                ###
                # for testing, the best model wrt validation loss is loaded
                print("=====================")
                print("Loading best model for test evaluation")
                self.load_model(sess,saver)
                
                test_batch = next_batch("test", complete_set = self.complete_set[2])
                print("=====================")
                if self.model_name=='self_transfer':
                        alpha_value = np.float32(0.5)
                        additional_feeds = {alpha:alpha_value}
                else:
                    additional_feeds = {}
                feed_dict = {x: test_batch[0], y_: test_batch[1], 
                                 keep_prob: 1, batch_norm_training: True}
                for k in additional_feeds:
                    feed_dict[k] = additional_feeds[k]
                if self.model_name=='self_transfer':
                    summary, acc, _,_, _,_= sess.run([merged, accuracy, 
                                                   loc_accuracy, loss, location_loss, 
                                                   classification_loss], 
                                           feed_dict = feed_dict)
                else:
                    summary, acc, _ = sess.run([merged, accuracy, loss], 
                                               feed_dict = feed_dict)
                test_writer.add_summary(summary,(self.num_epochs-1))
                print("Test Accuracy : %3.2f"%(acc))
                train_writer.close()
                validation_writer.close()
                test_writer.close()
                
            
        
    def save_model(self,sess, saver,val_acc):
        """
        - function to check whether we found a new best model based on the 
            current validation accuracy
        - if so, we overwrite our previous checkpoint
        
        Args:
            sess: current session
            saver: saver object belonging to this session
            val_acc: current validation accuracy
        """
        # check if we need to save model (no validation accuracy recorded yet
        # , so we must be at beginning of training or improved val_acc)
        if not hasattr(self,'best_val_acc') or val_acc > self.best_val_acc:
            print('Saving model')
            self.best_val_acc = val_acc
            saver.save(sess,os.path.join(self.model_path,"model.ckpt"))
            
    def load_model(self,sess, saver):
        """
        - function to load the model
        - function checks whether there is checkpoint that we can load
            saved variables from
        
        Args:
            - sess: current session
            - saver: saver object belonging to this session
        """
        for f in os.listdir(self.model_path):
            if ".ckpt" in f:
                saver.restore(sess, os.path.join(self.model_path,"model.ckpt"))
    
    
    
if __name__ == "__main__":
    pass
            
            
            
            
            
            