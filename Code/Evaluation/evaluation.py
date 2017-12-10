#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:43:38 2017

@author: nick
"""
import tensorflow as tf
import numpy as np
import os
from sys import path
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                            'HelperClasses'))
from DRAD_params import DRAD_Params
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                            'Networks','SelfTransfer'))
from backtrack_location import backtrack_locations

class Evaluation:
    
    def __init__(self, data, model_path, summary_folder, model_name, log_folder,
                 batch_size, datasets, loss, accuracy, localization, save_imgs, num_imgs):
        """
        Class for handling the evaluation part. Re-evaluates loss, accuracy
        and if applicable the intersection over union error.
        
        Args:
            data: instance of data for input, label, and segmentation Mask retrieval
            model_path: path to folder where the saved graph lies
            summary_folder: path to folder where summaries lie
            model_name: name of the model being evaluated (e.g. 'drad')
            log_folder: folder path where results are written to 
            batch_size: batch size during training (useful to compute accuracy and loss without OOM error)
            datasets: datasets to be evaluated (train,val,test)
            loss: flag indicating if loss shall be evaluated
            accuracy: flag indicating if accuracy shall be evaluated 
            bb_boxes: flag indicating if iou shall be evaluated and bounding boxes be drawn in
                (iou computation only possible in the case of segmentation masks being present)
            save_imgs: flag determining if images showing bounding box shall be
                        created and saved
            num_imgs: The amount of images that shall be saved
        """
        self.data = data
        self.model_path = model_path
        self.summary_folder = summary_folder
        self.model_name = model_name
        self.log_folder = log_folder
        self.batch_size = batch_size
        self.datasets= datasets
        self.loss = loss
        self.accuracy = accuracy
        self.localization = localization
        self.save_imgs = save_imgs
        self.num_imgs = num_imgs
        
    ###########################################################################  
    def evaluate(self):
        """
        Main evaluation method, where it is possible to evaluate accuracy, loss
        and intersection over union
        """
        
        if self.loss:
            self.computeLoss(self.datasets)
        
        if self.accuracy:
            self.computeAccuracy(self.datasets)
        
        if self.localization:
            self.localization_quality(self.datasets,self.num_imgs, self.save_imgs)
        
    ###########################################################################
    
    def localization_quality(self, datasets, num_saveImgs,saveImgs):
        """
        Method that evaluates complete test set on intersection over union error if segmentation 
        masks are available. Additionally, the accuracy is computed.
        The result is written out to Summaries folder.
        
        Args:
            datasets: the dataset to do evaluation on (per default that is the test set)
            num_saveImgs: optional parameter to request that only num_saveImgs
                            images shall be saved
            saveImgs: insert bounding boxes into image and save it
        """
        from reconstructBoundingBox import computeBBfromSM, computeBBfromParamsDRAD,\
                                            computeBBfromBB
        
        print('Computing Bounding Boxes...')
        
        def point_in_bounding_box(bb_true, location):
            """
            Check if computed index (y,x) is in true bounding box.
            
            Args:
                bb_true: ground truth bounding box of object in image
                location: tuple of coordinates
            """
            return int(bb_true.contains_point(*location))
            
        
        def intersectionOverUnion(bb_det, bb_true):
            """
            Compute intersection over Union area given two bounding boxes
            
            Parameters:
                bb_det: detected bounding box of object
                bb_true: ground truth bounding box
            """
            tmp_x = min(bb_det.x_max,bb_true.x_max)-max(bb_det.x_min, bb_true.x_min)
            tmp_y = min(bb_det.y_max,bb_true.y_max)-max(bb_det.y_min, bb_true.y_min)
            
            if tmp_x < 0 or tmp_y < 0:
                return 0
            else:
                intersection = tmp_x*tmp_y
            
            union = bb_det.area()+bb_true.area()-intersection                   
                               
            return intersection/union
        
        def insert_blob(img, obj_loc, save_path, bb_true = None):
            """
            Draw computed location in image as well as true bounding box
            if that information is available and save image
            
            Args:
                img: image
                obj_loc: (y,x) coordinates of object location
                save_path: where to save image to 
                bb_true: true bounding box if available
            """
            from matplotlib import pyplot as plt
            import matplotlib.patches as patches
            fig,ax = plt.subplots(1)
            # normalize image to [0,1] as float32 img -> otherwise strange display
            if np.max(img) > 1:
                img = img.astype(np.float32)
                img -= np.min(img)
                img /= np.max(img)
            ax.imshow(img)
            
            if np.any(bb_true):
                rect_true = patches.Rectangle((bb_true.x_min,bb_true.y_min),bb_true.x_max - bb_true.x_min,
                                         bb_true.y_max - bb_true.y_min, linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect_true)
                
            blob = patches.Circle((obj_loc[1],obj_loc[0]), radius = 10, linewidth = 3, edgecolor = 'b', facecolor = 'none')
            ax.add_patch(blob)
            
            plt.savefig(save_path)
            plt.close()
            
        
        def insertBoundingBox(img,bb_det, thick,save_path, bb_true = None):
            """
            Draw bounding box into image and save result
            
            Args:
                img: the image
                bb_true: the ground truth bounding box 
                bb_det: the detected bounding box
                thick: flag indicating that bounding box should be made thick
                        as this is the bounding box of interest
                save_path: path where result shall be saved
            """
            from matplotlib import pyplot as plt
            import matplotlib.patches as patches
            fig,ax = plt.subplots(1)
            # normalize image to [0,1] as float32 img -> otherwise strange display
            if np.max(img) > 1:
                img = img.astype(np.float32)
                img -= np.min(img)
                img /= np.max(img)
            # need to reshape gray scale images to avoid type error
            if img.shape[-1] == 1:
                img = np.reshape(img,img.shape[:-1])
            ax.imshow(img)
            
            if np.any(bb_true):
                rect_true = patches.Rectangle((bb_true.x_min,bb_true.y_min),bb_true.x_max - bb_true.x_min,
                                         bb_true.y_max - bb_true.y_min, linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect_true)
            
            if thick:
                rect_det = patches.Rectangle((bb_det.x_min,bb_det.y_min),bb_det.x_max - bb_det.x_min,
                                         bb_det.y_max - bb_det.y_min, linewidth=5,edgecolor='y',facecolor='none')
            else:
                rect_det = patches.Rectangle((bb_det.x_min,bb_det.y_min),bb_det.x_max - bb_det.x_min,
                                         bb_det.y_max - bb_det.y_min, linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect_det)
            plt.savefig(save_path)
            plt.close()
        
        
        with tf.Graph().as_default():
            with tf.Session() as sess:
                ###
                # load trained graph
                ###
                loader = tf.train.import_meta_graph(os.path.join(self.model_path,"model.meta"))
                loader.restore(sess, os.path.join(self.model_path,'model'))
                saver = tf.train.Saver()
                saver.restore(sess, os.path.join(self.model_path,"model.ckpt"))
                
                ##
                # get placeholders and tensor nodes to be evaluated
                # get test data to do evluation on
                # do evaluation
                ##
                
                x = tf.get_collection("x")[0]
                y_ = tf.get_collection("y_")[0]
                keep_prob = tf.get_collection('keep_prob')[0]
                batch_norm_training = tf.get_collection('batch_norm_training')[0]
                
                for dataset in datasets:
                
                    if self.data.hasData(dataset,segMap = True):
                        X, y, y_segMap = self.data.getData(dataset, segMap =True)
                    elif self.data.hasData(dataset,segMap=False,bb=True):
                        X, y, y_bb = self.data.getData(dataset, segMap = False, bb=True)
                    else:
                        X, y = self.data.getData(dataset)
                    
                    
                    ############################################################
                    # DRAD #
                    ########
                    if self.model_name.lower() =='drad':
                        gx = []
                        gy = []
                        sigma2 = []
                        delta = []
                        
                        # get attention and sequence length parameter
                        with open(os.path.join(self.model_path, 'info.txt'), 'r') as f:
                            info = f.read()
                            attention_N = np.int32(info.split('attention_N: ')[1].split('\n')[0])
                            seq_length = np.int32(info.split('seq_length: ')[1].split('\n')[0])
                        
                        # obtain the attention patch parameters from all time points
                        for i in range(seq_length):
                            gx.append(tf.get_collection('gx_'+str(i))[0])
                            gy.append(tf.get_collection('gy_'+str(i))[0])
                            sigma2.append(tf.get_collection('sigma2_'+str(i))[0])
                            delta.append(tf.get_collection('delta_'+str(i))[0])
                            
                        best_time_step = tf.get_collection('best_time_step')
                            
                        gx_, gy_, sigma2_, delta_, time_step = sess.run([gx,gy,sigma2,delta, best_time_step],
                                                              feed_dict = {x: X, y_: y,
                                                                           keep_prob:1, batch_norm_training:False})
    
                                    
                        ##
                        # compute IoUs for data if segmentation Masks present
                        ##
                        ious = np.zeros(y.shape[0])
                        ious.fill(np.nan)
                        # create folder if evaluated images shall be saved
                        if saveImgs:
                            if not os.path.exists(os.path.join(self.summary_folder,'EvaluatedImages')):
                                os.makedirs(os.path.join(self.summary_folder,'EvaluatedImages')) 
                
                        IMG_COUNTER = 0
                        # compute the bounding boxes for each timestep for each sample
                        for t in range(len(gx_)):
                            # in outer loop, iterate over time
                            for s in range(gx_[t].shape[0]):
                                # in inner loop. iterate over samples
                                bb_det = computeBBfromParamsDRAD(DRAD_Params(gx_[t][s][0],
                                                                    gy_[t][s][0],
                                                                    delta_[t][s][0],
                                                                    sigma2_[t][s][0],
                                                                    attention_N))
                                
                                if self.data.hasData(dataset, segMap = True):
                                    bb_true = computeBBfromSM(y_segMap[s])
                                    # only compute IoU for most confident timestep
                                    if (time_step[0][s]==t):
                                        ious[s] = intersectionOverUnion(bb_det,bb_true)
                                
                                    # if desired, insert bounding boxes into image and save it
                                    if saveImgs and IMG_COUNTER < num_saveImgs:
                                        folder = os.path.join(self.summary_folder,'EvaluatedImages',dataset,'Sample_'+str(s))
                                        if not os.path.exists(folder):
                                            os.makedirs(folder)
                                        insertBoundingBox(X[s], bb_det, time_step[0][s]==t,
                                                          os.path.join(folder,
                                                                       'time_'+str(t)+'.png'), bb_true=bb_true)
                                        IMG_COUNTER +=1
                                        
                                elif self.data.hasData(dataset,segMap=False,bb=True):
                                    bb_true = computeBBfromBB(y_bb[s])
                                    # only compute IoU for most confident timestep
                                    if (time_step[0][s]==t):
                                        ious[s] = intersectionOverUnion(bb_det,bb_true)
                                
                                    # if desired, insert bounding boxes into image and save it
                                    if saveImgs and IMG_COUNTER < num_saveImgs:
                                        folder = os.path.join(self.summary_folder,'EvaluatedImages',dataset,'Sample_'+str(s))
                                        if not os.path.exists(folder):
                                            os.makedirs(folder)
                                        insertBoundingBox(X[s], bb_det, time_step[0][s]==t,
                                                          os.path.join(folder,
                                                                       'time_'+str(t)+'.png'), bb_true=bb_true)
                                        IMG_COUNTER +=1
                                        
                                # save evaluated images with integrated bounding box without ground truth
                                elif saveImgs and IMG_COUNTER< num_saveImgs:
                                    folder = os.path.join(self.summary_folder,'EvaluatedImages',dataset,'Sample_'+str(s))
                                    if not os.path.exists(folder):
                                        os.makedirs(folder)
                                    insertBoundingBox(X[s], bb_det, time_step[0][s]==t,
                                                          os.path.join(folder,
                                                                       'time_'+str(t)+'.png'))
                                    IMG_COUNTER +=1
                                
                        iou = np.mean(ious)
                        with open(os.path.join(self.log_folder,'eval_iou.txt'), 'a') as log:
                            log.write('Average IOU (%s): %0.2f \n' %(dataset,iou))
                    #####################################################################################################
                    # SELF TRANSFER #
                    #################
                    elif self.model_name.lower()=='self_transfer':
                        # get indices of highest activations
                        num_argmax_pools = tf.get_collection('num_argmax_pools')[0]
                        argmax_pooling = []
                        for i in range(num_argmax_pools):
                            argmax_pooling.append(tf.get_collection('argmax_pooling_'+str(i))[0])
                        # get weight and bias tensors, as well as conv tensors
                        num_weights = tf.get_collection('num_weights')[0]
                        weight_tensors = []
                        bias_tensors = []
                        conv_tensors = []
                        for i in range(num_weights):
                            weight_tensors.append(tf.get_collection('weights_'+str(i))[0])
                            bias_tensors.append(tf.get_collection('bias_'+str(i))[0])
                            conv_tensors.append(tf.get_collection('conv_tensor_'+str(i))[0])
                        location_out = tf.get_collection('location_out')[0]
                        #guessed_class = tf.get_collection('guessed_class')[0]
                        argmax_indices, loc_out, weight_tensors_, bias_tensors_, conv_tensors_ = \
                                            sess.run([argmax_pooling, location_out, weight_tensors,
                                                      bias_tensors, conv_tensors],
                                                               feed_dict = {x: X, y_: y,
                                                  keep_prob:1, batch_norm_training:False})
                        class_indices = np.argmax(loc_out,axis=1)
                        # read in network information necessary to recover the location
                        params_dic = {}
                        model_params = ['kernel_sizes','pooling','padding_type',
                                  'kernel_size_loc','feature_maps','img_size',
                                  'label_size']
                        with open(os.path.join(self.model_path,'info.txt'),'r') as f:
                            for line in f:
                                k = line.lstrip('\t').split(':')[0]
                                if k in model_params:
                                    val = line.split(': ')[1]
                                    # convert lists 
                                    if '[' in val:
                                        val = list(map(int,val.replace('[','').replace(']','').split(', ')))
                                    # convert tuples
                                    elif '(' in val:
                                        val = tuple(map(int,val.replace('(','').replace(')','').split(', ')))
                                    # convert ints
                                    else:
                                        try:
                                            val = int(val)
                                        #just keep as string
                                        except:
                                            val = val.rstrip('\n')
                                    params_dic[k] = val
                        
                        # get object location for each image in batch
                        object_locations = [backtrack_locations([argmax_indices[j][i] for j in range(num_argmax_pools)],
                                            [conv_tensors_[j][i] for j in range(num_weights)],
                                            weight_tensors_, bias_tensors_, class_indices[i],params_dic)
                                            for i in range(X.shape[0])]
                        
                        in_bb = np.zeros(y.shape[0])
                        in_bb.fill(np.nan)
                        # create folder if evaluated images shall be saved
                        if saveImgs:
                            if not os.path.exists(os.path.join(self.summary_folder,'EvaluatedImages')):
                                os.makedirs(os.path.join(self.summary_folder,'EvaluatedImages')) 
                                
                        IMG_COUNTER = 0
                        # iterate over samples
                        for s in range(X.shape[0]):
                            if self.data.hasData(dataset, segMap = True):
                                bb_true = computeBBfromSM(y_segMap[s])
                                obj_loc = object_locations[s]
                                in_bb[s] = point_in_bounding_box(bb_true,obj_loc)
                            elif self.data.hasData(dataset, segMap = False, bb = True):
                                bb_true = computeBBfromBB(y_bb[s])
                                obj_loc = object_locations[s]
                                in_bb[s] = point_in_bounding_box(bb_true,obj_loc)
                            else:
                                bb_true = None
                                obj_loc = object_locations[s]
                                
                            # if desired, insert bounding boxes into image and save it
                            if saveImgs and IMG_COUNTER < num_saveImgs:
                                folder = os.path.join(self.summary_folder,'EvaluatedImages',dataset)
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                                insert_blob(X[s], obj_loc,
                                                  os.path.join(folder,
                                                               'img_'+str(s)+'.png'), bb_true=bb_true)
                                IMG_COUNTER +=1
                                
                        loc_accuracy = np.mean(in_bb)
                        with open(os.path.join(self.log_folder,'eval_loc.txt'), 'a') as log:
                            log.write('Average in BB (%s): %0.2f \n' %(dataset,loc_accuracy))
                            
                    #####################################################################################################    
                        
                    else:
                        raise RuntimeError('Evaluation of localization not available for %s'%(self.model_name))
                
                
                    
                   
    ###########################################################################           
                
    def computeAccuracy(self, datasets):
        """
        Method loads model to compute accuracy.
        
        Args:
            datasets: the datasets to evaluate accuracy on ({train,val,test})
        """
        print('Computing Accuracy')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                ###
                # load trained graph
                ###
                loader = tf.train.import_meta_graph(os.path.join(self.model_path,"model.meta"))
                loader.restore(sess, os.path.join(self.model_path,'model'))
                saver = tf.train.Saver()
                saver.restore(sess, os.path.join(self.model_path,"model.ckpt"))
                
                ##
                # get placeholders and tensor nodes to be evaluated
                # get test data to do evluation on
                # do evaluation
                ##
                
                x = tf.get_collection("x")[0]
                y_ = tf.get_collection("y_")[0]
                keep_prob = tf.get_collection('keep_prob')[0]
                batch_norm_training = tf.get_collection('batch_norm_training')[0]
                accuracy = tf.get_collection('accuracy')[0]
                
                accuracies = np.zeros(len(datasets))
                
                if self.model_name=='self_transfer':
                    alpha = tf.get_collection('alpha')[0]
                    additional_feeds = {alpha:np.float32(0.5)}
                else:
                    additional_feeds = {}
                
                for i,d in enumerate(datasets):
                    num_samples = self.data.getSize(d)
                    acc_batch=[]
                    b = 0
                    # evaluate accuracy batch by batch (avoids OOM)
                    while (b+1)*self.batch_size <= num_samples:
                        X,y = self.data.getData(d, indices = np.arange(b*self.batch_size,(b+1)*self.batch_size))
                        feed_dict = {x: X, y_: y, keep_prob:1, batch_norm_training:False}
                        for k in additional_feeds:
                            feed_dict[k] = additional_feeds[k]
                        acc = sess.run(accuracy, feed_dict=feed_dict)
                        acc_batch.append(acc)
                        b +=1
                    # process rest batch
                    """
                    X,y = self.data.getData(d, indices = np.arange(int(self.data.getSize(d)/self.batch_size)*self.batch_size, self.data.getSize(d)))
                    feed_dict = {x: X, y_: y, keep_prob:1, batch_norm_training:False}
                    for k in additional_feeds:
                        feed_dict[k] = additional_feeds[k]
                    acc = sess.run(accuracy, feed_dict=feed_dict)
                    acc_batch.append(acc)
                    """
                    # take average
                    accuracies[i] = np.mean(acc_batch)
                
                
                with open(os.path.join(self.log_folder,'accuracy.txt'), 'w') as log:
                    for t in zip(datasets,accuracies):
                        log.write('%s Accuracy : %0.2f\n' %(t[0].title(), t[1]))
                        
    ###########################################################################                  
                        
    def computeLoss(self, datasets=['test']):
        """
        Method loads model to compute loss.
        
        Args:
            datasets: the datasets to evaluate accuracy on ({train,val,test})
        """
        print('Computing loss')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                ###
                # load trained graph
                ###
                loader = tf.train.import_meta_graph(os.path.join(self.model_path,"model.meta"))
                loader.restore(sess, os.path.join(self.model_path,'model'))
                saver = tf.train.Saver()
                saver.restore(sess, os.path.join(self.model_path,"model.ckpt"))
                
                ##
                # get placeholders and tensor nodes to be evaluated
                # get test data to do evluation on
                # do evaluation
                ##
                
                x = tf.get_collection("x")[0]
                y_ = tf.get_collection("y_")[0]
                keep_prob = tf.get_collection('keep_prob')[0]
                batch_norm_training = tf.get_collection('batch_norm_training')[0]
                loss = tf.get_collection('loss')[0]
                
                losses = np.zeros(len(datasets))
                
                if self.model_name=='self_transfer':
                    alpha = tf.get_collection('alpha')[0]
                    additional_feeds = {alpha:np.float32(0.5)}
                else:
                    additional_feeds = {}
                    
                    
                for i,d in enumerate(datasets):
                    num_samples = self.data.getSize(d)
                    loss_batch=[]
                    b = 0
                    # evaluate accuracy batch by batch (avoids OOM)
                    while (b+1)*self.batch_size < num_samples:
                        X,y = self.data.getData(d, indices = np.arange(b*self.batch_size,(b+1)*self.batch_size))
                        feed_dict = {x: X, y_: y, keep_prob:1, batch_norm_training:False}
                        for k in additional_feeds:
                            feed_dict[k] = additional_feeds[k]
                        l = sess.run(loss, feed_dict=feed_dict)
                        loss_batch.append(l)
                        b +=1
                    losses[i] = np.mean(loss_batch)
                
                
                with open(os.path.join(self.log_folder,'loss.txt'), 'w') as log:
                    for t in zip(datasets,losses):
                        log.write('%s Loss : %0.2f\n' %(t[0].title(), t[1]))
                            
                
                
               
    
    
    
        
    
if __name__ == '__main__':
    pass
    