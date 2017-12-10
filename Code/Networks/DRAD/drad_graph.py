# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:20:40 2017

@author: nickv
"""



import tensorflow as tf
import numpy as np
import os
from sys import path
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'HelperConvNets'))
print(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'HelperConvNets'))
from CNN import CNN

class Drad:
    """
    The DRAD graph is responsible for building the tensorflow graph for object
    localization using the DRAD mechanism. There are different parameters,
    implemented as class attributes, that define the graph. In the end,
    the graph will be saved as .meta file to be picked up by the Network
    class, that is responsible for the training.
    """
    
    def __init__(self, save_model_to, img_size, label_size, attention_N = 21, 
                 hidden_n = 256, seq_length=3,
                 conv_pretrained = False, conv_weights='', 
                 kernel_sizes = [5,5], feature_maps = [32,64], pooling = [2,2],
                 dense_dims_cnn=[512,256], trainable = True, 
                  batch_normalization = False, padding_type = 'VALID',
                 local_response_normalization = True,learn_sigma = False,
                 leaky_relu_alpha = 0.3, limit_delta = False, delta_limit = 2,
                 debug = False, gru = False):
        """
        Args:
            save_model_to: - folder path where model shall be saved
            attention_N : - size of the attention patch (NxN)
            hidden_n : - number of hidden units in RNN 
            seq_length : - length of RNN
            img_size_h : - height of image
            img_size_w : - width of image
            no_color_channels: - the number of color channels in image
            label_size: -size of label vector
            denseInit: - helper flag for creation of graph
            rnnInit: - helper flag for creation of graph
            conv_pretrained: - flag indicating whether CNN has been pretrained
            conv_weights: - path to folder where the weights of the pretrained
                            CNN lie
            kernel_sizes,feature_maps,pooling,dense_dims_cnn: cnn parameters
            trainable: Flag indicating if pretrained CNN variables should
                        be trainable
            learn_sigma: - flag indicating whether sigma should be learned
                            independent of delta
            leaky_relu_alpha: alpha value in case a leaky relu activation is used
            limit_delta: put some more restrictions on delta to force it to smaller bounding box
            delta_limit: the upper limit of delta
            debug: flag to indicate if paramters should show up in tensorboard
            gru: flag to indicate if gru units should be used instead of lstm
        """
        self.attention_N = attention_N         
        self.hidden_n = hidden_n         
        self.seq_length = seq_length            
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]       
        self.no_color_channels = img_size[2]
        self.label_size = label_size            
        self.denseInit = True           
        self.rnnInit = True             
        self.save_model_to = save_model_to  
        self.conv_pretrained = conv_pretrained
        self.conv_weights = conv_weights
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        self.pooling = pooling
        self.dense_dims_cnn = dense_dims_cnn
        self.trainable = trainable
        self.padding_type = padding_type
        self.batch_normalization = batch_normalization
        self.local_response_normalization = local_response_normalization
        self.learn_sigma = learn_sigma
        self.leaky_relu_alpha = leaky_relu_alpha
        self.limit_delta = limit_delta
        self.delta_limit = delta_limit
        self.debug = debug
        self.gru = gru
        # create info.txt in folder with the parameter configuration
        with open(os.path.join(self.save_model_to,'info.txt'),'w') as info:
            info.write("Configuration of DRAD graph:\n")
            for k in self.__dict__.keys():
                info.write('\t'+k +": "+ str(self.__dict__[k])+"\n")
    
    
    def build_graph(self):
        """
        Method that sets up DRAD Graph
        """
        
        ###########################################################################
        #####################
        # Helper Operations #
        ##################### 
        
        def extract_cnn_parameters():
            """
            When a pretrained CNN is used, we need to extract the architecture
            parameters of the CNN in order to be able to reuse pretrained 
            weights.
            """
            lists = ['kernel_sizes','feature_maps','pooling']
            bools = ['trainable','batch_normalization', 'local_response_normalization']
            #strs = ['padding_type']
            def get(keyword, txt):
                if keyword in lists:
                    parts = txt.split(keyword+'\': [')[1].split(']')[0].split(',')
                    res = [int(p) for p in parts]
                elif keyword in bools:
                    part = txt.split(keyword+'\': ')[1]
                    if part[0].lower()=='t':
                        res = True
                    else:
                        res = False
                else:
                    part = txt.split(keyword+'\': \'')[1].split('\'')[0]
                    res = part                    
                return res
            with open(os.path.join(os.path.dirname(self.conv_weights),
                                               'info.txt'), 'r') as info:
                txt = info.read()
                return (get('pooling',txt), get('feature_maps',txt), get('kernel_sizes',txt),
                       get('padding_type',txt), get('batch_normalization',txt),
                        get('local_response_normalization',txt))
                
        
        
        def read(x, h_prev, batch_size, keep_prob, batch_norm_training):
            """
            Compute attention patch based on input image and previous hidden state
                x: input image
                h_prev: previous hidden state
                batch_size: batch size, necessary for reshape operations
                keep_prob: 1-dropout_rate for h_prev before being fed to dense layer
            """
            (Fx, Fy), (gx,gy,sigma2,delta) = attn_window(h_prev, batch_size, 
                                                keep_prob, batch_norm_training)
            
            # Apply filter on image
            
            x = tf.reshape(x, [batch_size,self.img_size_h,self.img_size_w, self.no_color_channels], 
                           name = "img_reshape_read")
            Fxt = tf.cast(tf.transpose(Fx, perm = [0,2,1], name = "Fx_transpose"),dtype=tf.float32)              # transpose each of the batch_size matrices
            
            """
            Do 4D matrix multiplication of x with Fx^t and Fy:
                reshape and transpose matrices such that in the end
                we have a multiplication of each sample in x and each color 
                channel in x_i with Fx^t_i from the right and Fy_i from the left
            """
            def filter_image_3d(inp):
                x = inp[0]
                Fxt = inp[1]
                Fy = inp[2]
                # transpose x (necessary to make reshape consistent)
                x = tf.transpose(x,[0,2,1])
                # reduce x to 2D (append channel dimension to height dimension)
                x = tf.reshape(x,(self.img_size_h*self.no_color_channels,
                                  self.img_size_w))
                # do right hand side multiplication
                tmp = tf.matmul(x,Fxt)
                # reshape result to 3d
                tmp_reshape = tf.reshape(tmp,(self.img_size_h,
                                              self.no_color_channels,self.attention_N))
                # tranpose tmp result to get the true temporary result where
                # each sample and color channel was multiplied with the respective Fx^t
                tmp_transpose = tf.transpose(tmp_reshape,perm=[0,2,1])
                # reshape temporary result to prepare for multiplication with Fy
                tmp_reshape_again = tf.reshape(tmp_transpose,(self.img_size_h,
                                                              self.attention_N*self.no_color_channels))
                # do the matrix multiplication from the left
                result = tf.matmul(Fy,tmp_reshape_again)
                # reshape the result to get the actual 3d outcome
                glimpse = tf.reshape(result,(self.attention_N,self.attention_N,
                                             self.no_color_channels))
                
                return (glimpse,Fxt,Fy)
            
            glimpse,_,_ = tf.map_fn(filter_image_3d, (x,Fxt,Fy),
                                dtype=(tf.float32,tf.float32,tf.float32))
            
            return (glimpse,(gx,gy,sigma2,delta))
            
       
        def attn_window(h_prev, batch_size, keep_prob, batch_norm_training):
            """
            Compute filterbanks Fx and Fy, as well as filter multiplier gamma 
            """
            with tf.variable_scope("Patch_Parameters") as scope:
                if not self.denseInit:
                    scope.reuse_variables()
                # apply batch normalization and dropout to h_prev
                if self.batch_normalization:
                    h_prev = tf.layers.batch_normalization(h_prev, training=batch_norm_training)
                h_prev = tf.nn.dropout(h_prev,keep_prob)
                parameters = tf.layers.dense(h_prev,3, activation = tf.tanh,
                                              name = "attn_patch_params")  # tanh dense layer to ensure g_x,g_y \in [-1,1]
        
                gx_, gy_, delta_ = tf.split(parameters,[1,1,1], axis = 1, name = "split_grid_center")
                
                if self.debug:
                    tf.summary.scalar('h_prev_mean',tf.reduce_mean(h_prev))
                    tf.summary.scalar('h_prev_nan',tf.cast(tf.reduce_any(tf.is_nan(h_prev)),dtype=tf.int32))
                    tf.summary.scalar('parameters_nan',tf.cast(tf.reduce_any(tf.is_nan(parameters)),dtype=tf.int32))
                    tf.summary.scalar('gx_nan',tf.cast(tf.reduce_any(tf.is_nan(gx_)),dtype=tf.int32))
                    tf.summary.scalar('gy_nan',tf.cast(tf.reduce_any(tf.is_nan(gy_)),dtype=tf.int32))
                    tf.summary.scalar('delta_nan',tf.cast(tf.reduce_any(tf.is_nan(delta_)),dtype=tf.int32))
            
                # compute grid stride delta (normalize according to image size)
                # delta = {(delta_+1) * (min(img_width,img_height)-attention_n)/(attention_n-1) * 0.5} + 1 
                # --> delta \in [1 , (min(img_w,img_h)-1)/(attention_n-1)] (range for meaningful stride length)
                if not self.limit_delta:
                    delta =tf.multiply((delta_+1)*0.5, tf.cast(tf.divide(tf.minimum(self.img_size_h,self.img_size_w)-1,
                                         self.attention_N-1),tf.float32)-1)+1
                else:
                    # restrict delta to [1,delta_limit] while keeping above restriction intact as well
                    delta = tf.minimum(((delta_+1)*(self.delta_limit-1)*0.5)+1,
                                       tf.multiply((delta_+1)*0.5, tf.cast(tf.divide(tf.minimum(self.img_size_h,self.img_size_w)-1,
                                         self.attention_N-1),tf.float32)-1)+1)
                
                if self.learn_sigma:
                    # ensure sigma is positive
                    sigma2_ = tf.layers.dense(h_prev,1,activation = tf.sigmoid,name = "sigma2_comp")
                    sigma2 = tf.multiply(sigma2_,2*delta)+.01        # sigma \in [.01, 2*delta] 
                                                                     #(sigma cannot become zero, otherwise architecture crashes as we divide by sigma later)
                else:
                    # set sigma2 equal to delta as simplifying measure
                    sigma2 = (delta/np.pi)**2
                             
                if self.denseInit:
                    self.denseInit = False
            
            # rescale g_x, g_y to image size, s.t. attention patch stays completely in image
            # g_x \in [0.5*((attention_N-1)*\delta+1), width - 0.5*((attention_N-1)*delta)+1]
            # g_y \in [0.5*((attention_N-1)*\delta+1), height - 0.5*((attention_N-1)*delta)+1]
            tmp = (tf.multiply(tf.cast(self.attention_N-1,tf.float32), delta)+1)*0.5
            gx = tf.multiply(gx_+1, self.img_size_w-2*tmp)*0.5 + tmp
            gy = tf.multiply(gy_+1, self.img_size_h-2*tmp)*0.5 + tmp
            
            ###
            # Debugging summaries
            ###
            if self.debug:
                with tf.name_scope('debug_summaries'):
                    tf.summary.scalar('delta_min', tf.reduce_min(delta))
                    tf.summary.scalar('delta_avg', tf.reduce_mean(delta))
                    tf.summary.scalar('delta_max', tf.reduce_max(delta))
                    
                    tf.summary.scalar('sigma2_min', tf.reduce_min(sigma2))
                    tf.summary.scalar('sigma2_avg', tf.reduce_mean(sigma2))
                    tf.summary.scalar('sigma2_max', tf.reduce_max(sigma2))
                    
                    tf.summary.scalar('gx_min', tf.reduce_min(gx))
                    tf.summary.scalar('gx_avg', tf.reduce_mean(gx))
                    tf.summary.scalar('gx_max', tf.reduce_max(gx))
                    
                    tf.summary.scalar('gy_min', tf.reduce_min(gy))
                    tf.summary.scalar('gy_avg', tf.reduce_mean(gy))
                    tf.summary.scalar('gy_max', tf.reduce_max(gy))
            
            
            return (filterbank(gx,gy,sigma2,delta,batch_size),(gx,gy,sigma2,delta))
        
        
        def filterbank(gx,gy,sigma2,delta, batch_size):
            """
            Given the center (gx,gy), the between filter distance delta, 
            as well as the spread gamma of each Gaussian filter, a 
            [attention_N x attention_N] image patch is constructed represented
            by a horizontal Gaussian Fx and a vertical Gaussian Fy
            """
            
            # construct 1xN matrix: [[0,...,N-1]] -> helper to compute grid centers
            grid = tf.reshape(tf.cast(tf.range(self.attention_N), tf.float32),[1, -1], 
                              name = "grid_reshape_filterbank")
            
            # compute all the x_values/y_values of the Gaussian means
            #   mu_x = [[mu_x_0,...,mu_x_{attention_N - 1}]]
            mu_x = gx + (grid - self.attention_N/2 + 0.5) * delta                  # DEVIATION FROM OTHER IMPLEMENTATION
            mu_y = gy + (grid - self.attention_N/2 + 0.5) * delta                  # This way (gx,gy) are actually in the center
            
            # reshape means to fit calculation format necessary later
            mu_x = tf.reshape(mu_x, [batch_size, self.attention_N, 1], 
                              name = "mu_x_reshape_filterbank")
            mu_y = tf.reshape(mu_y, [batch_size, self.attention_N, 1], 
                              name = "mu_y_reshape_filterbanK")
            
            # create 2D index array [[0,1,...,img_width/height-1],....,[0,1,...,imgwidth/height-1]] attention_N - times (batch_size times)
            img_x = tf.reshape(tf.tile(tf.cast(tf.range(self.img_size_w), tf.float32),
                                     [self.attention_N*batch_size]), 
                                    [batch_size,self.attention_N,self.img_size_w], 
                                    name = "img_index_arr_x")
            
            img_y = tf.reshape(tf.tile(tf.cast(tf.range(self.img_size_h), tf.float32),
                                     [self.attention_N*batch_size]), 
                                    [batch_size,self.attention_N,self.img_size_h], 
                                    name = "img_index_arr_y")
            
            # compute filterbanks according to paper 
            #   2D matrices of shape: attention_A x img_size
            sigma2 = tf.reshape(sigma2, [-1, 1, 1], name = "reshape_sigma")             #necessary reshape for division
            Fx = tf.exp(-tf.square((img_x-mu_x)/(2*sigma2)))
            Fy = tf.exp(-tf.square((img_y-mu_y)/(2*sigma2)))
            
            # Normalize rows of the filterbanks
            Fx = Fx / tf.reshape(tf.maximum(tf.reduce_sum(Fx,2),1e-8), [batch_size,self.attention_N,1])
            Fy = Fy / tf.reshape(tf.maximum(tf.reduce_sum(Fy,2),1e-8), [batch_size,self.attention_N,1])
            
            return tf.cast(Fx,dtype=tf.float32),tf.cast(Fy,dtype=tf.float32)
            
        ###########################################################################
        with tf.Graph().as_default():
            ##
            # placeholders for input data
            ##
            x = tf.placeholder(tf.float32, shape = [None,self.img_size_h,self.img_size_w,self.no_color_channels], name = 'image_placeholder')
            keep_prob = tf.placeholder(tf.float32) 
            batch_norm_training = tf.placeholder(tf.bool)
            batch_size = tf.shape(x)[0]
            
            ##
            # RNN Unit
            ##
            if self.gru:
                gru = tf.contrib.rnn.GRUCell(self.hidden_n)          
            else:
                lstm = tf.contrib.rnn.LSTMCell(self.hidden_n, num_proj = self.label_size)
            ###
            # Hidden state h
            ###
            if self.gru:
                state = tf.zeros([batch_size,self.hidden_n])                        # hidden_state
            else:
                state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size,self.hidden_n]), 
                                                      tf.zeros([batch_size, self.label_size]))     
            
            """
            Initialize parameters necessary to detect timepoint where CNN
            is the most confident to classify. 
            Analogy: RNN is a police officer showing a witness, which is the CNN,
            different pictures of suspects. Judging by the witness' reaction, the police officer
            adapts his choice of pictures. In the end, picture where witness
            was most confident is taken. 
            """
            best_max_activations = tf.zeros([batch_size],dtype=tf.float32)           # maximum activations
            best_max_class_index = tf.zeros([batch_size],dtype=tf.int64)   # corresponding class indices
            best_max_time_step = tf.zeros([batch_size],dtype=tf.int32)     # time step of maximum activation
            
            
            #######################
            # t time steps of RNN #
            #######################
            
            for t in range(self.seq_length):
                if self.gru:
                    r_t,(gx,gy,sigma2,delta) = read(x, state, batch_size, keep_prob, 
                                                        batch_norm_training)      # get image patch using the attention window
                else:
                    r_t,(gx,gy,sigma2,delta) = read(x, state[0], batch_size, keep_prob, 
                                                        batch_norm_training)      # get image patch using the attention window
                ##
                # Insertion of CNN (potentially pretrained)
                ##
                shape = (self.attention_N,self.attention_N, self.no_color_channels)
                # do zero padding to match input shape of pretrained CNN
                if self.conv_pretrained:
                    # get intended shape of CNN and also the architecture
                    self.pooling, self.feature_maps, self.kernel_sizes, \
                    self.padding_type, self.batch_normalization, \
                    self.local_response_normalization= extract_cnn_parameters()
                ####
                # push r_t through CNN
                ####
                cnn_out, weight_variables =\
                     CNN(r_t, shape, keep_prob, batch_norm_training,
                         batch_normalization = self.batch_normalization,
                         local_response_normalization = self.local_response_normalization,
                          trainable = self.trainable, kernel_sizes = self.kernel_sizes,
                          feature_maps = self.feature_maps, pooling = self.pooling,
                          padding_type = self.padding_type)
                # add dense layer
                with tf.name_scope('CNN_Dense'):
                    output_shape_flat = np.prod(np.array(cnn_out.get_shape().as_list()[1:],dtype=np.int32),dtype=np.int32)
                    W_fc1 = tf.Variable(tf.truncated_normal(shape=[output_shape_flat, 
                                                                   self.hidden_n], stddev=0.1, name = 'Weights_dense'))
                    b_fc1 = tf.Variable(tf.constant(0.1,shape=[self.hidden_n], 
                                                    name='Bias_dense'))
                    cnn_out_reshape = tf.reshape(cnn_out,[-1,output_shape_flat])
                    c_tmp = tf.matmul(cnn_out_reshape,W_fc1)+ b_fc1
                    c_t = tf.maximum(self.leaky_relu_alpha*c_tmp, c_tmp)
                    
                    if self.debug:
                        tf.summary.histogram('cnn_dense_weights',W_fc1)
                        tf.summary.histogram('cnn_feature_vector',c_t)
                        tf.summary.scalar('cnn_nan',tf.cast(tf.reduce_any(tf.is_nan(c_t)),dtype=tf.int32))
                    
                    c_t = tf.nn.dropout(c_t,keep_prob)
        
                with tf.variable_scope("rnn_cell") as scope:
                    if not self.rnnInit:
                        scope.reuse_variables()
                    if self.gru:
                        output, state = gru(c_t, state)             # push patch through GRU unit -> obtain output and hidden state
                    else:
                        output,state = lstm(c_t, state)    
                    
                    if self.debug:
                        if self.gru:
                            tf.summary.histogram('history_state',state)
                        else:
                            tf.summary.histogram('history_state',state[0])
                        tf.summary.histogram('output_vector',output)
                    
                    ###############################
                    # look for highest activation #
                    ###############################
                    def f_help(inp):
                        # helper function for map_fn function (to apply process elementwise)
                        # inp: (comparison, best, current)
                        return (inp[0],tf.cond(inp[0],lambda:inp[2],lambda:inp[1]),inp[2])
                    ##
                    max_activations = tf.reduce_max(output,axis=1)
                    max_indices = tf.argmax(output, axis=1)   # indices of highest activation
                    time_tensor = tf.fill([batch_size],t)
                    comparison = tf.less(best_max_activations,max_activations)
                    _,best_max_activations, _ = tf.map_fn(f_help,(comparison,best_max_activations,max_activations),
                                                         dtype = (tf.bool,tf.float32,tf.float32))
                    _,best_max_class_index, _= tf.map_fn(f_help,(comparison, best_max_class_index, max_indices),
                                                         dtype = (tf.bool,tf.int64,tf.int64))
                    _,best_max_time_step, _ = tf.map_fn(f_help,(comparison,best_max_time_step, time_tensor),
                                                        dtype=(tf.bool,tf.int32,tf.int32))
                    ####
                   
                    
                    
                if self.rnnInit:
                    # set flag to False, now that all layer parameters have been initialized
                    self.rnnInit = False
                # save last glimpse
                if t+1 == self.seq_length:
                    last_glimpse = tf.reshape(r_t,
                                 [-1,self.attention_N,self.attention_N,self.no_color_channels], name = "glimpse_reshape")
                    if self.debug:
                        tf.summary.image('last_glimpse', last_glimpse, 10)
                    # add highest confidence time step to summary
                    tf.add_to_collection('best_time_step', best_max_time_step)
                # add image patch parameters to collection to 
                # compute bounding box in evaluation
                tf.add_to_collection('gx_'+str(t), gx)
                tf.add_to_collection('gy_'+str(t), gy)
                tf.add_to_collection('sigma2_'+str(t), sigma2)
                tf.add_to_collection('delta_'+str(t), delta)
                
            ####################
                
            
            ##
            # Dense layer to push final output through to obtain unnormalized probability vector
            ##
            with tf.variable_scope("Dense_output"):
                if self.debug:
                    tf.summary.scalar('Output_nan',tf.cast(tf.reduce_any(tf.is_nan(output)),dtype=tf.int32))
                y_tmp = tf.layers.dense(output, self.label_size,name = "output_layer")
                y = tf.maximum(self.leaky_relu_alpha*y_tmp,y_tmp)
                if self.debug:
                    tf.summary.scalar('result_nan',tf.cast(tf.reduce_any(tf.is_nan(y)),dtype=tf.int32))
            
            ##
            # add necessary parameters to graph collection
            ##
            tf.add_to_collection("x",x)
            tf.add_to_collection("y",y)
            tf.add_to_collection('keep_prob',keep_prob)
            tf.add_to_collection('batch_norm_training', batch_norm_training)
            
            
            ##
            # save the graph
            ##
            def create_variable_dict(weight_variables):
                """
                Helper method to load pretrained CNN weights
                """
                dic = {}
                i = 0
                while(i<len(weight_variables)-1):
                    dic['W_'+str(i/2)] = weight_variables[i]
                    dic['b_'+str(i/2)] = weight_variables[i+1]
                    i +=2
                return dic
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver= tf.train.Saver()
                if self.conv_pretrained:
                    saver_conv = tf.train.Saver(create_variable_dict(
                            weight_variables))
                    saver_conv.restore(sess, self.conv_weights)
                saver.save(sess,os.path.join(self.save_model_to,"model"))
                saver.export_meta_graph(os.path.join(self.save_model_to,"model.meta"))
        

    
    
if __name__ == "__main__":
    ###
    # Parameters
    ###
    attention_N = 12
    hidden_n = 256
    seq_length = 10
    learning_rate = 1e-4
    img_size = (30,40)
    label_size = 10
    p = os.getcwd().rstrip(os.path.join("Code","Networks","DRAD"))
    targetDir = os.path.join(p,"Networks","DRAD")
    
    drad = Drad(targetDir, img_size)
    drad.buildGraph()
        