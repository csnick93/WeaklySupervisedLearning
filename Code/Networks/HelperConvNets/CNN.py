# -*- coding: utf-8 -*-
###
# Start interactive session
###

import tensorflow as tf
import numpy as np
import os

from sys import path
path.insert(1,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))),'DataLoader'))
from dataloader import DataLoader

# weight initialization
def weight_variable(shape, name, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable = trainable, name = name)

def bias_variable(shape, name, trainable = True):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, trainable = trainable, name=name)

# convolution and pooling
def conv2d(x,W, padding_type):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = padding_type)

def max_pool(x, padding_type, pool_size=2):
    return tf.nn.max_pool(x, ksize = [1,pool_size,pool_size,1], 
                          strides = [1,pool_size,pool_size,1], padding = padding_type)

def CNN(x, x_shape,keep_prob, batch_norm_training,batch_normalization = False,
        local_response_normalization = True, trainable=True, 
        kernel_sizes = [5], feature_maps= [32], 
        pooling = [2],dense_dims = [],padding_type='SAME'):
    """
    Method that builds a straightforward CNN (stacking of convolutional 
    layers with dense layers in the end and a softmax layer). The number 
    of layers is determined implicitly by looking at the other parameters, such
    as kernel_sizes, feature maps etc. However, these need to be consistent in
    length; otherwise a RuntimeError will be raised.
    
    Args:
        x: the input image (batch) to be processed
        x_shape: shape of the input 
        keep_prob: the placeholder for the dropout parameter
        batch_norm_training: boolean to indicate whether we are in training or inference mode
        batch_normalization: flag to indicate if batch normalization should be 
                                applied after each layer
        trainable: flag indicating whether variables shall be trained (in case of
                    pretrained variables, that might not be desirable)
        kernel_sizes: size of the filters in each convolutional layer
        feature_maps: number of feature maps per layer
        pooling: indicates for each layer if pooling should be done and what size
                (per default 2x2); if value <= 1, no pooling is done
        dense_dims: sizes of the dense layers stacked upon CNN
        padding_type: the type of padding done {SAME,VALID}
        
    Returns:
        tmp_out: the output of the CNN layers. Called tmp_out, because the 
                    output will have to be further processed in at least
                    one loss layer
        weight_variables: list of weight variables used, such that
                            saving and restoring can be done later
        
    Raises:
        RuntimeError: if length of kernel_sizes, feature_maps and pooling
                        is not the same
    """
    ###
    # Check consistency of parameters
    ###
    if (len(kernel_sizes)!=len(feature_maps)):
        raise RuntimeError('Input parameters are not consistent. The kernel_size \
                           parameter and feature_maps parameter have varying \
                           length')
    # make sure that parameters are integers
    kernel_sizes = [int(k) for k in kernel_sizes]
    feature_maps = [int(f) for f in feature_maps]
    pooling = [int(p) for p in pooling]
    dense_dims = [int(d) for d in dense_dims]
    
    ###
    # Build convolutional layers
    ###
    #print('Building Network')
    weight_variables = []
    for l in range(len(kernel_sizes)):
        with tf.name_scope('Conv_'+str(l)):
            if l == 0:
                ##
                # first layer
                ##
                W_conv = weight_variable([kernel_sizes[0],kernel_sizes[0],x_shape[2],feature_maps[0]], 
                                          trainable=trainable,name='Weights_0')
                b_conv = bias_variable([feature_maps[0]], trainable=trainable,
                                        name='Bias_0')
                
                # reshape input to 2D to enable convolutional operation
                x_image = tf.reshape(x,[-1,x_shape[0],x_shape[1],x_shape[2]], name='x_img_reshape')
                
                tmp_out = conv2d(x_image,W_conv, padding_type)+b_conv
                                
                # apply batch normalization
                if batch_normalization:
                    tmp_out = tf.layers.batch_normalization(tmp_out, axis=1, training=batch_norm_training)
                
                #apply relu activation AFTER batch normalization
                tmp_out = tf.nn.relu(tmp_out)
                
            else:
                W_conv = weight_variable([kernel_sizes[l],kernel_sizes[l],feature_maps[l-1],feature_maps[l]], 
                                          trainable=trainable,name='Weights_'+str(l))
                b_conv = bias_variable([feature_maps[l]], trainable=trainable,
                                        name='Bias_'+str(l))
                
                tmp_out = conv2d(tmp_out,W_conv, padding_type)+b_conv
                
                # apply batch normalization
                if batch_normalization:
                    tmp_out = tf.layers.batch_normalization(tmp_out, axis=1, training=batch_norm_training)
                
                #apply conv layer with relu activation function
                tmp_out = tf.nn.relu(tmp_out)
                
                
            # do pooling if appropriate
            if pooling[l] > 1:
                tmp_out = max_pool(tmp_out, padding_type,pool_size=pooling[l])
                
            if local_response_normalization:
                tmp_out = tf.nn.lrn(tmp_out, 5, bias = 1.0, alpha = 1e-4,
                                    beta = 0.75)
                
            # save weight variables in list
            weight_variables.append(W_conv)
            weight_variables.append(b_conv)
    
    
    
    ###
    # add dense layers
    ###
    if len(dense_dims)>0:
        ###
        # flatten last convolutional output
        ###
        flat_dim = int(tmp_out.get_shape()[1])*int(tmp_out.get_shape()[2])*int(tmp_out.get_shape()[3])
        tmp_out_reshape = tf.reshape(tmp_out,[-1,flat_dim], name='flatten')
        with tf.name_scope('Dense_0'):
            W_dense = weight_variable([flat_dim, dense_dims[0]],
                                            trainable = trainable,
                                              name='dense_weights_0')
            b_dense = bias_variable([dense_dims[0]], trainable = trainable,
                                    name = 'dense_bias_0')
            tmp_out = tf.matmul(tmp_out_reshape,W_dense)+ b_dense
            if batch_normalization:
                tmp_out = tf.layers.batch_normalization(tmp_out, training=batch_norm_training)              
            tmp_out = tf.nn.relu(tmp_out)
            weight_variables.append(W_dense)
            weight_variables.append(b_dense)
            
            tmp_out = tf.nn.dropout(tmp_out, keep_prob)
        
    for d in range(1,len(dense_dims)):
        with tf.name_scope('Dense_'+str(d)):
            W_dense = weight_variable([dense_dims[d-1], dense_dims[d]],
                                        trainable = trainable,
                                          name='dense_weights_'+str(d))
            b_dense = bias_variable([dense_dims[d]], trainable = trainable,
                                    name = 'dense_bias_'+str(d))
            tmp_out = tf.matmul(tmp_out,W_dense)+ b_dense
            if batch_normalization:
                tmp_out = tf.layers.batch_normalization(tmp_out, training=batch_norm_training)              
            tmp_out = tf.nn.relu(tmp_out)
            weight_variables.append(W_dense)
            weight_variables.append(b_dense)
            
            tmp_out = tf.nn.dropout(tmp_out, keep_prob)    
                          
    return tmp_out,weight_variables
                          
############################################################################################################################

def pretrainCNN(data,num_epochs, batch_size, opt, 
                opt_params, summary_intervals,complete_set,
                save_weights_to, summary_folder,
                model_path, cnn_params, dropout, l2_reg):
    """
    Train a CNN as specified by cnn_params on dataset data.
    The weights of the CNN are saved separately from the
    model in order to be reusable by other models.
    
    Args:    
        dataset: dataset to be loaded (cMNIST, embeddedMNIST,..)
        num_epochs: the number of epochs to do training
        batch_size: batch_size during training    
        opt: the type of optimizer to use
        opt_params : the optimizer parameters in the form of a dictionary
        summary_intervals: the interval length to produce summaries
        complete_set: flag that is useful for large datasets
                        when it would take too long to evaluate
                        whole dataset when producing summary
        save_weights_to: path where weights to be reused later
                            are saved
        summary_folder: folder where tensorflow summaries are saved
        cnn_params: parameters for CNN network architecture as dictionary
        dropout: dropout for dense layers
    
    Returns:
        None
        
    Raises:
        ValueError: if specified optimizer is not implemented
    """

    

    ###########################################################################
    # Helper Methods #
    ##################
    def next_batch(dataset, batch_size=None,complete_set = False):
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
            if dataset == 'train':
                if complete_set:
                    x_batch,y_batch = data.getData('train')
                else:
                    no_samples = data.getSize('train') 
                    indices = np.random.randint(0,no_samples, batch_size)
                    x_batch,y_batch = data.getData('train',batch_size, indices)
            else:
                x_batch,y_batch = data.getData(dataset)
                
            return (x_batch,y_batch)
    
    def save_model(sess, saver,val_acc, best_val_acc, model_path):
        """
        - function to check whether we found a new best model based on the 
            current validation accuracy
        - if so, we overwrite our previous checkpoint
        
        @Parameters:
            - sess: current session
            - saver: saver object belonging to this session
            - val_acc: current validation accuracy
        """
        # check if we need to save model (no validation accuracy recorded yet
        # , so we must be at beginning of training or improved val_acc)
        if val_acc > best_val_acc:
            print('Saving model')
            best_val_acc = val_acc
            saver.save(sess,os.path.join(model_path,"model.ckpt"))
        return best_val_acc
            
    def load_model(sess, saver, model_path):
        """
        - function to load the model
        - function checks whether there is checkpoint that we can load
            saved variables from
        
        @Parameters:
            - sess: current session
            - saver: saver object belonging to this session
        """
        for f in os.listdir(model_path):
            if ".ckpt" in f:
                saver.restore(sess, os.path.join(model_path,"model.ckpt"))
    
    ###########################################################################
    # preliminary work: calculate epoch_factor and create folders
    epoch_factor = np.int32(data.getSize('train')/ batch_size)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
        
    with tf.Graph().as_default():
        with tf.Session() as sess:
    
            # Define placeholders
            
            x = tf.placeholder(tf.float32, shape = ([None]+list(data.get_dimensions())))
            y_ = tf.placeholder(tf.float32, shape = [None,data.get_label_dimensions()])
            keep_prob = tf.placeholder(tf.float32)
            batch_norm_training = tf.placeholder(tf.bool)
            
            # define regularizer
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
            
            ##
            # get graph output and apply dropout
            ##
            tmp_out, weight_variables = CNN(x,(data.getImgHeight(), 
                                                           data.getImgWidth(),
                                                            data.get_num_color_channels()),
                                                            keep_prob, batch_norm_training,
                                                            **cnn_params)
            if len(cnn_params['dense_dims'])==0:
                output_shape_flat = np.prod(np.array(tmp_out.get_shape().as_list()[1:],dtype=np.int32),dtype=np.int32)
                tmp_out = tf.reshape(tmp_out,[-1,output_shape_flat])
                
            tmp_out = tf.nn.dropout(tmp_out, keep_prob)
            ###
            # add last dense softmax layer
            ###
            if len(cnn_params['dense_dims'])==0:
                W_fc2 = weight_variable(shape = ([output_shape_flat,data.get_label_dimensions()]), 
                                    name='softmax_weight')
            else:
                W_fc2 = weight_variable(shape = ([int(weight_variables[-1].get_shape()[0]),data.get_label_dimensions()]), 
                                        name='softmax_weight')
            b_fc2 = bias_variable([data.get_label_dimensions()], name='softmax_bias')
            
            y_out = tf.matmul(tmp_out,W_fc2)+b_fc2
            
                             
                             
            
            ##
            # specify loss
            ##
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_out), 
                                               name = "cross_entropy_loss")
                reg_vars = [tf_var for tf_var in tf.trainable_variables()
                                    if not ('noreg' in tf_var.name.lower() or 'bias' in tf_var.name.lower())]
                reg_term = tf.contrib.layers.apply_regularization(regularizer, 
                                                              reg_vars)
                loss = cross_entropy+reg_term
                tf.summary.scalar('loss' , loss)
            
            ###
            # training step
            ###
            if opt == "adam":
                with tf.name_scope('train'):
                    train_step = tf.train.AdamOptimizer(**opt_params).minimize(loss)
            elif opt == 'sgd':
                with tf.name_scope('train'):
                    train_step = tf.train.GradientDescentOptimizer(**opt_params).minimize(loss)
            elif opt == 'momentum':
                with tf.name_scope('train'):
                    train_step = tf.train.MomentumOptimizer(**opt_params).minimize(loss)
            else:
                raise ValueError('No valid optimizer specified!\nAborting...')
            
            correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            sess.run(tf.global_variables_initializer())
        
            ###
            # measuring performance
            ###
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name = "accuracy")
                tf.summary.scalar('accuracy',accuracy)
                
            saver = tf.train.Saver()
            tf.add_to_collection("y_out",y_out)
            tf.add_to_collection("accuracy",accuracy)
            tf.add_to_collection("loss", cross_entropy) 
                
            # Merge all the summaries and write them out 
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(summary_folder,'train'), sess.graph)
            validation_writer = tf.summary.FileWriter(os.path.join(summary_folder,'val'))
            test_writer = tf.summary.FileWriter(os.path.join(summary_folder,'test'))
            
            ###
            # Training loop
            ###
            best_val_acc = 0
            print()
            for i in range(num_epochs):
                # one epoch consists of epoch_factor many batch training iterations
                for j in range(epoch_factor):
                    batch = next_batch("train", batch_size = batch_size)
                    sess.run([train_step], feed_dict = {x: batch[0], y_: batch[1], 
                             keep_prob : 1-dropout, batch_norm_training:True})
                # determine validation error every summary_intervals
                if i% summary_intervals == (summary_intervals-1):
                    print("=====================")
                    print("Completed Epoch %d/%d"%(i+1,num_epochs))
                    
                    # Compute training performance
                    batch = next_batch('train', batch_size = batch_size,
                                          complete_set = complete_set)
                    summary, acc, _ = sess.run([merged, accuracy, cross_entropy], 
                                               feed_dict = {x:batch[0], y_ : batch[1], 
                                                            keep_prob:1, batch_norm_training:False})
                    train_writer.add_summary(summary,i)
                    print("Training Accuracy : %3.2f"%(acc))
                    
                    #Compute validation performance
                    batch = next_batch("val")
                    summary, acc, _  = sess.run([merged, accuracy, cross_entropy], 
                                                feed_dict = {x:batch[0], y_ : batch[1], 
                                                             keep_prob:1, batch_norm_training:False})
                    validation_writer.add_summary(summary, i)
                    print("Validation Accuracy : %3.2f"%(acc))
                    save_model(sess,saver,acc, best_val_acc, model_path)                
            
            ###
            # Testing trained network
            ###
            # for testing, the best model wrt validation loss is loaded
            print("=====================")
            print("Loading best model for test evaluation")
            load_model(sess,saver, model_path)
            
            test_batch = next_batch("test")
            print("=====================")
            summary, acc, _  = sess.run([merged, accuracy, cross_entropy], 
                                        feed_dict = {x:test_batch[0], y_ : test_batch[1], 
                                                     keep_prob:1, batch_norm_training:False})
            test_writer.add_summary(summary,(num_epochs-1))
            print("Test Accuracy : %3.2f"%(acc))
            train_writer.close()
            validation_writer.close()
            test_writer.close()
    
            ##
            # save weight variables
            ##
            def create_variable_dict(weight_variables):
                dic = {}
                i = 0
                while(i<len(weight_variables)-1):
                    dic['W_'+str(i/2)] = weight_variables[i]
                    dic['b_'+str(i/2)] = weight_variables[i+1]
                    i +=2
                return dic
                     
            dic = create_variable_dict(weight_variables)
            saver_weights = tf.train.Saver(dic)
            saver_weights.save(sess, save_weights_to)
            
    
if __name__=='__main__':
    import argparse
    
    
    ################################
    # Helper functions for parsing #
    ################################
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    def str2dict(d):
        def isInt(i):
            if not '.' in i:
                try:
                    int(i)
                    return True
                except:
                    return False
            return False
        def isfloat(f):
            try:
                float(f)
                return True
            except:
                return False
            
        # get from string to dict
        dic_string = d.lstrip('{').rstrip('}').replace(' ','').split(',')
        d_ = {}
        for entry in dic_string:
            # check if value is a complete Windows path by looking for second ':'
            if len(entry.split(':'))==3:
                key, val1, val2 = entry.split(':')
                val = val1+':'+val2
            else:
                key,val = entry.split(':')
            d_[key] = val
              
        # check for potential necessary conversions
        for key in d_:
            val = d_[key]
            # check for list:
            if '[' in val:
               val = val.replace('[','').replace(']','')
               if isInt(val.split(';')[0]) or len(val.split(';')[0])==0:
                   d_[key] = list(np.fromstring(val,dtype=int,sep=';'))
               elif isfloat(val.split(';')[0]):
                   d_[key] = list(np.fromstring(val,dtype=float,sep=';'))
               else:
                   d_[key] = val.split(';')
            # check for boolean
            elif val.lower() in ['false','true']:
                d_[key] = False if val.lower()=='false' else True
            elif val.lower() == 'none':
                d_[key] = None
            # check for number                
            elif isfloat(val):
                # check if float
                if '.' in val or 'e' in val:
                    d_[key] = float(val)
                else:
                    d_[key] = int(val)
            # otherwise, leave entry as string
            else:
                pass
        return d_
    
    #########################################
    # Definition of some default parameters #
    #########################################
    
    root_model_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.realpath(__file__)))))),'Networks',
                    'pretrainedCNN')
    default_model_path = os.path.join(root_model_path,
                            'model'+str(len(os.listdir(root_model_path))))
    default_summary_folder = os.path.join(default_model_path, 'Summaries')
    
    default_save_weights_to= os.path.join(default_model_path,'pretrained_weights')
    
    default_cnn_params = {'trainable':True,'kernel_sizes':[5,5],
                          'feature_maps':[32,64], 'pooling':[2,2],
                            'dense_dims': [512,256], 'padding_type':'SAME',
                            'batch_normalization':False,
                            'local_response_normalization': True}
    
    ######################
    # Argparse arguments #
    ######################
        
    
    parser = argparse.ArgumentParser(description=
                                     "Parser for arguments of CNN Trainer")
    
    parser.add_argument('-d', metavar = 'dataset', type = str,
                        nargs = 1, default = ['cMNIST'], dest = 'dataset',
                        help = 'The type of dataset to use for training (cMNIST'+
                                ', embMNIST, red_embMNIST)'+
                                '--> Default: cMNIST')
    
    
    parser.add_argument('-opt', metavar = 'optimizer', type = str,
                        nargs = 1, default = ['adam'], dest = 'opt',
                        help = 'The type of optimizer to use for training'+
                                ' --> Default: Adam')
    
    parser.add_argument('-opt_params', metavar = 'optimizer parameters', type = str2dict,
                        nargs = 1, default = [{}], dest= 'opt_params', 
                        help = 'The parameters for the optimizer. Hand over as tuple'+
                                ' in the order of the optimizers constructor'+
                                ' --> Default: {}')
    
    parser.add_argument('-num_epochs', metavar= 'number of epochs', type = int,
                        nargs = 1, default = [2], dest = 'num_epochs',
                        help = 'The number of epochs that model shall be trained.'+
                                ' Note that these are real epochs, so independent of the'+
                                ' chosen batch size'+
                                ' --> Default: 2')
    
    parser.add_argument('-batch_size', metavar = 'batch size', type = int,
                        nargs = 1, default = [64], dest = 'batch_size',
                        help = 'Batch size for gradient updates'+
                                ' --> Default: 64')
    
    parser.add_argument('-model_folder', metavar = 'model folder', type = str,
                        nargs = 1, default = [default_model_path], dest = 'model_folder',
                        help = 'The path to the folder, where model will be saved'+
                                ' --> Default: '+default_model_path)
    
    parser.add_argument('-save_weights_to', metavar = 'save_weights_to', type = str,
                        nargs = 1, default = [default_save_weights_to], dest = 'save_weights_to',
                        help = 'Filename where trained variables are saved for later usage'+
                                ' --> Default: '+default_save_weights_to)
    
    parser.add_argument('-summary_folder', metavar = 'summary folder', type = str,
                        nargs = 1, default = [default_summary_folder], dest = 'summary_folder',
                        help = 'The path to the folder, where the summaries of '+
                                'the model will be saved'+
                                ' --> Default: '+default_summary_folder)
    
    parser.add_argument('-summary_intervals', metavar='summary intervals', type = int,
                        nargs = 1, default = [1], dest = 'summary_intervals',
                        help = 'Number of epochs, after which the validation error ' +
                                'shall be calculated and summaries be updated'+
                                ' --> Default: 1')
    
    parser.add_argument('-complete_set', metavar = 'Complete Set', type = str2bool,
                        nargs = 1, default = [True], dest = 'complete_set', 
                        help = 'Boolean deciding if whole training set is used for \
                        computing training summary'+
                                ' --> Default: True')
    
    parser.add_argument('-dropout', metavar = 'Dropout', type = float,
                        nargs = 1, default = [0.5], dest = 'dropout', 
                        help = 'Dropout rate for dense layers'+
                                ' --> Default: %f'%(0.5))
    
    parser.add_argument('-l2_reg', metavar = 'L2-Regularization', type = float,
                        nargs = 1, default = [1e-5], dest = 'l2_reg', 
                        help = 'L2 regularization for network weights'+
                                ' --> Default: %f'%(1e-5))
    
    
    parser.add_argument('-cnn_params', metavar = 'CNN parameters', type = str2dict,
                        nargs = 1, default = [default_cnn_params], dest = 'cnn_params',
                        help = 'Model parameters necessary to configure specified '+
                                'CNN'+
                                ' --> Default: '+ str(default_cnn_params))

    args = parser.parse_args()
    
    ###############################################################
    # fill up CNN parameters with default arguments where neeeded #
    ###############################################################
    for k in default_cnn_params:
        if k not in args.cnn_params[0]:
            args.cnn_params[0][k] = default_cnn_params[k]
    
    ###################### 
    # document arguments #
    ######################
    if not os.path.exists(args.model_folder[0]):
        os.makedirs(args.model_folder[0])
    with open(os.path.join(args.model_folder[0],'info.txt'),'w') as f:
        f.write('Num_epochs: %i' %(args.num_epochs[0]))
        f.write('\nBatch_size: %i' %(args.batch_size[0]))
        f.write('\nDropout: %f' %(args.dropout[0]))
        f.write('\nL2-Reg: %f' %(args.l2_reg[0]))
        f.write('\nOptimizer: %s' %(args.opt[0]))
        f.write('\nOptimizer Parameters: %s' %(str(args.opt_params[0])))
        f.write('\nCNN Parameters: %s' %(str(args.cnn_params[0])))
        f.write('\nData: %s' %(args.dataset[0]))
        
    data = DataLoader(args.dataset[0]).load()
    
    pretrainCNN(data, args.num_epochs[0], args.batch_size[0],args.opt[0],
                args.opt_params[0], args.summary_intervals[0], 
                args.complete_set[0], args.save_weights_to[0],
                args.summary_folder[0], args.model_folder[0], 
                args.cnn_params[0], args.dropout[0], args.l2_reg[0])
    
    
        








