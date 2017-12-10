# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:50:27 2017

@author: nickv
"""
import numpy as np

def backtrack_locations(pooling_indices, conv_tensors, weight_tensors, bias_tensors, class_index,model_params):
    """
    Method that backtracks the highest activation back to location
    in input image for the self transfer model for guessed class.
    
    Approach:
        - use argmax indices to backtrack activation through pooling layers
        - use convolutional weights/bias to compute feature map that contributes
            the most$
    
    Args:
        pooling_indices: list of the index tensors output by max_pool_with_argmax
                            shape: [num_poolings,pooling_output_shape]
        conv_tensors: the data tensors being convoluted
        weight_tensors, bias_tensors : trained weight/bias tensors of the 
                                convolutional layers necessary to do backtracking
        class_index: index of the class in location output tensor with highest activation
                        --> only return location of object with highest activation
                        (int)
        model_params: dictionary containing the necessary informations about
                        the self transfer model to compute the locations
                        in input model (dic)
    
    Returns:
        2D index tuple (y,x) signifying the coordinates in the image where
        highest activation for classification.
    """
    def convert_flattened_index(index,height,width,channels):
        """
        Converts flattened index into (y,x,c) index
        
        Args:
            index: flattened index
            batch_index: index of sample in processed batch
            height: height of samples
            width: width of samples
            channels: number of channels in samples
            
        Returns:
            (height,width,channel) index
        """
        c = index % channels
        x = int((index-c)/channels) % width
        y = int((int((index-c)/channels)-x)/width)%height
               
        return np.array([y,x,c])
    
    ###
    # reconstruct shapes of tensors going through network
    ###
    shapes = [np.array(model_params['img_size'])]
    # go through joint cnn
    for i in range(len(model_params['kernel_sizes'])):
        #account for padding type
        kernel_size = model_params['kernel_sizes'][i]
        pooling = model_params['pooling'][i]
        features = model_params['feature_maps'][i]
        if model_params['padding_type'] == 'VALID':
            adj = kernel_size-1
            shape = shapes[-1]-np.array([adj,adj,0])
        else:
            shape = shapes[-1]
        shape[-1] = features
        shapes.append(shape)
        # account for pooling and feature maps/channels
        if pooling > 1:
            shape = np.array([int(shape[0]/pooling), int(shape[1]/pooling), features])
        # append to list
        shapes.append(shape)
    # account for localization convolution
    kernel_size = model_params['kernel_size_loc']
    if model_params['padding_type'] == 'VALID':
        adj = kernel_size-1
        shape = shapes[-1]-np.array([adj,adj,0])
    else:
        shape = shapes[-1]
    shape[-1] = model_params['label_size']
    shapes.append(shape)
    
    # shapes = [original_shape, shape_after_conv,*shape_after_pooling,
    #           shape_after_second_conv, *shape_after_pooling,...,
    #           shape_before_localization_conv, shape_after_localization_conv]
    # *: only present if pooling happened
    
    ###
    # backtrack global max pooling
    ###
    backtrack_counter = -1
    pooling_counter = -1
    conv_counter = -1
    # go from [1,1,K] global pooling layer one step back 
    global_pooling_index = pooling_indices[pooling_counter][0,0,class_index]
    previous_index = convert_flattened_index(global_pooling_index.flatten()[0],
                                             *shapes[backtrack_counter])
    backtrack_counter -= 1
    pooling_counter -=1
    ###
    # backtrack location network convolution
    ###
    # adjust index if padding was done in convolution
    if model_params['padding_type'] =='VALID':
        adj = int(model_params['kernel_size_loc']-1)/2
        previous_index += (adj,adj,0)
        backtrack_counter-=1
    else:
        adj = 0
        
    #backtrack convolution
    weight_tensor = weight_tensors[conv_counter]
    conv_tensor = conv_tensors[conv_counter]
    channel_of_interest = previous_index[-1]
    feature_map = np.argmax([np.sum(np.multiply(conv_tensor[previous_index[0]-adj:previous_index[0]+adj+1,
                                                            previous_index[1]-adj:previous_index[1]+adj+1,i],
                                                            weight_tensor[:,:,i,channel_of_interest]))
                                for i in range(weight_tensor.shape[-2])])
    previous_index[-1] = feature_map
    conv_counter -=1
    
    ##
    # backtrack through joint convolutions
    ##
    for i in range(len(model_params['kernel_sizes'])):        
        ##
        # check if pooling was done
        if model_params['pooling'][pooling_counter+1] > 1:
            previous_index = convert_flattened_index(pooling_indices[pooling_counter][previous_index[0],previous_index[1],previous_index[2]].flatten()[0],
                                                    *shapes[backtrack_counter])
            backtrack_counter -=1
            pooling_counter -=1
        ##
        # account for padding if necessary
        if model_params['padding_type'] =='VALID':
            adj = int(model_params['kernel_sizes'][-i]-1)/2
            previous_index += (adj,adj,0)
            backtrack_counter-=1
        else:
            adj = 0
        ##
        # backtrack convolutional layer
        # find feature map in previous layer contributing the most to 
        # the currently found highest activation index
        #   - which feature map most responsible for activation at previous_index
        weight_tensor = weight_tensors[conv_counter]
        conv_tensor = conv_tensors[conv_counter]
        channel_of_interest = previous_index[-1]
        
        feature_map = np.argmax([np.sum(np.multiply(conv_tensor[previous_index[0]-adj:previous_index[0]+adj+1,
                                                                previous_index[1]-adj:previous_index[1]+adj+1,i],
                                                                weight_tensor[:,:,i,channel_of_interest]))
                                    for i in range(weight_tensor.shape[-2])])
        previous_index[-1] = feature_map
        conv_counter -=1
        
            
    # finally return the [y,x] part of the [y,x,c] found index
    return (previous_index[0],previous_index[1])
    
    