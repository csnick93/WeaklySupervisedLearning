# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 19:06:58 2017

@author: nickv
"""
from PIL import Image
import os
import numpy as np

def prepare_graz(path, colorized = True):
    
    def get_imgs(p, height, width, num_images):
        """
        p: path to images
        width, height: dimensions of images
        num_images: the number of images (not segmentations masks in folder)
        """
        if colorized:
            imgs = np.zeros((num_images,height,width,3))
        else:
            imgs = np.zeros((num_images,height,width,1))
        seg_masks = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2]))
        c = 0        
        for i,img in enumerate(os.listdir(p)):
            # check if img is actually image
            if not 'image' in img:
                continue
            # read in image
            if colorized:
                im = np.array(Image.open(os.path.join(p,img)),dtype=np.float32)
            else:
                im = np.array(Image.open(os.path.join(p,img)).convert('L'),dtype=np.float32)
                # add artificial color channel
                im = np.reshape(im,(im.shape[0],im.shape[1],1))
            # transpose image if necessary
            color_channels = 3 if colorized else 1
            im = np.transpose(im,axes=[1,0,2]) if im.shape==(width,height,color_channels) else im
            # normalize image to [0,1]
            im = im-np.min(im)
            im = im/np.max(im)
            #
            imgs[c] = im
            # get seg mask to image
            seg_masks[c] = get_segMask(p,img.rstrip('image.png'),height,width)
            c+=1
        return imgs, seg_masks
    
    def get_segMask(root, img,height,width):
        # get segmasks (potentially multiple)
        mask_paths = []
        mask_name = img+'.mask'
        for f in os.listdir(root):
            if mask_name in f:
                mask_paths.append(os.path.join(root,f))
        # read in masks
        masks = []
        for m_p in mask_paths:
            mask = np.array(Image.open(m_p))
            mask = np.transpose(mask) if mask.shape==(width,height) else mask
            masks.append(mask)

        # convert to single 0,1 binary mask
        mask = np.zeros((height,width))
        for m in masks:
            indices = np.where(m>0)
            mask[indices]=1
        return mask
        
        
    
    def get_bike_imgs():
        return get_imgs(os.path.join(path,'bikes'),
                        480,640,369)
    def get_car_imgs():
        return get_imgs(os.path.join(path,'cars'),
                        480,640,424)
    def get_people():
        return get_imgs(os.path.join(path,'people'),
                        480,640,316)
        
    
    bikes, bikes_gt = get_bike_imgs()
    cars, cars_gt = get_car_imgs()
    person, person_gt = get_people()
    
    ##
    # separate into train, val, and test set 
    ##
    def getLabels(num_entries,value,num_labels):
        a = np.empty((num_entries,num_labels))
        a[:,value] = 1
        return a
    
    bike_train, bike_gt_train, bike_val, bike_gt_val, bike_test, bike_gt_test = \
        bikes[:int(0.8*len(bikes))], bikes_gt[:int(0.8*len(bikes))],\
        bikes[int(0.8*len(bikes)):int(0.9*len(bikes))], bikes_gt[int(0.8*len(bikes)):int(0.9*len(bikes))],\
        bikes[int(0.9*len(bikes)):], bikes_gt[int(0.9*len(bikes)):]
    bike_train_label, bike_val_label, bike_test_label = \
        getLabels(bike_train.shape[0],0,3),getLabels(bike_val.shape[0],0,3),getLabels(bike_test.shape[0],0,3)
        
    cars_train, cars_gt_train, cars_val, cars_gt_val, cars_test, cars_gt_test = \
        cars[:int(0.8*len(cars))], cars_gt[:int(0.8*len(cars))],\
        cars[int(0.8*len(cars)):int(0.9*len(cars))], cars_gt[int(0.8*len(cars)):int(0.9*len(cars))],\
        cars[int(0.9*len(cars)):], cars_gt[int(0.9*len(cars)):]
    cars_train_label, cars_val_label, cars_test_label = \
        getLabels(cars_train.shape[0],1,3),getLabels(cars_val.shape[0],1,3),getLabels(cars_test.shape[0],1,3)
        
    person_train, person_gt_train, person_val, person_gt_val, person_test, person_gt_test = \
        person[:int(0.8*len(person))], person_gt[:int(0.8*len(person))],\
        person[int(0.8*len(person)):int(0.9*len(person))], person_gt[int(0.8*len(person)):int(0.9*len(person))],\
        person[int(0.9*len(person)):], person_gt[int(0.9*len(person)):]
    person_train_label, person_val_label, person_test_label = \
        getLabels(person_train.shape[0],2,3),getLabels(person_val.shape[0],2,3),getLabels(person_test.shape[0],2,3)
        
    
    X_train = np.concatenate((bike_train,cars_train,person_train))
    y_train = np.concatenate((bike_train_label, cars_train_label, person_train_label))
    y_train_seg = np.concatenate((bike_gt_train,cars_gt_train, person_gt_train))
    
    X_val = np.concatenate((bike_val,cars_val,person_val))
    y_val = np.concatenate((bike_val_label, cars_val_label, person_val_label))
    y_val_seg = np.concatenate((bike_gt_val,cars_gt_val, person_gt_val))
    
    X_test = np.concatenate((bike_test,cars_test,person_test))
    y_test = np.concatenate((bike_test_label, cars_test_label, person_test_label))
    y_test_seg = np.concatenate((bike_gt_test,cars_gt_test, person_gt_test))
     
    
    print('Saving to file')
    save_to_file = os.path.join(path,'graz_color.npz') if colorized else os.path.join(path,'graz.npz')
    np.savez(save_to_file, X_train = X_train, y_train = y_train, y_train_seg = y_train_seg,
             X_valid = X_val, y_valid = y_val, y_valid_seg=y_val_seg, 
             X_test = X_test, y_test = y_test, y_test_seg = y_test_seg)
    print('Done')
    
    
if __name__=='__main__':
    colorized = False
    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Datasets','GRAZ')
    prepare_graz(path,colorized)
    
    
    
    
    
    
    