# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:00:00 2017

@author: nickv
"""

import os
import matplotlib.image as mpimg
import numpy as np

DIC = {'cow': np.array([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'bus': np.array([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'diningtabl': np.array([ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'bird': np.array([ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'boat': np.array([ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'head': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0 ],dtype=np.float32), 'pottedplant': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'cat': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0 ],dtype=np.float32), 'chair': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0 ],dtype=np.float32), 'tvmonitor': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'sheep': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'bottl': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'car': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'otorbik': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'trai': np.array([ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'hors': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0 ],dtype=np.float32), 'ropl': np.array([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'dog': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0 ],dtype=np.float32), 'bicycl': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'sof': np.array([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ],dtype=np.float32), 'perso': np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1 ],dtype=np.float32)}

def get_voc_image(path, index):
    """
    Get Pascal VOC images from folder
        limit: only get limit many samples
    """
    img_path = os.listdir(path)[index]
    print(img_path)
    im = mpimg.imread(os.path.join(path,img_path))
    im = im.astype(np.float32)
    im -= np.min(im)
    im = im/np.max(im)
    im_shape = im.shape
    # do zero padding to get 500x500 image
    def pad_img(im):
        h,w,_ = im.shape
        padding = [[int((500-h)/2),500-h-int((500-h)/2)],
                   [int((500-w)/2),500-w-int((500-w)/2)],
                    [0,0]]
        return np.pad(im,padding,mode='constant',constant_values=0)
    im = pad_img(im)
    return im,im_shape

def get_voc_label(path,index,img_shape):
    """
    Read in labels of images. Take label that takes up the biggest area in image.
    """
    def mask_img(coordinates,img_shape):
        """
        Create image mask for bounding box coordinates
        """
        def pad_mask(mask):
            # pad according to image
            mask = np.pad(mask,[[coordinates[2],img_shape[0]-coordinates[3]],
                                [coordinates[0],img_shape[1]-coordinates[1]]],
                            mode = 'constant', constant_values = False)
            # pad to 500x500
            h = img_shape[0]
            w = img_shape[1]
            mask = np.pad(mask,[[int((500-h)/2),500-h-int((500-h)/2)],
                                [int((500-w)/2),500-w-int((500-w)/2)]],
                                mode = 'constant', constant_values = False) 
            return mask
        mask = np.empty((coordinates[3]-coordinates[2],coordinates[1]-coordinates[0]),dtype=bool)
        mask.fill(True)
        mask = pad_mask(mask)
        return mask
    
    def adjusted_coordinates(xmin,xmax,ymin,ymax, img_shape):
        h = img_shape[0]
        w = img_shape[1]
        xmin += int((500-w)/2)
        xmax += 500-w-int((500-w)/2)
        ymin += int((500-h)/2)
        ymax += 500-h-int((500-h)/2)
        return np.array([xmin,xmax,ymin,ymax])
    
    def object_w_biggest_area(f,img_shape):
        """
        Get label of object with biggest area in image.
        """
        objects = []
        coordinates = []
        areas = []
        lines = f.replace('\t','').split('\n')
        for i, l in enumerate(lines):
            if '<name>' in l:
                objects.append(lines[i].lstrip('<name>').rstrip('</name>'))
                coordinates_flag = [False,False,False,False]
                for l_ in lines[i:]:
                    if '<xmin>' in l_:
                        xmin = int(float((l_.strip('<xmin>').rstrip('</xmin>'))))
                        coordinates_flag[0] = True
                    elif '<xmax>' in l_:
                        xmax = int(float(l_.strip('<xmax>').rstrip('</xmax>')))
                        coordinates_flag[1] = True
                    elif '<ymin>' in l_:
                        ymin = int(float(l_.strip('<ymin>').rstrip('</ymin>')))
                        coordinates_flag[2] = True
                    elif '<ymax>' in l_:
                        ymax = int(float(l_.strip('<ymax>').rstrip('</ymax>')))
                        coordinates_flag[3] = True
                    if np.all(coordinates_flag):
                        break
                coordinates.append(adjusted_coordinates(xmin,xmax,ymin,ymax, img_shape))
                area = (xmax-xmin)*(ymax-ymin)
                areas.append(area)
        biggest_object = np.argmax(areas)
        return objects[biggest_object], coordinates[biggest_object]
                
    
    
    annotation = os.listdir(label_path)[index]
            
    with open(os.path.join(path,annotation),'r') as f:
        txt = f.read()
        label, bb_coord = object_w_biggest_area(txt, img_shape)
            
    return label, bb_coord


def create_dataset(dest_path,img_path, label_path, limit_range,split_factor = .8):
    imgs = []
    labels = []
    bb_coords = []
      
    for i in limit_range:
        print(i)
        img, im_shape = get_voc_image(img_path,i)
        label, bb_coord = get_voc_label(label_path,i,im_shape)
        imgs.append(img)
        labels.append(label)
        bb_coords.append(bb_coord)
    imgs = np.array(imgs, dtype=np.int8)
    bb_coords = np.array(bb_coords)
    # turn labels into 0-1 vector
    labels = [DIC[l] for l in labels] 
    labels=np.array(labels, dtype=np.int8)
    
    size = imgs.shape[0]
    X_train = imgs[:int(size*split_factor)]
    y_train = labels[:int(size*split_factor)]
    y_train_bb = bb_coords[:int(size*split_factor)]
    
    val_factor = split_factor + (1-split_factor)/2
    X_val = imgs[int(size*split_factor):int(size*val_factor)]
    y_val = labels[int(size*split_factor):int(size*val_factor)]
    y_val_bb = bb_coords[int(size*split_factor):int(size*val_factor)]
    
    X_test = imgs[int(size*val_factor):]
    y_test = labels[int(size*val_factor):]
    y_test_bb = bb_coords[int(size*val_factor):]
    
    print('Saving to file')
    np.savez(dest_path, X_train = X_train, y_train = y_train, y_train_bb = y_train_bb,
             X_valid = X_val, y_valid = y_val, y_valid_bb = y_val_bb,
             X_test = X_test, y_test = y_test, y_test_bb = y_test_bb)
    
    #return imgs, labels, bb_coords

if __name__ == '__main__':
    def findValue(dic,v):
        for k in dic:
            if v==dic[k]:
                return k
        return 'Nothing'
    
   
    limit_range = range(12000,12991)
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.getcwd()))),'VOC2012','JPEGImages')
    label_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.getcwd()))),'VOC2012','Annotations')
    dest_path = os.path.join(os.path.dirname(os.path.dirname(
            os.getcwd())),'Datasets','VOC')
    dest_path = os.path.join(dest_path,'voc_'+str(len(os.listdir(dest_path))+1))
    
    create_dataset(dest_path,img_path,label_path,limit_range)    

    
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    for i in range(len(imgs)):
        fig,ax = plt.subplots(1)
        ax.imshow(imgs[i])
        coords = (bb_coords[i][0], bb_coords[i][1],bb_coords[i][2], bb_coords[i][3])
        print(coords)
        rect_true = patches.Rectangle((coords[0],coords[2]),coords[1] - coords[0],
                                         coords[3] - coords[2], linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect_true)
        print(labels[i])
    """
    
    
    