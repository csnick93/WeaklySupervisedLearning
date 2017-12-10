# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:28:34 2017

@author: nickv
"""

"""
Script to create embedded MNIST
"""

import os
import tarfile
import gzip
import numpy as np
from PIL import Image

def mergeImageNetMNIST(imageNet_source, mnist_source, saveTo, color = False, num_images = 10000): #TODO:
    """
    Embed the MNIST images into the ImageNet images
        - go through all MNIST images
        - for each MNIST image, find random ImageNET image
        - randomly insert mnist number into imageNet image
    """
    
    def getMinShape(imageNet):
        """
        find out smallest dimensions of image net images
        """
        min_height = min(imageNet, key = lambda x:x.shape[0]).shape[0]
        min_width = min(imageNet,key = lambda x:x.shape[1]).shape[1]
        c = 3 if color else 1
        
        return min_height,min_width, c
    
    def unifyImageNetDimensions(imageNet):
        """
        Crop all imageNet images to minimum dimensions to 
        get uniform dimensions for all images
        """
        min_h, min_w, c = getMinShape(imageNet)
        imageNet_uniform = np.zeros((imageNet.shape[0],min_h,min_w,c))
        
        for i,img in enumerate(imageNet):
            imageNet_uniform[i] = img[:min_h,:min_w]
        
        return imageNet_uniform
    
    def mergeInstance(image_net_img, mnist_img):
        """
        #Embed a mnist image into a imageNet image
        #    - randomly choose coordinate for upper left corner of 
        #        mnist image
        """
        height,width,channels = image_net_img.shape
        h,w,c = mnist_img.shape
        x = np.random.randint(0,width-w)
        y = np.random.randint(0,height-h)
        
        def embed(background, img,x,y, intensity):
            """
            #Embed image into background at coordinate x,y
            """
            for c, col in enumerate(img):
                for r, val in enumerate(col):
                    if color:
                        if np.any(val>0):
                            background[y+c][x+r] = np.array([intensity,intensity,intensity])
                    else:
                        if val > 0:
                            background[y+c][x+r] = intensity
            return background
        
        def embed_segMask(background, img,x,y, intensity):
            """
            #Embed image into background at coordinate x,y
            """
            for c, col in enumerate(img):
                for r, val in enumerate(col):
                    if val > 0:
                        background[y+c][x+r] = intensity
            return background
        
        def normalizeImg(img):
            """
            Normalize image to range [0,1]
            """
            img -= np.min(img)
            img *= (1./np.max(img))
            
            return img
        
        embeddedImg = embed(image_net_img, mnist_img,x,y, np.mean(image_net_img)+2*np.std(image_net_img))
        segmentationMap = embed_segMask(np.zeros((image_net_img.shape[0],image_net_img.shape[1])), mnist_img,x,y,1)
        
        embeddedImg = normalizeImg(embeddedImg)
        
        # reshape to obtain additional color channel dimension
        if not color:
            embeddedImg = np.reshape(embeddedImg,(embeddedImg.shape[0],embeddedImg.shape[1],1))
        
        return embeddedImg, segmentationMap
    
    ###########################################################################
        
    print('Getting MNIST and ImageNet data')
    img_net = getImageNet(imageNet_source,color, num_images)    # TODO:  
    image_net = unifyImageNetDimensions(img_net)    
    (X_train,y_train,X_val,y_val, X_test,y_test) = getMNIST(mnist_source, num_images)
    

    print('Starting to merge the datasets')

    X = np.concatenate((X_train,np.concatenate((X_val,X_test),axis=0)),axis=0)
    if color:
        X_merged = np.zeros(shape=(num_images,image_net[0].shape[0],image_net[0].shape[1],3))  #TODO:
        #X_merged = np.zeros(shape=(X.shape[0],image_net[0].shape[0],image_net[0].shape[1],3))     
    else:
        X_merged = np.zeros(shape=(num_images,image_net[0].shape[0],image_net[0].shape[1],1))  #TODO:  
        #X_merged = np.zeros(shape=(X.shape[0],image_net[0].shape[0],image_net[0].shape[1],1))     
    y_segMask = np.zeros((X_merged.shape[0],X_merged.shape[1],X_merged.shape[2]))

    
    for i,x in enumerate(X): 
        if i == num_images:  #TODO:
            break
        index = np.random.randint(image_net.shape[0])
        # make sure cropped image is not single intensity
        while (np.max(image_net[index])<0.001 or np.max(image_net[index])-np.min(image_net[index])<0.001):
            index = np.random.randint(image_net.shape[0])
        embeddedImg, segMap = mergeInstance(np.copy(image_net[index]),X[i])
        X_merged[i] = embeddedImg
        y_segMask[i] = segMap
                 
                 
    
    print('Saving to file')
    np.savez(saveTo,X_train=X_merged[0:X_train.shape[0]], y_train = y_train, y_train_seg = y_segMask[0:X_train.shape[0]],
            X_valid = X_merged[X_train.shape[0]:X_train.shape[0]+X_val.shape[0]], y_valid = y_val, y_valid_seg = y_segMask[X_train.shape[0]:X_train.shape[0]+X_val.shape[0]],
            X_test = X_merged[-X_test.shape[0]:], y_test = y_test, y_test_seg = y_segMask[-X_test.shape[0]:])
    print('DONE!!!')
    


def getImageNet(sourceFile,color, num_images): #TODO: 
    """
    extract ImageNet images into numpy arrays
    """
    imgs = []
    with tarfile.open(sourceFile) as tf:
        for e,entry in enumerate(tf):
            if e == num_images: #TODO:
                break
            fileobj = tf.extractfile(entry)
            if not color:
                im = Image.open(fileobj).convert('L')
                (width, height) = im.size
                greyscale_map = list(im.getdata())
                greyscale_map = np.array(greyscale_map)
                # normalize to [0,1] from [0,255]
                greyscale_map = greyscale_map/255
                img = greyscale_map.reshape((height, width,1))
            else:
                im = Image.open(fileobj)
                img = np.array(im)
                if (img.ndim == 2):
                    continue
                # normalize to [0,1] from [0,255]
                img = img / 255
            imgs.append(img)
    return np.array(imgs)


def getMNIST(sourceFile,num_images):
    """
    get mnist data
    """
    mnist = np.load(sourceFile)
    X_train = mnist['X_train'][:int(num_images*0.8)]
    y_train = mnist['y_train'][:int(num_images*0.8)]
    X_valid = mnist['X_valid'][:int(num_images*0.1)]
    y_valid = mnist['y_valid'][:int(num_images*0.1)]
    X_test = mnist['X_test'][:int(num_images*0.1)]
    y_test = mnist['y_test'][:int(num_images*0.1)]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
if __name__=="__main__":
    root = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                               'Datasets'))
    mnist_source = os.path.join(root,'MNIST','mnist.npz')
    imgnet_source = os.path.join(root,'EmbeddedMNIST','imageNet.tar')
    targetPath = os.path.join(root,'EmbeddedMNIST','embeddedMNIST.npz')
    mergeImageNetMNIST(imgnet_source,mnist_source,targetPath) 
    
    """
    color = False
    num_images = 10
    X, y = mergeImageNetMNIST(imgnet_source,mnist_source,targetPath, color = color, num_images = num_images)
    
    
    import matplotlib.pyplot as plt
    for i in range(num_images):
        if color:
            plt.imshow(X[i])
            plt.show()
            plt.close()
            plt.imshow(y[i])
            plt.show()
            plt.close()
        else:
            plt.imshow(X[i,:,:,0])
            plt.show()
            plt.close()
            plt.imshow(y[i])
            plt.show()
            plt.close()
     
    
    print(X.shape)
    print(y.shape)
    """