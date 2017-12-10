# WeaklySupervisedLearning

This is a small tensorflow based framework intended for playing around with different weakly supervised deep learning algorithms. Given images and class level labels, the algorithms attempt to not only classify the object(s) in the image correctly, but also to localize them. The framework has the following structure:

![Screenshot](class_hierarchy.png)

## Components

### Manager
Class that is responsible to connect all the subcomponents: loading data, creating model, training model and evaluating the trained model.

### DataLoader
Class that is responsible for loading the .npz data and feeding it back to the manager.

### GraphBuilder
Class that is responsible for creating the tensorflow model, by initiliazing the model class with the specified parameters.

#### HelperCNN
Small CNN helper class used for creating convolutional networks needed within the weakly supervised models.

### Network
Class that is responsible for doing the training and creating the tensorflow summary file.

### Evaluation
Class that loads trained model and evaluates the perfomance on the dataset (e.g. accuracy, IoU), and also saves out visualization of the detection performance in the specified way (e.g. bounding box, circle,..).

### Configuration
Configuration of the training process can be done in the config.xml.

## Implemented Networks

Currently there are two implemented networks.

### Deep Recurrent Attentive Detector (DRAD)

DRAD was inspired by the following paper: https://arxiv.org/abs/1502.04623. It is a recurrent network with an integrated attention mechanism.


### Self Transfer

The self transfer model is an implemented tensorflow version of the following paper: https://arxiv.org/abs/1602.01625. Its basic idea is to train a CNN that in the final layers forks into a classification output layer and a localization output layer. Both layers have a classification loss function, which is linearly combined. By tracing back the highest activations from the localization input, it is possible to visualize the area in the image that was a major cause for the classification decision. Thus, it is possible to obtain the object location just by using class labels. 


## Available Datasets

For quick testing of the implemented models, two datasets were used: cluttered MNIST and embedded MNIST. However, due space restrictions on github, I only uploaded a very small fraction of them. If needed, I will be happy to make them fully available to you (just send me an email for that).

### Cluttered MNIST

This is a very simplistic toy example serving as a sanity check. It consists of 40x40x3 MNIST images that contain some minor clutter. The goal is to correctly classify the number present in the image and to localize the number.

### Embedded MNIST

This dataset was created using images from ImageNet and pasting MNIST digits into the images. The digit's intensity value was adapted according to the mean and standard deviation of the respective ImageNet image that it was pasted in. The image dimensions are 59x100x3.

## Results

In the Results folder, some results of each model are presented. If desired, I can also provide the trained models files (which were also not uploaded due to space requirements).
