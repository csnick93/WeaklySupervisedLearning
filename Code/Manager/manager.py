# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:48:44 2017

@author: nickv
"""

"""
Class Manager

This class is the overall manager that the user interacts with to do the
whole training process:
    - The manager will tell the Dataloader, what data to load
    - hand the data over to the Network Manager
    - tell Network Manager what network to load (location of the .meta file)
    - lastly tell the Evaluation class how to do the evaluation.
"""

import os


from sys import path
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                                            , "DataLoader"))
from dataloader import DataLoader
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                                            , "NetworkManager"))
from network_manager import Network
path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                                            , "Networks"))
from graph_builder import GraphBuilder
path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                                            , "Evaluation"))
from evaluation import Evaluation
from load_config import load_configuration


class Manager:
    """
    Class handling the whole pipeline from dataloader->network manager -> evaluation.
    The class that needs to be run in order to start the pipeline.
    """
    
    def __init__(self, dataset, do_training, opt, opt_params,num_epochs, keep_prob,
                 l2_reg, clip_gradient, clip_value,batch_size, 
                 summary_intervals, complete_set, model_name, model_params, tensorboard,
                 do_eval,eval_params):
        """
        Args:
            dataset: The dataset to be loaded 
                        - either a default dataset: cMNIST,...
                        - or a local dataset: provide path
            
            opt: type of optimizer {adam, adagrad,...}
            opt_params: - parameters for optimizer saved as a tuple
                        - the parameters are inserted in order of constructor
                        - e.g. for Adam: (learning_rate, beta1,beta2,epsilon, use_locking, name)
            do_training: - flag that indicates whether training should be done
                         - this way, it is possible to only do evaluation
            num_epochs : the number of epochs for training
            keep_prob: dropout rate for training
            l2_reg: l2 -regularization rate for network weights
            clip_gradient: flag indicating whether gradient should be clipped to prevent exploding gradient
            clip_value: threshold value where gradient clipping should happen
            batch_size: batch size used for training
            summary_intervals: number of epochs after which validation error
                               shall be calculated and summaries be updated 
            complete_set: flag deciding if whole training set is used for computing training accuracy
            model_name: name of model (e.g. drad), in case model needs to be created
            model_params: model parameters for tensorflow graph in case the
                            desired graph that shall be trained does not exist
                            yet and needs to be created
            tensorboard: - flag to indicate whether tensorboard shall be openend
                            when training is finished
            do_eval: - boolean indicating whether evaluation shall be done or not
            eval_params: parameters necessary for evaluation
        """
        ###############
        # Data loader #
        ###############
        self.dataset = dataset
        
        ###########
        # Network #
        ###########
        self.do_training = do_training
        # create folder model_folder/model_i 
        self.model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                            'Networks','Tmp_ModelFolder') 
        if not os.path.exists(self.model_folder):
            self.model_folder = os.path.join(self.model_folder,'model1')
            os.makedirs(self.model_folder)
        elif os.path.exists(os.path.join(self.model_folder,'model.meta')):
            pass
        else:
            self.model_folder = os.path.join(self.model_folder,'model'+
                                           str(len(os.listdir(self.model_folder))+1))
            os.mkdir(self.model_folder)
            
        self.opt = opt
        self.opt_params = opt_params
        self.num_epochs = num_epochs
        self.keep_prob = keep_prob
        self.l2_reg = l2_reg
        self.clip_gradient = clip_gradient
        self.clip_value = clip_value
        self.batch_size = batch_size
        
        # create summary folder
        self.summary_folder = os.path.join(self.model_folder,'Summaries') 
        if not os.path.exists(self.summary_folder):
            self.summary_folder = os.path.join(self.summary_folder,'run1')
            os.makedirs(self.summary_folder)
        else:
            self.summary_folder = os.path.join(self.summary_folder,'run'+
                                     str(len(os.listdir(self.summary_folder))+1))
            os.makedirs(self.summary_folder)
        self.summary_intervals = summary_intervals
        self.complete_set = complete_set
        self.model_name = model_name
        self.model_params = model_params
        if not 'save_model_to' in self.model_params:
            self.model_params['save_model_to'] = self.model_folder
        self.tensorboard = tensorboard
        
        #############
        # Evaluator #
        #############
        self.do_eval = do_eval
        self.eval_params = eval_params
        
        ##
        # save training info to info file
        ##
        with open(os.path.join(self.summary_folder,'training_info.txt'),'w') as info:
            info.write("Training configuration:\n")
            for k in self.__dict__.keys():
                info.write('\t'+k +": "+ str(self.__dict__[k])+"\n")
        

###############################################################################

    def main(self):
        """
        Method executing the whole pipeline.
        """        
        ##
        # get data
        ##
        if (self.dataset in DataLoader.default_datasets or
            os.path.exists(self.dataset)):
            dataLoader = DataLoader(DataLoader.default_datasets[self.dataset])
            data = dataLoader.load()
            self.model_params['img_size'] = data.get_dimensions()
            self.model_params['label_size'] = data.get_label_dimensions()
        else:
            print("Dataset "+self.dataset + " does not exist. Aborting...")
            return -1
        
        ###
        # Potential Graph creation
        ###
        if not os.path.exists(os.path.join(self.model_folder,'model.meta')):
            builder = GraphBuilder()
            builder.build_graph(self.model_name, self.model_params)
    
        ###
        # Network training
        ###
        if self.do_training:
            network = Network(self.model_name,
                              self.model_folder, self.opt, self.opt_params,
                              self.num_epochs, self.batch_size, data, 
                              self.summary_folder, self.summary_intervals,
                              self.complete_set, self.keep_prob, self.l2_reg,
                              self.clip_gradient, self.clip_value)
            network.load_and_train()
        
        ###
        # Evaluation
        ###
        if self.do_eval:
            evaluator = Evaluation(data,self.model_folder, self.summary_folder,
                                   self.model_name, self.summary_folder,self.batch_size,   
                                   **self.eval_params)
            evaluator.evaluate()
            print('Finished Evaluation.')
        
        ###
        # Tensorboard
        ###
        if self.tensorboard and self.do_training:
            print("Opening Tensorboard")
            os.system("tensorboard --logdir="+ self.summary_folder)
            #start_new_thread(webbrowser.open,("localhost:6006",))
            
            
            
if __name__ == "__main__":   
    
    ###
    # default dictionary parameter
    ###
    default_eval_params = {'datasets':['train','val','test'],                                   
                           'loss':True, 'accuracy':True, 'localization':True, 
                           'save_imgs':True,'num_imgs':10}
    
    
    def fill_up_input_dict(default_dic, dic):
        for k in default_dic:
            if not k in dic:
                dic[k] = default_dic[k]
        return dic
    
    args = load_configuration()
    args['eval_params'] = fill_up_input_dict(default_eval_params,args['eval_params'])
    
    manager = Manager(**args)
    manager.main()
    
        