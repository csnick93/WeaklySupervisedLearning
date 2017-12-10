# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:36:46 2017

@author: nickv
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 10:35:55 2017

    $MySpotifyPwd?
    @author: nickv
"""

import os
import csv

def trained_configs(root):
    """
    Writes out a csv file summary for trained configurations.
    
    Args:
        Root: root directory for trained models
    """
    # extract infos
    header_general = ['Folder', 'num_epochs', 'keep_prob', 'l2_reg', 'clip_gradient']
    header_params = ['learning_rate','conv_pretrained', 'trainable', 'attention_N', 
                         'seq_length', 'learn_sigma', 'kernel_sizes', 'pooling', 'feature_maps',
                         'gru', 'local_response_normalization', 'hidden_n', 
                         'batch_normalization', 'limit_delta','delta_limit']
    losses = ['Train Loss ', 'Val Loss ','Test Loss ']
    accuracies = ['Train Accuracy', 'Val Accuracy ', 'Test Accuracy ']
    ious = ['Average IOU (train)', 'Average IOU (val)', 'Average IOU (test)']
    
    def find_value(txt,key):
        """
        Finds value to key in training config file.
        
        Args:
            txt: list of lines of config file
            key: key that we need the value for
        """
        for line in txt:
            if key in line:
                return line.lstrip(key).lstrip(': ')
        return 'n/a'
        
    def find_model_params(txt,key):
        """
        Finds value to model parameter keys.
        
        Args:
            txt: list of lines of config file
            key: key that we need the value for
        """
        for line in txt:
            if key in line:
                tmp = line.split('\''+key+'\': ')[1]
                if tmp[0] == '[':
                    tmp = tmp.split(']')[0]+']'
                elif tmp[-1]=='}' and not ',' in tmp:
                    tmp = tmp.rstrip('}')
                else:
                    tmp = tmp.split(',')[0]
                return tmp
    
    infos = []
    for r, dirs,files in os.walk(root):
        if 'training_info.txt' in files:
            info = [os.path.basename(os.path.dirname(os.path.dirname(r)))]
            with open(os.path.join(r,'training_info.txt'),'r') as f:
                txt = f.read()
                txt = txt.split('\n')
                txt = [line.lstrip('\t') for line in txt]
                for h in header_general[1:]:
                    info.append(find_value(txt,h))
                for h in header_params:
                    info.append(find_model_params(txt,h))
            filenames = ['loss.txt','accuracy.txt','eval_iou.txt']
            for i, cat in enumerate([losses,accuracies,ious]):
                filename = os.path.join(os.path.dirname(os.path.dirname(r)),filenames[i])
                if os.path.exists(filename):
                    with open(filename,'r') as f:
                        txt = f.read()
                        txt = txt.split('\n')
                        for k in cat:
                            info.append(find_value(txt,k))
                else:
                    info += ['n/a','n/a','n/a']
            infos.append(info)
                
            
            
    # write to csv file
    with open(os.path.join(root,'summary.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar ='|', quoting = 
                            csv.QUOTE_MINIMAL)
        writer.writerow(header_general+header_params+losses+accuracies+ious)
        for info in infos:
            writer.writerow(info)
    
if __name__=='__main__':
    root = r'D:\MasterProjekt\WeaklySupervisedLearning\Networks\emb_MNIST\LSTM\untrainedCNN'
    trained_configs(root)
        
        