# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:58:44 2017

@author: nickv
"""
import os
import xml.etree.ElementTree as ET

XML_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'Configuration','config.xml')
print(XML_CONFIG)

def str2bool(s):
    '''
    Args:
        v: string encoding a boolean
    Returns:
        Boolean encoded by v
    '''
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False
        
def str2dict(d):
    '''
    Convert string to encoded dictionary.
    
    Args:
        d: string describing a dictionary
    Returns:
        dictionary described by d
    '''
    def isBool(s):
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        else:
            return False
        
    def isInt(s):
        if not '.' in s:
            try:
                int(s)
                return True
            except:
                return False
        return False 
    
    def isFloat(s):
        try:
            float(s)
            return True
        except:
            return False
        
    dic_string = d.lstrip('{').rstrip('}').replace(' ','')
    num_entries = dic_string.count(':')
    dic = {}
    
    for n in range(num_entries):
        key = dic_string.split(':')[0]
        dic_string = dic_string.lstrip(key+':')
        val = dic_string.split(':')[0]
        # check for list
        if '[' in val:
            val = val.split(']')[0].lstrip('[')
            val = str2list(val)
            dic_string = dic_string.lstrip(str(val).replace(' ',''))
        else:
            val = val.split(',')[0]
            # check for int
            if isInt(val):
                val = int(val)
            #check for float
            elif isFloat(val):
                val = float(val)
            # check for bool
            elif isBool(val):
                val = str2bool(val)
            # else remains string
            else:
                pass
            dic_string = dic_string.lstrip(str(val))
        dic[key] = val
    return dic
        
        
   

def str2list(s):
    '''
    Args:
        s: string encoding a list
    Returns:
        list encoded by string s
    '''
    elements = s.replace('[','').replace(']','').split(',')
    # check for bool
    if elements[0].lower() in ['true','false']:
        elements = [str2bool(e) for e in elements]
    # check for ints
    try:
        elements = [int(e) for e in elements]
    except:
        pass
    return elements

def load_configuration():
    '''
    Helper method that loads the configuration parameters from the
    config.xml file to initialize the manager instance.
    
    Returns:
        All the necessary configuration parameters.
    '''
    
    tree = ET.parse(XML_CONFIG)
    root = tree.getroot()
    data_params = root.find('DataParameters')
    model_params = root.find('ModelParameters')
    training_params = root.find('TrainingParameters')
    summary_params = root.find('SummaryParameters')
    eval_params = root.find('EvaluationParameters')
    
    params = {}
    
    params['dataset'] = data_params.find('dataset').text
    params['complete_set'] = str2list(data_params.find('complete_set').text)
    
    params['model_name'] = model_params.find('model_name').text
    params['model_params'] = str2dict(model_params.find('model_args').text)
    
    params['do_training'] = training_params.find('do_training').text
    params['num_epochs'] = int(training_params.find('epochs').text)
    params['keep_prob'] = float(training_params.find('keep_prob').text)
    params['l2_reg'] = float(training_params.find('l2_regularization').text)
    params['clip_gradient'] = str2bool(training_params.find('clip_gradient').text)
    params['clip_value'] = float(training_params.find('clip_value').text)
    params['opt'] = training_params.find('optimization').text
    params['opt_params'] = str2dict(training_params.find('optimization_parameters').text)
    params['batch_size'] = int(training_params.find('batch_size').text)
    
    params['tensorboard'] = str2bool(summary_params.find('tensorboard').text)
    params['summary_intervals'] = int(summary_params.find('summary_intervals').text)
    
    params['do_eval'] = str2bool(eval_params.find('do_eval').text)
    params['eval_params'] = str2dict(eval_params.find('eval_args').text)
    
    return params


if __name__=='__main__':
    print(load_configuration())