# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 22:15:32 2017

@author: nickv
"""

def f(a):
    print(a)
    
if __name__=='__main__':
    import argparse
    
    def str2list(s):
        def convert_bool(s):
            if s.lower() == 'true':
                return True
            else:
                return False
        elements = s.replace('[','').replace(']','').split(',')
        # check for bool
        if elements[0].lower() in ['true','false']:
            elements = [convert_bool(e) for e in elements]
        # check for ints
        try:
            elements = [int(e) for e in elements]
        except:
            pass
        return elements
    
    
    parser = argparse.ArgumentParser(description=
                                     "Parser for arguments of manager")
    
    parser.add_argument('-d', metavar = 'dataset', type = str2list,
                        nargs = 1, default = [[True,False]], dest = 'dataset')
    
    args = parser.parse_args()
    f(args.dataset[0])