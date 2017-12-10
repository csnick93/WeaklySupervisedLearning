# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:45:44 2017

@author: nickv
"""

"""
Test script
"""

import argparse


def f(x,y):
    if x == None:
        print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for integers")
    parser.add_argument('-x,', metavar = 'x', type = str, nargs =1,
                        default = [None], dest = 'x',
                        help = 'x value for the function f')
    parser.add_argument('-y,', metavar = 'y', type = str, nargs =1,
                        default = ['y'], dest = 'y',
                        help = 'y value for the function f')
    args = parser.parse_args()
    f(args.x[0], args.y[0])