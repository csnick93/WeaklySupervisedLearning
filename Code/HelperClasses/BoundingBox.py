# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:44:00 2017

@author: nickv
"""


class BoundingBox:
    """
    Datastructure to encapsule bounding box coordinates
    """
    def __init__(self, x_min, x_max, y_min, y_max):
        """
        Args:
            x_min,x_max,y_min,y_max: coordinates of bounding box
        """
        if x_min > x_max or y_min > y_max:
            raise ValueError('Invalid arguments for bounding box object')
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
    def area(self):
        """
        Returns:
            computed area of bounding box
        """
        return (self.x_max-self.x_min)*(self.y_max-self.y_min)
    
    def contains_point(self, y, x):
        if x in range(self.x_min, self.x_max+1) and y in range(self.y_min,self.y_max+1):
            return True
        else:
            return False
    
    def print_params(self):
        print('X_min: %i' %self.x_min)
        print('X_max: %i' %self.x_max)
        print('Y_min: %i' %self.y_min)
        print('Y_max: %i' %self.y_max)
        
    
    def cout(self):
        """
        Print function for a bounding box object.
        """
        from matplotlib import pyplot as plt
        plt.plot([self.x_min,self.x_min,self.x_max,self.x_max,self.x_min],
                [self.y_min, self.y_max,self.y_max,self.y_min, self.y_min])
        plt.show()
        plt.close()
        
if __name__=='__main__':
    pass