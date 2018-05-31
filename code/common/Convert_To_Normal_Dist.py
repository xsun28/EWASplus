#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:49:03 2017

@author: Xiaobo
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm
import math
class AnyToNormal(BaseEstimator, TransformerMixin):
    
    def __init__(self,col):
        self.col = col
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        column = X[self.col]
        value_counts = column.value_counts().sort_index()
        value_cumsum = value_counts.cumsum()
        prob = value_cumsum/(len(column)+0.5)
        norm_dist_xs = norm.ppf(prob)       
        value_mapping = pd.DataFrame()
        value_mapping['original_'+str(self.col)] = value_counts.index
        value_mapping['norm_'+str(self.col)] = norm_dist_xs
        group = column.groupby(column)  #group by value
        X[self.col] = group.apply(lambda x: self.change_value(x, value_mapping))
        return value_mapping
    
    def evaluate(self,x,mapping):
        original_col = mapping['original_'+self.col]
        norm_col = mapping['norm_'+self.col]
        index = np.searchsorted(original_col,x,side='left')
        if index == 0:
            return float(norm_col[index])
        original_lower_bound = float(original_col[index-1])
        original_upper_bound = float(original_col[index])
        norm_lower_bound = float(norm_col[index-1])
        norm_upper_bound = float(norm_col[index])
        original_diff = original_upper_bound - original_lower_bound
        norm_x = (x-original_lower_bound)/original_diff*norm_upper_bound + (original_upper_bound-x)/original_diff*norm_lower_bound
        return norm_x
    
    def change_value(self,x,mapping):
        original_val = x[x.index[0]] #the first index of grouped series x is not always 0
        norm_val = self.evaluate(original_val,mapping)
        if math.isnan(norm_val):
            print(original_val)
        x[:] =  norm_val
        return x 


  