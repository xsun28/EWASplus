#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:17:58 2017

@author: Xiaobo
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
#-------------------------------------------------
class reduce_vif(BaseEstimator,TransformerMixin):
    def __init__(self,thresh=5):    ###thresh=5 when 80% variance in a variable can be explained by other variables
        self.thresh = thresh
    
    def fit(self,X,y=None):
         return self

    def transform(self,X,y=None):
        return self.calculate(X)

    
    def calculate(self,X):       
        stop = False
        while not stop:
            columns = X.columns
            scores = np.array([vif(X[columns].values,columns.get_loc(col)) for col in columns])
            if scores.max()>self.thresh:
                max_index = scores.argmax()
                max_col = columns[max_index]
                X = X.drop(max_col,axis=1)
                continue
            else:
                stop = True
                            
        return columns