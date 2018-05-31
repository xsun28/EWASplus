#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:01:06 2017

@author: Xiaobo
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class reduce_corr(BaseEstimator,TransformerMixin):
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        cols = X.columns
        for col in cols:
            corr_matrix = X[cols].corr()
            corr_matrix.values[[np.arange(corr_matrix.shape[0])]*2] = 0.
            if(corr_matrix>self.threshold).sum().sum() == 0:
                return cols
            max_corr_index = np.argmax((corr_matrix>self.threshold).sum())
            cols = cols.drop(max_corr_index)
            
        
        return None                        