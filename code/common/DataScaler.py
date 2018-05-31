#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:17:43 2017

@author: Xiaobo
"""

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

class DataScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self, scaler='standard',feature_range=(0,1)):
        if scaler == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=feature_range)
        return
    
    def fit(self,X,y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self,X,y=None):       
        return pd.DataFrame(self.scaler.transform(X),columns=X.columns,index=X.index)
    
    def parameters(self):
        return self.scaler.get_params()