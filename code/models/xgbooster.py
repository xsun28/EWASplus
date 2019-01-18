#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:42:10 2017

@author: Xiaobo
"""

import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
class xgbooster(BaseEstimator):
    class_weight = None
    def __init__(self,**params):
        self.params = params
      
        
    def fit(self,X,y=None,sample_weight=None):
        if len(y.unique()) > 2:
            objective = 'multi:softmax'
        else:
            objective = 'binary:logistic'
        self.params['objective'] = objective
        if ('class_weight' in self.params) or (sample_weight is not None):
            params = self.params.copy()                          
            weights = pd.Series(np.ones(len(y)),index=y.index)
            if ('class_weight' in self.params):
                class_weight = params.pop('class_weight')
                if class_weight is not None:
                    for key,value in class_weight.items():
                        weights[y==key] = value
            else:
                weights = sample_weight
            self.xgb = xgb.XGBClassifier(**params)
            self.xgb.fit(X,y,sample_weight=weights)
        else:
            self.xgb = xgb.XGBClassifier(**self.params)

            return self.xgb.fit(X,y)
        return self  
    
    def predict(self,X):
        return self.xgb.predict(X)
    
    def predict_proba(self,X):
        return self.xgb.predict_proba(X)
    
    def score(self,X,y):
        pred_y = self.xgb.predict_proba(X)
        return log_loss(y,pred_y)
    

    def get_params(self,deep=True):
        return self.params
    
    def set_params(self,**params):
        self.params.update(params)
        return self