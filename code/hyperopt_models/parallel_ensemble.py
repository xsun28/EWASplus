#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:14:08 2018

@author: Xiaobo
"""


''''
Call the run-opt.sh first !!
'''''
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import xgbooster 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import deep_network_estimator as dne
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib

class Ensemble(BaseEstimator):
    

#------------------------------------------------------------------------------    
    def __init__(self,methods=None,home='/home/ec2-user/CpGPython/',dataset='AD_CpG'):
        self.methods = methods
        self.home = home
        self.models = self.get_train_models(methods)
        self.dataset = dataset+"/"
        self.best_estimators_ = {}
        self.best_params_ = {}
            
    def get_train_models(self,models=['LogisticRegression','RandomForestClassifier','SVC','xgbooster','tensor_DNN']):
        methods = {}
        if 'LogisticRegression' in models:
            l = LogisticRegression
            methods['LogisticRegression' ] = l
        if 'RandomForestClassifier' in models:
            rf = RandomForestClassifier
            methods['RandomForestClassifier'] = rf
        if 'SVC' in models:
            svc = SVC
            methods['SVC'] = svc
        if 'xgbooster' in models:
            xg = xgbooster.xgbooster
            methods['xgbooster'] = xg
        if 'tensor_DNN' in models:
            dnn = dne.tensor_DNN
            methods['tensor_DNN'] = dnn
        if 'MLPClassifier' in models:
            methods['MLPClassifier'] = MLPClassifier
        return methods
        
#------------------------------------------------------------------------------    
    
    def fit(self,X,y,sample_weight=None):
        train_x = pd.DataFrame(X).copy()
        self.class_num = len(y.unique())
        self.labels = y.unique()
        train_label = pd.Series(y).copy()
        sample_weights_train = pd.Series(sample_weight).copy() if sample_weight is not None else pd.Series(np.ones_like(y))
        for method in self.methods:
            model_params = joblib.load(self.home+"models/"+self.dataset+method+".pkl")
            weight_factor = model_params.pop('weight_factor') if 'weight_factor' in model_params else 1
            estimator = self.models[method](**model_params)
            if method == 'MLPClassifier':
                estimator.fit(train_x,train_label)
            else:
                estimator.fit(train_x,train_label,sample_weight=np.power(sample_weights_train,weight_factor))
            self.best_estimators_[method] = estimator            
            self.best_params_[method] = model_params
        return self

    
#------------------------------------------------------------------------------    

    def voting(self,X):
        probs = np.zeros((X.shape[0],self.class_num))
        for method,best_estimator in self.best_estimators_.items():
            try:
                prob = best_estimator.predict_proba(X)
            except (NotFittedError,AttributeError):
                best_estimator.fit(Ensemble.train_x, Ensemble.train_label,Ensemble.sample_weights_train)
                prob = best_estimator.predict_proba(X)
            probs = np.add(probs,prob)
        labels = np.argmax(probs,axis=1)
        return probs,labels
        
    def predict(self,X):
        probs,labels = self.voting(X)                   
        return labels
    
    def predict_proba(self,X):
        probs,labels = self.voting(X)
        return probs/float(len(self.best_estimators_))
    
    def score (self,X,y):
        probs,pred_labels = self.voting(X)
        pred_probs = probs/float(len(self.best_estimators_))
        logloss_score = log_loss(y,pred_probs)
        if self.class_num > 2:
            f1_avg_score = f1_score(y,pred_labels,average='macro')
            recall_avg_score = recall_score(y,pred_labels,average='macro')
            precision_avg_score = precision_score(y,pred_labels,average='macro')
        else:
            f1_avg_score = f1_score(y,pred_labels)
            recall_avg_score = recall_score(y,pred_labels)
            precision_avg_score = precision_score(y,pred_labels)
        return logloss_score,f1_avg_score,recall_avg_score,precision_avg_score
    
    def label_conversion(self,y):
        y_copy = y.copy()
        for i,label in enumerate(self.labels):
            y_copy[y_copy==label] = i
        return y_copy   
           
    def true_label_conversion(self,preds):       
        preds_s = pd.Series(preds)
        for i,label in enumerate(self.labels):
            preds_s[preds_s==i] = self.labels[i]
        return preds_s
        
        
                
        