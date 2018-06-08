#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:16:42 2017

@author: Xiaobo
"""
##############################################################################
# usage:  select = ms.ModelSelection(scoring='neg_log_loss',class_weights={-1:400,1:100,0:1},
#                                    logistic_params=log_params,svm_params=svm_params,mlp_params=mlp_params,xgboost_params=xgb_params)
#
#  svm_params = {'C':0.01,'gamma':5}
#  mlp_params = {'alpha':100,'hidden_layer':(100,80,50,25,10) }
#  xgb_params = {'learning_rate':0.01,'lambda':100,'gamma':0.5}
#  log_params = {'C':0.01}
###############################################################################
import sys
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score
from functools import partial
import numpy as np
import pandas as pd
from models import deep_network_estimator as dne 

class ModelSelection(BaseEstimator):

    def __init__(self,scoring,class_weights,sample_weight,methods=['logistic','svm','random_forest','xgboost','mlp','dnn'],**kwargs):
        self.kwargs = kwargs
        self.class_weights = class_weights
        self.sample_weight = sample_weight
        self.scoring = scoring
        self.methods = { 'logistic': False,
                   'svm': False,
                   'random_forest':False,
                   'xgboost': False,
                   'mlp': False
                }
        self.initialized_methods = {}
        self.mean_score = {}        
        for method in methods:
            self.methods[method] = True
                        
        if self.methods['logistic']:
            params_map = kwargs['logistic_params']
            c = params_map['C']
            if len(class_weights.keys()) > 2:
                log_reg = LogisticRegression(class_weight=class_weights,C=c,multi_class='multinomial',solver='lbfgs')
            else:
                log_reg = LogisticRegression(class_weight=class_weights,C=c)
            self.initialized_methods['logistic'] = log_reg
        if self.methods['svm']:            
            params_map = kwargs['svm_params']
            c = params_map['C']
            g = params_map['gamma'] if 'gamma' in params_map.keys() else 'auto'
            svc = SVC(kernel='rbf',gamma=g,C=c,class_weight=class_weights,probability=True)
            self.initialized_methods['svm'] = svc
        if self.methods['random_forest']:
            rf = RandomForestClassifier(class_weight=class_weights,n_estimators=1000,n_jobs=-1)
            self.initialized_methods['random_forest'] = rf
        if self.methods['mlp']:
            params_map = kwargs['mlp_params']
            hidden_layer = params_map['hidden_layer']
            alpha = params_map['alpha']  ##alpha==lambda
            mlp = MLPClassifier(solver='adam',alpha=alpha,hidden_layer_sizes=hidden_layer)
            self.initialized_methods['mlp'] = mlp
        if self.methods['dnn']:
            params_map = kwargs['dnn_params']
            dnn = dne.tensor_DNN(**params_map)
            self.initialized_methods['dnn'] = dnn
        
        
    def fit(self,X,y):       

        cross_val = partial(cross_val_score,X=X,y=y,cv=5,n_jobs=-1,scoring=self.scoring,fit_params={'sample_weight':self.sample_weight})
        if self.methods['logistic']:
            log_reg = self.initialized_methods['logistic']
            logistic_results = cross_val(log_reg)
            self.mean_score['LogisticRegression'] = logistic_results.mean()
            log_reg.fit(X,y,sample_weight=self.sample_weight)
        if self.methods['svm']:
            svm = self.initialized_methods['svm']
            svm_results = cross_val(svm)
            self.mean_score['SVC'] = svm_results.mean()
            svm.fit(X,y,sample_weight=self.sample_weight)
        if self.methods['random_forest']:
            rf = self.initialized_methods['random_forest']
            rf_results = cross_val(rf)
            self.mean_score['RandomForestClassifier'] = rf_results.mean()
            rf.fit(X,y,sample_weight=self.sample_weight)
        if self.methods['xgboost']:
            params_map = self.kwargs['xgboost_params']
            rate = params_map['learning_rate'] if 'learning_rate' in params_map.keys() else 0.3
            lam = params_map['lambda'] if 'lambda' in params_map.keys() else 1
            g = params_map['gamma'] if 'gamma' in params_map.keys() else 0 
            iteration_round = params_map['iteration_round'] if 'iteration_round' in params_map.keys() else 3000   
            weights = pd.Series(np.ones(X.shape[0]),index=y.index)
            if self.class_weights is not None:
                for cls,weight in self.class_weights.items():
                    weights[y==cls] = weight
            elif self.sample_weight is not None:
                weights = self.sample_weight                
            data_matrix = xgb.DMatrix(X,label=y,weight=weights)
            self.class_num = len(self.class_weights.keys())
            params = {}
            if self.class_num >2 :
                params['objective'] = 'multi:softmax'
                params['num_class'] = self.class_num
            else:
                params['objective'] = 'binary:logistic'
            params.update({'eta':rate,'gamma':g,'lambda':lam})
            metrics = ['mlogloss'] if self.class_num > 2 else ['logloss']
            xgb_result = xgb.cv(params,data_matrix,num_boost_round=iteration_round,
                                nfold=5,stratified=True,metrics=metrics,
                                callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
            self.mean_score['xgbooster'] = xgb_result[xgb_result.keys()[0]].mean() #test-mlogloss-mean or test-auc_mean
            #if self.class_num > 2:
            self.mean_score['xgbooster'] = -self.mean_score['xgbooster']
            self.initialized_methods['xgbooster'] = xgb.train(params,data_matrix,num_boost_round=iteration_round)
        if self.methods['mlp']:
            mlp = self.initialized_methods['mlp']
            mlp_results = cross_val_score(mlp,X=X,y=y,cv=5,n_jobs=-1,scoring=self.scoring)
            self.mean_score['MLPClassifier'] = mlp_results.mean()
            mlp.fit(X,y)
        if self.methods['dnn']:
            dnn = self.initialized_methods['dnn']
            dnn_results = cross_val(dnn,n_jobs=1)
            self.mean_score['tensor_DNN'] = dnn_results.mean()
            dnn.fit(X,y,sample_weight=self.sample_weight)
        return self.mean_score
        
    def best_models(self,n=1):      
        best_ix = np.argsort(self.mean_score.values())[::-1][:n]
        best_models = {self.mean_score.keys()[ix]: self.mean_score.values()[ix] for ix in best_ix}
        return best_models
         
    def label_conversion(self,y):
        y_copy = y.copy()
        labels = np.unique(y_copy)
        for i,label in enumerate(labels):
            y_copy[y_copy==label] = i
        return y_copy 

    def estimators(self):
        return self.initialized_methods        
            
        
            
            
            
            
            