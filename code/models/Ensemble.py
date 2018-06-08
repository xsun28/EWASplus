#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:04:59 2017

@author: Xiaobo
"""
##############################################################################
#Usage: 
#  l = LogisticRegression()
#  rf = RandomForestClassifier()
#  svc = SVC()
#  xgb = xgb.xgbooster()
#  mlp = MLPClassifier()
#  methods = [l ,rf,svc,xg,mlp]
#  l_param=[{'C':np.linspace(0.01, 0.5,20),'class_weight':[{0:500,1:1,2:100}]}]
#  rf_param = [{'max_depth':np.linspace(5,20,6,dtype='i8'),'min_samples_split': np.linspace(2,12,4,dtype='i8'),'min_samples_leaf': np.linspace(1,3,3,dtype='i8'),'class_weight':[{0:500,1:1,2:100}]}]
#  svc_param = [{'C':np.linspace(0.01,0.5,10),'gamma':np.linspace(0.02,2,10),'class_weight':[{0:500,1:1,2:100}]}]
#  mlp_param = [{'alpha':np.linspace(10,100,10),'hidden_layer_sizes':[(100,80,50,25,10),(200,120,80,40),(300,200,100),(400,200)]}]
#  xgb_param = [{'learning_rate':[0.01],'max_depth': np.linspace(3,13,6,dtype='i8'),'n_estimators':np.linspace(100,200,11,dtype='i8'),'reg_lambda': np.linspace(1,100,20),'gamma':np.linspace(0,10,11),'class_weight':[{0:500,1:1,2:100}],'search':['random',] }]
#  params = {'LogisticRegression': l_param,'RandomForestClassifier': rf_param, 'SVC': svc_param,'xgbooster':xgb_param,'MLPClassifier':mlp_param}
##############################################################################
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, roc_auc_score
from copy import deepcopy

class Ensemble(BaseEstimator):

    def __init__(self,methods=None,params={}):
        self.methods = {}
        self.params = {}
        self.best_estimator_ = {}
        self.best_params_ = {}
        self.best_score_ = {}
        self.cv_results_ = {}
        self.model_pred_probs = dict()
        self.model_scores = dict()

        for method in methods:
            name = type(method).__name__
            self.methods[name] = method
            if params is not None:
                if name in params.keys():
                    self.params[name] = params[name]
            else:
                class_weight = {0:1,1:30}
                l_param=[{'C':np.linspace(0.1, 10,20),'class_weight':[class_weight]}]
                rf_param = [{'max_depth':np.linspace(5,30,6,dtype='i8'),'min_samples_split': np.linspace(5,30,5,dtype='i8'),'min_samples_leaf': np.linspace(1,5,5,dtype='i8'),'class_weight':[class_weight]}]
                svc_param = [{'C':np.linspace(0.01,0.2,5),'gamma':np.linspace(0.001,0.5,5),'class_weight':[class_weight]}]
                mlp_param = [{'alpha':np.linspace(0.01,2,10),'max_iter':[2000],'hidden_layer_sizes':[(100,80,50,25,10),(200,120,80,40),(300,200,100),(400,200)]}]
                xgb_param = [{'learning_rate':[0.1],'max_depth': np.linspace(3,13,6,dtype='i8'),'n_estimators':np.linspace(500,2000,5,dtype='i8'),'reg_lambda': np.linspace(5,100,10),'gamma':np.linspace(2,20,4),'class_weight':[class_weight],'search':['random',] }]
                dnn_param = [{'batch_normalization': [True],
                             'l2_reg': np.linspace(0.001,0.05,5),                            
                             'drop_out':np.linspace(0.1,0.5,3),
                             'weight_factor':np.linspace(1,2,3),
                             'steps':np.linspace(200,2000,10,dtype='i8'),
                             'batch_size':[30],
                             'scoring':['precision'],
                             'search':['random',]
                             }]  
                params = {'RandomForestClassifier': rf_param, 'SVC': svc_param,'MLPClassifier':mlp_param,'tensor_DNN':dnn_param,'xgbooster':xgb_param,'LogisticRegression': l_param}

                if name in params.keys():
                    self.params[name] = params[name]
                    
    def fit(self,X,y):
        self.class_num = len(y.unique())
        self.labels = y.unique()
        #scoring = 'neg_log_loss' #if self.class_num > 2 else 'f1'
        scoring = 'f1_macro' if self.class_num > 2 else 'f1'                                       
        estimators = []
        for name,method in self.methods.items():
            print (name)
            param_grid = deepcopy(self.params[name])
            search_method = param_grid[0].pop('search')[0] if 'search' in param_grid[0] else 'grid'
            n_iter = param_grid[0].pop('n_iter')[0] if 'n_iter' in param_grid[0] else 50
            sample_weight = param_grid[0].pop('sample_weight')[0] if 'sample_weight' in param_grid[0] else None
            if name == 'LogisticRegression' and self.class_num > 2:
                param_grid[0]['solver'] = np.array(['lbfgs'])
                param_grid[0]['multi_class'] = np.array(['multinomial'])
                print(param_grid)
            if name == 'SVC':
                param_grid[0]['probability'] = np.array([True])
            if name == 'tensor_DNN': 
                if 'n_classes' not in param_grid[0]:
                    param_grid[0]['n_classes'] = [self.class_num]
                if 'hidden_layers' not in param_grid[0]:
                    feature_num = X.shape[1]
                    param_grid[0]['hidden_layers'] = [[int(feature_num*5),int(feature_num*3),int(feature_num*1)],[int(feature_num*4),int(feature_num*3),int(feature_num*2),int(feature_num*1)],[int(feature_num*3),int(feature_num*2.5),int(feature_num*2),int(feature_num*1.5),int(feature_num*1)],[int(feature_num*6),int(feature_num*3)]]
            search = GridSearchCV(method,param_grid,cv=2,n_jobs=-1,scoring=scoring) if search_method == 'grid' else RandomizedSearchCV(method,param_distributions=param_grid[0],n_iter=n_iter,cv=2,scoring=scoring)
            print (type(search).__name__)
            if name == 'MLPClassifier':
                search.fit(X,y)
            else:
                search.fit(X,y,sample_weight=sample_weight)
            self.best_estimator_[name] = search.best_estimator_
            self.best_params_[name] = search.best_params_
            self.best_score_[name] = search.best_score_
            self.cv_results_[name] = search.cv_results_
            estimators.extend([(name,search.best_estimator_)])
        self.voting_clf = VotingClassifier(estimators,voting='soft',n_jobs=-1)
        return self
    
    def voting(self,X,y=None):
        probs = np.zeros((X.shape[0],self.class_num))
        for method,best_estimator in self.best_estimator_.items():
            prob = best_estimator.predict_proba(X)
            self.model_pred_probs[method] = prob
            if y is not None:
                self.model_scores[method] = self.model_score(y,prob)
            probs = np.add(probs,prob)
        labels = np.argmax(probs,axis=1)
        return probs,labels
        
    def predict(self,X):
        probs,labels = self.voting(X)                   
        return labels
    
    def predict_proba(self,X):
        probs,labels = self.voting(X)
        return probs/float(len(self.best_estimator_))
    
    def score (self,X,y):
        probs,pred_labels = self.voting(X,y)
        pred_probs = probs/float(len(self.best_estimator_))
        logloss_score = log_loss(y,pred_probs)
        if self.class_num > 2:
            f1_avg_score = f1_score(y,pred_labels,average='macro')
            recall_avg_score = recall_score(y,pred_labels,average='macro')
            precision_avg_score = precision_score(y,pred_labels,average='macro')
            return logloss_score,f1_avg_score,recall_avg_score,precision_avg_score
        else:
            f1_avg_score = f1_score(y,pred_labels)
            recall_avg_score = recall_score(y,pred_labels)
            precision_avg_score = precision_score(y,pred_labels)
            positive_prob = pred_probs[:,1] 
            auc_score = roc_auc_score(y,positive_prob)
            return logloss_score,f1_avg_score,recall_avg_score,precision_avg_score,auc_score
    
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
        
    def model_probs(self):
        if len(self.model_pred_probs) == 0:
            print('No model probs')
            return None
        return self.model_pred_probs
                
    def model_score(self,y,pred_probs):
        pred_labels = np.argmax(pred_probs,axis=1)
        logloss_score = log_loss(y,pred_probs)
        if self.class_num > 2:
            f1_avg_score = f1_score(y,pred_labels,average='macro')
            recall_avg_score = recall_score(y,pred_labels,average='macro')
            precision_avg_score = precision_score(y,pred_labels,average='macro')
            return logloss_score,f1_avg_score,recall_avg_score,precision_avg_score
        else:
            f1_avg_score = f1_score(y,pred_labels)
            recall_avg_score = recall_score(y,pred_labels)
            precision_avg_score = precision_score(y,pred_labels)
            positive_prob = pred_probs[:,1] 
            auc_score = roc_auc_score(y,positive_prob)
            return logloss_score,f1_avg_score,recall_avg_score,precision_avg_score,auc_score
        
    def get_model_scores(self):
        if len(self.model_scores) == 0:
            print('No model scores')
            return None
        return self.model_scores    
    
    def best_params(self):
        return self.best_params_
        