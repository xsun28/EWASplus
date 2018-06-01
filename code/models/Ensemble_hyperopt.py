#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:14:08 2018

@author: Xiaobo
"""
import sys
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score,roc_auc_score
from hyperopt import fmin,tpe,hp, STATUS_OK,Trials
from models import xgbooster 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate
from models import deep_network_estimator as dne
from sklearn.exceptions import NotFittedError
import math
from sklearn.calibration import CalibratedClassifierCV
class Ensemble(BaseEstimator):

    #static fields
    estimators = dict()
    train_x = None
    train_label = None
    sample_weights_train = None
    scoring = None


    @staticmethod
    def cal_opt_score(estimator,weight_factor=None):
        print(weight_factor)
        if weight_factor is not None:
            results = cross_validate(estimator,Ensemble.train_x,Ensemble.train_label,scoring=Ensemble.scoring,cv=3,return_train_score=False,fit_params={'sample_weight':np.power(Ensemble.sample_weights_train,weight_factor)},n_jobs=-1)
        else:
            results = cross_validate(estimator,Ensemble.train_x,Ensemble.train_label,scoring=Ensemble.scoring,cv=3,return_train_score=False,n_jobs=-1)
        if 'f1_macro' in Ensemble.scoring:
            score = results['test_f1_macro'].mean()
        elif 'f1' in  Ensemble.scoring:
            score = results['test_f1'].mean()
        if math.isnan(score) or math.isinf(score):
            score = -np.Infinity
        return score
    
    @staticmethod
    def logistic_loss(params):
        weight_factor = params.pop('weight_factor')
        l_estimator = LogisticRegression(**params)
        score = Ensemble.cal_opt_score(l_estimator,weight_factor)
        Ensemble.estimators['LogisticRegression'].extend([l_estimator])
        return {'loss':-score,'status':STATUS_OK}
    
    @staticmethod
    def rf_loss(params):
        weight_factor = params.pop('weight_factor')
        global estimators
        rf_estimator = RandomForestClassifier(**params,n_jobs=-1)
        score = Ensemble.cal_opt_score(rf_estimator,weight_factor)
        Ensemble.estimators['RandomForestClassifier'].extend([rf_estimator])
        return {'loss':-score,'status':STATUS_OK}

    @staticmethod
    def svc_loss(params):
        weight_factor = params.pop('weight_factor')
        global estimators
        svc_estimator = SVC(**params)
        score = Ensemble.cal_opt_score(svc_estimator,weight_factor)
        Ensemble.estimators['SVC'].extend([svc_estimator])
        return {'loss':-score,'status':STATUS_OK}
    
    @staticmethod
    def xgb_loss(params):
        weight_factor = params.pop('weight_factor')
        global estimators
        params['n_jobs'] = -1
        xgb_estimator = xgbooster.xgbooster(**params)
        score = Ensemble.cal_opt_score(xgb_estimator,weight_factor)
        Ensemble.estimators['xgbooster'].extend([xgb_estimator])
        return {'loss':-score,'status':STATUS_OK}
    
    @staticmethod
    def dnn_loss(params):
        weight_factor = params.pop('weight_factor')
        global estimators
        dnn_estimator = dne.tensor_DNN(**params)
        score = Ensemble.cal_opt_score(dnn_estimator,weight_factor)
        Ensemble.estimators['tensor_DNN'].extend([dnn_estimator])
        return {'loss':-score,'status':STATUS_OK}
    
    @staticmethod
    def mlp_loss(params):
        mlp_estimator = MLPClassifier(**params)
        score = Ensemble.cal_opt_score(mlp_estimator)
        Ensemble.estimators['MLPClassifier'].extend([mlp_estimator])
        return {'loss':-score,'status':STATUS_OK}
    
    @staticmethod
    def lsvc_loss(params):
        weight_factor = params.pop('weight_factor')
        lsvc_estimator = CalibratedClassifierCV(LinearSVC(**params,dual=False))
        score = Ensemble.cal_opt_score(lsvc_estimator,weight_factor)
        Ensemble.estimators['LinearSVC'].extend([lsvc_estimator])
        return {'loss':-score,'status':STATUS_OK}
    
    def best_estimator(self,trials,estimators):
        best_ix = np.argmin(trials.losses())
        return estimators[best_ix] 
    
#------------------------------------------------------------------------------    
    def __init__(self,methods=None,params=None):

        if 'scoring' in params:
            Ensemble.scoring = params.pop('scoring')
        self.params = {}
        self.trials = {}
        self.best_estimators_ = {}
        self.best_params_ = dict()
        self.best_score_ = dict()
        self.model_pred_probs = dict()
        self.model_preds = dict()
        self.model_scores = dict()
        for name in methods:
            if params is not None:
                if name in params.keys():
                    self.params[name] = params[name]
            else:
                l_param = {'C': hp.uniform('C',0.05,10),'weight_factor':hp.uniform('weight_factor',1,1)}
                rf_param = {'max_depth':3+hp.randint('max_depth',30), 'min_samples_split': 3+hp.randint('min_samples_split',30),'min_samples_leaf': 1+hp.randint('min_samples_leaf',5),'weight_factor':hp.uniform('weight_factor',1,1)}
                svc_param = {'C': hp.uniform('C',0.005,1),'gamma': hp.uniform('gamma',0.001,1),'probability':hp.choice('probability',[True]),'weight_factor':hp.uniform('weight_factor',1,1)}
                xgb_param = {'learning_rate':hp.choice('learning_rate',[0.1]),'max_depth': 3+hp.randint('max_depth',15),'n_estimators':500+hp.randint('n_estimators',3000),'reg_lambda': hp.uniform('reg_lambda',1,100),'gamma': hp.uniform('gamma',1,30),'weight_factor':hp.uniform('weight_factor',1,1)}
                dnn_param = {'batch_normalization': hp.choice('batch_normalization',[True]),
                             'l2_reg': hp.uniform('l2_reg',0.001,0.05),                            
                             'drop_out':hp.uniform('drop_out',0.1,0.5),
                             'weight_factor':hp.uniform('weight_factor',1,1),
                             'steps':200+hp.randint('steps',2000),
                             'batch_size':hp.choice('batch_size',[30]),
                             'scoring':hp.choice('scoring',['precision']),
                             }
                mlp_param = {'alpha':hp.uniform('alpha',0.001,5),'max_iter':2000+hp.randint('max_iter',1000)}
                lsvc_param = {'C': hp.uniform('C',0.1,10),'weight_factor':hp.uniform('weight_factor',1,1)}                
                params = {'RandomForestClassifier': rf_param, 'LinearSVC':lsvc_param,'SVC': svc_param,'tensor_DNN':dnn_param,'xgbooster':xgb_param,'LogisticRegression': l_param,'MPClassifier':mlp_param}
                if name in params.keys():
                    self.params[name] = params[name]
        
        for name in params.keys():
            Ensemble.estimators[name] = []                
            self.trials[name] = Trials()
            
#------------------------------------------------------------------------------    
    
    def fit(self,X,y,sample_weight=None,max_iter=50):
        self.class_num = len(y.unique())
        self.labels = y.unique()
        if Ensemble.scoring is None:
            if self.class_num <= 2:
                Ensemble.scoring = ['precision','recall','f1','roc_auc','neg_log_loss']
            else:
                Ensemble.scoring = ['precision_macro','recall_macro','f1_macro','neg_log_loss']
        for value in Ensemble.estimators.values():
            value = []
        Ensemble.train_x = pd.DataFrame(X)
        Ensemble.train_label = pd.Series(y)
        Ensemble.sample_weights_train = pd.Series(sample_weight) if sample_weight is not None else pd.Series(np.ones_like(y))
        if 'LogisticRegression' in self.params.keys() and self.class_num > 2:
            l_param = self.params['LogisticRegression'] 
            l_param['solver'] = hp.choice('solver',['lbfgs'])
            l_param['multi_class'] = hp.choice('multi_class',['multinomial'])
            print(l_param)
        if 'tensor_DNN' in self.params.keys():
            feature_num = X.shape[1]
            dnn_param = self.params['tensor_DNN']
            dnn_param['n_classes'] = hp.choice('n_classes',[self.class_num])
            dnn_param['hidden_layers'] = hp.choice('hidden_layers',[[int(feature_num*50),int(feature_num*30),int(feature_num*10)],[int(feature_num*40),int(feature_num*30),int(feature_num*20),int(feature_num*10)],[int(feature_num*30),int(feature_num*25),int(feature_num*20),int(feature_num*15),int(feature_num*10)],[int(feature_num*60),int(feature_num*30)]])
        if 'MLPClassifier' in self.params.keys():
            feature_num = X.shape[1]
            mlp_param = self.params['MLPClassifier']
            mlp_param['hidden_layer_sizes'] = hp.choice('hidden_layer_sizes',[(int(feature_num*50),int(feature_num*30),int(feature_num*10)),(int(feature_num*40),int(feature_num*30),int(feature_num*20),int(feature_num*10)),(int(feature_num*30),int(feature_num*25),int(feature_num*20),int(feature_num*15),int(feature_num*10)),(int(feature_num*60),int(feature_num*30))])
            
        score_fns = {'LogisticRegression':Ensemble.logistic_loss,
                     'RandomForestClassifier': Ensemble.rf_loss,
                     'SVC': Ensemble.svc_loss,
                     'xgbooster': Ensemble.xgb_loss,
                     'tensor_DNN': Ensemble.dnn_loss,
                     'MLPClassifier': Ensemble.mlp_loss,
                     'LinearSVC': Ensemble.lsvc_loss
                     }
        
        for name,param in self.params.items():
            score_fn = score_fns[name] 
            best = fmin(score_fn,param,algo=tpe.suggest,max_evals=max_iter,trials=self.trials[name])
            print(best)
            self.best_estimators_[name] = self.best_estimator(self.trials[name],Ensemble.estimators[name])
            self.best_params_[name] = self.best_estimators_[name].get_params()
            self.best_params_[name]['weight_factor'] = best['weight_factor']
            self.best_score_[name] = -np.min(self.trials[name].losses())
        self.voting_clf = VotingClassifier(self.best_estimators_,voting='soft',n_jobs=-1)
        return self
    
    
#------------------------------------------------------------------------------    

    def voting(self,X,y=None):
        probs = np.zeros((X.shape[0],self.class_num))
        for method,best_estimator in self.best_estimators_.items():
            try:
                prob = best_estimator.predict_proba(X)
                pred = best_estimator.predict(X)
            except (NotFittedError,AttributeError):
                if method == 'MLPClassifier':
                    best_estimator.fit(Ensemble.train_x, Ensemble.train_label)
                else:
                    weight_factor = self.best_params_[method]['weight_factor']
                    print("{},{}".format(method,weight_factor))
                    best_estimator.fit(Ensemble.train_x, Ensemble.train_label,np.power(Ensemble.sample_weights_train,weight_factor))
                prob = best_estimator.predict_proba(X)
                pred = best_estimator.predict(X)
            self.model_pred_probs[method] = prob
            self.model_preds[method] = pred
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
        return probs/float(len(self.best_estimators_))
    
    def score (self,X,y):
        probs,pred_labels = self.voting(X,y)
        pred_probs = probs/float(len(self.best_estimators_))
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
    
    def model_predicts(self):
        if len(self.model_preds) == 0:
            print('No model probs')
            return None
        return self.model_preds
    
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
        print(self.model_scores)
        return self.model_scores
        
    def best_params(self):
        return self.best_params_
        
                
        