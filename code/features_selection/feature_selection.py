#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:13:18 2017

@author: Xiaobo
"""

from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif,SelectKBest,f_classif,SelectFromModel
from features_selection import CorrFeatureSelector as cfs
from features_selection import reduce_vif as rv
from functools import partial
import xgboost as xgb
import itertools
from scipy.stats import rankdata
import re

class FeatureSelection(BaseEstimator,TransformerMixin):

    def __init__(self,class_num=2,class_weights=None, methods=['random_forest','xgboost','extra_forest','mutual_information','fisher_score','logistic_regression'],all_intersect=True,**kwargs):
        self.class_num =class_num
        self.class_weights = class_weights
        self.methods = methods
        self.selected_methods = {'random_forest':False,
                        'extra_forest': False,
                        'xgboost':False,
                        'mutual_information': False,
                        'logistic_regression': False,
                        'correlation_reduction': False,
                        'VIF': False,
                        'fisher_score':False,
                        'linear_SVC':False
                        }
        self.initialized_methods = {}
        self.selected_features = {}
        self.feature_scores ={}
        for method in methods:
            self.selected_methods[method] = True
        self.all_intersect = all_intersect
        self.kwargs = kwargs
        return
    
    def fit(self,X=None,y=None,sample_weight=None,random_state=43):
        self.sample_weights = sample_weight
        if self.selected_methods['random_forest']:
            rf_params = self.kwargs['random_forest']
            rf_params['random_state'] = random_state
            rf = RandomForestClassifier(**rf_params)
            self.initialized_methods['random_forest'] = rf
        if self.selected_methods['xgboost']:   
            xgb_params = self.kwargs['xgboost']
            xgb_params['objective'] = 'multi:softmax' if self.class_num>2 else 'binary:logistic'
            xgb_params['random_state'] = random_state
            booster = xgb.XGBClassifier(**xgb_params)
            self.initialized_methods['xgboost'] = booster
#        if self.selected_methods['extra_forest']:
#            ef = ExtraTreesClassifier(class_weight=self.class_weights,n_estimators=10000,n_jobs=-1)
#            self.initialized_methods['extra_forest'] = ef
        if self.selected_methods['mutual_information']:
            mutual_info_clf = partial(mutual_info_classif,random_state=91)
            mi_params = self.kwargs['mutual_information']
            mi_params['random_state'] = random_state
            mi = SelectKBest(mutual_info_clf,**mi_params)
            self.initialized_methods['mutual_information'] = mi
        if self.selected_methods['fisher_score']:
            fisher_params = self.kwargs['fisher_score']
            f = SelectKBest(f_classif,**fisher_params)
            self.initialized_methods['fisher_score'] = f
        if self.selected_methods['linear_SVC']:
            lsvc_params = self.kwargs['linear_SVC']
            lsvc_params['random_state'] = random_state
            lsvc = LinearSVC(**lsvc_params)
            self.initialized_methods['linear_SVC'] = lsvc
        if self.selected_methods['logistic_regression']:
            log_reg_params = self.kwargs['logistic_regression']
            log_reg_params['random_state'] = random_state
            if self.class_num == 2:
                log_reg = LogisticRegression(**log_reg_params)
                self.initialized_methods['logistic_regression'] = log_reg
            else:
                class_weights = [{0:self.class_weights[0],1:self.class_weights[1]},
                                 {1:self.class_weights[1],2:self.class_weights[2]},
                                 {0:self.class_weights[0],2:self.class_weights[2]}]
                log_regs = {i:LogisticRegression(**log_reg_params,class_weight=weight) for i,weight in zip(['01','12','02'],class_weights)}
                self.initialized_methods['logistic_regression'] = log_regs
        
        return self
                                        
    def transform(self,X,y=None): 
        X_std = np.std(X,axis=0)                      
        if self.selected_methods['random_forest']:
            rf = self.initialized_methods['random_forest']
            rf.fit(X,y,sample_weight = self.sample_weights)
            rf_scores = pd.DataFrame({'feature':X.columns,'score':rf.feature_importances_})
            rf_scores.sort_values('score',inplace=True,ascending=False)
            self.selected_features['random_forest'] = pd.DataFrame(rf_scores['feature'][:100])
            self.feature_scores['random_forest'] = rf_scores
        if self.selected_methods['xgboost']:
            eval_metric = 'mlogloss' if self.class_num >2 else 'logloss'     
            booster = self.initialized_methods['xgboost']
#            weights = pd.Series(np.ones(X.shape[0]),index=y.index)
#            if type(self.class_weights) == dict:
#                for cls,weight in self.class_weights.items():  #weight per class
#                    weights[y==cls] = weight
#            elif type(self.class_weights) == pd.Series:        # weight per instance
#                weights = self.class_weights
            weights = self.sample_weights             
            booster.fit(X,y,sample_weight=weights,eval_metric=eval_metric)
            booster_scores = pd.DataFrame({'feature':X.columns,'score':booster.feature_importances_}).sort_values('score',ascending=False)
            self.selected_features['xgboost'] = pd.DataFrame(booster_scores['feature'][:100])
            self.feature_scores['xgboost'] = booster_scores
        if self.selected_methods['extra_forest']:
            ef = self.initialized_methods['extra_forest']
            ef.fit(X,y,sample_weight = self.sample_weights)
            ef_scores = pd.DataFrame({'feature':X.columns,'score':rf.feature_importances_})
            ef_scores.sort_values('score',inplace=True,ascending=False)
            self.selected_features['extra_forest'] = pd.DataFrame(ef_scores['feature'][:100])
            self.feature_scores['extra_forest'] = ef_scores
        if self.selected_methods['mutual_information']:
            mi = self.initialized_methods['mutual_information']
            mi.fit_transform(X,y)
            self.selected_features['mutual_information'] = pd.DataFrame([X.columns[col] for col in np.argsort(mi.scores_)[::-1]][:self.kwargs['num']],columns=['feature'])
            mi_scores = pd.DataFrame({'feature':X.columns,'score':mi.scores_}).sort_values('score',ascending=False)
            self.feature_scores['mutual_information'] = mi_scores
        if self.selected_methods['fisher_score']:
            f = self.initialized_methods['fisher_score']
            f.fit_transform(X,y)
            self.selected_features['fisher_score'] = pd.DataFrame([X.columns[col] for col in np.argsort(f.scores_)[::-1]][:self.kwargs['num']],columns=['feature'])
            fs_scores = pd.DataFrame({'feature':X.columns,'score':f.scores_}).sort_values('score',ascending=False)
            self.feature_scores['fisher_score'] = fs_scores
        if self.selected_methods['linear_SVC']:
            lsvc = self.initialized_methods['linear_SVC']
            lsvc.fit(X,y,sample_weight = self.sample_weights)
            self.selected_features['linear_SVC'] = pd.DataFrame(X.columns[SelectFromModel(lsvc,prefit=True).get_support()],columns=['feature'])
            print(lsvc.coef_*X_std.values)
            lsvc_scores = pd.DataFrame({'feature':X.columns.values,'score':np.abs((lsvc.coef_*X_std.values)[0])})
            self.feature_scores['linear_SVC'] = lsvc_scores.query('score>0')
        if self.selected_methods['logistic_regression']:
            if self.class_num == 2:
                log_reg = self.initialized_methods['logistic_regression']
                log_reg.fit(X,y,sample_weight = self.sample_weights)
                self.selected_features['logistic_regression'] = pd.DataFrame(X.columns[SelectFromModel(log_reg,prefit=True).get_support()],columns=['feature'])
                log_reg_scores = pd.DataFrame({'feature':X.columns.values,'score':np.abs((log_reg.coef_*X_std.values)[0])})
                self.feature_scores['logistic_regression'] = log_reg_scores.query('score>0')
            else:
                log_selected_features = {key:self.get_features_logistic_regression(X,y,key,classes) for key, classes in zip(['01','12','02'],[(0,1),(1,2),(0,2)]) }
                self.selected_features['logistic_regression'] = pd.DataFrame(list(set.union(*log_selected_features.values())),columns=['feature'])
        if self.selected_methods['correlation_reduction']:
            threshold = self.kwargs['threshold']
            intersect_features = self.intersection()
            reducer = cfs.reduce_corr(threshold=threshold)
            corr_reduced_features = reducer.fit_transform(X[intersect_features.iloc[:,0]])
            self.selected_features['correlation_reduction'] = pd.DataFrame(corr_reduced_features,columns=['feature'])
            if not self.selected_methods['VIF']:
                return self.selected_features['correlation_reduction']
        if self.selected_methods['VIF']:
            vif = rv.reduce_vif()
            if self.selected_methods['correlation_reduction']:
                vif_selected_features = vif.fit_transform(X[self.selected_features['correlation_reduction'].iloc[:,0]]).values                    
            else:
                intersect_features = self.intersection()
                vif_selected_features = vif.fit_transform(X[intersect_features.iloc[:,0]]).values    
            self.selected_features['VIF'] = pd.DataFrame(vif_selected_features,columns=['feature'])
            return self.selected_features['VIF']
                                                         
        return self.intersectionNxN()
    
    
    def get_features_logistic_regression(self,X,y,key,classes):
        log_reg = self.initialized_methods['logistic_regression'][key]
        log_reg.fit(X,y,sample_weight = self.sample_weights)
        return set(X.columns[SelectFromModel(log_reg,prefit=True).get_support()])
    
        
    def intersection(self):
        features = pd.DataFrame()
        for feature in self.selected_features.values():
            feature.rename({0:'feature'},axis=1,inplace=True)
            if len(features) == 0:
                features = feature
                continue
            features = pd.merge(features,feature,how='inner',on='feature')
        return features
    
    def intersect(self,a,b):
        if len(a) == 0:
            return b
        elif len(b) == 0:
            return a
        else:
            return pd.merge(a,b,how='inner',on='feature')
        
        
    def intersectionNxN(self):
        sets = pd.DataFrame(columns=['feature','n'])
        n = len(self.methods)
        for i in range(2,n+1):
            s = set()
            iterator = itertools.combinations(self.methods,i)
            for combination in iterator:
                s1 = set()
                first = True
                for m in combination:
                    feature_set = set(self.selected_features[m]['feature'])
                    if first:
                        s1 = feature_set
                        first=False
                    s1 = s1.intersection(feature_set)
                s = s.union(s1)
            sets = sets.append(pd.DataFrame({'feature':list(s),'n':i*np.ones(len(s))}),ignore_index=True)
        sets = sets.drop_duplicates(['feature'],keep='last')
        return sets

    
    def feature_rank(self):
        start = True
        for method,df in self.feature_scores.items():            
            if start:
                features_ranks = df
                start = False
            df['score'] = 1.0/rankdata(df['score'])
            df.set_index('feature',inplace=True)
            features_ranks = features_ranks.join(df,how='outer',rsuffix="_"+method)
        features_ranks.fillna(0,inplace=True)
        pattern = re.compile('score.*')
        features_ranks['score_sum'] = features_ranks[[c for c in filter(pattern.search,features_ranks.columns)]].sum(axis=1)
        features_ranks.sort_values(['score_sum'],ascending=False,inplace=True)
        features_ranks.reset_index(inplace=True)
        return features_ranks[:60]
