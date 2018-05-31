#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:39:31 2018

@author: Xiaobo
"""

import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
import pandas as pd
import numpy as np
from hyperopt import fmin,tpe,hp, STATUS_OK,Trials 
from sklearn.model_selection import cross_validate
import math
from sklearn.externals import joblib
import Logger
import argparse
from sklearn.svm import SVC
def cal_opt_score(estimator,weight_factor):
    results = cross_validate(estimator,train_x,train_label,scoring=scoring,cv=cv,return_train_score=False,fit_params={'sample_weight':np.power(sample_weights_train,weight_factor)})
    if 'f1_macro' in scoring:
        score = results['test_f1_macro'].mean()
    elif 'f1' in  scoring:
        score = results['test_f1'].mean()
    if math.isnan(score) or math.isinf(score):
        score = - np.Infinity
    return score

def svc_loss(params):
    weight_factor = params.pop('weight_factor')
    global estimators
    svc_estimator = SVC(**params)
    score = cal_opt_score(svc_estimator,weight_factor)
    estimators.extend([svc_estimator]) 
    return {'loss':-score,'status':STATUS_OK}

parser = argparse.ArgumentParser(description='Logistic Regression')
parser.add_argument('-f',required=False,default='normal',help='feature set',dest='features',metavar='normal/autoencoder')
parser.add_argument('-i',required=False,default=30,help='max iteration',dest='maxiter',metavar='30',type=int)
parser.add_argument('-c',required=False,default=3,help='cv number',dest='cv',metavar='3',type=int)
parser.add_argument('-d',required=False,default='AD_CpG',help='dataset type',dest='dataset',metavar='AD_CpG/RICHS')
args = parser.parse_args()
feature_dataset = args.features
cv = args.cv
max_iter = args.maxiter  
dataset = args.dataset+'/' 
home='/home/ec2-user/CpGPython/'
log_dir = home+'logs/'
model_dir = home+'models/'+dataset
logger = Logger.Logger(log_dir,False).get_logger()
if feature_dataset == 'normal':
##features selecetd by traditional methods
    with pd.HDFStore(home+'data/'+dataset+'selected_features','r') as h5s:
        train_x =h5s['train_x'] 
        train_label = h5s['train_label'] 
        test_x = h5s['test_x'] 
        test_label = h5s['test_label']   
        sample_weights_train = h5s['sample_weights_train'] 
        sample_weights_test = h5s['sample_weights_test'] 
    logger.info('Features used in training are from traditional feature selection')
##features selected by sparse autoencoder
elif feature_dataset == 'autoencoder':
    with pd.HDFStore(home+'data/'+dataset+'new_features','r') as h5s:
        train_x =h5s['train_x'] 
        train_label = h5s['train_label'] 
        test_x = h5s['test_x'] 
        test_label = h5s['test_label']
        sample_weights_train = h5s['sample_weights_train'] 
        sample_weights_test = h5s['sample_weights_test']
    logger.info('Features used in training are from sparse autoencoder')
if dataset == 'AD_CpG/':
    weight_factor = hp.uniform('weight_factor',1,1.5)
elif dataset == 'RICHS/':
    weight_factor = hp.uniform('weight_factor',1.0/3,0.5)
train_x = pd.DataFrame(train_x)
train_label = pd.Series(train_label)
sample_weights_train = pd.Series(sample_weights_train)

labels = train_label.unique()
class_num = len(labels)

svc_param = {'C': hp.uniform('C',0.005,1),'gamma': hp.uniform('gamma',0.001,1),'probability':hp.choice('probability',[True]),'weight_factor':weight_factor}
if class_num <= 2:
    scoring = ['precision','recall','f1','roc_auc','neg_log_loss']
else:
    scoring = ['precision_macro','recall_macro','f1_macro','neg_log_loss']

trial = Trials()
estimators = []
best = fmin(svc_loss,svc_param,algo=tpe.suggest,max_evals=max_iter,trials=trial) 
best_ix = np.argmin(trial.losses())
best_estimator = estimators[best_ix]
best_params = best_estimator.get_params()
best_params['weight_factor'] = best['weight_factor']
joblib.dump(best_params,model_dir+'SVC.pkl')    