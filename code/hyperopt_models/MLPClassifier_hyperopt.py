#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:10:54 2018

@author: Xiaobo
"""
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
import pandas as pd
import numpy as np
from hyperopt import fmin,tpe,hp, STATUS_OK,Trials 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import math
from sklearn.externals import joblib
import Logger
import argparse

def cal_opt_score(estimator):
    results = cross_validate(estimator,train_x,train_label,scoring=scoring,cv=cv,return_train_score=False)
    if 'f1_macro' in scoring:
        score = results['test_f1_macro'].mean()
    elif 'f1' in  scoring:
        score = results['test_f1'].mean()
    if math.isnan(score) or math.isinf(score):
        score = - np.Infinity
    return score

def mlp_loss(params):
    mlp_estimator = MLPClassifier(**params)
    score = cal_opt_score(mlp_estimator)
    estimators.extend([mlp_estimator])
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
train_x = pd.DataFrame(train_x)
train_label = pd.Series(train_label)
sample_weights_train = pd.Series(sample_weights_train)

labels = train_label.unique()
class_num = len(labels)
feature_num = train_x.shape[1]
mlp_param = {'alpha':hp.uniform('alpha',0.001,5),'max_iter':2000+hp.randint('max_iter',1000),'hidden_layer_sizes': hp.choice('hidden_layer_sizes',[(int(feature_num*50),int(feature_num*30),int(feature_num*10)),(int(feature_num*40),int(feature_num*30),int(feature_num*20),int(feature_num*10)),(int(feature_num*30),int(feature_num*25),int(feature_num*20),int(feature_num*15),int(feature_num*10)),(int(feature_num*60),int(feature_num*30))])}

if class_num <= 2:
    scoring = ['precision','recall','f1','roc_auc','neg_log_loss']
else:
    scoring = ['precision_macro','recall_macro','f1_macro','neg_log_loss']
    
trial = Trials()
estimators = []
best = fmin(mlp_loss,mlp_param,algo=tpe.suggest,max_evals=max_iter,trials=trial)
best_ix = np.argmin(trial.losses())
best_estimator = estimators[best_ix]
best_params = best_estimator.get_params()
joblib.dump(best_params,model_dir+'MLPClassifier.pkl')