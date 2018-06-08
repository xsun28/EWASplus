#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:26:06 2017

@author: Xiaobo
"""
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
import pandas as pd
import numpy as np
import ModelSelection as MS
import Ensemble as es
import xgbooster
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score,roc_curve,roc_auc_score,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.externals import joblib
from sklearn.model_selection import learning_curve
#-----------------------------------------------------------------------------
def learn_curve(model,train_x,train_label,cv=3,scoring='neg_log_loss'):
    model_c = clone(model)
    N,train_score,test_score = learning_curve(model_c, 
                                            train_x,train_label,cv=cv,train_sizes=np.linspace(0.3,1,10),
                                            scoring=scoring)
    
    plt.figure(figsize=(7,4))
    plt.title('{}'.format(type(model).__name__))
    plt.plot(N,np.mean(train_score,1),color='blue', label='training score')
    plt.plot(N,np.mean(test_score,1),color='red',label='validation score')
    plt.xlabel('training sample')
    plt.ylabel(scoring)
    plt.legend(loc=0)
    plt.show()
#-----------------------------------------------------------------------------    
def error_analysis(estimator,test_x,label,types='confusion_matrix'):
    predict = estimator.predict(test_x)
    class_num = len(label.unique())
    if types == 'confusion_matrix':
        conf_mat = confusion_matrix(label,predict)
        row_sums = conf_mat.sum(axis=1,keepdims=True)
        norm_conf_mat = conf_mat/row_sums
        np.fill_diagonal(norm_conf_mat,0)
        plt.matshow(norm_conf_mat,cmap=plt.cm.gray)        
    elif types == 'precision_recall_curve' and class_num<=2:
        precision,recall,threshold = precision_recall_curve(label,predict)
        plot_curve(recall,precision,type(estimator).__name__,types)
    elif types == 'roc_curve'and class_num<=2:
        fpr,tpr, threshold = roc_curve(label,predict)
        plot_curve(fpr,tpr,type(estimator).__name__,types)
    else:
        print("{0} classes can't use {1}".format(class_num,types))
    plt.show()
#---------------------------------------------------------------------------    
def plot_curve(score1,score2,label,types):
    plt.figure(figsize=(7,5))
    plt.title(types)
    plt.plot(score1,score2,linewidth=2,label=label)
    plt.plot([0,1],[1,1],'k--')
    plt.axis([0,1,0,1])
    if types == 'precision_recall_curve':        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    if types == 'roc_curve':
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
 #--------------------------------------------------------------------------
def scores(estimator,x,y,average):
    class_num = len(y.unique())
    predicts = estimator.predict(x)
    score_map = {}
    recall = recall_score(y,predicts,average=average)
    score_map['recall'] = recall
    precision = precision_score(y,predicts,average=average)
    score_map['precision'] = precision
    accuracy = accuracy_score(y,predicts)
    score_map['accuracy'] = accuracy
    f1 = f1_score(y,predicts,average=average)
    score_map['f1'] = f1
    if class_num <=2:
        auc = roc_auc_score(y,predicts,average=average)
        score_map['auc'] = auc
    return score_map
##############################################################################
dir='/home/ec2-user/CpGPython/'
with pd.HDFStore(dir+'data/selected_features','r') as h5s:
    train_x =h5s['train_x'] 
    train_label = h5s['train_label'] 
    test_x = h5s['test_x'] 
    test_label = h5s['test_label']   


#Cross-validation based model selection
score = 'neg_log_loss'
svm_params = {'C':0.01,'gamma':0.01}
mlp_params = {'alpha':100,'hidden_layer':(100,80,50,25,10) }
xgb_params = {'learning_rate':0.01,'lambda':100,'gamma':0.5}
log_params = {'C':0.01}
select = MS.ModelSelection(scoring=score,class_weights={0:2,2:2,1:1},
                           logistic_params=log_params,svm_params=svm_params,
                           mlp_params=mlp_params,xgboost_params=xgb_params)

select.fit(train_x,train_label)
model_scores = select.transform(train_x,train_label)

#Model Hyperparameter tuning and Evaluation
l = LogisticRegression()
rf = RandomForestClassifier()
svc = SVC()
xg = xgbooster.xgbooster()
mlp = MLPClassifier()
methods = [rf,svc,mlp]
class_w = {0:100,1:1,2:20}
l_param=[{'C':np.linspace(0.01, 0.5,20),'class_weight':[class_w]}]
rf_param = [{'max_depth':np.linspace(5,20,6,dtype='i8'),'min_samples_split': np.linspace(2,12,4,dtype='i8'),'min_samples_leaf': np.linspace(1,3,3,dtype='i8'),'class_weight':[class_w]}]
svc_param = [{'C':np.linspace(0.01,0.5,5),'gamma':np.linspace(0.01,1,5),'class_weight':[class_w],'search':['random',]}]
mlp_param = [{'alpha':np.linspace(10,100,10),'max_iter':[400],'hidden_layer_sizes':[(100,80,50,25,10),(200,120,80,40),(300,200,100),(400,200)]}]
xgb_param = [{'learning_rate':[0.01],'max_depth': np.linspace(3,13,6,dtype='i8'),'n_estimators':np.linspace(100,200,11,dtype='i8'),'reg_lambda': np.linspace(1,100,20),'gamma':np.linspace(0,10,11),'class_weight':[class_w],'search':['random',] }]
params = {'RandomForestClassifier': rf_param, 'SVC': svc_param,'MLPClassifier':mlp_param}
ensemble = es.Ensemble(methods=methods,params=params)
ensemble.fit(train_x,train_label)
ensemble.score(test_x,test_label)
joblib.dump(ensemble,'dir+models/ensemble.pkl')
#Model evaluation with learning curve
score_map={}
for name,estimator in ensemble.best_estimator.items():
    learn_curve(estimator,train_x,train_label,scoring='f1_macro')
    error_analysis(estimator,test_x,test_label)
    score_map[name] = scores(estimator,test_x,test_label,'macro')
