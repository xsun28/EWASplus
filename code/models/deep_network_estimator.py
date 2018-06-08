#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:52:29 2018

@author: Xiaobo
"""
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from log import Logger
from functools import partial
import re
from datetime import datetime
from sklearn.base import BaseEstimator

class tensor_DNN(BaseEstimator):
    
    @staticmethod
    def dnn_model(features,labels,mode,params):
        if mode == tf.estimator.ModeKeys.TRAIN:
            training = True
        else:
            training = False
        if mode != tf.estimator.ModeKeys.PREDICT:
            sample_weights = features.pop('sample_weights')
            print(sample_weights)
        if_batch_norm = params['batch_normalization'] if 'batch_normalization' in params else True
        l2_reg = params['l2_reg'] if 'l2_reg' in params else 0
        n_classes = params['n_classes'] if 'n_classes' in params else 2
        hidden_layers = params['hidden_layers']
        dropout_rate = params['dropout'] if 'dropout' in params else 0
        activation = tf.nn.elu
        he_init = tf.contrib.layers.variance_scaling_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        dense_layer = partial(tf.layers.dense,kernel_regularizer=regularizer,kernel_initializer=he_init)
        dropout = partial(tf.layers.dropout,rate=dropout_rate,training=training)
        batch_norm = partial(tf.layers.batch_normalization,training=training,momentum=0.9)
        net = tf.feature_column.input_layer(features,params['feature_columns'])
        #if type(hidden_layers) == tuple:
            #hidden_layers = list(hidden_layers[0])
        for units in hidden_layers:
            net_drop = dropout(net)
            if if_batch_norm:
                hidden = dense_layer(net_drop,units)
                bn = batch_norm(hidden)
                net = activation(bn)
            else:
                net = dense_layer(net_drop,units,activation=activation)
        
        logits_before_bn = dense_layer(net,n_classes)
        logits = batch_norm(logits_before_bn)
        #prediction 
        predicts = tf.arg_max(logits,1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                    'class_ids':predicts[:,tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits,}
            return tf.estimator.EstimatorSpec(mode,predictions=predictions)
        
        ##loss
        unweighted_base_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='base_loss')
        #tf.summary.scalar('unweighted base losses',tf.reduce_mean(unweighted_base_losses))
        base_losses = tf.reduce_mean(tf.multiply(tf.cast(sample_weights,dtype=tf.float32),unweighted_base_losses))
        #tf.summary.scalar('weighted base losses',base_losses)
        weight_max = tf.reduce_max(sample_weights)
        weight_min = tf.reduce_min(sample_weights)
        #tf.summary.scalar('max weight',weight_max)
        #tf.summary.scalar('min weight',weight_min)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_losses]+reg_loss,name='loss')
        
        #evaluation 
        if  n_classes <= 2:
            accuracy = tf.metrics.accuracy(labels=labels,predictions=predicts,name='acc_op')
            recall = tf.metrics.recall(labels=labels,predictions=predicts,name='recall_op')
            precision = tf.metrics.precision(labels=labels,predictions=predicts,name='precision_op')
            auc = tf.metrics.auc(labels,predicts,name='auc_op')
            f1 = tf.multiply(2.0,tf.divide(tf.multiply(recall,precision),tf.add(recall,precision),name='f1'))
            metrics = {'accuracy':accuracy,'recall':recall,'precision':precision,'auc':auc,'f1':f1}
        else:
            metrics = {'losses': base_losses}
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode,loss=base_losses,eval_metric_ops=metrics)
        
        #training
        assert mode == tf.estimator.ModeKeys.TRAIN
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
    
    @staticmethod
    def train_input_fn(data,labels,batch_size):  #data contains sample weights
        data = data.copy()
        #data['sample_weights'] = sample_weights  #sample weights go together with data
        dataset = tf.data.Dataset.from_tensor_slices((dict(data),labels))
        shuffle_len = int(len(labels)*2)
        return dataset.shuffle(shuffle_len).repeat().batch(batch_size)
    
    @staticmethod
    def eval_input_fn(data,labels,batch_size): #data contains sample_weights
        features = dict(data)
        if labels is None:
            inputs = features
        else:
            inputs = (dict(data),labels)
        
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        assert batch_size is not None
        return dataset.batch(batch_size)
    
    def __init__(self,home='/home/ec2-user/CpGPython/',**params):
        tf.logging.set_verbosity(tf.logging.WARN)
        self.home = home
        log_dir = home+'logs/'
        self.logger = Logger.Logger(log_dir,False).get_logger()
        self.tensorboard_log = home+'tensor_logs/'+datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.params = params
        self.scoring = self.params['scoring'] if 'scoring' in params else 'precision'

    
    def fit(self, X,y,sample_weight=None):
        X = pd.DataFrame(X).copy()
        sample_weight = pd.Series(sample_weight).copy()
        X['sample_weights'] = sample_weight
        y = pd.Series(y).copy()
        train_x, train_label = self.data_preprocessing(X,y)
        feature_cols = []
        for key in train_x.keys()[train_x.keys()!='sample_weights']:
            feature_cols.append(tf.feature_column.numeric_column(key=key))
        self.params['feature_columns'] = feature_cols
#        train_num = train_x.shape[0]
        self.estimator = tf.estimator.Estimator(model_fn=tensor_DNN.dnn_model,params=self.params,model_dir=self.tensorboard_log)
        n_steps = self.params['steps'] if 'steps' in self.params else 2000
        self.batch_size = self.params['batch_size'] if 'batch_size' in self.params else len(train_x)
        self.estimator.train(input_fn=lambda:tensor_DNN.train_input_fn(train_x,train_label,self.batch_size),steps=n_steps)    

#        sample_weights_train = np.power(sample_weights_train,1.5)
#        sample_weights_test = np.power(sample_weights_test,1.5)
        
        #test_label = test_label.astype('i8')

    def data_preprocessing(self,X,y=None):
        X = X.copy()
        if y is not None:
            y = y.copy()
            y = y.astype('i8')
            weight_factor = self.params['weight_factor'] if 'weight_factor' in self.params else 1 
            X['sample_weights'] = np.power(X['sample_weights'],weight_factor)
        pattern = ".*[+].*"
        reg = re.compile(pattern)
        for key in X.keys():
            if isinstance(key,str) and len(reg.findall(key))>0:
                key1 = key.replace('+','plus')
                X.rename({key:key1},axis=1,inplace=True)                
        return X,y
            
    
    def get_params(self,deep=True):
        return self.params
    
    def set_params(self,**params):
        self.params.update(params)
        return self
    
    def score(self,X,y,sample_weight=None):
        X = pd.DataFrame(X).copy()
        sample_weight = pd.Series(sample_weight).copy()
        X['sample_weights'] = sample_weight
        y = pd.Series(y).copy()
        test_x, test_label = self.data_preprocessing(X,y)
        self.eval_results = self.estimator.evaluate(input_fn=lambda:tensor_DNN.eval_input_fn(test_x,test_label,self.batch_size))
        #return 0.8*self.eval_results[self.scoring]+0.2*self.eval_results['recall']
        return self.eval_results[self.scoring]
    
    def predict_proba(self,X):
        X = pd.DataFrame(X).copy()
        test_x,_ = self.data_preprocessing(X)
        pred_results = self.estimator.predict(input_fn=lambda: tensor_DNN.eval_input_fn(test_x,None,self.batch_size)) 
        probs = []
        for predicts in pred_results:
            probs.extend([predicts['probabilities']])
        return np.array(probs)
    
    def predict(self,X):
        probs = self.predict_proba(X)
        return np.argmax(probs,axis=1)
    
    def evaluate(self,X,y,sample_weight=None,scorings=['precision','recall','auc']): #scoring is 'losses' for class_num >2
        X = pd.DataFrame(X).copy()
        sample_weight = pd.Series(sample_weight).copy()
        X['sample_weights'] = sample_weight
        y = pd.Series(y).copy()
        test_x, test_label = self.data_preprocessing(X,y)
        self.eval_results = self.estimator.evaluate(input_fn=lambda:tensor_DNN.eval_input_fn(test_x,test_label,self.batch_size))
        return {key:val for key,val in self.eval_results.items() if key in scorings }
##features selecetd by traditional methods
#with pd.HDFStore(home+'data/selected_features','r') as h5s:
#    train_x =h5s['train_x'] 
#    train_label = h5s['train_label'] 
#    test_x = h5s['test_x'] 
#    test_label = h5s['test_label']   
#    sample_weights_train = h5s['sample_weights_train'] 
#    sample_weights_test = h5s['sample_weights_test'] 

#logger.info('Features used in training are from traditional feature selection')  



 