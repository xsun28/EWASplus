#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:20:36 2018

@author: Xiaobo
"""


import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator,TransformerMixin
from functools import partial
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from datetime import datetime
from common import commons
home = commons.home

class sparse_autoencoder(BaseEstimator,TransformerMixin):

    def __init__(self,home=home,**params):
        self.params = params
        self.model_path = home+'tensor_model/sparse_autoencoder.ckpt'
        self.root_log_dir = home+'tensor_logs/'
        self.sparsity_target = params['sparsity_target'] if 'sparsity_target' in params else 0.1
        self.sparsity_weight = params['sparsity_weight'] if 'sparsity_weight' in params else 0.2
        self.hidden_layers = params['hidden_layers'] 
        self.n_epochs = params['n_epochs'] if 'n_epochs' in params else 50
        self.n_batch = params['n_batch'] if 'n_batch' in params else 20      
        self.l2_reg = params['l2_reg'] if 'l2_reg' in params else 0.01   
        self.top_k = params['top_k'] if 'top_k' in params else 50 
        
    def kl_divergence(self,p,q):
        return p*tf.log(p/q)+(1-p)*tf.log((1-p)/(1-q))
    
    
    def get_learning_rate(self,initial_rate=0.1,decay_steps=10000,decay_rate=1/10):
        global_step = tf.Variable(0,trainable=False,name='global_step')
        learning_rate = tf.train.exponential_decay(initial_rate,global_step,decay_steps,decay_rate)
        return learning_rate

    def fit(self,X,y=None,sample_weight=None):
        tf.reset_default_graph()
        log_dir = "{}run-{}".format(self.root_log_dir,datetime.utcnow().strftime("%Y%m%d%H%M%S"))
        log_writer = tf.summary.FileWriter(log_dir,tf.get_default_graph())
        n_input = X.shape[1]
        n_output = n_input
        train_x = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(X)   
        sample_weights_train = sample_weight
        regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
        he_init = tf.contrib.layers.variance_scaling_initializer()
        dense_layer = partial(tf.layers.dense,kernel_regularizer=regularizer,kernel_initializer=he_init)
        inputs = tf.placeholder(tf.float32,shape=(None,n_input),name='inputs')
        tf.add_to_collection('inputs',inputs)
        sample_weights = tf.placeholder(tf.float32,shape=(None),name='weights')
        tf.add_to_collection('weights',sample_weights)
        training = tf.placeholder_with_default(False,shape=(),name='training')
        
        with tf.name_scope("dnn"):
            net = inputs
            num_hidden_layer = len(self.hidden_layers)
            for i,units in enumerate(self.hidden_layers):
                hidden = dense_layer(net,units)
                bn = tf.layers.batch_normalization(hidden,training=training,momentum=0.9)
                if i < num_hidden_layer-1:
                    net = tf.nn.sigmoid(bn)
            new_feature = tf.nn.sigmoid(bn,name='new_feature')
            tf.add_to_collection('new_feature',new_feature)
            net_mean = tf.reduce_mean(net,axis=0)
            net = new_feature
            for units in self.hidden_layers[-2::-1]:
                hidden = dense_layer(net,units)
                bn = tf.layers.batch_normalization(hidden,training=training,momentum=0.9)
                net = tf.nn.sigmoid(bn)
            logits_before_bn = dense_layer(net,n_output,name='outputs',activation=None)
            logits = tf.layers.batch_normalization(logits_before_bn,training=training,momentum=0.9)
      
        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)    
            reconstruction_loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=inputs,logits=logits,pos_weight=sample_weights),name='reconstruction_loss')
            #unweighted_base_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=logits)
            #base_losses = tf.reduce_mean(tf.multiply(tf.cast(sample_weights,dtype=tf.float32),unweighted_base_losses)) 
            #tf.add_to_collection('reconstruction_loss',base_losses)
            tf.add_to_collection('reconstruction_loss',reconstruction_loss)
            sparsity_loss = tf.reduce_sum(self.kl_divergence(self.sparsity_target,net_mean))
            loss = tf.add_n([reconstruction_loss,sparsity_loss]+reg_loss,name='loss')
            reconstruction_loss_summary = tf.summary.scalar('logloss',reconstruction_loss)

        with tf.name_scope('train'):
            learning_rate = self.get_learning_rate()
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True)
            training_op = optimizer.minimize(loss)
        
        init = tf.global_variables_initializer()
        batch_size = train_x.shape[0]//self.n_batch
        saver = tf.train.Saver() 
        sample_weights_train.reset_index(drop=True,inplace=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.Session() as sess:
            
            init.run()
            for epoch in range(self.n_epochs):
                print('epoch: {}'.format(epoch))
                random_ix = np.random.permutation(train_x.shape[0]) 
                for iteration in range(self.n_batch):
                    start_ix = iteration*batch_size 
                    end_ix = np.minimum((iteration+1)*batch_size,train_x.shape[0])
                    ixs = random_ix[start_ix:end_ix]
                    x_batch = train_x[ixs,:]
                    weights_batch = sample_weights_train[ixs].values.reshape([-1,1])
                    if iteration == 19:
                        summary_str = reconstruction_loss_summary.eval(feed_dict={inputs:x_batch,sample_weights:weights_batch,training:True})
                        step = self.n_epochs*self.n_batch
                        log_writer.add_summary(summary_str,step)
                    #print(loss.eval(feed_dict={inputs:x_batch,sample_weights:weights_batch,training:True}))
                   # print(hidden1_mean.eval(feed_dict={inputs:x_batch,sample_weights:weights_batch,training:True}))
                    sess.run([training_op,update_ops],feed_dict={inputs:x_batch,sample_weights:weights_batch,training:True})
            saver.save(sess,self.model_path)
      
    
    def transform(self,X,y=None,sample_weight=None):
        scaled_inputs = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(X)
        weights = sample_weight.values.reshape([-1,1])
        saver = tf.train.Saver()
        new_feature = tf.get_collection('new_feature')[0]  ##for variable specified under a name_scope, don't specify the scope parameter.
        inputs = tf.get_collection('inputs',scope='inputs')[0]
        sample_weights = tf.get_collection('weights',scope='weights')[0]
        with tf.Session() as sess:
            saver.restore(sess,self.model_path)
            new_features = new_feature.eval(feed_dict={inputs:scaled_inputs,sample_weights:weights})
            print(new_features)
        new_features_mean = np.mean(new_features,axis=0)
        max_active_ix = np.argpartition(new_features_mean,self.top_k)[:-self.top_k-1:-1]
        new_features1 = new_features[:,max_active_ix]
        new_features1 = StandardScaler().fit_transform(new_features1)
        return new_features1
    
    def score(self,X,y=None,sample_weight=None):
        scaled_inputs = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(X)
        weights = sample_weight.values.reshape([-1,1])        
        reconstruction_loss = tf.get_collection('reconstruction_loss')[0]  ##for variable specified under a name_scope, don't specify the scope parameter.
        inputs = tf.get_collection('inputs',scope='inputs')[0]
        sample_weights = tf.get_collection('weights',scope='weights')[0]
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,self.model_path)
            loss = reconstruction_loss.eval(feed_dict={inputs:scaled_inputs,sample_weights:weights})
            print('reconstruction loss is: '+str(loss))
        return -loss
    
        
    def get_params(self,deep=True):
        return self.params
    
    def set_params(self,**params):
        self.params.update(params)
        return self
            