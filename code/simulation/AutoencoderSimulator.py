#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:19:27 2017

@author: Xiaobo
"""

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from functools import partial
import tensorflow as tf
import numpy as np
from common import commons
home = commons.home
class dataset_simulator(BaseEstimator, TransformerMixin):
    def __init__(self,num=71*5,n_inputs=100,n_hidden1=500,n_hidden2=500,n_hidden3=200,learning_rate=0.001,n_epochs=100,batch_size=71):
        self.num = num
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.n_hidden4 = n_hidden2
        self.n_hidden5 = n_hidden1
        self.n_output = n_inputs
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.scaler= MinMaxScaler(feature_range=(0,1))
    
    def fit(self,X,y=None):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        scaled_X = self.scaler.fit_transform(X)
        dense_layer = partial(tf.layers.dense,activation=tf.nn.elu,kernel_initializer=initializer)
        X_holder = tf.placeholder(tf.float32,shape=[None,self.n_inputs])
        hidden1 = dense_layer(X_holder,self.n_hidden1)
        hidden2 = dense_layer(hidden1,self.n_hidden2)
        hidden3_mean = dense_layer(hidden2,self.n_hidden3,activation=None)
        hidden3_gamma = dense_layer(hidden2,self.n_hidden3,activation=None)
        noise = tf.random_normal(tf.shape(hidden3_gamma),dtype=tf.float32)
        hidden3 = tf.identity(hidden3_mean + tf.exp(0.5*hidden3_gamma)*noise,name='hidden3')
        tf.add_to_collection('hidden',hidden3)                                                             
        hidden4 = dense_layer(hidden3,self.n_hidden4)
        hidden5 = dense_layer(hidden4,self.n_hidden5)
        logits = dense_layer(hidden5,self.n_output,activation=None)
        outputs = tf.sigmoid(logits,name='output')
        tf.add_to_collection('hidden',outputs)
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X_holder,logits=logits)
        reconstruction_loss = tf.reduce_sum(xentropy)
        latent_loss = 0.5*tf.reduce_sum(tf.exp(hidden3_gamma)+tf.square(hidden3_mean)-1-hidden3_gamma)                             
        loss = reconstruction_loss+latent_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init.run()
            for epoch in range(self.n_epochs):
                for n_batch in range(X.shape[0]//self.batch_size):
                    X_batch = scaled_X[epoch*self.batch_size:epoch*self.batch_size+n_batch,:]
                    sess.run(train_op,feed_dict={X_holder:X_batch})
                loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X_holder: X_batch})
                print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)  # not shown
                saver.save(sess, home+"autoencoder_variation.ckpt")  # not shown
        return self
    
    def transform(self,X=None,y=None):
        saver = tf.train.Saver()
        hidden3 = tf.get_collection('hidden',scope='hidden3')[0]
        outputs = tf.get_collection('hidden',scope='output')[0]
        rdn_num = np.random.normal(size=[self.num,self.n_hidden3])
        with tf.Session() as sess:
            saver.restore(sess,homme+"autoencoder_variation.ckpt")
            sim = outputs.eval(feed_dict={hidden3: rdn_num})
            sim_data = self.scaler.inverse_transform(sim)
        return sim_data
            