#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:16:26 2017

@author: Xiaobo
"""
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import Convert_To_Normal_Dist as ctnd
import TTest as tt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
from matplotlib import pyplot as plt
import CorrFeatureSelector as cfs
from sklearn.preprocessing import StandardScaler
import AutoencoderSimulator as AS
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from functools import partial 
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import Feature_Selection as FS
import DataScaler as ds

#------------------------------------------------------------------------------

#--------------------------------------------------------------------- 
def normalize_feature(all_features):
    count_thresh = 10
    unique_value_counts = [len(all_features[col].unique()) for col in all_features.columns[2:]] #ignore chr and coordinate
    normalize_all_features = all_features.copy()
    mappings = pd.DataFrame()
    for col,num in zip(all_features.columns[2:],unique_value_counts):
        if num >= count_thresh:
            print(col)
            atn = ctnd.AnyToNormal(col)
            mapping = atn.transform(normalize_all_features)
            mappings[mapping.columns[0]] = mapping.ix[:,0]
            mappings[mapping.columns[1]] = mapping.ix[:,1]
        else: 
            continue
    h5s = pd.HDFStore('/home/ec2-user/CpGPython/data/normalize_mapping','w')
    h5s['mappings'] = mappings
    h5s['normal_features'] = normalize_all_features
    h5s.close()
    return '/home/ec2-user/CpGPython/data/normalize_mapping',normalize_all_features,mappings
   
#-------------------------------------------------------------------------------
def data_selection(data,classes=[0,1,2],*args):              #which classes of data to keep
    if not isinstance(data,pd.DataFrame):
        cols = args
        data = pd.DataFrame(data,columns=cols)    
    ret_data = data.query('label in @classes')
    for i,label in enumerate(classes):
        ret_data[ret_data['label']==label] = i
    return ret_data

#------------------------------------------------------------------------------                  
 
def TSNEPlot(data,class_labels,param_map):##param_map like 1:('r','o','80')
    data1 = data.reset_index()
    tsne = TSNE(n_components=3,random_state=91)
    feature_reduced = tsne.fit_transform(data1.drop(['label','index'],axis=1))
    fig = plt.figure(figsize=(18,15))
    ax = fig.gca(projection='3d')
    for i in class_labels:
        c,marker,size = param_map[i]
        print(c,marker,size)
        index = data1.query('label == @i').index.values
        ax.scatter(feature_reduced[index,0],feature_reduced[index,1],feature_reduced[index,2],c=c,marker=marker,s=size)
    plt.show()
    return None

#-------------------------------------------------------------------------------
def simulate(data,label,**argkw):
    data = data[data['label']==label]
    encoder = AS.dataset_simulator(**argkw)
    encoder.fit(np.array(data.drop('label',axis=1)))
    sim_data = pd.DataFrame(encoder.transform(),columns=data.columns.drop('label'))
    sim_data['label'] = label
    return sim_data

#-----------------------------------------------------------------------------
def train_test_split(data):
    total_dataset = data.copy()
    total_dataset = total_dataset.reset_index().drop('index',axis=1)   #reset index or split below will generate filtered index and NAN values
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=17)
    for train_index, test_index in split.split(total_dataset,total_dataset['label']):
        train_set = total_dataset.ix[train_index]
        test_set = total_dataset.ix[test_index]
    scaler = ds.DataScaler(scaler='standard')
    train_x = scaler.fit_transform(train_set[train_set.columns.drop('label')])
    train_label = train_set['label']
    test_x = scaler.transform(test_set[test_set.columns.drop('label')])
    test_label = test_set['label']
    return train_x,train_label,test_x,test_label              
#-----------------------------------------------------------------------------
def subset_control(data,num):
    ewas_index = data.query('label!=0').index.values
    control_sub_index = np.random.permutation(data.query('label==0').index.values)[:len(ewas_index)*num]
    select_index = np.concatenate((ewas_index,control_sub_index))
    return data.loc[select_index,:]


################################################################################
dir='/home/ec2-user/CpGPython/'
h5s = pd.HDFStore(dir+'/data/all_features','r')
all_data = h5s['all_features']
h5s.close()
all_data['coordinate'] = all_data['coordinate'].astype('i8')
all_data.drop(['coordinate','chr'],axis=1,inplace=True)
all_data['dist_to_nearest_tss'] = all_data['dist_to_nearest_tss'].astype('i8')
all_data = data_selection(all_data,classes=[0,1,2])
#all_features = subset_control(all_data,30)
all_features = all_data ##only for superior temporal gyrus dataset
#------------------------------------------------------------------------------
#split train test data and scaling on train data
train_x,train_label,test_x,test_label = train_test_split(all_features)

#----------------------------------------------------------------
#feature selection based on train dataset
fs = FS.FeatureSelection(class_weights={0:2,1:1,2:2},methods=['logistic_regression','random_forest','linear_SVC'],C=0.01)
fs.fit()
selected_features = fs.transform(train_x,train_label)
reduced_train_x = train_x[selected_features['feature']]
reduced_test_x = test_x[selected_features['feature']]
plot_data = reduced_train_x.copy()
plot_data['label'] = train_label
TSNEPlot(plot_data,class_labels=[0,1,2],param_map={0:('g','^',20),1:('b','*',20),2:('r','o',20)})
with pd.HDFStore(dir+'data/selected_features','w') as h5s:
    h5s['train_x'] = reduced_train_x
    h5s['train_label'] = train_label
    h5s['test_x'] = reduced_test_x
    h5s['test_label'] = test_label   

##using t-test
#pos_set = all_features.query('label==1')
#neg_set = all_features.query('label==0')
#stats_pvalues = []
#for col in all_features.columns[:-1]:
#    stats_pvalues.extend([tt.FeatureTTest(col).transform(pos_set,neg_set)])
#
#stats_p_values_df = pd.DataFrame(stats_pvalues,index=all_features.columns[:-1])    
#stats_p_values_df.sort_values('pvalue',inplace=True)
#t_selected_features = stats_p_values_df.head(100).index


#-------------------------------------------------------------------------

                    
#simulate class=1 data
sample_num = (all_features['label']==2).sum()
sim_pos_data = simulate(all_features,label=2,num=sample_num*3,n_inputs=all_features.columns.shape[0]-1,
                        batch_size=sample_num,n_hidden1=500,n_hidden2=500,
                        n_hidden3=200,learning_rate=0.001,n_epochs=100)
sim_pos_data['label'] = 1
pos_data = all_features.query('label==2')
merged_data = pd.concat([sim_pos_data,pos_data],ignore_index=True)
TSNEPlot(merged_data,class_labels=[1,2],param_map={1:('b','*',80),2:('r','o',80)})
sim_pos_data['label'] = 2  
            
#Split dataset into training and testing using StratifiedShuffleSplit


#combined_train_set = pd.concat([train_set,sim_data],ignore_index=True)
#train_x = combined_train_set[combined_train_set.columns.drop('label')]
#train_label = combined_train_set['label']

