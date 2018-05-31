#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:51:52 2017

@author: Xiaobo
"""

import pandas as pd
import numpy as np
from sklearn import utils 
import os
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
import re


#--------------------------
def nearest_tss(tss,sites_df):
    merged = pd.merge(sites_df,tss,how='outer',on=['chr','coordinate'])
    merged.sort_values(['chr','coordinate'],inplace=True)
    merged.rename(columns={'strand':'before_tss'},inplace=True)
    merged.ix[merged['before_tss'].isnull()==False, 'before_tss'] = merged.ix[merged['before_tss'].isnull()==False,'coordinate']
    merged['after_tss'] = merged['before_tss']
    merged['before_tss'].fillna(method='ffill', inplace=True)
    merged['after_tss'].fillna(method='bfill',inplace=True)
    merged['dist_to_before_tss'] = np.abs(merged['coordinate']-merged['before_tss'])
    merged['dist_to_after_tss'] = np.abs(merged['coordinate']-merged['after_tss'])
    merged['tss'] = None
    before_ix = (merged['dist_to_before_tss'] < merged['dist_to_after_tss']) | (merged['dist_to_after_tss'].isnull())
    merged.ix[before_ix,'tss'] = merged.ix[before_ix,'before_tss']
    after_ix = (merged['dist_to_before_tss'] >= merged['dist_to_after_tss']) | (merged['dist_to_before_tss'].isnull())
    merged.ix[after_ix,'tss'] = merged.ix[after_ix,'after_tss']
    merged['dist_to_nearest_tss'] = np.abs(merged['coordinate']-merged['tss']) 
    merged.dropna(axis=0,subset=['id'],inplace=True)
    return merged
#-----------------------------------------------------------------------    
def sampling(x,control,rate=100):
    index = x.index[0]
    bin_num = x['bin'][index]
    chr_num = x['chr'][index]
    control1 = control.query('(bin == @bin_num) & (chr == @chr_num)')
    sample_num = rate*len(x) if len(control1) >= rate*len(x) else len(control1)
    sample = control1.sample(sample_num,replace=False)
    return sample
#-----------------------------------------------------------------------
def rename_features(x):   #rename repetitive features
    features = np.array(x.columns)
    features_count = pd.Series(index=x.columns.unique())
    features_count = features_count.fillna(int(0))
    for i,name in enumerate(x.columns):
        if features_count[name] == 0:
            features_count[name] += 1
        else:
            features[i] = name+str(features_count[name])
            features_count[name] += 1
    x.columns = features
    return 

#------------------------------------------------------------------------
def convert_chr_to_num(data):
    data['chr'].where(data['chr']!='X','23',inplace=True)
    data['chr'].where(data['chr']!='Y','24',inplace=True)
    data['chr'].where(data['chr']!='M','25',inplace=True)
    data['chr'] = data['chr'].astype('i8')
    return data

#------------------------------------------------------------------------

       
dir='/home/ec2-user/CpGPython/'

#ewas_columns = ['EWAS_ID', 'ID_REF', 'chr','position','Gene_symbol','group1_valid_size','group2_valid_size','group1_average','group2_average','T_statistics','T_Pvalue']
#ewas = pd.read_csv(dir+'ewas/ewas_AD.txt',names=ewas_columns,header=None,sep='\t').reset_index()
ewas = pd.read_csv(dir+'ewas/ewas_web.csv',sep='$',header=None,skiprows=1,names=['ewas_id','id','chr','coordinate','gene',
			 'group1','group2','group1_beta','group2_beta',
			't-stats','pvalue'])

ewas['group'] = 1  # if norm is in group 2
ewas['group'][ewas['group1'].apply(lambda x: 'Normal' in x)] = -1 #if norm is in group1
ewas['adj-t-statistics'] = ewas['group']*ewas['t-stats']
ewas.drop('group',axis=1,inplace=True)
     
#ewas = ewas[[ 'ID_REF','chr','position','T_statistics']]
#ewas.rename({'position':'coordinate', 'ID_REF':'id'},inplace=True,axis=1)
ewas['label'] = 2     
ewas['label'][ewas['adj-t-statistics']<0] = 0
ewas['chr'] = ewas['chr'].astype('str')
ewas = ewas[['id','chr','coordinate','label']]

control = pd.read_csv(dir+'sites.txt',sep="\s+",header=0)
control.rename(columns={'coordinates':'coordinate'},inplace=True)
control.dropna(axis=0,inplace=True)
control = control.reset_index()
control.sort_values(['chr','coordinate'],inplace=True)
ewas_all = pd.read_csv(dir+'ewas/ewas_AD.txt',usecols=[2,3],names=['chr','coordinate'],sep='\t')
ewas_all.sort_values(['chr','coordinate'],inplace=True)
ewas_all.drop_duplicates(inplace=True)
overlaps_index = pd.merge(control,ewas_all,on=['chr','coordinate'],how='inner')['index']
control.set_index('index',inplace=True)
control.drop(overlaps_index,inplace=True)

positive = pd.read_csv(dir+'PosCpG_diff.csv',usecols=[0,1,2,3],names=['id','chr','coordinate','score'])
positive['chr']=positive['chr'].astype('str')
positive['label'] = 2
positive['label'].where(positive['score']>0,0,inplace=True)  ## if AD>control, positive and assign 1, otherwise assign 0
positive.drop('score',axis=1,inplace=True)

stg = pd.read_excel(dir+'superior_temporal_gyrus_sites.xlsx',usecols=[0,1,2,3,13,14,15],names=['id','chr','start','end','state','pvalue','qvalue'],
                    skiprows=3,header=None) #superior temporal gyrus dataset
stg_pos = stg.query('state=="Hyper"')
stg_neg = stg.query('state=="Hypo"')
stg_pos.sort_values(['pvalue'],inplace=True)
stg_neg.sort_values(['pvalue'],inplace=True)
stg_positive = stg_pos.head(int(stg_pos.shape[0]*0.1))
stg_positive['label'] = 2
stg_negative = stg_neg.head(int(stg_neg.shape[0]*0.1))
stg_negative['label'] = 0
stg_control = pd.concat([stg_pos.tail(int(stg_pos.shape[0]*0.1)),stg_neg.tail(int(stg_neg.shape[0]*0.1))],ignore_index=True)
stg_control['label'] = 1
stg_control['coordinate'] = ((stg_control['start']+stg_control['end'])/2).astype('i8')
stg_diff = pd.concat([stg_positive,stg_negative],ignore_index=True)
stg_diff['coordinate'] = ((stg_diff['start']+stg_diff['end'])/2).astype('i8')
       
cols=['chr', 'coordinates','strand']
tss =  pd.read_csv(dir+'tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss['chr'] = tss['chr'].str[3:]
tss.rename(columns={'coordinates':'coordinate'},inplace=True)

ewas = ewas.sort_values(['chr','coordinate'])
control = control.sort_values(['chr','coordinate'])
positive = positive.sort_values(['chr','coordinate'])

tss = tss.sort_values(['chr','coordinate'])
stg_diff.sort_values(['chr','coordinate'],inplace=True)
stg_control.sort_values(['chr','coordinate'],inplace=True)
stg_diff['chr'] = stg_diff['chr'].apply(lambda x: x[3:])
stg_control['chr'] = stg_control['chr'].apply(lambda x: x[3:])

ewas_tss = nearest_tss(tss,ewas)
ewas_tss = ewas_tss[ewas_tss['chr'].notnull()]
ewas_tss = convert_chr_to_num(ewas_tss)

positive_tss = nearest_tss(tss,positive)
positive_tss['chr'] = positive_tss['chr'].astype('i8')

stg_diff_tss = nearest_tss(tss,stg_diff)
stg_control_tss = nearest_tss(tss,stg_control)
stg_diff_tss = convert_chr_to_num(stg_diff_tss)
stg_control_tss = convert_chr_to_num(stg_control_tss)



overlap_tss = pd.merge(positive_tss,ewas_tss,on=['chr','coordinate'],how='left')
missing_ix = overlap_tss[overlap_tss.isnull().sum(axis=1)>0].index
missings = overlap_tss.ix[missing_ix,:].dropna(axis=1)
missings.columns = [x if '_x' not in x  else x[:len(x)-2] for x in missings.columns]
ewas_tss = ewas_tss.append(missings,ignore_index=True)
ewas_tss.sort_values(['chr','coordinate'],inplace=True)
 
control_tss = nearest_tss(tss, control)
control_tss = control_tss[control_tss['chr'].notnull()]
control_tss = convert_chr_to_num(control_tss)
ewas_tss['bin'],boundaries = pd.qcut(ewas_tss['dist_to_nearest_tss'],10,retbins=True,labels=np.arange(10))

boundaries[0]= -np.Inf
boundaries[-1] = np.Inf

control_tss['bin'] = pd.cut(control_tss['dist_to_nearest_tss'],boundaries,labels=np.arange(10))  
control_samples = ewas_tss.groupby(['chr','bin']).apply(lambda x: sampling(x,control_tss))
control_samples.sort_values(['chr','coordinate'],inplace=True)
ewas_tss.sort_values(['chr','coordinate'],inplace=True)
ewas_tss['start'] = (ewas_tss['coordinate']/200).apply(lambda x: int(np.ceil(x-1))*200+1)
control_samples['start'] = (control_samples['coordinate']/200).apply(lambda x: int(np.ceil(x-1))*200+1)

stg_diff_tss['start'] = (stg_diff_tss['coordinate']/200).apply(lambda x: int(np.ceil(x-1))*200+1)
stg_control_tss['start'] = (stg_control_tss['coordinate']/200).apply(lambda x: int(np.ceil(x-1))*200+1)
#-----------------------
#load feature coordinates
#-------------
wincols=['chr','start','end']
feature_wins = pd.read_csv(dir+'wins.txt',sep='\s+',usecols=[0,1,2],header=None,names=wincols,skiprows=1)
feature_wins.reset_index(inplace=True)
feature_wins['index'] = feature_wins['index']+1
kept_chr = []
for i in np.arange(1,23):
    kept_chr.extend(['chr'+str(i)])
kept_chr.extend(['chrX','chrY'])

feature_wins = feature_wins.query('chr in @kept_chr')
feature_wins['chr'] = feature_wins['chr'].apply(lambda x: x[3:])
feature_wins = convert_chr_to_num(feature_wins)   
feature_wins.sort_values(['chr','start'],inplace=True)

#-----------------------------------
#Merge feature coordinates with positive/negative sites
#-----------------------------------
ewas_bin = pd.merge(ewas_tss,feature_wins, on=['chr','start'],how='left')
ewas_bin.rename(columns={'index':'winid'},inplace=True)
control_bin = pd.merge(control_samples,feature_wins, on =['chr','start'],how='left')
control_bin.rename(columns={'index':'winid'},inplace=True)

stg_diff_bin = pd.merge(stg_diff_tss,feature_wins,on=['chr','start'],how='left')
stg_diff_bin.rename(columns={'index':'winid'},inplace=True)
stg_control_bin = pd.merge(stg_control_tss,feature_wins,on=['chr','start'],how='left')
stg_control_bin.rename(columns={'index':'winid'},inplace=True)
stg_diff_bin = stg_diff_bin.dropna(axis=0).reset_index().drop(['index'],axis=1)
stg_control_bin = stg_control_bin.dropna(axis=0).reset_index().drop(['index'],axis=1)
#--------------------
#export positive/negative winid for R to load and find corresponding features for export
#--------------------

ewas_bin['winid'].to_csv(dir+'pos_winid.csv',index=False)
control_bin['winid'].to_csv(dir+'neg_winid.csv',index=False)

stg_diff_bin['winid'].to_csv(dir+'pos_winid.csv',index=False)
stg_control_bin['winid'].to_csv(dir+'neg_winid.csv',index=False)

#import features of all positive sites
feature_dir = dir+'data/features/'
files = os.listdir(feature_dir)
pattern = '.*Pos.csv$'
reg = re.compile(pattern)
files = [name for name in files if len(reg.findall(name))>0]


for file in files:    
    feature = pd.read_csv(feature_dir+file)
    print(len(feature.columns))
#    ewas_bin = pd.concat([ewas_bin,feature],axis=1)
    stg_diff_bin = pd.concat([stg_diff_bin,feature],axis=1)
    
rename_features(ewas_bin)
rename_features(stg_diff_bin)

ewas_bin['bin'] = ewas_bin['bin'].astype('int64')
h5s = pd.HDFStore(dir+'/data/ewas_features','w')
h5s['ewas_features'] = ewas_bin
h5s.close()
with pd.HDFStore(dir+'/data/stg_features','w') as h5s:
    h5s['stg_features'] = stg_diff_bin

ewas_bin.drop(ewas_bin.columns[[0,4,5,6,7,8,10,11,12,13]],inplace=True,axis=1)
stg_diff_bin.drop(stg_diff_bin.columns[[0,1,2,3,4,5,6,8,9,10,11,12,13,15,16]],inplace=True,axis=1)               
#import features of all negative sites

files = os.listdir(feature_dir)
pattern = '.*neg.csv$'
reg = re.compile(pattern)
files = [name for name in files if len(reg.findall(name))>0]
# files = list(name for name in itertools.ifilter(lambda x: len(reg.findall())>0,files)) 
for file in files:    
    feature = pd.read_csv(feature_dir+file)
#    control_bin = pd.concat([control_bin,feature],axis=1)
    stg_control_bin = pd.concat([stg_control_bin,feature],axis=1)
    
rename_features(control_bin)
rename_features(stg_control_bin)

control_bin['label'] = 1
control_bin['bin'] = control_bin['bin'].astype('i8')
h5s = pd.HDFStore(dir+'/data/negative_features','w')
h5s['control_features'] = control_bin
h5s.close()

with pd.HDFStore(dir+'/data/negative_features','w') as h5s:
    h5s['stg_control_features'] = stg_control_bin

control_bin.drop(control_bin.columns[[0,3,4,5,6,7,9,10,11,12]],axis=1,inplace=True)
stg_control_bin.drop(stg_control_bin.columns[[0,1,2,3,4,5,6,8,9,10,11,12,13,15,16]],axis=1,inplace=True)

#--------------------
#merge positive and negative data 
#---------------------

all_features = pd.concat([ewas_bin,control_bin],ignore_index=True)
stg_all_features = pd.concat([stg_diff_bin,stg_control_bin],ignore_index=True)
h5s = pd.HDFStore(dir+'/data/all_features','w')
h5s['all_features'] = all_features
h5s.close()
with pd.HDFStore(dir+'/data/all_features','w') as h5s:
    h5s['all_features'] = stg_all_features
       

###############################################################################
#############################################################################
                

