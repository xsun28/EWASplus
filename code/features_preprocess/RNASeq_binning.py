#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:50:08 2018

@author: Xiaobo
"""


import multiprocessing as mp
import subprocess
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import BED_binning
import os
import re
import gc
import pandas as pd

def RNASeq_Preprocessing(file):
    data_dir = extra_storage+'RNASeq/'
    script_path = data_dir+'RNASeq_download_to_bed.sh'
    RNASeq_h5s = home+'data/RNASeq/single_file/'
    single_file = file.split('/')[-1].split('.')[0]
    if not os.path.exists(data_dir+single_file+'.bed'):
        subprocess.check_call([script_path,file,data_dir])
    else:
        print(single_file+'.bed already exists')
    if os.path.exists(RNASeq_h5s+single_file):
        print(single_file+" already exists, skipping")
    else:
        print("binning "+single_file+" ...")
        rnaseq_binning = BED_binning.BED_binning(data_type='RNASeq',data_dir=extra_storage+'RNASeq/',output=RNASeq_h5s)
        rnaseq_binning.binning(single_file=single_file)
        gc.collect()
    if os.path.exists(data_dir+single_file+'.bam'):
        os.remove(data_dir+single_file+'.bam')
    #os.remove(data_dir+single_file+'.bed')


def process_meta_info_table(meta_info_table,downloaded_files):
    print('downloaded fies are: '+str(downloaded_files))
    meta_data_df = pd.read_csv(meta_info_table, sep='\t')
    selected_meta_data_df = meta_data_df[(meta_data_df['File format']=='bam') & (meta_data_df['Output type']=='alignments') & (meta_data_df['Assembly']=='hg19')]
    file_exp_dict = {}
    for ind, row in selected_meta_data_df.iterrows():
        if row['File accession'] in downloaded_files:
            if not row['Experiment accession'] in file_exp_dict.keys():
                file_exp_dict[row['Experiment accession']] = [] 
            file_exp_dict[row['Experiment accession']].append(row['File accession'])
    return file_exp_dict    
    
    
data_dir = extra_storage+'RNASeq/'
pool = mp.Pool(processes=2)
files = [line.rstrip('\n') for line in open(data_dir+'files.txt','r')]
pool.map(RNASeq_Preprocessing,files)

####combine RNASeq samples from same experiment
print('combine RNASeq features by expriments')
exp_dir = home+'data/RNASeq/'
h5s_dir = home+'data/RNASeq/single_file/'
print('combined experiment output dir: '+exp_dir)
meta_file = data_dir+'metadata_rnaseq.tsv'
files = [file.split('/')[-1].split('.')[0] for file in files]
file_exp_dict = process_meta_info_table(meta_file,files)
print(file_exp_dict)
for exp_key in file_exp_dict:
    print("Processing %s" % exp_key)
    if os.path.exists(os.path.join(exp_dir, exp_key)):
        print('experiment: '+exp_key+' already exists, skipping...')
        continue

    file_list = file_exp_dict[exp_key]
    files_count = len(file_list)
    print("%i files for this exp" % files_count)
    for file_ind in range(files_count):
        print(file_list[file_ind])
        if (file_ind==0):
            res_df = pd.read_hdf(os.path.join(h5s_dir, file_list[file_ind]))
            res_df.set_index('winid', inplace=True)
        else:
            tmp_df = pd.read_hdf(os.path.join(h5s_dir, file_list[file_ind]))
            tmp_df.set_index('winid', inplace=True)
            res_df = res_df.join(tmp_df, how='outer')
    res_df.fillna(0, inplace=True)
    res_df = res_df.mean(axis=1).to_frame().reset_index()
    res_df.columns = ["winid", "%s_RNASeq_counts"%exp_key]
    with pd.HDFStore(os.path.join(exp_dir, exp_key),'w') as h5s:
        h5s[exp_key] = res_df   


#######run binning file by file for server with limit memory
#files = os.listdir(data_dir)
#pattern = '.*\.bed$'
#reg = re.compile(pattern)
#files = [f for f in files if len(reg.findall(f))>0]
#RNASeq_h5s = home+'data/RNASeq/'
#for f in files:
#    single_file = f.split('/')[-1].split('.')[0]
#    if os.path.exists(RNASeq_h5s+single_file):
#        print(single_file+" already exists, skipping")
#        continue
#    print("binning "+f+" ...")
#    rnaseq_binning = BED_binning.BED_binning(data_type='RNASeq',data_dir=data_dir,output=RNASeq_h5s)    
#    rnaseq_binning.binning(single_file=single_file)
#    gc.collect()