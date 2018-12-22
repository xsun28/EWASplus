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

def RNASeq_Preprocessing(file):
    data_dir = extra_storage+'RNASeq/'
    script_path = data_dir+'RNASeq_download_to_bed.sh'
    single_file = file.split('/')[-1].split('.')[0]
    if not os.path.exists(data_dir+single_file+'.bed'):
        subprocess.check_call([script_path,file,data_dir])
    else:
        print(single_file+'.bed already exists')
    os.remove(data_dir+single_file+'.bam')
    #os.remove(data_dir+single_file+'.bed')

data_dir = extra_storage+'RNASeq/'
print('here')
pool = mp.Pool(processes=2)
files = [line.rstrip('\n') for line in open(data_dir+'files.txt','r')][1:]
pool.map(RNASeq_Preprocessing,files[3:])

files = os.listdir(data_dir)
pattern = '.*\.bed$'
reg = re.compile(pattern)
files = [f for f in files if len(reg.findall(f))>0]
RNASeq_h5s = home+'data/RNASeq/'
for f in files:
    single_file = f.split('/')[-1].split('.')[0]
    if os.path.exists(RNASeq_h5s+single_file):
        print(single_file+" already exists, skipping")
        continue
    print("binning "+f+" ...")
    rnaseq_binning = BED_binning.BED_binning(data_type='RNASeq',data_dir=data_dir,output=RNASeq_h5s)    
    rnaseq_binning.binning(single_file=single_file)
    gc.collect()