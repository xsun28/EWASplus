#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:27:39 2018
@author: Xiaobo
"""

import pandas as pd
import numpy as np
import sys
import os
from common import commons
home = commons.home
logger = commons.logger
extra_storage = commons.extra_storage
from features_preprocess import get_winid
import pysam
from features_preprocess.get_winid import convert_num_to_chrstr,convert_chrstr_to_num

#----------------------------------------------------

#
class Eigen_Preprocess(object):
    def __init__(self,data_dir = extra_storage+'Eigen/',sites_file = home+'data/commons/all_sites_winid.csv',additional_feature_file = home+'data/features/addtional_features'):
        self.data_dir = data_dir
        self.sites_file = sites_file
        self.additional_feature_file = additional_feature_file 
        logger.info('Process Eigen features for sites in file {}, to be output to {}'.format(sites_file,additional_feature_file))

    
    def process(self):
        all_sites = pd.read_csv(self.sites_file)
        all_sites = get_winid.convert_chr_to_num(all_sites)
        #all_sites.sort_values(['chr','coordinate'],inplace=True)

        #reg = re.compile('^Eigen.*bgz$')
        #reg1 = re.compile('chr[0-9]{1,2}')
        #files = os.listdir(data_dir)
        #files = [f for f in files if (len(reg.findall(f))>0) and (reg1.findall(f)[0][3:] in chrs)]
        eigen_scores = []
        i = 0
        for site in all_sites.values:
            #raw_scores_one_site = []
            phred_one_site = []
            #pc_raw_scores_one_site = []
            pc_phred_one_site = []
            chrm = convert_num_to_chrstr(int(site[1]))
            pos = int(site[2])
            left = pos
            right = pos-1
            eigen_file = self.data_dir+'Eigen_hg19_noncoding_annot_chr'+chrm+'.tab.bgz'
            tabix = pysam.Tabixfile(eigen_file)
            while len(phred_one_site) == 0:
                left = left-1
                right = right+1
                for row in tabix.fetch(chrm,left,right,parser=pysam.asTuple()):
                    #raw_scores_one_site.extend([float(row[-4])])
                    phred_one_site.extend([float(row[-3])])
                    #pc_raw_scores_one_site.extend([float(row[-2])])
                    pc_phred_one_site.extend([float(row[-1])])
            #average_raw = np.mean(raw_scores_one_site)
            #max_raw = np.max(raw_scores_one_site)
            average_phred = np.mean(phred_one_site)
            max_phred = np.max(phred_one_site)
            #average_pc_raw = np.mean(pc_raw_scores_one_site)
            #max_pc_raw = np.max(pc_raw_scores_one_site)
            average_pc_phred = np.mean(pc_phred_one_site)
            max_pc_phred = np.max(pc_phred_one_site)
            eigen_scores.extend([[convert_chrstr_to_num(chrm),pos,max_phred,average_phred,max_pc_phred,average_pc_phred]])
            #eigen_scores.extend([[chrm,pos,max_raw,average_raw,max_phred,average_phred,max_pc_raw,average_pc_raw,max_pc_phred,average_pc_phred]])
            i += 1
            if i%1000 == 0:
                #print([chrm,pos,max_raw,average_raw,max_phred,average_phred,max_pc_raw,average_pc_raw,max_pc_phred,average_pc_phred])
                logger.info('Eigen raw file for chromsome {} is {}'.format(chrm,eigen_file))
                logger.info('Processed {} sites...'.format(i))
                #print([chrm,pos,max_phred,average_phred,max_pc_phred,average_pc_phred]) 
                
        with pd.HDFStore(self.additional_feature_file,'a') as h5s:
            #h5s['Eigen'] = pd.DataFrame(eigen_scores,columns=['chr','coordinate','eigen_max_raw','eigen_avg_raw','eigen_max_phred','egien_avg_phred','eigen_max_pc_raw','eigen_avg_pc_raw','eigen_max_pc_phred','egien_avg_pc_phred'])
            h5s['Eigen'] = pd.DataFrame(eigen_scores,columns=['chr','coordinate','eigen_max_phred','egien_avg_phred','eigen_max_pc_phred','egien_avg_pc_phred'])
            logger.info('Eigen features of sites in {} are outputted to {}'.format(self.sites_file,self.additional_feature_file))



#data_dir = '/Users/Xiaobo/Desktop/test.tsv'
#sites_file = '/Users/Xiaobo/Jobs/CpG/data/all_sites_winid.csv'