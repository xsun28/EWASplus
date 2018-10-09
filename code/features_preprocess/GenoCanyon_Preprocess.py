#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:18:44 2018

@author: Xiaobo

Run the GenoCanyon_Preprocess.R script first to get the scores on desired locations
"""

import pandas as pd
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage

##################################################

class GenoCanyon_Preprocess(object):
    
    def __init__(self,data_dir=extra_storage+'GenoCanyon/Results/', sites_file = home+'data/all_sites_winid.csv',additional_feature_file = home+'data/features/addtional_features'):        
        self.data_dir = data_dir
        self.sites_file = sites_file
        self.additional_feature_file = additional_feature_file
    
    def process(self,score_file='selected_site_scores.txt'):
        all_sites = pd.read_csv(self.sites_file)
        scores = pd.read_csv(self.data_dir+score_file,header=None)
        genocanyon_scores = pd.DataFrame(all_sites[['chr','coordinate','winid']])
        genocanyon_scores['genocanyon_score'] = scores
        
        with pd.HDFStore(self.additional_feature_file,'a') as h5s:
            h5s['GenoCanyon'] = genocanyon_scores
                

#----------------------------------------




