#running script: run as python all450k_feature_preprocess.py -r False
import os
import sys
import pandas as pd
import numpy as np
from common import commons
home = commons.home
extra_storage = commons.extra_storage
dataset = commons.dataset
logger = commons.logger
from features_preprocess import BED_binning
from features_preprocess import BED_Preprocess,CADD_Preprocess,DANN_Preprocess, Eigen_Preprocess,GenoCanyon_Preprocess,WGBS_preprocess,GWAVA_Preprocess
import subprocess
import argparse
import gc
from features_preprocess import get_winid
from common.commons import rename_features,check_genocaynon
import re
from sklearn.externals import joblib


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
    merged.drop(['before_tss','after_tss','tss','dist_to_before_tss','dist_to_after_tss'],axis=1,inplace=True)
    merged.dropna(axis=0,subset=['id'],inplace=True)
    return merged


#####all 450k sites features process, ONLY NEED TO RUN ONCE
#parser = argparse.ArgumentParser(description='Adding all features to all 450K sites')
#parser.add_argument('-d',required=True,default='AD_CpG',help='disease dataset',dest='dataset',metavar='AD or Cd?')
#args = parser.parse_args()
#dataset = args.dataset # AD_CpG or Cd
additional_feature_file = home+'data/features/'+dataset+'/all_450k_addtional_features'
#if dataset == 'AD_CpG':
#    type_name = commons.type_name  ## amyloid, cerad, tangles
#    with_cell_type = commons.with_cell_type ## with or without
#    dataset = dataset+'/'+type_name+with_cell_type
sites_file = home+'data/'+dataset+'/all_450k_sites_winid.csv'
logger.info('Concatening features to all 450k sites of {}'.format(dataset))
logger.info('Read 450k sites file of {} with window id at {}'.format(dataset,sites_file))


parser = argparse.ArgumentParser(description='Adding all features to all 450k sites')
parser.add_argument('-r',required=False,default='False',help='reset feature processing tracker',dest='reset_tracker',metavar='True/False')
args = parser.parse_args()
reset_tracker = (args.reset_tracker=='True')
if reset_tracker:
    logger.info('reset features preprocess progress tracker')
    tracker = {'1806features':0,'wgbs':0,'atac':0,'rnaseq':0,'cadd':0,'dann':0,'eigen':0,'genocanyon':0,'gwava':0}
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')

tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['1806features'] == 1:
    logger.info("1806 features for {} 450k sites already processed".format(dataset))
else:
    logger.info("start processing 1806 features for {} 450k sites".format(dataset))
    subprocess.call([home+'code/features_preprocess/Feature_export.R',home+'data',dataset,'True'])
    logger.info("finish processing 1806 features for {} 450k sites".format(dataset))
    tracker['1806features'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')

#single sites WGBS
WGBS_h5s = home+'data/commons/WGBS_single_H5S'
tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['wgbs'] == 1:
    logger.info("WGBS features for {} 450k sites already processed".format(dataset))
else:
    WGBS_proc = WGBS_preprocess.WGBS_Preprocess(h5s_file=WGBS_h5s,data_dir=extra_storage+'WGBS/',sites_file=sites_file,additional_feature_file=additional_feature_file,hg19_file= home+'data/WGBS/hg19_WGBS.csv')
    logger.info("start processing WGBS features for {} 450k sites".format(dataset))
    if not os.path.exists(WGBS_h5s):
        logger.info('start converting all WGBS files from hg38 to hg19 coordinates')
        WGBS_proc.process()
    WGBS_proc.scores()
    logger.info("Got all WGBS features for for {} 450k sites".format(dataset))
    tracker['wgbs'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')


ATAC_h5s = home+'data/commons/ATAC_H5S'
tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['atac'] == 1:
    logger.info("ATAC features for {} 450k sites already processed".format(dataset))
else:
    if not os.path.exists(ATAC_h5s):
        logger.info("Binned ATAC score file {} doesn't exist, regenerating...".format(ATAC_h5s))
        atac_binning = BED_binning.BED_binning(data_type='ATAC',data_dir=extra_storage+'ATAC/',output=ATAC_h5s,sorted=True)
        atac_binning.binning()
    atac_process = BED_Preprocess.BED_Preprocessing(h5s_file=ATAC_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file,data_type='ATAC')
    logger.info("Start processing ATAC features for {} 450k sites".format(dataset))
    atac_process.process()
    logger.info("Got all ATAC features for {} 450k sites".format(dataset))
    tracker['atac'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')
    
    
RNASeq_h5s = home+'data/RNASeq/'
tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['rnaseq'] == 1:
    logger.info("RNASeq for {} 450k sites already processed".format(dataset))
else:
    RNASeqOutput = subprocess.check_output(['python',home+'code/features_preprocess/RNASeq_binning.py'])
    logger.info('RNASeq binning returned message: {}'.format(RNASeqOutput))
    rnaseq_process = BED_Preprocess.BED_Preprocessing(h5s_file=RNASeq_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file, data_type='RNASeq')
    logger.info("Start processing RNASeq features for {} 450k sites".format(dataset))
    rnaseq_process.process()
    logger.info("Got all RNASeq features for {} 450k sites".format(dataset))
    tracker['rnaseq'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')
    
    
tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['cadd'] == 1:
    logger.info("CADD for {} 450k sites already processed".format(dataset))
else:        
    cadd_preprocess = CADD_Preprocess.CADD_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
    logger.info("Start processing CADD features for {} 450k sites".format(dataset))
    cadd_preprocess.process()
    logger.info("Got all CADD features for {} 450k sites".format(dataset))
    tracker['cadd'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')

    
    
tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['dann'] == 1:
    logger.info("DANN for {} 450k sites already processed".format(dataset))
else:    
    dann_preprocess = DANN_Preprocess.DANN_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
    logger.info("Start processing DANN features for {} 450k sites".format(dataset))
    dann_preprocess.process()
    logger.info("Got all DANN features for {} 450k sites".format(dataset))
    tracker['dann'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')


tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['eigen'] == 1:
    logger.info("Eigen for {} 450k sites already processed".format(dataset))
else: 
    eigen_preprocess = Eigen_Preprocess.Eigen_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
    logger.info("Start processing Eigen features for {} 450k sites".format(dataset))
    eigen_preprocess.process()
    logger.info("Got all Eigen features for {} 450k sites".format(dataset))
    tracker['eigen'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')
    

tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['genocanyon'] == 1:
    logger.info("genocanyon for {} 450k sites already processed".format(dataset))
else:  
    genocanyon_scores = extra_storage+'GenoCanyon/Results/'+dataset+'/selected_site_all_450k_scores.txt'
    data_dir=extra_storage+'GenoCanyon/Results/'+dataset+'/'
    if not (os.path.exists(genocanyon_scores) and check_genocaynon(genocanyon_scores,sites_file)):
        logger.info('Genocaynon score file error, running GenoCanyon R script to regenerate Genocaynon scores for {} ...'.format(sites_file))
        subprocess.call([home+'code/features_preprocess/GenoCanyon_Preprocess.R',"TRUE",home,extra_storage,dataset])
        logger.info('Complete generating genocaynon score files for {}'.format(sites_file))

    genocanyon_preprocess = GenoCanyon_Preprocess.GenoCanyon_Preprocess(data_dir=data_dir,sites_file=sites_file,additional_feature_file=additional_feature_file)
    logger.info("Start processing Genocaynon features for {} 450k sites".format(dataset))
    genocanyon_preprocess.process('selected_site_all_450k_scores.txt')
    logger.info("Got all Genocaynon features for {} 450k sites".format(dataset))
    tracker['genocanyon'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')

    
tracker = joblib.load(home+'data/'+dataset+'/450k_tracker.pkl')
if tracker['gwava'] == 1:
    logger.info("GWAVA for {} 450k sites already processed".format(dataset))
else:
    gwava_preprocess = GWAVA_Preprocess.GWAVA_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
    logger.info("Start processing GWAVA features for {} 450k sites".format(dataset))
    gwava_preprocess.process()
    logger.info("Got all GWAVA features for {} 450k sites".format(dataset))
    tracker['gwava'] = 1
    joblib.dump(tracker,home+'data/'+dataset+'/450k_tracker.pkl')
        
gc.collect()
    
    
feature_dir = home+'data/features/'+dataset+'/'
all_450_features = home+'data/'+dataset+'/all_450k_features'    
with pd.HDFStore(home+'data/'+dataset+'/all_450k_sites_winid','r') as h5s:
    all_sites = h5s['all_450k_sites_winid']
all_sites.reset_index(drop=True,inplace=True)
logger.info('{} 450k sites with window id are in {}'.format(dataset,home+'data/'+dataset+'/all_450k_sites_winid'))
logger.info('1806 features for {} 450K sites are in {}'.format(dataset,feature_dir))
files = os.listdir(feature_dir)
pattern = '.*all_450k.csv$'
reg = re.compile(pattern)
files = [name for name in files if len(reg.findall(name))>0]

for file in files:    
    feature = pd.read_csv(feature_dir+file)
    logger.info('Concatenating {} of 1806 features'.format(file))
    logger.info('Its feature number is {}'.format(len(feature.columns)))
    all_sites = pd.concat([all_sites,feature],axis=1)
    
rename_features(all_sites)
all_sites.drop(['start','end'],axis=1,inplace=True)
logger.info('complete concatenating 1806 features to {} 450k sites'.format(dataset))                
     
                
logger.info('Concatenating additional features( ATAC, CADD, etc)')
additional_features = ['ATAC','CADD','DANN','Eigen','GenoCanyon','RNASeq','WGBS','GWAVA']
#merge with additional features
with pd.HDFStore(feature_dir+'all_450k_addtional_features','r') as h5s:
    for feature in additional_features:
        feature_frame = h5s[feature]
        all_sites = pd.concat([all_sites,feature_frame],axis=1)
        logger.info('{} feature concatenated'.format(feature))

all_sites = all_sites.loc[:,~all_sites.columns.duplicated()]
all_sites['chr'] = all_sites['chr'].astype('i8')

                
#nearest tss distance
logger.info('Calcuating distance to nearest tss for each {} 450k sites'.format(dataset))
chrs = all_sites['chr'].unique()
cols=['chr', 'coordinate','strand']
tss =  pd.read_csv(home+'data/commons/tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss = get_winid.convert_chr_to_num(tss,chrs)
tss.sort_values(['chr','coordinate'],inplace=True)
all_sites = nearest_tss(tss,all_sites)

with pd.HDFStore(all_450_features,'w') as h5s:
    h5s['all_450k_features'] = all_sites.drop(['pvalue','beta'],axis=1)
    logger.info('Combined features of {} 450k sites are saved in {}'.format(dataset,all_450_features))

