##running script: match all features to all WGBS sites. Called as python WGBS_all_sites_feature_preprocess.py -a True -r False

import os
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import BED_binning
#from features_preprocess import BED_Preprocess,CADD_Preprocess_bedtools as CADD_Preprocess,DANN_Preprocess_bedtools as DANN_Preprocess,Eigen_Preprocess_bedtools as Eigen_Preprocess,GenoCanyon_Preprocess
from features_preprocess import BED_Preprocess,CADD_Preprocess ,DANN_Preprocess,Eigen_Preprocess,GenoCanyon_Preprocess,GWAVA_Preprocess
import subprocess
import pandas as pd
from features_preprocess import get_winid
import numpy as np
import re
from features_preprocess import WGBS_preprocess
from common.commons import rename_features, check_genocaynon,logger
import gc
import argparse


def wgbs_sites_selection(tss,allsites):
    tss = tss.sort_values(['chr','coordinate'])
    allsites = all_sites.sort_values(['chr','coordinate']).reset_index(drop=True)
    i = 0
    selected_sites = []
    #selected_sites = pd.DataFrame(columns=['chr','coordinate','tss_coordinate'])
    tss['before'] = tss['coordinate']-100000
    tss['after'] = tss['coordinate']+100000
    for row in allsites.iterrows():
        if i >= len(tss):
            break
        chr = row[1]['chr']
        coordinate = row[1]['coordinate']
        winid = row[1]['winid']
        if chr==tss.ix[i,'chr'] and coordinate>=tss.ix[i,'before'] and coordinate<=tss.ix[i,'after']:
            selected_sites.extend([[winid,chr,coordinate,tss.ix[i,'chr'],tss.ix[i,'coordinate']]])
        else:
            while  i<len(tss) and (chr>tss.ix[i,'chr'] or (chr==tss.ix[i,'chr'] and coordinate>tss.ix[i,'after'])):
                i += 1
            if i<len(tss) and chr==tss.ix[i,'chr'] and coordinate>=tss.ix[i,'before'] and coordinate<=tss.ix[i,'after']:
                selected_sites.extend([[winid,chr,coordinate,tss.ix[i,'chr'],tss.ix[i,'coordinate']]])
    df = pd.DataFrame(selected_sites,columns=['winid','chr','coordinate','tss_chr','tss_coordinate'])
    df['chr'] = df['chr'].astype('i8')
    return df


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
    merged.dropna(axis=0,inplace=True)
    return merged

logger.info('Concatening features to WGBS sites')
parser = argparse.ArgumentParser(description='Adding all features to all WGBS sites')
parser.add_argument('-a',required=False,default='True',help='feature set',dest='all',metavar='All WGBS sites?')
parser.add_argument('-r',required=False,default='False',help='reset feature processing tracker',dest='reset_tracker',metavar='True/False')
args = parser.parse_args()

dataset = 'WGBS'
all_wgbs_sites_file = home+'data/'+dataset+'/all_wgbs_sites_winid.csv'
all_sites = pd.read_csv(all_wgbs_sites_file)
logger.info('Read all WGBS sites file with window id at '+all_wgbs_sites_file)

chrs = all_sites['chr'].unique()
logger.info('chromosomes of all WGBS sites are: '+str(chrs))
cols=['chr', 'coordinate','strand']
tss =  pd.read_csv(home+'data/commons/tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss = get_winid.convert_chr_to_num(tss,chrs)

all_wgbs_sites = (args.all=='True')
reset_tracker = (args.reset_tracker=='True')

if all_wgbs_sites:
    logger.info('Using all WGBS sites')
    selected_wgbs_tss = all_sites[['winid','chr','coordinate']]
elif not os.path.exists(home+'data/'+dataset+'/all_selected_wgbs_sites'):
    logger.info('Selecting WGBS sites within 100k of tss sites')
    selected_wgbs_tss = wgbs_sites_selection(tss,all_sites)
    with pd.HDFStore(home+'data/'+dataset+'/all_selected_wgbs_sites','w') as h5s:
        h5s['all_wgbs'] = selected_wgbs_tss
else:
    logger.info('Using selected wgbs sites')
    with pd.HDFStore(home+'data/'+dataset+'/all_selected_wgbs_sites','r') as h5s:
        selected_wgbs_tss = h5s['all_wgbs']

    
start_pos = 0
end_pos = len(selected_wgbs_tss)-1
logger.info('total selected wgbs sites number: '+str(end_pos+1))
subprocess.call(['sed','-i','s/tss_end =.*/tss_end = '+str(end_pos)+'/',home+'code/prediction/prediction_commons.py'])
ranges = np.arange(start_pos,end_pos,2000000)
if reset_tracker:
    logger.info('reset features preprocess progress tracker')
    tracker = pd.DataFrame({'start':ranges,'1806features':np.zeros_like(ranges),'wgbs':np.zeros_like(ranges),'atac':np.zeros_like(ranges),
                           'rnaseq':np.zeros_like(ranges),'cadd':np.zeros_like(ranges),'dann':np.zeros_like(ranges),
                            'eigen':np.zeros_like(ranges),'genocanyon':np.zeros_like(ranges),'gwava':np.zeros_like(ranges)})
    tracker = tracker.set_index('start')
    tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
ranges = np.append(ranges,end_pos)    

for i in np.arange(len(ranges)-1):
    start = ranges[i]
    end = ranges[i+1]
    logger.info("start processing WGBS sites with range %d to %d"%(start,end))
    selected_wgbs = selected_wgbs_tss[start:end]
    sites_file = home+'data/'+dataset+'/all_sites_winid.csv'
    selected_wgbs.to_csv(sites_file,index=False)
    selected_wgbs.to_csv(home+'data/'+dataset+'/selected_pos_winid.csv',columns=['winid'],index=False,header=None)
    additional_feature_file = home+'data/features/'+dataset+'/addtional_features_'+str(start)+'_'+str(end)
    logger.info("Additional features (WGBS,ATAC,etc) for range {} to {} are saved at {}".format(start,end,additional_feature_file))
    
    ###1806 features
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'1806features'] == 1:
        logger.info("1806 features "+str(start)+"_"+str(end)+" already processed")
    else:
        logger.info("start processing 1806 features for wgbs sites range %d to %d"%(start,end))
        subprocess.call([home+'code/features_preprocess/Feature_export.R',home+'data',dataset,'False'])
        logger.info("finish processing 1806 features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'1806features'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()

        
    ###WGBS 
    WGBS_h5s = home+'data/commons/WGBS_single_H5S'
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'wgbs'] == 1:
        logger.info("WGBS features for "+str(start)+"_"+str(end)+" already processed")
    else:
        logger.info("start processing WGBS features for wgbs sites range %d to %d"%(start,end))
        WGBS_proc = WGBS_preprocess.WGBS_Preprocess(h5s_file=WGBS_h5s,data_dir=extra_storage+'WGBS/',sites_file=sites_file,additional_feature_file=additional_feature_file,hg19_file= home+'data/WGBS/hg19_WGBS.csv')
        if not os.path.exists(WGBS_h5s):
            logger.info('start converting all WGBS files from hg38 to hg19 coordinates')
            WGBS_proc.process()
        WGBS_proc.scores()
        logger.info("Got all WGBS features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'wgbs'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()
        
    ###ATAC
    ATAC_h5s = home+'data/commons/ATAC_H5S'
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'atac'] == 1:
        logger.info("ATAC for "+str(start)+"_"+str(end)+" already processed")
    else:
        if not os.path.exists(ATAC_h5s):
            logger.info("Binned ATAC score file {} doesn't exist, regenerating...".format(ATAC_h5s))
            atac_binning = BED_binning.BED_binning(data_type='ATAC',data_dir=extra_storage+'ATAC/',output=ATAC_h5s,sorted=True)
            atac_binning.binning()
        atac_process = BED_Preprocess.BED_Preprocessing(h5s_file=ATAC_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file,data_type='ATAC')
        logger.info("Start processing ATAC features for wgbs sites range %d to %d"%(start,end))
        atac_process.process()
        logger.info("Got all ATAC features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'atac'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect() 
        
    
    ###RNASeq
    RNASeq_h5s = home+'data/RNASeq/'
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'rnaseq'] == 1:
        logger.info("RNASeq for "+str(start)+"_"+str(end)+" already processed")
    else:
        RNASeqOutput = subprocess.check_output(['python',home+'code/features_preprocess/RNASeq_binning.py'])
        logger.info('RNASeq binning returned message: {}'.format(RNASeqOutput))
        rnaseq_process = BED_Preprocess.BED_Preprocessing(h5s_file=RNASeq_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file, data_type='RNASeq')
        logger.info("Start processing RNASeq features for wgbs sites range %d to %d"%(start,end))
        rnaseq_process.process()
        logger.info("Got all RNASeq features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'rnaseq'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()
        
    
    ###CADD
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'cadd'] == 1:
        logger.info("CADD for "+str(start)+"_"+str(end)+" already processed")
    else:    
        cadd_preprocess = CADD_Preprocess.CADD_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
        logger.info("Start processing CADD features for wgbs sites range %d to %d"%(start,end))
        cadd_preprocess.process()
        logger.info("Got all CADD features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'cadd'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()
    
    ###DANN
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'dann'] == 1:
        logger.info("DANN for "+str(start)+"_"+str(end)+" already processed")
    else: 
        dann_preprocess = DANN_Preprocess.DANN_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
        logger.info("Start processing DANN features for wgbs sites range %d to %d"%(start,end))
        dann_preprocess.process()
        logger.info("Got all DANN features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'dann'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()
    
    ###eigen
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'eigen'] == 1:
        logger.info("Eigen for "+str(start)+"_"+str(end)+" already processed")
    else: 
        eigen_preprocess = Eigen_Preprocess.Eigen_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
        logger.info("Start processing Eigen features for wgbs sites range %d to %d"%(start,end))
        eigen_preprocess.process()
        logger.info("Got all Eigen features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'eigen'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()
    
    ###genocanyon
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'genocanyon'] == 1:
        logger.info("genocanyon for "+str(start)+"_"+str(end)+" already processed")
    else:      
        genocanyon_scores = extra_storage+'GenoCanyon/Results/'+dataset+'/selected_site_scores.txt'
        data_dir=extra_storage+'GenoCanyon/Results/'+dataset+'/'
        if (not os.path.exists(genocanyon_scores)) or (not check_genocaynon(genocanyon_scores,sites_file)):
            logger.info('Genocaynon score file error, running GenoCanyon R script to regenerate Genocaynon scores for {} ...'.format(sites_file))
            subprocess.call([home+'code/features_preprocess/GenoCanyon_Preprocess.R',"FALSE",home,extra_storage,dataset])
            logger.info('Complete generating genocaynon score files for {}'.format(sites_file))
        genocanyon_preprocess = GenoCanyon_Preprocess.GenoCanyon_Preprocess(data_dir=data_dir,sites_file=sites_file,additional_feature_file=additional_feature_file)
        logger.info("Start processing Genocaynon features for wgbs sites range %d to %d"%(start,end))
        genocanyon_preprocess.process('selected_site_scores.txt')
        logger.info("Got all Genocaynon features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'genocanyon'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()
        
    ###gwava
    tracker = pd.read_pickle(home+'data/'+dataset+'/tracker.pkl')
    if tracker.loc[start,'gwava'] == 1:
        logger.info("GWAVA for "+str(start)+"_"+str(end)+" already processed")
    else:
        gwava_preprocess = GWAVA_Preprocess.GWAVA_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
        logger.info("Start processing GWAVA features for wgbs sites range %d to %d"%(start,end))
        gwava_preprocess.process()
        logger.info("Got all GWAVA features for wgbs sites range %d to %d"%(start,end))
        tracker.loc[start,'gwava'] = 1
        tracker.to_pickle(home+'data/'+dataset+'/tracker.pkl')
        gc.collect()  
    
    selected_wgbs = pd.read_csv(home+'data/'+dataset+'/all_sites_winid.csv')
    logger.info('WGBS sites with windows id of range from {} to {} are in {}'.format(start,end,selected_wgbs))
    feature_dir = home+'data/features/'+dataset+'/'
    logger.info('1806 features for WGBS sites of range from {} to {} are in {}'.format(start,end,feature_dir))
    files = os.listdir(feature_dir)
    pattern = '.*all.csv$'
    reg = re.compile(pattern)
    files = [name for name in files if len(reg.findall(name))>0]
    
    for file in files:    
        feature = pd.read_csv(feature_dir+file)
        logger.info('Concatenating {} of 1806 features'.format(file))
        logger.info('Its feature number is {}'.format(len(feature.columns)))
        selected_wgbs = pd.concat([selected_wgbs,feature],axis=1)
    
    rename_features(selected_wgbs)
    logger.info('complete concatenating 1806 features to WGBS sites of range from {} to {}'.format(start,end))
    additional_features = ['ATAC','CADD','DANN','Eigen','GenoCanyon','RNASeq','WGBS','GWAVA']
    
    #merge with additional features
    logger.info('Concatenating additional features( ATAC, CADD, etc)')
    with pd.HDFStore(additional_feature_file,'r') as h5s:
        for feature in additional_features:
            feature_frame = h5s[feature]
            selected_wgbs = pd.concat([selected_wgbs,feature_frame],axis=1)
            logger.info('{} feature concatenated'.format(feature))
    selected_wgbs = selected_wgbs.loc[:,~selected_wgbs.columns.duplicated()]
    selected_wgbs['chr'] = selected_wgbs['chr'].astype('i8')
    
    #nearest tss distance
    logger.info('Calcuating distance to nearest tss for each WGBS site within range from {} to {}'.format(start,end))
    chrs = selected_wgbs['chr'].unique()
    cols=['chr', 'coordinate','strand']
    tss =  pd.read_csv(home+'data/commons/tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
    tss = get_winid.convert_chr_to_num(tss,chrs)
    tss.sort_values(['chr','coordinate'],inplace=True)
    selected_wgbs = nearest_tss(tss,selected_wgbs)
    
    
    with pd.HDFStore(home+'data/'+dataset+'/all_features_'+str(start)+'_'+str(end),'w') as h5s:
        h5s['all_features'] = selected_wgbs
        logger.info('Combined features of WGBS sites within range from {} to {} are saved in {}'.format(start,end,home+'data/'+dataset+'/all_features_'+str(start)+'_'+str(end)))
    gc.collect()    