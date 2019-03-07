#running script run as python WGBS_alltraits_prediction_AD.py -m LogisticRegression xgbooster
import sys
from common import commons
home = commons.home
logger = commons.logger
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.externals import joblib
from importlib import reload
from log import Logger
from models import Ensemble_hyperopt as eh 
from features_preprocess import get_winid
import gc
import argparse
#############################################Get top 500 sites predicted commonly by all traits
parser = argparse.ArgumentParser(description='WGBS top 500 and 450 k sites methylation prediction commonly by all traits')
parser.add_argument('-m',required=False,default=['LogisticRegression','xgbooster'],help='prediction methods',dest='methods',metavar='LogisticRegression xgbooster',nargs='+')
args = parser.parse_args()
methods = '-'.join(args.methods)

pred_probs = [home+'data/'+dataset+'/pred_probs' for dataset in ['AD_CpG/amyloidwith','AD_CpG/ceradwith','AD_CpG/tangleswith','AD_CpG/gpathwith','AD_CpG/braakwith','AD_CpG/cogdecwith']]
traits = ['amyloidwith','ceradwith','tangleswith','gpathwith','braakwith','cogdecwith']
dataset_common_predictions = None
for pred_prob,trait in zip(pred_probs,traits):
    first = True
    with pd.HDFStore(pred_prob,'r') as h5s:
        for key in h5s.keys():
            if first:
                predicted_positives = h5s[key]
                first = False
            else:
                predicted_positives = pd.concat([predicted_positives,h5s[key]],ignore_index=True)
    if dataset_common_predictions is None:
        dataset_common_predictions = predicted_positives
    else:
        dataset_common_predictions = dataset_common_predictions.merge(predicted_positives,on=['chr','coordinate'],how='inner')
    dataset_common_predictions.rename({'positive':'positive_'+trait,'negative':'negative_'+trait},axis=1,inplace=True)
    

dataset_common_predictions['mean_positive'] = dataset_common_predictions[['positive_'+trait for trait in traits]].mean(axis=1)
tenfold_test_results = {}
weight_scale = 'f1' #'auc','recall','precision','f1','accuracy'
scale_sum = 0
weight_score = None
total_weight = 0;
for trait in traits:
    trait_result = joblib.load(home+'data/AD_CpG/'+trait+'/10fold_test_results.pkl')
    tenfold_test_results[trait] = trait_result
    #dataset_common_predictions['weight_'+weight_scale+'_'+trait] = trait_result['LogisticRegression-xgbooster'][weight_scale]
    #total_weight += trait_result['LogisticRegression-xgbooster'][weight_scale]
    dataset_common_predictions['weight_'+weight_scale+'_'+trait] = trait_result[methods][weight_scale]
    total_weight += trait_result[methods][weight_scale]
    if weight_score is None:
        #weight_score = dataset_common_predictions['positive_'+trait]*trait_result['LogisticRegression-xgbooster'][weight_scale]
        weight_score = dataset_common_predictions['positive_'+trait]*trait_result[methods][weight_scale]
    else:
        #weight_score += dataset_common_predictions['positive_'+trait]*trait_result['LogisticRegression-xgbooster'][weight_scale]        
        weight_score += dataset_common_predictions['positive_'+trait]*trait_result[methods][weight_scale]

dataset_common_predictions['weighted_positive'] = weight_score/total_weight

dataset_common_predictions_top500_mean = dataset_common_predictions.sort_values(['mean_positive'],ascending=False)[:500]
dataset_common_predictions_top500_weighted = dataset_common_predictions.sort_values(['weighted_positive'],ascending=False)[:500]

#nearest tss distance    
chrs = dataset_common_predictions_top500_mean['chr'].unique()
cols=['chr', 'coordinate','strand']
tss =  pd.read_csv(home+'data/commons/tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss = get_winid.convert_chr_to_num(tss,chrs)
tss.sort_values(['chr','coordinate'],inplace=True)
dataset_common_predictions_top500_mean = nearest_tss(tss,dataset_common_predictions_top500_mean).sort_values(['mean_positive'],ascending=False)

chrs = dataset_common_predictions_top500_weighted['chr'].unique()
cols=['chr', 'coordinate','strand']
tss =  pd.read_csv(home+'data/commons/tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss = get_winid.convert_chr_to_num(tss,chrs)
tss.sort_values(['chr','coordinate'],inplace=True)
dataset_common_predictions_top500_weighted = nearest_tss(tss,dataset_common_predictions_top500_weighted).sort_values(['weighted_positive'],ascending=False)


common_top_500_mean = home+'data/AD_CpG/pred_positive_500_commmon_mean.csv'
dataset_common_predictions_top500_mean.to_csv(common_top_500_mean,index=False)

common_top_500_weighted = home+'data/AD_CpG/pred_positive_500_commmon_weighted.csv'
dataset_common_predictions_top500_weighted.to_csv(common_top_500_weighted,index=False)

all_sites = None
for trait in traits:
    with pd.HDFStore(home+'data/AD_CpG/'+trait+'/all_450k_sites_winid','r') as h5s:
        site = h5s['all_450k_sites_winid'][['id','chr','coordinate','pvalue','beta']]
        site.rename({'pvalue':trait+'_pvalue','beta':trait+'_beta'},axis=1,inplace=True)
    if all_sites is None:
        all_sites = site
    else:
        all_sites = all_sites.merge(site,on=['id','chr','coordinate'])
top500_nearest_cpgs_mean = commons.find_nearest_450ksites(5000,all_sites,dataset_common_predictions_top500_mean)
top500_nearest_cpgs_weighted = commons.find_nearest_450ksites(5000,all_sites,dataset_common_predictions_top500_weighted)
top500_nearest_cpgs_mean.to_csv(home+'data/AD_CpG/common_top500_mean_nearest_450k.csv')
top500_nearest_cpgs_weighted.to_csv(home+'data/AD_CpG/common_top500_weighted_nearest_450k.csv')


####all 450k sites predicted probs by all traits
pred_probs = [home+'data/'+dataset+'/pred_probs_450k' for dataset in ['AD_CpG/amyloidwith','AD_CpG/ceradwith','AD_CpG/tangleswith','AD_CpG/gpathwith','AD_CpG/braakwith','AD_CpG/cogdecwith']]
#traits = ['amyloidwith','ceradwith','tangleswith','gpathwith','braakwith','cogdecwith']
first = True
for pred_prob,trait in zip(pred_probs,traits):
    with pd.HDFStore(pred_prob,'r') as h5s:
        if first:
            all_probs_450k = h5s['pred_probs_450k']
            first = False
        else:
            all_probs_450k = pd.concat([all_probs_450k,h5s['pred_probs_450k']],axis=1)
    all_probs_450k.rename({'positive':'positive_'+trait,'negative':'negative_'+trait},axis=1,inplace=True)
    
all_probs_450k = all_probs_450k.loc[:,~all_probs_450k.columns.duplicated()]
all_450k_features = home+'data/AD_CpG/all_450k_features'
with pd.HDFStore(all_450k_features,'r') as h5s:
    all_450k_data = h5s['all_450k_features']
all_probs_450k = pd.merge(all_probs_450k,all_450k_data[['id','chr','coordinate']],on=['chr','coordinate'],how='left')

trait_pvalues = [home+'data/'+dataset+'/all_450k_sites_winid.csv' for dataset in ['AD_CpG/amyloidwith','AD_CpG/ceradwith','AD_CpG/tangleswith','AD_CpG/gpathwith','AD_CpG/braakwith','AD_CpG/cogdecwith']]
all_probs_450k['mean_positive'] = all_probs_450k[['positive_'+trait for trait in traits]].mean(axis=1)
tenfold_test_results = {}
weight_scale = 'f1' #'auc','recall','precision','f1','accuracy'
scale_sum = 0
weight_score = None
total_weight = 0;
for trait,pvalue in zip(traits,trait_pvalues):
    trait_result = joblib.load(home+'data/AD_CpG/'+trait+'/10fold_test_results.pkl')
    tenfold_test_results[trait] = trait_result
    trait_pvalue = pd.read_csv(pvalue,usecols=[1,2,3,4],header=None,skiprows=1,names=['chr','coordinate',trait+'_pvalue',trait+'_beta'])
    #all_probs_450k['weight_'+weight_scale+'_'+trait] = trait_result['LogisticRegression-xgbooster'][weight_scale]
    all_probs_450k['weight_'+weight_scale+'_'+trait] = trait_result[methods][weight_scale]
    all_probs_450k = all_probs_450k.merge(trait_pvalue,on=['chr','coordinate'],how='left')
    #total_weight += trait_result['LogisticRegression-xgbooster'][weight_scale]
    total_weight += trait_result[methods][weight_scale]
    if weight_score is None:
        #weight_score = all_probs_450k['positive_'+trait]*trait_result['LogisticRegression-xgbooster'][weight_scale]
        weight_score = all_probs_450k['positive_'+trait]*trait_result[methods][weight_scale]
    else:
        #weight_score += all_probs_450k['positive_'+trait]*trait_result['LogisticRegression-xgbooster'][weight_scale]
        weight_score += all_probs_450k['positive_'+trait]*trait_result[methods][weight_scale]
all_probs_450k['weighted_positive'] = weight_score/total_weight

trait_pvalue = pd.read_csv(home+'data/AD_CpG/amyloidwith/all_450k_sites_winid.csv',usecols=[1,2,3,4],header=None,skiprows=1,names=['chr','coordinate',trait+'_pvalue',trait+'_beta'])

all_probs_450k.to_csv(home+'data/AD_CpG/450kwithpredictedprob.csv',index=False)