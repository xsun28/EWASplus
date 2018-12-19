#running script run as python WGBS_prediction.py -r True -u True -m LogisticRegression xgbooster
import sys
from common import commons
home = commons.home
#sys.path.append('/home/ec2-user/anaconda3/lib/python3.6/site-packages')
import pandas as pd
import numpy as np
from models.model_commons import get_hyperopt_params
from sklearn.externals import joblib
from importlib import reload
from log import Logger
from models import Ensemble_hyperopt as eh
import prediction_commons
from sklearn.externals import joblib 
import get_winid
import gc
import argparse

parser = argparse.ArgumentParser(description='WGBS sites methylation prediction')
parser.add_argument('-r',required=False,default=False,help='retrain prediction mode',dest='retrain',metavar='True or False?')
parser.add_argument('-u',required=False,default=True,help='upsampling data',dest='upsampling',metavar='True or False?')
parser.add_argument('-m',required=False,default=['LogisticRegression','xgbooster'],help='prediction methods',dest='methods',metavar='LogisticRegression xgbooster')
args = parser.parse_args()
retrain = args.retrain
up_sampling = args.upsampling
methods = args.methods

dataset = commons.dataset
if dataset == 'AD_CpG':
    type_name = commons.type_name  ## amyloid, cerad, tangles
    with_cell_type = commons.with_cell_type ## with or without
    dataset = dataset+'/'+type_name+with_cell_type
model_path = home+'data/'+dataset+'/prediction_model.pkl'
pred_probs = home+'data/'+dataset+'/pred_probs'

##features selecetd by traditional methods

if up_sampling:
    wtf_lo = 0.05 if dataset=="Cd" else 0.2
    wtf_hi = 0.1 if dataset=="Cd" else 0.3
else:
    wtf_lo = 1.0/3 if dataset=="Cd" else 1 
    wtf_hi = 0.5 if dataset=="Cd" else 1.5
    
log_dir = home+'logs/'
logger = Logger.Logger(log_dir,False).get_logger()
with pd.HDFStore(home+'data/'+dataset+'/selected_features','r') as h5s:
    train_x =h5s['train_x'] 
    train_label = h5s['train_label'] 
    test_x = h5s['test_x'] 
    test_label = h5s['test_label']
    sample_weights_train = h5s['sample_weights_train'] 
    sample_weights_test = h5s['sample_weights_test']
total_x = pd.concat([train_x,test_x],ignore_index=True)
total_label = pd.concat([train_label,test_label],ignore_index=True)
total_sample_weights = pd.concat([sample_weights_train,sample_weights_test],ignore_index=True)
selected_features = total_x.columns

######model training
if (not os.path.exists(model_path)) or retrain:
    print('retraning prediction model...')
    #methods = ['LogisticRegression','xgbooster']
    params = get_hyperopt_params(methods,wtf_lo=wtf_lo,wtf_hi=wtf_hi)
    ensemble_hyopt = eh.Ensemble(methods,params)
    ensemble_hyopt.fit(total_x,total_label,sample_weight=total_sample_weights,max_iter=100)
    joblib.dump(ensemble_hyopt,model_path)


###########
print('Starting WGBS prediction of '+dataset+'...')
ranges = np.arange(prediction_commons.tss_start,prediction_commons.tss_end,1000000)
ranges = np.append(ranges,prediction_commons.tss_end)
print('sites ranges are '+ranges)

for i in np.arange(len(ranges)-1):
    ensemble_hyopt = joblib.load(model_path)
    for method,best_estimator in ensemble_hyopt.best_estimators_.items():
        best_estimator.fit(total_x,total_label,total_sample_weights)
    start = ranges[i]
    end = ranges[i+1]
    print("start processing range %d to %d"%(start,end))
    all_features = home+'data/WGBS/all_features_'+str(start)+'_'+str(end)
    print(all_features)
    with pd.HDFStore(all_features,'r') as h5s:
        wgbs_all_data = h5s['all_features']
    scaler = joblib.load(home+'data/'+dataset+'/scaler.pkl')
    wgbs_data = pd.DataFrame(scaler.transform(wgbs_all_data[selected_features]),columns=selected_features,index=wgbs_all_data.index)
    pred = ensemble_hyopt.predict(wgbs_data)
    pred_prob = ensemble_hyopt.predict_proba(wgbs_data)
    pred_prob = pd.DataFrame(pred_prob,columns=['negative','positive'])
    target_sites = pred_prob.sort_values(['positive'],ascending=False).query('positive >= 0.5')
    target_sites_coordinate = wgbs_all_data.ix[target_sites.index,['chr','coordinate']]
    target_sites = target_sites.join(target_sites_coordinate)
    with pd.HDFStore(pred_probs,'a') as h5s:
        h5s[str(start)+'_'+str(end)] = target_sites
        print(h5s.keys())
    print(target_sites)
    gc.collect()



print('Get top 500 predicted sites of trait '+dataset)    
all_sites_probs = None
with pd.HDFStore(pred_probs,'r') as h5s:
    for key in h5s.keys():
        if all_sites_probs is None:
            all_sites_probs = h5s[key]
        else:
            all_sites_probs = pd.concat([all_sites_probs,h5s[key]],ignore_index=True)

top_500 = all_sites_probs.sort_values(['positive'],ascending=False).reset_index(drop=True)
top_500 = top_500[:500]
top_500['coordinate'] = top_500['coordinate'].astype('i8')

#nearest tss distance    
chrs = top_500['chr'].unique()
cols=['chr', 'coordinate','strand']
tss =  pd.read_csv(home+'data/commons/tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss = get_winid.convert_chr_to_num(tss,chrs)
tss.sort_values(['chr','coordinate'],inplace=True)
top_500 = nearest_tss(tss,top_500).sort_values(['positive'],ascending=False)

pred_positive_500 = home+'data/'+dataset+'/pred_positive_500.csv'
top_500.to_csv(pred_positive_500,index=False)
pred_positive_500_bed = home+'data/'+dataset+'/pred_positive_500.bed'
top_500_bed = top_500.copy()
top_500_bed['end'] = top_500_bed['coordinate']+1
top_500_bed['chr'] = top_500_bed['chr'].apply(lambda x: 'chr'+str(x))
top_500_bed[['chr','coordinate','end','positive']].to_csv(pred_positive_500_bed,index=False,header=None,sep='\t')

trait_all_sites = pd.read_csv(home+'data/'+dataset+'/all_450k_sites_winid.csv')
top500_nearest_cpgs = commons.find_nearest_450ksites(5000,trait_all_sites,top_500)
top500_nearest_cpgs.drop(['start','winid','end'],axis=1).to_csv(home+'data/'+dataset+'/top500_nearest_450k.csv')


######predict 450k array sites
pred_probs_450k = home+'data/'+dataset+'/pred_probs_450k'
selected_features = total_x.columns
all_450k_features = home+'data/AD_CpG/all_450k_features'
with pd.HDFStore(all_450k_features,'r') as h5s:
    all_450k_data = h5s['all_450k_features']
scaler = joblib.load(home+'data/'+dataset+'/scaler.pkl')
all_450k_data_features = pd.DataFrame(scaler.transform(all_450k_data[selected_features]),columns=selected_features,index=all_450k_data.index)

#ensemble_hyopt = joblib.load(model_path)
#for method,best_estimator in ensemble_hyopt.best_estimators_.items():
    #best_estimator.fit(total_x,total_label,total_sample_weights)
pred_450k = ensemble_hyopt.predict(all_450k_data_features)
pred_prob_450k = ensemble_hyopt.predict_proba(all_450k_data_features)

pred_prob_450k = pd.DataFrame(pred_prob_450k,columns=['negative','positive'])
pred_prob_450k['chr'] = all_450k_data['chr']
pred_prob_450k['coordinate'] = all_450k_data['coordinate']

with pd.HDFStore(pred_probs_450k,'w') as h5s:
    h5s['pred_probs_450k'] = pred_prob_450k
    
    