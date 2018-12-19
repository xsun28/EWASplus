#running script
import sys
from common import commons
home = commons.home
import pandas as pd
import numpy as np
from features_selection import Feature_Selection as FS
from log import Logger
from features_selection import  WilcoxonRankSums
from sklearn.externals import joblib
from features_selection import feature_selection_commons as fsc

dataset = commons.dataset
if dataset == 'AD_CpG':
    type_name = commons.type_name  ## amyloid, cerad, tangles
    with_cell_type = commons.with_cell_type ## with or without
    dataset = dataset+'/'+type_name+with_cell_type

log_dir = home+'logs/'
logger = Logger.Logger(log_dir,False).get_logger()
with pd.HDFStore(home+'data/'+dataset+'/all_features','r') as h5s:
    all_data = h5s['all_features']
all_data['beta_sign'] = all_data['label']
#all_data['coordinate'] = all_data['coordinate'].astype('i8')
all_data.drop(['coordinate','chr'],axis=1,inplace=True)
all_data['dist_to_nearest_tss'] = all_data['dist_to_nearest_tss'].astype('i8')
all_data = fsc.data_selection(all_data,classes=[0,1,-1],combine=True)
#all_data = fsc.data_selection(all_data,classes=[0,1,-1],combine=True)
all_features = all_data
#all_features = fsc.subset_control(all_data,30)
#all_features = all_data.query('beta_sign>0') ##only for hypermethylated sites in RICHS dataset, for AD dataset, hyper/hypo status can't be determined from beta
#logger.info('only keep heypermethylated sites')
if dataset == 'Cd':
    type_weight_factor = 0.4
else:
    if type_name == 'cerad':
        type_weight_factor = 0.23
    elif type_name == 'amyloid':
        type_weight_factor = 0.3
    elif type_name == 'cogdec':
        type_weight_factor = 0.4
    elif type_name == 'gpath':
        type_weight_factor = 0.3
    elif type_name == 'braak':
        type_weight_factor = 0.3
    elif type_name == 'tangles':
        type_weight_factor = 0.3
    else:
        type_weight_factor = 0.3
    
    
#split train test data and scaling on train data
scaler_type='MinMax'
all_features.drop(['id','winid','beta','beta_sign'],axis=1,inplace=True)
train_x,train_label,test_x,test_label,_ = commons.train_test_split(all_features,scaler=scaler_type)
train_x.reset_index(drop=True,inplace=True)
train_label.reset_index(drop=True,inplace=True)
test_x.reset_index(drop=True,inplace=True)
test_label.reset_index(drop=True,inplace=True)


sample_weights_train = commons.sample_weights(train_x,train_label,factor=1)
sample_weights_test = commons.sample_weights(test_x,test_label,factor=1)
weight_min_max_ratio = sample_weights_train.max()/sample_weights_train.min()
print('trait %s weight max ratio: %f',type_name+with_cell_type,weight_min_max_ratio)
train_x.drop(['pvalue'],axis=1,inplace=True)
test_x.drop(['pvalue'],axis=1,inplace=True)
fs_sample_weights = np.power(sample_weights_train, type_weight_factor) 

print('scaled trait %s max weight ratio: %f',type_name+with_cell_type,fs_sample_weights.max()/fs_sample_weights.min())

methods = ['random_forest','xgboost','logistic_regression','linear_SVC']
all_intersect = False
fs_params = fsc.method_params(methods)
fs = FS.FeatureSelection(class_num=2,methods=methods,all_intersect=all_intersect,**fs_params)
logger.info('Feature selection methods are: '+str(methods))
logger.info('All intersected features: '+str(all_intersect))
fs.fit(sample_weight=fs_sample_weights)
selected_features = fs.transform(train_x,train_label)
logger.info('selected features number is: %d\n',selected_features.shape[0])
logger.info(selected_features)
reduced_train_x = train_x[selected_features['feature']]
reduced_test_x = test_x[selected_features['feature']]
total_x = pd.concat([reduced_train_x,reduced_test_x],ignore_index=True)
total_label = pd.concat([train_label,test_label],ignore_index=True)
total_weights = pd.concat([sample_weights_train,sample_weights_test],ignore_index=True)

feature_diff_stats = fsc.selected_feature_analysis(selected_features['feature'],total_x,total_label)
feature_diff_stats = pd.merge(feature_diff_stats,selected_features,on='feature')
selected_features_100 = feature_diff_stats if len(feature_diff_stats) <=50 else feature_diff_stats.sort_values(['n','pvalue'],ascending=[False,True])[:60]
reduced_train_x = train_x[selected_features_100['feature']]
reduced_test_x = test_x[selected_features_100['feature']]
total_x = pd.concat([reduced_train_x,reduced_test_x],ignore_index=True)
total_label = pd.concat([train_label,test_label],ignore_index=True)
total_weights = pd.concat([sample_weights_train,sample_weights_test],ignore_index=True)

_,_,_,_,scaler = commons.train_test_split(all_features[['label','pvalue']+list(selected_features_100['feature'])],scaler=scaler_type)
joblib.dump(scaler,home+'data/'+dataset+'/scaler.pkl')
print('Data scaler type is: %s'%scaler_type)

selected_features_100.to_csv(home+'data/'+dataset+'/feature_stats.csv')