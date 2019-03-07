#!/bin/bash


rm ../logs/logging.conf
PYTHONPATH=$(pwd)


dataset=AD

if [[ "$dataset" == "AD" ]]; then
        traits=(amyloid cerad ceradaf tangles cogdec gpath braak)
        sed -i "s/^dataset = .*/dataset = 'AD_CpG'/" common/commons.py
        for trait in ${traits[@]}; do
        	echo "Predict WGBS methylation for $trait"
                retrain=True
		models="LogisticRegression xgbooster"
		echo $models
                python prediction/WGBS_prediction.py -r $retrain -u True -m $models
        done
        #python prediction/WGBS_alltraits_prediction_AD -m $models
fi
