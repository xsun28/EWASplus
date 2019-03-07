#!/bin/bash
PYTHONPATH=$(pwd)        
traits=(amyloid cerad ceradaf tangles cogdec gpath braak)
sed -i "s/^dataset = .*/dataset = 'AD_CpG'/" common/commons.py
echo "Preprocess features for all 450K WGBS sites"
python features_preprocess/all450k_feature_preprocess.py -r True
for trait in ${traits[@]}; do
	echo "start processing $trait..."
        sed -i "s/\([[:space:]]\+type_name\) = .*/\1 = '$trait'/" common/commons.py
	echo "Start ${trait} training sites selection"
        python sites_selection/AD_sites_selection.py
	echo "Preprocess features for selected ${trait} training sites"
        python features_preprocess/all_features_preprocess.py
done
