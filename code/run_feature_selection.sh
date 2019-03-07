#!/bin/bash
PYTHONPATH=$(pwd)
traits=(amyloid cerad ceradaf tangles cogdec gpath braak)
sed -i "s/^dataset = .*/dataset = 'AD_CpG'/" common/commons.py

for trait in ${traits[@]}; do
	sed -i "s/\([[:space:]]\+type_name\) = .*/\1 = '$trait'/" common/commons.py
	python features_selection/feature_selection.py
done
