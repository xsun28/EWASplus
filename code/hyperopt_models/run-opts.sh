#!/bin/bash

methods=("LogisticRegression" "RandomForestClassifier" "SVC" "xgbooster" "tensor_DNN" "MLPClassifier")
#methods=("MLPClassifier")
feature=$1
max_iter=$2
cv=$3
dataset=$4
i=0
FAIL=0
for method in "${methods[@]}"; do
python ${method}_hyperopt.py -f ${feature} -i ${max_iter} -c ${cv} -d ${dataset} &
done

for job in $(jobs -p); do
echo $job
wait $job || FAIL=$((FAIL+1)) 
done
echo ${FAIL}
