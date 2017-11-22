#!/usr/bin/env bash
model_dir=models/ICASSP/Books
config_dir=models/ICASSP/Books
eval_dir=${model_dir}/eval

mkdir -p ${eval_dir}

cat ${model_dir}/AST.3/log.txt | grep AST -A1 | grep -v "MT\|ASR" > ${model_dir}/AST.3/AST.log
cat ${model_dir}/AST.3/log.txt | grep ASR -A1 | grep -v "MT\|AST" > ${model_dir}/AST.3/ASR.log
cat ${model_dir}/AST.3/log.txt | grep MT -A1 | grep -v "ASR\|AST" > ${model_dir}/AST.3/MT.log

cat ${model_dir}/ASR.1/log.txt | grep "score=" | cut -d' ' -f4,4 | cut -d'=' -f2,2 > ${eval_dir}/ASR.scores
cat ${model_dir}/ASR.1/log.txt | grep "step " | cut -d' ' -f4,4 > ${eval_dir}/ASR.steps

cat ${model_dir}/MT.1/log.txt | grep "dev score=" | cut -d' ' -f4,4 | cut -d'=' -f2,2 > ${eval_dir}/MT.scores
cat ${model_dir}/MT.1/log.txt | grep "step " | cut -d' ' -f4,4 > ${eval_dir}/MT.steps

cat ${model_dir}/AST.3/ASR.log | grep "score=" | cut -d' ' -f7,7 | cut -d'=' -f2,2 > ${eval_dir}/ASR.multitask.scores
cat ${model_dir}/AST.3/ASR.log | grep "step " | cut -d' ' -f5,5 | awk '{ print $1 + 54000 }' > ${eval_dir}/ASR.multitask.steps

cat ${model_dir}/AST.3/MT.log | grep "dev score=" | cut -d' ' -f5,5 | cut -d'=' -f2,2 > ${eval_dir}/MT.multitask.scores
cat ${model_dir}/AST.3/MT.log | grep "step " | cut -d' ' -f5,5 | awk '{ print $1 + 61000 }'  > ${eval_dir}/MT.multitask.steps

paste -d',' ${eval_dir}/ASR.{steps,scores} > ${eval_dir}/ASR.csv
paste -d',' ${eval_dir}/MT.{steps,scores} > ${eval_dir}/MT.csv

paste -d',' ${eval_dir}/ASR.multitask.{steps,scores} > ${eval_dir}/ASR.multitask.csv
paste -d',' ${eval_dir}/MT.multitask.{steps,scores} > ${eval_dir}/MT.multitask.csv

${config_dir}/plot-multitask.py ${eval_dir}

cat ${model_dir}/AST.1/log.txt | grep "dev score=" | cut -d' ' -f4,4 | cut -d'=' -f2,2 > ${eval_dir}/AST.1.scores
cat ${model_dir}/AST.1/log.txt | grep "step " | cut -d' ' -f4,4 > ${eval_dir}/AST.1.steps

cat ${model_dir}/AST.2/log.txt | grep "dev score=" | cut -d' ' -f4,4 | cut -d'=' -f2,2 > ${eval_dir}/AST.2.scores
cat ${model_dir}/AST.2/log.txt | grep "step " | cut -d' ' -f4,4 > ${eval_dir}/AST.2.steps

cat ${model_dir}/AST.3/log.txt | grep "dev score=" | cut -d' ' -f4,4 | cut -d'=' -f2,2 > ${eval_dir}/AST.3.scores
cat ${model_dir}/AST.3/log.txt | grep "step " | cut -d' ' -f4,4 > ${eval_dir}/AST.3.steps

paste -d',' ${eval_dir}/AST.1.{steps,scores} > ${eval_dir}/AST.1.csv
paste -d',' ${eval_dir}/AST.2.{steps,scores} > ${eval_dir}/AST.2.csv
paste -d',' ${eval_dir}/AST.3.{steps,scores} > ${eval_dir}/AST.3.csv

${config_dir}/plot-AST.py ${eval_dir}
