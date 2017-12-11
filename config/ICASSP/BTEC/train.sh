#!/usr/bin/env bash

config_dir=config/ICASSP/BTEC
model_dir=models/ICASSP/BTEC

./seq2seq.sh ${config_dir}/ASR.yaml --model-dir ${model_dir}/ASR.1 --train -v
./seq2seq.sh ${config_dir}/ASR.yaml --model-dir ${model_dir}/ASR.2 --train -v

./seq2seq.sh ${config_dir}/MT.yaml --model-dir ${model_dir}/MT.1 --train -v
./seq2seq.sh ${config_dir}/MT.yaml --model-dir ${model_dir}/MT.2 --train -v

./seq2seq.sh ${config_dir}/AST.yaml --model-dir ${model_dir}/AST.1 --train -v
./seq2seq.sh ${config_dir}/AST.yaml --model-dir ${model_dir}/AST.2 --train -v --checkpoints ${model_dir}/{MT.1,ASR.1}/checkpoints/best
./seq2seq.sh ${config_dir}/Multi-Task.yaml --model-dir ${model_dir}/AST.3 --train -v --checkpoints ${model_dir}/{MT.1,ASR.1}/checkpoints/best
