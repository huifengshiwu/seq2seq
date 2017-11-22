#!/usr/bin/env bash

model_dir=models/ICASSP/BTEC
eval_dir=${model_dir}/eval
data_dir=data/BTEC
gpu_id=0

mkdir -p ${eval_dir}

function run
{
./seq2seq.sh ${model_dir}/MT.1/config.yaml --eval $1 -v --output ${eval_dir}/$1.MT.1.greedy.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/MT.1/config.yaml --eval $1 -v --beam-size 8 --output ${eval_dir}/$1.MT.1.beam8.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/MT.1/config.yaml --eval $1 -v --beam-size 8 --ensemble --checkpoints ${model_dir}/MT.{1,2}/checkpoints/best --output ${eval_dir}/$1.MT.1-2.ensemble.beam8.out --gpu-id ${gpu_id}

./seq2seq.sh ${model_dir}/ASR.1/config.yaml --eval $1 -v --output ${eval_dir}/$1.ASR.1.greedy.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/ASR.1/config.yaml --eval $1 -v --beam-size 8 --output ${eval_dir}/$1.ASR.1.beam8.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/ASR.1/config.yaml --eval $1 -v --beam-size 8 --ensemble --checkpoints ${model_dir}/ASR.{1,2}/checkpoints/best --output ${eval_dir}/$1.ASR.1-2.ensemble.beam8.out --gpu-id ${gpu_id}

./seq2seq.sh ${model_dir}/AST.1/config.yaml --eval $1 -v --output ${eval_dir}/$1.AST.1.greedy.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/AST.2/config.yaml --eval $1 -v --output ${eval_dir}/$1.AST.2.greedy.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/AST.3/config.yaml --eval $1 -v --output ${eval_dir}/$1.AST.3.greedy.out --gpu-id ${gpu_id}

./seq2seq.sh ${model_dir}/AST.1/config.yaml --eval $1 -v --beam-size 8 --output ${eval_dir}/$1.AST.1.beam8.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/AST.2/config.yaml --eval $1 -v --beam-size 8 --output ${eval_dir}/$1.AST.2.beam8.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/AST.3/config.yaml --eval $1 -v --beam-size 8 --output ${eval_dir}/$1.AST.3.beam8.out --gpu-id ${gpu_id}

./seq2seq.sh ${model_dir}/MT.1/config.yaml --eval ${eval_dir}/$1.ASR.1.greedy.out ${data_dir}/$1.en -v --output ${eval_dir}/$1.ASR.1.MT.1.cascaded.greedy.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/MT.1/config.yaml --eval ${eval_dir}/$1.ASR.1.beam8.out ${data_dir}/$1.en -v --beam-size 8 --output ${eval_dir}/$1.ASR.1.MT.1.cascaded.beam8.out --gpu-id ${gpu_id}
./seq2seq.sh ${model_dir}/MT.1/config.yaml --eval ${eval_dir}/$1.ASR.1-2.ensemble.beam8.out ${data_dir}/$1.en -v --beam-size 8 --ensemble --checkpoints ${model_dir}/MT.{1,2}/checkpoints/best --output ${eval_dir}/$1.ASR.1-2.MT.1-2.cascaded.ensemble.beam8.out --gpu-id ${gpu_id}
}

run test >> ${eval_dir}/test.log.txt 2>&1
run dev >> ${eval_dir}/dev.log.txt 2>&1
