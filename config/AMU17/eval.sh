#!/bin/bash
model_dir=$1
output_dir=${model_dir}/eval
gpu_id=0

set -e

mkdir -p ${output_dir}

printf "Greedy test:           "; ./seq2seq.sh ${model_dir}/config.yaml --eval test --beam-size 1 --output ${output_dir}/test.greedy.ape --gpu-id ${gpu_id} 2>&1 | tail -n1
printf "Greedy dev:            "; ./seq2seq.sh ${model_dir}/config.yaml --eval dev --beam-size 1 --output ${output_dir}/dev.greedy.ape --gpu-id ${gpu_id} 2>&1 | tail -n1

#parameters=`cat ${model_dir}/log.txt | grep "dev eval" | awk '{ print NR*10000 " " $0 }' | sed "s/\\s\+/ /g" | sort -t' ' -k7,7V | head -n8 | cut -d' ' -f1,1 | xargs printf "${model_dir}/checkpoints/translate-%s "`
parameters=`cat ${model_dir}/log.txt | grep "score=" | awk '{ print NR*10000 " " $0 }' | sort -t' ' -k5,5V | head -n8 | cut -d' ' -f1,1 | xargs printf "${model_dir}/checkpoints/translate-%s "`
printf "Beam 5 test avg(loss): "; ./seq2seq.sh ${model_dir}/config.yaml --eval test --beam-size 5 --checkpoints ${parameters} --average --output ${output_dir}/test.avg.ape --gpu-id ${gpu_id} 2>&1 | tail -n1
printf "Beam 5 dev avg(loss):  "; ./seq2seq.sh ${model_dir}/config.yaml --eval dev --beam-size 5 --checkpoints ${parameters} --average 2>&1 --output ${output_dir}/dev.avg.ape --gpu-id ${gpu_id} | tail -n1

printf "Beam 5 test:           "; ./seq2seq.sh ${model_dir}/config.yaml --eval test --beam-size 5 --output ${output_dir}/test.beam.ape --gpu-id ${gpu_id} 2>&1 | tail -n1
printf "Beam 5 dev:            "; ./seq2seq.sh ${model_dir}/config.yaml --eval dev --beam-size 5 --output ${output_dir}/dev.beam.ape --gpu-id ${gpu_id} 2>&1 | tail -n1

#parameters=`cat ${model_dir}/log.txt | grep "score=" | awk '{ print NR*10000 " " $0 }' | sort -t' ' -k5,5V | head -n8 | cut -d' ' -f1,1 | xargs printf "${model_dir}/checkpoints/translate-%s "`
parameters=`cat ${model_dir}/log.txt | grep "score=" | awk '{ print NR*10000 " " $0 }' | sort -t' ' -k7,7V | head -n8 | cut -d' ' -f1,1 | xargs printf "${model_dir}/checkpoints/translate-%s "`
printf "Beam 5 test avg(ter):  "; ./seq2seq.sh ${model_dir}/config.yaml --eval test --beam-size 5 --checkpoints ${parameters} --average --output ${output_dir}/test.avg.ter.ape --gpu-id ${gpu_id} 2>&1 | tail -n1
printf "Beam 5 dev avg(ter):   "; ./seq2seq.sh ${model_dir}/config.yaml --eval dev --beam-size 5 --checkpoints ${parameters} --average --output ${output_dir}/dev.avg.ter.ape --gpu-id ${gpu_id} 2>&1 | tail -n1
