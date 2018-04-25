#!/usr/bin/env bash

root_dir=models/APE/$1
size=$2
gpu_id=$3
eval_dir=${root_dir}/eval
log_file=${eval_dir}/${size}.log

rm -f ${log_file}
mkdir -p ${eval_dir}

for index in 1 2 3 4
do
    model=${size}.${index}
    model_dir=${root_dir}/${model}
    rm -rf ${model_dir}.avg
    checkpoints=`find ${model_dir}/checkpoints/best-* -printf "%f\n" | cut -d'.' -f1,1 | sort | uniq | cut -d'-' -f2,2 | xargs printf " %s|" | sed s/\|$//`
    checkpoints=`cat ${model_dir}/checkpoints/scores.txt | grep -P "${checkpoints}" | sed s/-// | sort -g | head -n4 | cut -d' ' -f2,2 | xargs printf "${model_dir}/checkpoints/best-%s "`
    echo ${checkpoints}
    ./seq2seq.sh ${model_dir}/config.yaml --average --checkpoints ${checkpoints} --save --model-dir ${model_dir}.avg --no-gpu >/dev/null 2>&1
    rename "s/translate-[0-9]*/average/" ${model_dir}.avg/checkpoints/translate-*
    mv ${model_dir}.avg/checkpoints/average.* ${model_dir}/checkpoints/
    rm -rf ${model_dir}.avg
done

function header {
    printf "%s %-40s" `date +"%H:%M:%S"` $1 >> ${log_file}
}

function filter {
    tail -n1 | grep -Po "(ter|bleu1|bleu|wer|penalty|ratio)=[0-9]*.?[0-9]*" | xargs printf "%s " | sed "s/ $/\n/" >> ${log_file}
}

for beam_size in 1 6
do
    for corpus in dev test test.2017
    do
        for index in 1 2 3 4
        do
            model=${size}.${index}
            model_dir=${root_dir}/${model}

            output=${corpus}.${model}.beam${beam_size}
            header ${output}
            ./seq2seq.sh ${model_dir}/config.yaml --eval ${corpus} --beam-size ${beam_size} --gpu-id ${gpu_id} --raw-output --output ${eval_dir}/${output}.raw 2>&1 | filter
            scripts/post_editing/reverse-edits.py data/APE/${corpus}.mt ${eval_dir}/${output}.raw > ${eval_dir}/${output}.out

            output=${corpus}.${model}.avg.beam${beam_size}
            header ${output}
            ./seq2seq.sh ${model_dir}/config.yaml --eval ${corpus} --beam-size ${beam_size} --checkpoints ${model_dir}/checkpoints/average --gpu-id ${gpu_id} --raw-output --output ${eval_dir}/${output}.raw 2>&1 | filter
            scripts/post_editing/reverse-edits.py data/APE/${corpus}.mt ${eval_dir}/${output}.raw > ${eval_dir}/${output}.out
        done

        output=${corpus}.${size}.ensemble.beam${beam_size}
        header ${output}
        ./seq2seq.sh ${root_dir}/${size}.1/config.yaml --eval ${corpus} --beam-size ${beam_size} --ensemble --checkpoints ${root_dir}/${size}.{1,2,3,4}/checkpoints/best --gpu-id ${gpu_id} --raw-output --output ${eval_dir}/${output}.raw 2>&1 | filter
        scripts/post_editing/reverse-edits.py data/APE/${corpus}.mt ${eval_dir}/${output}.raw > ${eval_dir}/${output}.out

        output=${corpus}.${size}.ensemble.avg.beam${beam_size}
        header ${output}
        ./seq2seq.sh ${root_dir}/${size}.1/config.yaml --eval ${corpus} --beam-size ${beam_size} --ensemble --checkpoints ${root_dir}/${size}.{1,2,3,4}/checkpoints/average --gpu-id ${gpu_id} --raw-output --output ${eval_dir}/${output}.raw 2>&1 | filter
        scripts/post_editing/reverse-edits.py data/APE/${corpus}.mt ${eval_dir}/${output}.raw > ${eval_dir}/${output}.out
    done
done
