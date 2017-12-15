#!/usr/bin/env bash

set -e

if [[ $# -lt 4 ]]
then
    echo "wrong number of arguments supplied: $#"
    exit 0
fi

if [ -z ${MOSES} ] || [ -z ${GIZA} ]
then
    echo "variables MOSES and/or GIZA undefined"
    exit 0
fi

config_file=`readlink -f $1`
temp_dir=`readlink -f $2`
filename=`readlink -f $3`
output_filename=$4

cores=`lscpu | grep "^CPU(s):" | sed "s/CPU(s):\\s*//"`

if [ -d "${temp_dir}" ]
then
    echo "directory ${temp_dir} already exists"
    exit 0
fi

mkdir -p ${temp_dir}/data
mkdir -p ${temp_dir}/output

printf "started: "; date
scripts/split-corpus.py ${filename} ${temp_dir}/data --splits ${cores} --tokens

${MOSES}/scripts/training/filter-model-given-input.pl ${temp_dir}/model ${config_file} ${filename} 2>/dev/null

for i in `ls ${temp_dir}/data`
do
    echo "${temp_dir}/data/$i => ${temp_dir}/output/$i"
    cat ${temp_dir}/data/${i} | sed "s/|//g" | ${MOSES}/bin/moses -f ${temp_dir}/model/moses.ini -threads 1 > ${temp_dir}/output/${i} 2>/dev/null &
done

finished=false
while [ finished = false ]
do
    finished=true
    for i in `ls ${temp_dir}/data`
    do
        src_lines=`wc -l ${temp_dir}/data/${i}`
        mt_lines=`wc -l ${temp_dir}/output/${i}`

        if [ src_lines != mt_line ]
        then
            finished=false
            break
        fi
    done
    echo "test"
    sleep 30
done

cat ${temp_dir}/output/* > ${output_filename}
#rm -rf ${temp_dir}
printf "finished: "; date
