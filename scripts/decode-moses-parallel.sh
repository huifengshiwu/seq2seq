#!/usr/bin/env bash

set -e

if [[ $# -lt 4 ]]
then
    echo "wrong number of arguments supplied: $#"
    exit 0
fi

if [ -z ${MOSES} ]
then
    echo "variable MOSES undefined"
    exit 0
fi

config_file=`readlink -f $1`
temp_dir=`readlink -f $2`
filename=`readlink -f $3`
output_filename=$4

cores=`lscpu | grep "^CPU(s)\|Processeur(s)" | sed "s/\(CPU(s):\|Processeur(s).:\)\\s*//"`

if [ -d "${temp_dir}" ]
then
    echo "directory ${temp_dir} already exists"
    exit 0
fi

mkdir -p ${temp_dir}/data
mkdir -p ${temp_dir}/output

printf "started: "; date
scripts/split-corpus.py ${filename} ${temp_dir}/data --splits ${cores} --tokens

${MOSES}/scripts/training/filter-model-given-input.pl ${temp_dir}/model ${config_file} ${filename} >/dev/null 2>/dev/null

for i in `ls ${temp_dir}/data`
do
    cat ${temp_dir}/data/${i} | sed "s/|//g" | ${MOSES}/bin/moses -f ${temp_dir}/model/moses.ini -threads 1 > ${temp_dir}/output/${i} 2>/dev/null &
    echo "$!: ${temp_dir}/data/$i => ${temp_dir}/output/$i"
done

function count {
    if [ -f $1 ]
    then
        wc -l $1 | cut -d' ' -f1,1
    else
        echo 0
    fi
}

finished=false
while [ ${finished} = false ]
do
    sleep 60
    finished=true
    for i in `ls ${temp_dir}/data`
    do
        src_lines=`count ${temp_dir}/data/${i}`
        mt_lines=`count ${temp_dir}/output/${i}`

        if [ ${src_lines} != ${mt_lines} ]
        then
            finished=false
        fi
    done
done

cat ${temp_dir}/output/* > ${output_filename}
rm -rf ${temp_dir}
printf "finished: "; date
