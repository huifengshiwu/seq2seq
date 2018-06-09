#!/usr/bin/env bash

set -e

if [[ $# -lt 8 ]]
then
    echo "wrong number of arguments supplied: $#"
    exit 0
fi

if [ -z ${MOSES} ] || [ -z ${GIZA} ]
then
    echo "variables MOSES and/or GIZA undefined"
    exit 0
fi

model_dir=`readlink -f $1`
data_dir=`readlink -f $2`
corpus=${data_dir}/$3
dev_corpus=${data_dir}/$4
src_ext=$5
trg_ext=$6
lm_path=${data_dir}/$7
lm_corpus=`basename ${lm_path}`
lm_order=$8
cores=`lscpu | grep -Po "^(CPU\(s\)|Processeur\(s\)).?:\s+\K\d+$"`

echo "training on ${cores} CPUs"

rm -rf ${model_dir}
mkdir -p ${model_dir}

echo "training language model, corpus=${lm_corpus}, order=${lm_order}" | ts
${MOSES}/bin/lmplz -o ${lm_order} --discount_fallback < ${lm_path}.${trg_ext} > ${model_dir}/${lm_corpus}.${trg_ext}.arpa 2>${model_dir}/train.log

echo "training moses, corpus=${corpus}" | ts
${MOSES}/scripts/training/train-model.perl -root-dir ${model_dir} \
-corpus ${corpus} -f ${src_ext} -e ${trg_ext} -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:${lm_order}:${model_dir}/${lm_corpus}.${trg_ext}.arpa \
-mgiza -external-bin-dir ${GIZA} \
-mgiza-cpus ${cores} -cores ${cores} --parallel 2>&1 | ts >> ${model_dir}/train.log

echo "tuning moses, corpus=${dev_corpus}" | ts
${MOSES}/scripts/training/mert-moses.pl ${dev_corpus}.${src_ext} ${dev_corpus}.${trg_ext} \
${MOSES}/bin/moses ${model_dir}/model/moses.ini --mertdir ${MOSES}/bin/ \
--decoder-flags="-threads ${cores}" --working-dir ${model_dir}/mert-work 2>&1 | ts > ${model_dir}/tuning.log

echo "finished" | ts
mv ${model_dir}/mert-work/moses.ini ${model_dir}/moses.tuned.ini
rm -rf ${model_dir}/mert-work

