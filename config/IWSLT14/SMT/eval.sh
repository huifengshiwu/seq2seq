#!/usr/bin/env bash

data_dir=data/IWSLT14
model_dir=models/IWSLT14

if [ -z ${MOSES} ]
then
    echo "variables MOSES and/or GIZA undefined"
    exit 0
fi

${MOSES}/bin/moses -f ${model_dir}/SMT/moses.tuned.ini < ${data_dir}/test.de > ${model_dir}/SMT/test.mt 2>/dev/null
scripts/score.py ${model_dir}/SMT/test.mt ${data_dir}/test.en

${MOSES}/bin/moses -f ${model_dir}/SMT_subwords/moses.tuned.ini < ${data_dir}/test.jsub.de > ${model_dir}/SMT_subwords/test.jsub.mt 2>/dev/null
cat ${model_dir}/SMT_subwords/test.jsub.mt | sed "s/@@ //g" | sed "s/@@//g" > ${model_dir}/SMT_subwords/test.mt
scripts/score.py ${model_dir}/SMT_subwords/test.mt ${data_dir}/test.en

${MOSES}/bin/moses -f ${model_dir}/SMT_LM/moses.tuned.ini < ${data_dir}/test.de > ${model_dir}/SMT_LM/test.mt 2>/dev/null
scripts/score.py ${model_dir}/SMT_LM/test.mt ${data_dir}/test.en

${MOSES}/bin/moses -f ${model_dir}/SMT_LM_subwords/moses.tuned.ini < ${data_dir}/test.jsub.de > ${model_dir}/SMT_LM_subwords/test.jsub.mt 2>/dev/null
cat ${model_dir}/SMT_LM_subwords/test.jsub.mt | sed "s/@@ //g" | sed "s/@@//g" > ${model_dir}/SMT_LM_subwords/test.mt
scripts/score.py ${model_dir}/SMT_LM_subwords/test.mt ${data_dir}/test.en
