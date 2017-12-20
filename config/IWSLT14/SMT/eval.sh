#!/usr/bin/env bash

set -e

data_dir=data/IWSLT14
model_dir=models/IWSLT14

if [ -z ${MOSES} ]
then
    echo "variable MOSES undefined"
    exit 0
fi

new_dir=`mktemp -d`
tmp_dir=${new_dir}/moses

rm -rf ${tmp_dir}
scripts/decode-moses.sh ${model_dir}/SMT/moses.tuned.ini ${tmp_dir} ${data_dir}/test.de ${model_dir}/SMT/test.mt 1>/dev/null 2>/dev/null
scripts/score.py ${model_dir}/SMT/test.mt ${data_dir}/test.en --bleu

rm -rf ${tmp_dir}
scripts/decode-moses.sh ${model_dir}/SMT_subwords/moses.tuned.ini ${tmp_dir} ${data_dir}/test.de ${model_dir}/SMT_subwords/test.jsub.mt 1>/dev/null 2>/dev/null
cat ${model_dir}/SMT_subwords/test.jsub.mt | sed "s/@@ //g" | sed "s/@@//g" > ${model_dir}/SMT_subwords/test.mt
scripts/score.py ${model_dir}/SMT_subwords/test.mt ${data_dir}/test.en --bleu

rm -rf ${tmp_dir}
scripts/decode-moses.sh ${model_dir}/SMT_LM/moses.tuned.ini ${tmp_dir} ${data_dir}/test.de ${model_dir}/SMT_LM/test.mt 1>/dev/null 2>/dev/null
scripts/score.py ${model_dir}/SMT_LM/test.mt ${data_dir}/test.en --bleu

rm -rf ${tmp_dir}
scripts/decode-moses.sh ${model_dir}/SMT_LM_subwords/moses.tuned.ini ${tmp_dir} ${data_dir}/test.de ${model_dir}/SMT_LM_subwords/test.jsub.mt 1>/dev/null 2>/dev/null
cat ${model_dir}/SMT_LM_subwords/test.jsub.mt | sed "s/@@ //g" | sed "s/@@//g" > ${model_dir}/SMT_LM_subwords/test.mt
scripts/score.py ${model_dir}/SMT_LM_subwords/test.mt ${data_dir}/test.en --bleu

rm -rf ${tmp_dir}
scripts/decode-moses.sh ${model_dir}/SMT_huge_LM/moses.tuned.ini ${tmp_dir} ${data_dir}/test.de ${model_dir}/SMT_huge_LM/test.mt 1>/dev/null 2>/dev/null
scripts/score.py ${model_dir}/SMT_huge_LM/test.mt ${data_dir}/test.en --bleu

rm -rf ${new_dir}
