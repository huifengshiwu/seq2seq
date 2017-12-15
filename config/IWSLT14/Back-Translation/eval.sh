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

scripts/decode-moses.sh ${model_dir}/Back_Translation/moses.tuned.ini ${tmp_dir} ${data_dir}/test.en ${model_dir}/Back_Translation/test.mt 1>/dev/null 2>/dev/null
scripts/score.py ${model_dir}/Back_Translation/test.mt ${data_dir}/test.de --bleu

scripts/decode-moses.sh ${model_dir}/Back_Translation_LM/moses.tuned.ini ${tmp_dir} ${data_dir}/test.en ${model_dir}/Back_Translation_LM/test.mt 1>/dev/null 2>/dev/null
scripts/score.py ${model_dir}/Back_Translation_LM/test.mt ${data_dir}/test.de --bleu

rm -rf ${new_dir}
