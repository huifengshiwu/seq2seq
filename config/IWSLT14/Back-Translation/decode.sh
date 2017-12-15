#!/usr/bin/env bash

set -e

data_dir=data/IWSLT14
model_dir=models/IWSLT14/Back_Translation_LM

file_id=$1

input_filename=${model_dir}/data/${file_id}
output_filename=${model_dir}/output/${file_id}

new_dir=`mktemp -d`
tmp_dir=${new_dir}/moses
scripts/decode-moses.sh ${model_dir}/moses.tuned.ini ${tmp_dir} ${input_filename} ${output_filename} 1>/dev/null 2>/dev/null
rm -rf ${new_dir}
