#!/usr/bin/env bash

set -e

data_dir=data/IWSLT14
model_dir=models/IWSLT14
train_script=scripts/train-moses.sh

# model_dir data_dir corpus dev_corpus src_ext trg_ext lm_corpus lm_order
${train_script} ${model_dir}/SMT ${data_dir} train dev de en train 3
${train_script} ${model_dir}/SMT_subwords ${data_dir} train.jsub dev.jsub de en train.jsub 3

cat ${data_dir}/{train,TED}.en > ${data_dir}/train+TED.en
${train_script} ${model_dir}/SMT_LM ${data_dir} train dev de en train+TED 3
cat ${data_dir}/{train,TED}.jsub.en > ${data_dir}/train+TED.jsub.en
${train_script} ${model_dir}/SMT_LM_subwords ${data_dir} train.jsub dev.jsub de en train+TED.jsub 3

${train_script} ${model_dir}/SMT_huge_LM ${data_dir} train dev de en OpenSubtitles 5

