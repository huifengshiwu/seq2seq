#!/usr/bin/env bash

data_dir=data/IWSLT14
model_dir=models/IWSLT14
train_script=scripts/train-moses.sh

# model_dir data_dir corpus dev_corpus src_ext trg_ext lm_corpus lm_order
${train_script} ${model_dir}/Back_Translation ${data_dir} train dev en de train 3
cat ${data_dir}/{train,OpenSubtitles}.de > ${data_dir}/train+OpenSubtitles.de
${train_script} ${model_dir}/Back_Translation_LM ${data_dir} train dev en de train+OpenSubtitles 3
