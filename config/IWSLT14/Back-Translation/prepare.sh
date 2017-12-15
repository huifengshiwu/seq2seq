#!/usr/bin/env bash

data_dir=data/IWSLT14

cat ${data_dir}/TED.de > ${data_dir}/train.TED.de
cat ${data_dir}/TED.en > ${data_dir}/train.TED.en

for i in {1..10}
do
    cat ${data_dir}/train.de >> ${data_dir}/train.TED.de
    cat ${data_dir}/train.en >> ${data_dir}/train.TED.en
done

scripts/prepare-data.py ${data_dir}/train.TED de en ${data_dir} --mode vocab --vocab-size 30000 --vocab-prefix vocab.TED

scripts/prepare-data.py ${data_dir}/train.TED de en ${data_dir} --subwords --bpe-path ${data_dir}/bpe.joint \
--output train.TED.jsub --vocab-size 0 --vocab-prefix vocab.TED.jsub --no-tokenize

cp ${data_dir}/train.TED.de ${data_dir}/train.TED.char.de
cp ${data_dir}/train.TED.en ${data_dir}/train.TED.char.en

cp ${data_dir}/vocab.char.de ${data_dir}/vocab.TED.char.de
cp ${data_dir}/vocab.char.en ${data_dir}/vocab.TED.char.en
