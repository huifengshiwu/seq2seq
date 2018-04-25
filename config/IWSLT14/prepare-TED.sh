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

for ext in de en
do
    scripts/bpe/apply_bpe.py -c ${data_dir}/bpe.joint.${ext} --vocabulary ${data_dir}/bpe-vocab.${ext} --vocabulary-threshold 10 < ${data_dir}/train.TED.${ext} > ${data_dir}/train.TED.jsub.${ext}
done

cp ${data_dir}/train.TED.de ${data_dir}/train.TED.char.de
cp ${data_dir}/train.TED.en ${data_dir}/train.TED.char.en
cp ${data_dir}/vocab.char.de ${data_dir}/vocab.TED.char.de
cp ${data_dir}/vocab.char.en ${data_dir}/vocab.TED.char.en

