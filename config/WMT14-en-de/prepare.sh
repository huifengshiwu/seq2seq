#!/usr/bin/env bash

raw_data=raw_data/WMT14_en_de
data_dir=data/WMT14_en_de

mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/train de en ${data_dir} --no-tokenize --vocab-size 30000 --shuffle --seed 1234

cat ${raw_data}/newstest{2012,2013}.en > ${data_dir}/dev.en
cat ${raw_data}/newstest{2012,2013}.de > ${data_dir}/dev.de
cat ${raw_data}/newstest2014.de > ${data_dir}/test.de
cat ${raw_data}/newstest2014.en > ${data_dir}/test.en
cat ${raw_data}/newstest2015.de > ${data_dir}/test.2015.de
cat ${raw_data}/newstest2015.en > ${data_dir}/test.2015.en

for size in 30000 60000
do
name=$(( size / 1000 ))k

cat ${raw_data}/train.{de,en} > ${data_dir}/train.concat
scripts/learn_bpe.py -i ${data_dir}/train.concat -o ${data_dir}/bpe.${name} -s ${size}
cp ${data_dir}/bpe.${name} ${data_dir}/bpe.${name}.de
cp ${data_dir}/bpe.${name} ${data_dir}/bpe.${name}.en
rm ${data_dir}/train.concat

scripts/prepare-data.py ${data_dir}/train de en ${data_dir} --no-tokenize \
--subwords --bpe-path ${data_dir}/bpe.${name} \
--dev-corpus ${data_dir}/dev --dev-prefix dev.${name} \
--test-corpus ${data_dir}/test --test-prefix test.${name} \
--output train.${name} --vocab-prefix vocab.${name} --vocab-size 0

scripts/apply_bpe.py -c ${data_dir}/bpe.${name} -i ${data_dir}/test.2015.en -o ${data_dir}/test.2015.${name}.en
scripts/apply_bpe.py -c ${data_dir}/bpe.${name} -i ${data_dir}/test.2015.de -o ${data_dir}/test.2015.${name}.de

cat ${data_dir}/train.${name}.{de,en} > ${data_dir}/train.concat.${name}
scripts/prepare-data.py ${data_dir}/train concat.${name} ${data_dir} --vocab-size 0 --mode vocab
cp ${data_dir}/vocab.concat.${name} ${data_dir}/vocab.concat.${name}.de
cp ${data_dir}/vocab.concat.${name} ${data_dir}/vocab.concat.${name}.en
rm ${data_dir}/train.concat.*
done
