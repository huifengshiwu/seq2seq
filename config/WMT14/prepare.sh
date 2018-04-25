#!/usr/bin/env bash

# Filtered WMT14 data, available on http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/

raw_data=raw_data/WMT14
data_dir=data/WMT14

rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/WMT14.fr-en fr en ${data_dir} --no-tokenize \
--dev-corpus ${raw_data}/ntst1213.fr-en \
--test-corpus ${raw_data}/ntst14.fr-en \
--vocab-size 30000 --shuffle --seed 1234

cat ${raw_data}/WMT14.fr-en.{fr,en} > ${data_dir}/train.concat
scripts/bpe/learn_bpe.py -i ${data_dir}/train.concat -o ${data_dir}/bpe.joint -s 30000
cp ${data_dir}/bpe.joint ${data_dir}/bpe.joint.fr
cp ${data_dir}/bpe.joint ${data_dir}/bpe.joint.en
rm ${data_dir}/train.concat

scripts/prepare-data.py ${raw_data}/WMT14.fr-en fr en ${data_dir} --no-tokenize \
--subwords --bpe-path ${data_dir}/bpe.joint \
--dev-corpus ${raw_data}/ntst1213.fr-en --dev-prefix dev.jsub \
--test-corpus ${raw_data}/ntst14.fr-en --test-prefix test.jsub \
--shuffle --seed 1234 --output train.jsub --mode prepare

cat ${data_dir}/train.jsub.{fr,en} > ${data_dir}/train.concat.jsub
scripts/prepare-data.py ${data_dir}/train concat.jsub ${data_dir} --vocab-size 0 --mode vocab
cp ${data_dir}/vocab.concat.jsub ${data_dir}/vocab.jsub.fr
cp ${data_dir}/vocab.concat.jsub ${data_dir}/vocab.jsub.en
rm ${data_dir}/*.concat.*

