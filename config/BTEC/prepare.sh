#!/usr/bin/env bash

# speech data preparation script
# this script assumes that the BTEC raw files are in the ${raw_data} dir
raw_data=raw_data/BTEC
data_dir=data/BTEC   # output directory for the processed files (text and audio features)

rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/speech/extract.py ${raw_data}/train-{Fabienne,Helene,Loic,Marion,Michel,Philippe}.tar ${data_dir}/train.concat.npz
scripts/speech/extract.py ${raw_data}/dev-Agnes.tar ${data_dir}/dev.npz
scripts/speech/extract.py ${raw_data}/test-Agnes.tar ${data_dir}/test.npz

rm -f ${data_dir}/train.raw.{fr,en}
for i in {1..6}
do
    cat ${raw_data}/train.fr >> ${data_dir}/train.raw.fr
    cat ${raw_data}/train.en >> ${data_dir}/train.raw.en
done

scripts/prepare-data.py ${data_dir}/train.raw fr en ${data_dir} --lowercase --output train.concat --mode prepare
scripts/prepare-data.py ${raw_data}/dev fr en ${data_dir} --lowercase --output dev --mode prepare
scripts/prepare-data.py ${raw_data}/test fr en ${data_dir} --lowercase --output test --mode prepare
scripts/prepare-data.py ${raw_data}/train fr en ${data_dir} --lowercase

scripts/prepare-data.py ${raw_data}/dev mref.en ${data_dir} --lowercase --output dev --mode prepare --lang en
scripts/prepare-data.py ${raw_data}/test mref.en ${data_dir} --lowercase --output test --mode prepare --lang en

scripts/speech/shuf.py ${data_dir}/train.concat.npz --input-txt ${data_dir}/train.concat.{fr,en}

scripts/prepare-data.py ${data_dir}/train fr en ${data_dir} --mode vocab --character-level --no-tokenize --vocab-prefix vocab.char

for corpus in train.concat train dev test
do
    cp ${data_dir}/${corpus}.fr ${data_dir}/${corpus}.char.fr
    cp ${data_dir}/${corpus}.en ${data_dir}/${corpus}.char.en
done

