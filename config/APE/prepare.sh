#!/usr/bin/env bash

raw_data=raw_data/APE
data_dir=data/APE

max_vocab_size=30000

rm -rf ${data_dir}
mkdir -p ${data_dir}

for ext in mt pe src
do
    cat ${raw_data}/train.${ext} > ${data_dir}/train.small.${ext}
    cat ${raw_data}/{train,train.2017}.${ext} > ${data_dir}/train.${ext}
    cat ${raw_data}/500K.${ext} > ${data_dir}/train.large.${ext}
    for i in {1..10}   # oversample PE data
    do
        cat ${raw_data}/{train,train.2017}.${ext} >> ${data_dir}/train.large.${ext}
    done

    cp ${raw_data}/dev.${ext} ${data_dir}/dev.${ext}
    cp ${raw_data}/test.${ext} ${data_dir}/test.${ext}
    cp ${raw_data}/test.2017.${ext} ${data_dir}/test.2017.${ext}
done

for corpus in train.small train train.large dev test test.2017
do
    scripts/post_editing/extract-edits.py ${data_dir}/${corpus}.{mt,pe} > ${data_dir}/${corpus}.edits
done

cat ${data_dir}/train.small.{mt,pe} > ${data_dir}/train.small.de
cat ${data_dir}/train.{mt,pe} > ${data_dir}/train.de
cat ${data_dir}/train.large.{mt,pe} > ${data_dir}/train.large.de

scripts/prepare-data.py ${data_dir}/train.small src de edits ${data_dir} --mode vocab --vocab-size 0 --vocab-prefix vocab.small
scripts/prepare-data.py ${data_dir}/train src de edits ${data_dir} --mode vocab --vocab-size 0
scripts/prepare-data.py ${data_dir}/train.large src de edits ${data_dir} --mode vocab --vocab-size 0 --vocab-prefix vocab.large --vocab-size ${max_vocab_size}

for vocab in vocab vocab.small vocab.large  # joint vocabularies
do
    cp ${data_dir}/${vocab}.de ${data_dir}/${vocab}.mt
    cp ${data_dir}/${vocab}.de ${data_dir}/${vocab}.pe
done
rm ${data_dir}/*.de
