#!/usr/bin/env bash

data_dir=data/LibriSpeech
raw_data=raw_data/LibriSpeech
mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/train fr en google.fr ${data_dir} --lowercase --no-tokenize en \
--dev-corpus ${raw_data}/dev --test-corpus ${raw_data}/test --normalize-punk fr google.fr --lang fr en fr \
--mode prepare

scripts/speech/extract.py ${raw_data}/train.tar ${data_dir}/train.npz
scripts/speech/extract.py ${raw_data}/dev.tar ${data_dir}/dev.npz
scripts/speech/extract.py ${raw_data}/test.tar ${data_dir}/test.npz
scripts/speech/extract.py ${raw_data}/other.tar ${data_dir}/other.npz

cat ${data_dir}/{train,train.google}.fr > ${data_dir}/train+google.fr
cat ${data_dir}/{train,train}.en > ${data_dir}/train+google.en
scripts/speech/cat.py ${data_dir}/{train,train}.npz ${data_dir}/train+google.npz
scripts/speech/shuf.py ${data_dir}/train+google.npz --input-txt ${data_dir}/train+google.{fr,en} 
scripts/speech/shuf.py ${data_dir}/train.npz --input-txt ${data_dir}/train.{fr,en,google.fr} 

# prepare BPE
scripts/bpe/learn_bpe.py -i ${data_dir}/train.en -s 30000 -o ${data_dir}/bpe.en

# apply BPE
scripts/prepare-data.py ${data_dir}/train en ${data_dir} --no-tokenize --subwords --bpe-path ${data_dir}/bpe \
--output train.sub --dev-prefix dev.sub --test-prefix test.sub --vocab-prefix vocab.sub \
--dev-corpus ${data_dir}/dev --test-corpus ${data_dir}/test 

scripts/prepare-data.py ${data_dir}/train+google en ${data_dir} --no-tokenize --subwords \
--bpe-path ${data_dir}/bpe --output train+google.sub --mode prepare

# prepare word-level vocabs
scripts/prepare-data.py ${data_dir}/train+google fr en ${data_dir} --mode vocab --vocab-size 30000

# prepare character-level vocabs
scripts/prepare-data.py ${data_dir}/train+google fr en ${data_dir} --mode vocab --character-level \
--vocab-size 200 --vocab-prefix vocab.char

for corpus in train+google train dev test
do
    cp ${data_dir}/${corpus}.fr ${data_dir}/${corpus}.char.fr
    cp ${data_dir}/${corpus}.en ${data_dir}/${corpus}.char.en
done

