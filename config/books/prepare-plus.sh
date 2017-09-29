#!/usr/bin/env bash

data_dir=data/books_plus
raw_data=raw_data/books
mkdir -p ${data_dir}

cat ${raw_data}/{train,other}.fr | perl -pe 's/^(-+)([^\s-])/$1 $2/g' > ${data_dir}/train.raw.fr
cat ${raw_data}/{train,other}.en > ${data_dir}/train.raw.en

scripts/prepare-data.py ${data_dir}/train.raw fr en ${data_dir} --lowercase --no-tokenize en \
--dev-corpus ${raw_data}/dev --test-corpus ${raw_data}/test --normalize-punk fr --shuffle --seed 1234 \
--vocab-size 30000

rm ${data_dir}/train.raw.{fr,en}
cat ${data_dir}/train.{fr,en} > ${data_dir}/train.joint

scripts/learn_bpe.py -i ${data_dir}/train.joint -s 30000 -o ${data_dir}/bpe.joint.fr
cp ${data_dir}/bpe.joint.fr ${data_dir}/bpe.joint.en
scripts/learn_bpe.py -i ${data_dir}/train.fr -s 30000 -o ${data_dir}/bpe.fr
scripts/learn_bpe.py -i ${data_dir}/train.en -s 30000 -o ${data_dir}/bpe.en

scripts/prepare-data.py ${data_dir}/train fr en ${data_dir} --no-tokenize --subwords --bpe-path ${data_dir}/bpe.joint \
--output train.jsub --dev-prefix dev.jsub --test-prefix test.jsub --vocab-prefix vocab.jsub --vocab-size 0 \
--dev-corpus ${data_dir}/dev --test-corpus ${data_dir}/test

scripts/prepare-data.py ${data_dir}/train fr en ${data_dir} --no-tokenize --subwords --bpe-path ${data_dir}/bpe \
--output train.sub --dev-prefix dev.sub --test-prefix test.sub --vocab-prefix vocab.sub --vocab-size 0 \
--dev-corpus ${data_dir}/dev --test-corpus ${data_dir}/test

scripts/prepare-data.py ${data_dir}/train fr en ${data_dir} --mode vocab --character-level --vocab-prefix vocab.char \
--vocab-size 200
