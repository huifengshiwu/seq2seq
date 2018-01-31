#!/usr/bin/env bash

data_dir=data/ICASSP/books
raw_data=raw_data/audiobooks
mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/train fr en google.fr ${data_dir} --lowercase --no-tokenize en \
--dev-corpus ${raw_data}/dev --test-corpus ${raw_data}/test --normalize-punk fr google.fr --lang fr en fr \
--mode prepare

cat ${data_dir}/{train,train.google}.fr > ${data_dir}/train+google.fr
cat ${data_dir}/{train,train}.en > ${data_dir}/train+google.en
scripts/speech/audio-features-cat.py ${raw_data}/{train,train}.feats41 ${data_dir}/train+google.feats41

scripts/speech/audio-features-shuf.py ${data_dir}/train+google.feats41 ${data_dir}/train+google.shuf.feats41 \
--input-txt ${data_dir}/train+google.{fr,en} --output-txt ${data_dir}/train+google.shuf.{fr,en}
rename -f s/.shuf// ${data_dir}/train+google.*

scripts/speech/audio-features-shuf.py ${raw_data}/train.feats41 ${data_dir}/train.shuf.feats41 \
--input-txt ${data_dir}/train.{fr,en} --output-txt ${data_dir}/train.shuf.{fr,en}
rename -f s/.shuf// ${data_dir}/train.*

cp ${raw_data}/{dev,test}.feats41 ${data_dir}

# prepare BPE
cat ${data_dir}/train.{fr,en} > ${data_dir}/train.joint
scripts/learn_bpe.py -i ${data_dir}/train.joint -s 30000 -o ${data_dir}/bpe.joint.fr
cp ${data_dir}/bpe.joint.fr ${data_dir}/bpe.joint.en
scripts/learn_bpe.py -i ${data_dir}/train.fr -s 30000 -o ${data_dir}/bpe.fr
scripts/learn_bpe.py -i ${data_dir}/train.en -s 30000 -o ${data_dir}/bpe.en
rm ${data_dir}/train.joint

# apply BPE
scripts/prepare-data.py ${data_dir}/train fr en ${data_dir} --no-tokenize --subwords --bpe-path ${data_dir}/bpe.joint \
--output train.jsub --dev-prefix dev.jsub --test-prefix test.jsub --mode prepare \
--dev-corpus ${data_dir}/dev --test-corpus ${data_dir}/test

scripts/prepare-data.py ${data_dir}/train fr en ${data_dir} --no-tokenize --subwords --bpe-path ${data_dir}/bpe \
--output train.sub --dev-prefix dev.sub --test-prefix test.sub --mode prepare \
--dev-corpus ${data_dir}/dev --test-corpus ${data_dir}/test

scripts/prepare-data.py ${data_dir}/train+google fr en ${data_dir} --no-tokenize --subwords \
--bpe-path ${data_dir}/bpe.joint --output train+google.jsub --mode prepare

scripts/prepare-data.py ${data_dir}/train+google fr en ${data_dir} --no-tokenize --subwords \
--bpe-path ${data_dir}/bpe --output train+google.sub --mode prepare

# prepare word-level vocabs
scripts/prepare-data.py ${data_dir}/train+google fr en ${data_dir} --mode vocab --vocab-size 30000

# prepare character-level vocabs
scripts/prepare-data.py ${data_dir}/train+google fr en ${data_dir} --mode vocab --character-level \
--vocab-size 200

# copy files
for corpus in train train+google dev test
do
    cp ${data_dir}/${corpus}.fr ${data_dir}/${corpus}.char.fr
    cp ${data_dir}/${corpus}.en ${data_dir}/${corpus}.char.en
done

# create google dev and test set
for corpus in test dev
do
    cp ${data_dir}/${corpus}.en ${data_dir}/${corpus}.google.en
    cp ${data_dir}/${corpus}.en ${data_dir}/${corpus}.google.char.en
    cp ${data_dir}/${corpus}.google.fr ${data_dir}/${corpus}.google.char.fr
    cp ${data_dir}/${corpus}.feats41 ${data_dir}/${corpus}.google.feats41
    cp ${data_dir}/${corpus}.sub.en ${data_dir}/${corpus}.google.sub.en
    cp ${data_dir}/${corpus}.jsub.en ${data_dir}/${corpus}.google.jsub.en
    scripts/apply_bpe.py --codes ${data_dir}/bpe.fr < ${data_dir}/${corpus}.google.fr > ${data_dir}/${corpus}.google.sub.fr
    scripts/apply_bpe.py --codes ${data_dir}/bpe.joint.fr < ${data_dir}/${corpus}.google.fr > ${data_dir}/${corpus}.google.jsub.fr
done
