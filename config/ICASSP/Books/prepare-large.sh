#!/usr/bin/env bash

data_dir=data/audiobooks
raw_data=raw_data/audiobooks

scripts/prepare-data.py ${raw_data}/other fr en google.fr ${data_dir} --mode prepare --output other --lowercase \
--no-tokenize en --normalize-punk fr google.fr --lang fr en fr

cat ${data_dir}/{train,other}.fr > ${data_dir}/train+other.fr
cat ${data_dir}/{train,other}.en > ${data_dir}/train+other.en
scripts/speech/audio-features-cat.py ${raw_data}/{train,other}.feats41 ${data_dir}/train+other.feats41

scripts/speech/audio-features-shuf.py ${data_dir}/train+other.feats41 ${data_dir}/train+other.shuf.feats41 \
--input-txt ${data_dir}/train+other.{fr,en} --output-txt ${data_dir}/train+other.shuf.{fr,en}
rename -f s/.shuf// ${data_dir}/train+other.*

cat ${data_dir}/{train,train.google,other,other.google}.fr > ${data_dir}/train+other+google.fr
cat ${data_dir}/{train,train,other,other}.en > ${data_dir}/train+other+google.en
scripts/speech/audio-features-cat.py ${raw_data}/{train,train,other,other}.feats41 ${data_dir}/train+other+google.feats41

scripts/speech/audio-features-shuf.py ${data_dir}/train+other+google.feats41 ${data_dir}/train+other+google.shuf.feats41 \
--input-txt ${data_dir}/train+other+google.{fr,en} --output-txt ${data_dir}/train+other+google.shuf.{fr,en}
rename -f s/.shuf// ${data_dir}/train+other+google.*

# apply BPE
scripts/prepare-data.py ${data_dir}/train+other fr en ${data_dir} --no-tokenize --subwords \
--bpe-path ${data_dir}/bpe.joint --output train+other.jsub --mode prepare
scripts/prepare-data.py ${data_dir}/train+other+google fr en ${data_dir} --no-tokenize --subwords \
--bpe-path ${data_dir}/bpe.joint --output train+other+google.jsub --vocab-prefix vocab.jsub --vocab-size 0
scripts/prepare-data.py ${data_dir}/train+other fr en ${data_dir} --no-tokenize --subwords \
--bpe-path ${data_dir}/bpe --output train+other.sub --mode prepare
scripts/prepare-data.py ${data_dir}/train+other+google fr en ${data_dir} --no-tokenize --subwords \
--bpe-path ${data_dir}/bpe --output train+other+google.sub --vocab-prefix vocab.sub --vocab-size 0

# prepare word-level vocabs
scripts/prepare-data.py ${data_dir}/train+other+google fr en ${data_dir} --mode vocab --vocab-size 30000

# prepare character-level vocabs
scripts/prepare-data.py ${data_dir}/train+other+google fr en ${data_dir} --mode vocab --character-level \
--vocab-size 200

# copy files
for corpus in train+other train+other+google
do
    cp ${data_dir}/${corpus}.fr ${data_dir}/${corpus}.char.fr
    cp ${data_dir}/${corpus}.en ${data_dir}/${corpus}.char.en
done
