#!/usr/bin/env bash

data_dir=data/audiobooks_plus
raw_data=raw_data/audiobooks
mkdir -p ${data_dir}

config/audiobooks/rename-audio-files.py ${raw_data}/other/audiofiles ${raw_data}/other/alignments.meta ${raw_data}/other/renamed

find ${raw_data}/other/renamed | tail -n+2 | scripts/speech/extract-audio-features.py --output ${data_dir}/other.feats41

scripts/speech/audio-features-cat.py data/audiobooks/train.feats41 ${data_dir}/{other,train.full}.feats41
cp data/audiobooks/{dev,test}.feats41 ${data_dir}

cat ${raw_data}/{train,other}.fr | perl -pe 's/^(-+)([^\s-])/$1 $2/g' > ${data_dir}/train.raw.fr
cat ${raw_data}/{train,other}.en > ${data_dir}/train.raw.en

scripts/prepare-data.py ${data_dir}/train.full.raw fr en ${data_dir} --lowercase --no-tokenize en \
--dev-corpus ${raw_data}/dev --test-corpus ${raw_data}/test --normalize-punk fr --vocab-size 30000

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

scripts/speech/audio-features-shuf.py ${data_dir}/train.feats41 ${data_dir}/train.shuf.feats41 \
--input-txt ${data_dir}/train.{fr,en} ${data_dir}/train.sub.{fr,en} ${data_dir}/train.jsub.{fr,en} \
--output-txt ${data_dir}/train.shuf.{fr,en} ${data_dir}/train.shuf.sub.{fr,en} ${data_dir}/train.shuf.jsub.{fr,en}
