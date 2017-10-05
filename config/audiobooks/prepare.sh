#!/usr/bin/env bash

data_dir=data/audiobooks
raw_data=raw_data/audiobooks
mkdir -p ${data_dir}

config/audiobooks/rename-audio-files.py ${raw_data}/dev/audiofiles ${raw_data}/dev/alignments.meta ${raw_data}/dev/renamed
config/audiobooks/rename-audio-files.py ${raw_data}/test/audiofiles ${raw_data}/test/alignments.meta ${raw_data}/test/renamed
config/audiobooks/rename-audio-files.py ${raw_data}/train/audiofiles ${raw_data}/train/alignments.meta ${raw_data}/train/renamed

find ${raw_data}/dev/renamed | tail -n+2 | scripts/speech/extract-audio-features.py --output ${data_dir}/dev.feats41
find ${raw_data}/test/renamed | tail -n+2 | scripts/speech/extract-audio-features.py --output ${data_dir}/test.feats41
find ${raw_data}/train/renamed | tail -n+2 | scripts/speech/extract-audio-features.py --output ${data_dir}/train.feats41

cat ${raw_data}/train.fr | perl -pe 's/^(-+)([^\s-])/$1 $2/g' > ${data_dir}/train.raw.fr
cp ${raw_data}/train.en ${data_dir}/train.raw.en

scripts/prepare-data.py ${data_dir}/train.raw fr en ${data_dir} --lowercase --no-tokenize en \
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

cp ${data_dir}/train.en ${data_dir}/train.char.en
cp ${data_dir}/train.fr ${data_dir}/train.char.fr
cp ${data_dir}/dev.en ${data_dir}/dev.char.en
cp ${data_dir}/dev.fr ${data_dir}/dev.char.fr
cp ${data_dir}/test.en ${data_dir}/test.char.en
cp ${data_dir}/test.fr ${data_dir}/test.char.fr

cp ${data_dir}/train.shuf.en ${data_dir}/train.shuf.char.en
cp ${data_dir}/train.shuf.fr ${data_dir}/train.shuf.char.fr

# File stats (min length to cover 90%, 95%, 98% or 99% of the corpus)

# train.feats41:
# 90%: 1488, 95%: 1867, 98%: 2385, 99%: 2880
# train.char.en:
# 90%:  222, 95%:  278, 98%:  354, 99%:  420
# train.char.fr:
# 90%:  286, 95%:  359, 98%:  464, 99%:  558

# dev.feats41:
# 90%: 1249, 95%: 1516, 98%: 1759, 99%: 1855
# dev.char.en:
# 90%:  201, 95%:  245, 98%:  316, 99%:  523
# dev.char.fr:
# 90%:  212, 95%:  277, 98%:  367, 99%:  445
