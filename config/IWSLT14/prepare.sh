#!/usr/bin/env bash

data_dir=data/IWSLT14
mkdir -p ${data_dir}

config/IWSLT14/prepare-mixer.sh
mv prep/*.{en,de} ${data_dir}
rename s/.de-en// ${data_dir}/*
rename s/valid/dev/ ${data_dir}/*
rm -rf prep orig

scripts/prepare-data.py ${data_dir}/train de en ${data_dir} --mode vocab --vocab-size 30000

scripts/bpe/learn_joint_bpe_and_vocab.py --input ${data_dir}/train.{de,en} -s 30000 -o ${data_dir}/bpe.joint.en --write-vocabulary ${data_dir}/bpe-vocab.de ${data_dir}/bpe-vocab.en
cp ${data_dir}/bpe.joint.en ${data_dir}/bpe.joint.de

cat ${data_dir}/train.{de,en} > ${data_dir}/train.concat
scripts/prepare-data.py ${data_dir}/train concat ${data_dir} --mode vocab --vocab-size 0 --character-level
mv ${data_dir}/vocab.concat ${data_dir}/vocab.char.en
cp ${data_dir}/vocab.char.en ${data_dir}/vocab.char.de
rm ${data_dir}/train.concat

for ext in de en
do
    for corpus in train dev test
    do
        scripts/bpe/apply_bpe.py -c ${data_dir}/bpe.joint.${ext} --vocabulary ${data_dir}/bpe-vocab.${ext} --vocabulary-threshold 10 < ${data_dir}/${corpus}.${ext} > ${data_dir}/${corpus}.jsub.${ext}
    done
done

cat ${data_dir}/train.jsub.{en,de} > ${data_dir}/train.jsub.concat
scripts/prepare-data.py ${data_dir}/train jsub.en jsub.de ${data_dir} --mode vocab --vocab-size 0
scripts/prepare-data.py ${data_dir}/train.jsub concat ${data_dir} --mode vocab --vocab-size 0
mv ${data_dir}/vocab.concat ${data_dir}/vocab.joint.jsub.en
cp ${data_dir}/vocab.joint.jsub.{en,de}
rm ${data_dir}/train.jsub.concat

cp ${data_dir}/train.en ${data_dir}/train.char.en
cp ${data_dir}/train.de ${data_dir}/train.char.de
cp ${data_dir}/dev.en ${data_dir}/dev.char.en
cp ${data_dir}/dev.de ${data_dir}/dev.char.de

wget http://opus.nlpl.eu/download/TED2013/mono/TED2013.en.gz -O ${data_dir}/TED2013.en.gz
#wget http://opus.nlpl.eu/download/OpenSubtitles2018/mono/OpenSubtitles2018.de.gz -O ${data_dir}/OpenSubtitles2018.de.gz
#wget http://opus.nlpl.eu/download/OpenSubtitles2018/mono/OpenSubtitles2018.en.gz -O ${data_dir}/OpenSubtitles2018.en.gz

function filter {
filename=`mktemp`
cat > ${filename} << EOF
import sys
lines = set(list(open('${data_dir}/dev.$1')) + list(open('${data_dir}/test.$1')))
for line in sys.stdin:
    if line not in lines:
        sys.stdout.write(line)
EOF
python3 ${filename}
rm ${filename}
}

gunzip ${data_dir}/TED2013.en.gz --stdout | scripts/moses/lowercase.perl | filter en > ${data_dir}/TED.en
rm ${data_dir}/TED2013.en.gz
#gunzip ${data_dir}/OpenSubtitles2018.de.gz --stdout  | scripts/moses/lowercase.perl | filter de > ${data_dir}/OpenSubtitles.de
#gunzip ${data_dir}/OpenSubtitles2018.en.gz --stdout  | scripts/moses/lowercase.perl | filter en > ${data_dir}/OpenSubtitles.en
