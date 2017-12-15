#!/usr/bin/env bash

set -e

data_dir=data/IWSLT14

rm -rf fast_align-master

wget https://github.com/clab/fast_align/archive/master.zip
unzip master.zip
rm master.zip
cd fast_align-master
mkdir build
cd build
cmake ..
make
cd ../..

corpus=train
fast_align=fast_align-master/build

scripts/join.py ${data_dir}/${corpus}.{de,en} > ${data_dir}/${corpus}.de-en
${fast_align}/fast_align -i ${data_dir}/${corpus}.de-en -d -o -v > ${data_dir}/${corpus}.forward.align
${fast_align}/fast_align -i ${data_dir}/${corpus}.de-en -d -o -v -r > ${data_dir}/${corpus}.reverse.align
${fast_align}/atools -i ${data_dir}/${corpus}.forward.align -j ${data_dir}/${corpus}.reverse.align -c grow-diag-final-and > ${data_dir}/${corpus}.align

scripts/extract-lexicon.py ${data_dir}/${corpus}.{de,en,align} > ${data_dir}/${corpus}.lexicon
python3 -c "print('\n'.join(line.rstrip() for line in open('${data_dir}/${corpus}.lexicon') if not line[0].isupper() and not line.split()[0] == line.split()[1]))" > ${data_dir}/${corpus}.lexicon.purged

rm -rf fast_align-master
rm ${data_dir}/${corpus}.de-en
