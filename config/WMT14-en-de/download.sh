#!/usr/bin/env bash

raw_data=raw_data/WMT14_en_de

mkdir -p ${raw_data}

function process {
    sed s/##AT##/@/g | sed s/##UNDERSCORE##/_/g | sed s/##STAR##/*/g | sed s/@-@/-/g | sed "s/&quot;/\"/g" |\
    sed "s/&apos;/\'/g" | sed "s/&gt;/>/g" | sed "s/&lt;/</g" | sed "s/&#124;/\|/g" | sed "s/&#91;/[/g" |\
    sed "s/&#93;/]/g" | sed "s/&amp;/\&/g" | sed "s/\s*& nbsp ;//g" | sed "s/\s*& nbsp ;//g" |\
    sed "s/\s*& # 160 ;//g" | sed "s/& amp ;/\&/g" | sed "s/& # 45 ;/-/g"
}

for corpus in train newstest2012 newstest2013 newstest2014 newstest2015
do
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/${corpus}.en -O- | process > ${raw_data}/${corpus}.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/${corpus}.de -O- | process > ${raw_data}/${corpus}.de
done
