#!/usr/bin/env bash

data_dir=data/IWSLT14
mkdir -p ${data_dir}

config/IWSLT14/prepare-mixer.sh
mv prep/*.{en,de} ${data_dir}
rename s/.de-en// ${data_dir}/*
rename s/valid/dev/ ${data_dir}/*

scripts/prepare-data.py ${data_dir}/train de en ${data_dir} --mode vocab --vocab-size 0 --min-count 3

rm -rf prep orig
