#!/usr/bin/env bash

# Filtered WMT14 data, available on http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/

raw_data=raw_data/WMT14
data_dir=data/WMT14

rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/WMT14.fr-en fr en ${data_dir} --no-tokenize \
--dev-corpus ${raw_data}/ntst1213.fr-en \
--test-corpus ${raw_data}/ntst14.fr-en \
--vocab-size 30000 --shuffle --seed 1234
