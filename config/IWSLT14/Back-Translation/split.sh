#!/usr/bin/env bash

data_dir=data/IWSLT14
model_dir=models/IWSLT14/Back_Translation_LM
scripts/split-corpus.py ${data_dir}/TED.en ${model_dir}/data --splits 12 --tokens
mkdir -p ${model_dir}/output
