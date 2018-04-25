#!/usr/bin/env bash

# start by downloading all the data files from "http://www.statmt.org/wmt17/ape-task.html", "EN-DE" language pair
# extract all the text files into the same "raw_data/APE" directory (no sub-directories)
# also copy the "true.{en,de}" and "{en,de}.bpe" files
# then run the following commands
raw_data=raw_data/APE
cur_dir=`pwd`
cd ${raw_data}

for ext in src mt pe
do
    mv en-de.train.${ext} train.2017.${ext}
    mv en-de.${ext}.test.2017 test.2017.${ext}
done
cd ${cur_dir}
# then run the pre-processing scripts "config/{APE,AMU}/prepare.sh"
