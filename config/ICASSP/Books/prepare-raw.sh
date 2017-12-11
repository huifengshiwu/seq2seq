#!/usr/bin/env bash

archive_dir=raw_data/audiobooks/archives
raw_data=~/audiobooks
mkdir -p ${raw_data}

tar xzf ${archive_dir}/corpus_with_gtranslate.tar.gz -C ${raw_data}
for corpus in dev test train other
do
    cp ${raw_data}/corpus_with_gtranslate/${corpus}/${corpus}.en ${raw_data}
    cp ${raw_data}/corpus_with_gtranslate/${corpus}/${corpus}_gtranslate.txt ${raw_data}/${corpus}.google.fr
    cat ${raw_data}/corpus_with_gtranslate/${corpus}/${corpus}.fr | perl -pe 's/^(-+)([^\s-])/$1 $2/g' > ${raw_data}/${corpus}.fr
done
rm -rf ${raw_data}/corpus_with_gtranslate

tar xzf ${archive_dir}/corpus_icassp_dev.tar.gz -C ${raw_data}
config/audiobooks/rename-audio-files.py ${raw_data}/dev/audiofiles ${raw_data}/dev/alignments.meta ${raw_data}/dev/renamed
find ${raw_data}/dev/renamed | tail -n+2 | sort | scripts/speech/extract-audio-features.py --output ${raw_data}/dev.feats41
rm -rf ${raw_data}/dev

tar xzf ${archive_dir}/corpus_icassp_test.tar.gz -C ${raw_data}
sed -i '1743,1744d' ${raw_data}/test/alignments.meta   # remove problematic lines
config/audiobooks/rename-audio-files.py ${raw_data}/test/audiofiles ${raw_data}/test/alignments.meta ${raw_data}/test/renamed
find ${raw_data}/test/renamed | tail -n+2 | sort | scripts/speech/extract-audio-features.py --output ${raw_data}/test.feats41
rm -rf ${raw_data}/test

tar xzf ${archive_dir}/corpus_icassp_train.tar.gz -C ${raw_data}
config/audiobooks/rename-audio-files.py ${raw_data}/train/audiofiles ${raw_data}/train/alignments.meta ${raw_data}/train/renamed
find ${raw_data}/train/renamed | tail -n+2 | sort | scripts/speech/extract-audio-features.py --output ${raw_data}/train.feats41
rm -rf ${raw_data}/train

tar xzf ${archive_dir}/corpus_icassp_other.tar.gz -C ${raw_data}
config/audiobooks/rename-audio-files.py ${raw_data}/other/audiofiles ${raw_data}/other/alignments.meta ${raw_data}/other/renamed
find ${raw_data}/other/renamed | tail -n+2 | sort | scripts/speech/extract-audio-features.py --output ${raw_data}/other.feats41
rm -rf ${raw_data}/other
