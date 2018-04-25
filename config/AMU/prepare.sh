#!/usr/bin/env bash

raw_data=raw_data/APE
data_dir=data/AMU

mkdir -p ${data_dir}

for ext in src mt pe
do
    if [ ${ext} = "src" ]
    then
        lang=en
    else
        lang=de
    fi

    for corpus in train train.2017 500K 4M dev test test.2017
    do
        cat ${raw_data}/${corpus}.${ext} | scripts/moses/escape-special-chars.perl | scripts/moses/truecase.perl --model ${raw_data}/true.${lang} > ${data_dir}/${corpus}.true.${ext}
        cat ${data_dir}/${corpus}.true.${ext} | scripts/bpe/apply_bpe.py -c ${raw_data}/${lang}.bpe > ${data_dir}/${corpus}.tmp.${ext}
    done

    mv ${data_dir}/dev.tmp.${ext} ${data_dir}/dev.XL.${ext}
    mv ${data_dir}/test.tmp.${ext} ${data_dir}/test.XL.${ext}
    mv ${data_dir}/test.2017.tmp.${ext} ${data_dir}/test.2017.XL.${ext}

    cat ${data_dir}/train.tmp.${ext} | scripts/bpe/get_vocab.py > ${data_dir}/bpe-vocab.small.${ext}
    cat ${data_dir}/{train,train.2017}.tmp.${ext} | scripts/bpe/get_vocab.py > ${data_dir}/bpe-vocab.medium.${ext}
    cat ${data_dir}/{train,train.2017,500K}.tmp.${ext} | scripts/bpe/get_vocab.py > ${data_dir}/bpe-vocab.large.${ext}

    cat ${data_dir}/{4M,500K}.tmp.${ext} > ${data_dir}/train.XL.${ext}
    rm ${data_dir}/{4M,500K}.tmp.${ext}
    cp ${data_dir}/train.XL.${ext} ${data_dir}/train.XXL.${ext}

    for i in {1..20}; do
        cat ${data_dir}/train.tmp.${ext} >> ${data_dir}/train.XL.${ext}
        cat ${data_dir}/{train,train.2017}.tmp.${ext} >> ${data_dir}/train.XXL.${ext}
    done
    rm ${data_dir}/{train,train.2017}.tmp.${ext}

    for size in small medium large
    do
        for corpus in train train.2017 500K dev test test.2017
        do
            cat ${data_dir}/${corpus}.true.${ext} | scripts/bpe/apply_bpe.py -c ${raw_data}/${lang}.bpe --vocabulary-threshold 5 --vocabulary ${data_dir}/bpe-vocab.${size}.${ext} > ${data_dir}/${corpus}.${size}.${ext}
        done
    done 
    rm -f ${data_dir}/*.tmp.* ${data_dir}/*.true.*
    cat ${data_dir}/train.2017.medium.${ext} >> ${data_dir}/train.medium.${ext}
    for i in {1..20}; do
        cat ${data_dir}/{train,train.2017}.large.${ext} >> ${data_dir}/500K.large.${ext}
    done
    mv ${data_dir}/500K.large.${ext} ${data_dir}/train.large.${ext}
    rm -f ${data_dir}/{train.2017,500K}.{small,medium,large}.${ext}
done

for size in small medium large XL
do
    cp ${raw_data}/dev.pe ${data_dir}/dev.${size}.pe.ref
    cp ${raw_data}/test.pe ${data_dir}/test.${size}.pe.ref
    cp ${raw_data}/test.2017.pe ${data_dir}/test.2017.${size}.pe.ref
done

for size in small medium
do
    cat ${data_dir}/train.${size}.{src,mt,pe} > ${data_dir}/train.${size}.all
    scripts/prepare-data.py ${data_dir}/train.${size} all ${data_dir} --mode vocab --vocab-size 0 --vocab-prefix vocab.${size}
    cp ${data_dir}/vocab.${size}.all ${data_dir}/vocab.${size}.mt
    cp ${data_dir}/vocab.${size}.all ${data_dir}/vocab.${size}.pe
    mv ${data_dir}/vocab.${size}.all ${data_dir}/vocab.${size}.src
done
for size in large XL
do
    cat ${data_dir}/train.${size}.{mt,pe} > ${data_dir}/train.${size}.de
    scripts/prepare-data.py ${data_dir}/train.${size} src de ${data_dir} --mode vocab --vocab-size 0 --vocab-prefix vocab.${size}
    cp ${data_dir}/vocab.${size}.de ${data_dir}/vocab.${size}.mt
    cp ${data_dir}/vocab.${size}.de ${data_dir}/vocab.${size}.pe
    rm ${data_dir}/train.${size}.de ${data_dir}/vocab.${size}.de
done

scripts/prepare-data.py ${data_dir}/train.XXL src mt pe ${data_dir} --mode vocab --vocab-size 0 --vocab-prefix vocab.XXL
rename s/\.medium// ${data_dir}/*

