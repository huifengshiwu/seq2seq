#!/usr/bin/env bash

# first download the Augmented LibriSpeech zip files inside `archive_dir`
archive_dir=raw_data/LibriSpeech/archives
raw_data=raw_data/LibriSpeech
mkdir -p ${raw_data}

#unzip -q ${archive_dir}/dev.zip -d ${raw_data}
#unzip -q ${archive_dir}/test.zip -d ${raw_data}
#unzip -q ${archive_dir}/train_100h.zip -d ${raw_data}
#unzip -q ${archive_dir}/train_130h_additional.zip -d ${raw_data}
#unzip -q ${archive_dir}/database.zip -d ${raw_data}

function clean-dash {
    perl -pe 's/^(-+)([^\s-])/$1 $2/g'
}
function clean-quotes {
    perl -pe "s/\"\s*\"/\"/g"
}

for corpus in dev test train other
do
    cp ${raw_data}/${corpus}/${corpus}.en ${raw_data}
    if [ ${corpus} = train ] || [ ${corpus} = other ]
    then
        cat ${raw_data}/${corpus}/${corpus}.fr | clean-quotes | clean-dash > ${raw_data}/${corpus}.fr
    else
        cat ${raw_data}/${corpus}/${corpus}.fr | clean-dash > ${raw_data}/${corpus}.fr
    fi
    cat ${raw_data}/${corpus}/${corpus}_gtranslate.fr | clean-quotes > ${raw_data}/${corpus}.google.fr

    rm -f ${raw_data}/${corpus}.orig.en

    alignments=${raw_data}/${corpus}/alignments.meta
    database=${raw_data}/TA-LibriSpeechCorpus.db
    var=1
    lines=`tail -n+2 ${alignments} | wc -l`
    len=`python -c "import math; print(1 + int(math.log10(${lines})))"`
    python3 -c "import sqlite3; conn = sqlite3.connect('${database}'); c = conn.cursor(); c.execute('SELECT audio_filename, source_segment FROM alignments'); d = dict(c.fetchall()); print('\n'.join(d[x.split()[-2]].strip() for x in open('${alignments}').readlines()[1:]))" | clean-quotes > ${raw_data}/${corpus}.orig.en
    
    for filename in `tail -n+2 ${alignments} | cut -f5,5`
    do
        name=`printf "%0${len}d" ${var}`
        #cp ${raw_data}/${corpus}/audiofiles/${filename}.wav ${raw_data}/${corpus}/${name}.wav
        ((var++));
    done

    #rm -r ${raw_data}/${corpus}/audiofiles
    find raw_data/LibriSpeech/${corpus}/ -maxdepth 1 -name "*.wav" > /tmp/files.txt
    tar -cf ${raw_data}/${corpus}.tar -T /tmp/files.txt
    #rm -r ${raw_data}/${corpus}
done

sed -i '1743,1744d' ${raw_data}/test.fr
sed -i '1743,1744d' ${raw_data}/test.en
