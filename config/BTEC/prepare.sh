#!/usr/bin/env bash

# speech data preparation script
# this script assumes that the BTEC raw files (btec-{train, dev-concat, test1, test2}.{fr,en,mref.en})
# are in ${raw_data_dir}
# and the Voxygen credentials (wsclient.cred) are in experiments/btec_speech/voxygen
raw_data_dir=raw_data/BTEC
raw_audio_dir=raw_data/BTEC_speech
speech_dir=experiments/BTEC_speech
data_dir=data/BTEC_speech   # output directory for the processed files (text and audio features)

mkdir -p ${raw_audio_dir} ${data_dir}

# use Voxygen to generate synthetic speech
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Fabienne Fabienne
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Helene Helene
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Loic Loic
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Marion Marion
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Philippe Philippe

${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Fabienne Fabienne
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Helene Helene
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Loic Loic
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Marion Marion
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Philippe Philippe

${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1-Marion Marion
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2-Marion Marion

# extract 40 MFCC features + frame energy
scripts/speech/extract-audio-features.py ${raw_audio_dir}/train-Agnes/* ${data_dir}/train.Agnes.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/train-Fabienne/* ${data_dir}/train.Fabienne.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/train-Helene/* ${data_dir}/train.Helene.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/train-Loic/* ${data_dir}/train.Loic.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/train-Marion/* ${data_dir}/train.Marion.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/train-Michel/* ${data_dir}/train.Michel.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/train-Philippe/* ${data_dir}/train.Philippe.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/dev-Agnes/* ${data_dir}/dev.Agnes.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/test1-Agnes/* ${data_dir}/test1.Agnes.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/test2-Agnes/* ${data_dir}/test2.Agnes.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/dev-Michel/* ${data_dir}/dev.Michel.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/test1-Michel/* ${data_dir}/test1.Michel.feats41
scripts/speech/extract-audio-features.py ${raw_audio_dir}/test2-Michel/* ${data_dir}/test2.Michel.feats41

# pre-process text data
scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat fr en ${data_dir} --max 0 --lowercase --output dev --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test1 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test2 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-train fr en ${data_dir} --max 0 --lowercase --output train

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat mref.en ${data_dir} --max 0 --lowercase --output dev --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test1 --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test2 --mode prepare --lang en

# Agnes is only used for development/testing, to test model's ability to generalize to other voices
scripts/speech/audio-features-cat.py ${data_dir}/train.{Helene,Fabienne,Loic,Marion,Michel,Philippe}.feats41 ${data_dir}/train.concat.feats41
cat ${data_dir}/train.{fr,fr,fr,fr,fr,fr} > ${data_dir}/train.concat.fr
cat ${data_dir}/train.{en,en,en,en,en,en} > ${data_dir}/train.concat.en

cp ${data_dir}/dev.Agnes.feats41 ${data_dir}/dev.feats41
cp ${data_dir}/test1.Agnes.feats41 ${data_dir}/test1.feats41
cp ${data_dir}/test2.Agnes.feats41 ${data_dir}/test2.feats41

scripts/speech/audio-features-shuf.py ${data_dir}/train.{concat,concat.shuf}.feats41 --input-txt ${data_dir}/train.concat.{fr,en} --output-txt ${data_dir}/train.concat.shuf.{fr,en}

scripts/prepare-data.py ${data_dir}/train.concat.shuf fr en ${data_dir} --mode vocab --character-level --no-tokenize --vocab-prefix vocab.char
cp ${data_dir}/vocab.fr ${data_dir}/vocab.w2c.fr
cp ${data_dir}/vocab.char.en ${data_dir}/vocab.w2c.en


for corpus in dev test1 test2 train train.concat train.concat.shuf
do
    cat ${data_dir}/${corpus}.fr | tr -d "[,\!.\?:;\"]" | tr -s " " > ${data_dir}/${corpus}.nopunk.fr
done

scripts/prepare-data.py ${data_dir}/train.concat.shuf.nopunk fr ${data_dir} --mode vocab --character-level --no-tokenize --vocab-prefix vocab.nopunk.char
scripts/prepare-data.py ${data_dir}/train.concat.shuf.nopunk fr ${data_dir} --mode vocab --no-tokenize --vocab-prefix vocab.nopunk
cp ${data_dir}/vocab.nopunk.fr ${data_dir}/vocab.nopunk.w2c.fr
cp ${data_dir}/vocab.nopunk.char.en ${data_dir}/vocab.nopunk.w2c.en
