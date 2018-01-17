

# Speech processing

## Install Yaafe

~~~
sudo apt-get install cmake cmake-curses-gui libargtable2-0 libargtable2-dev
libsndfile1 libsndfile1-dev libmpg123-0 libmpg123-dev libfftw3-3 libfftw3-dev
liblapack-dev libhdf5-serial-dev

wget https://sourceforge.net/projects/yaafe/files/yaafe-v0.64.tgz/download -O yaafe-v0.64.tgz

tar xzf yaafe-v0.64.tgz
cd yaafe-v0.64

# fix bug in the official release
cat src_cpp/yaafe-core/Ports.h | sed "s/\tpush_back/\tthis->push_back/g" > src_cpp/yaafe-core/Ports.h.fixed
mv src_cpp/yaafe-core/Ports.h.fixed src_cpp/yaafe-core/Ports.h

mkdir build
cd build
cmake ..
make
sudo make install

echo "export PYTHONPATH=/usr/local/python_packages/:\$PYTHONPATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/lib/:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export YAAFE_PATH=/usr/local/yaafe_extensions" >> ~/.bashrc
~~~

## Pre-process your data

Here is a dummy example of audio data pre-processing with the BTEC corpus (please send me an email if you'd like to get the audio data)
~~~
train_audio_data=raw_data/BTEC_train
test_audio_data=raw_data/BTEC_test
dev_audio_data=raw_data/BTEC_dev
raw_data=raw_data/BTEC

data_dir=data/BTEC
ls ${train_audio_data}/*.wav -v | scripts/speech/extract-audio-features.py -o ${data_dir}/train.feats41
ls ${dev_audio_data}/*.wav -v | scripts/speech/extract-audio-features.py -o ${data_dir}/dev.feats41
ls ${test_audio_data}/*.wav -v | scripts/speech/extract-audio-features.py -o ${data_dir}/test.feats41

scripts/prepare-data.py ${raw_data} train fr en ${data_dir} --no-tokenize --lowercase --vocab-size 0 --test-corpus test --dev-corpus dev
~~~

where `raw_data/BTEC_train` contains a wav file for each line in the training corpus. These files should be named so that their alphanumerical order is the same as the corresponding lines in `raw_data/BTEC/train.{fr,en}`. Check the output of the `ls -v` command to see if it gets the order right.

The `scripts/speech/extract-audio-features.py` depends on Yaafe. It assumes that your audio files use a sample rate of 16 kHz. You'll need to modify the `sample_rate`, `step_size` and `block_size` variables accordingly if you're using a different sample rate. If you want to produce more features, you can change the `mfcc_coeffs` and `mfcc_filters` variables. If you want to get the first-order and second-order derivatives (and get features of size 123 instead of 41), you can use the `--derivatives` argument.

The text pre-processing here assumes that the raw data is already tokenized, remove the `--no-tokenize` parameter if this is not the case. The `--vocab-size 0` option sets no limit to the vocabulary size. With larger datasets, you may want to set a limit there (e.g., 30000). You can also produce character-level vocabularies with the following command:

~~~
scripts/prepare-data.py ${raw_data} train fr en ${data_dir} --mode vocab --character-level --vocab-size 0 --vocab-prefix vocab.char
~~~

## Configuration files

Examples of configuration files for ASR and AST are: `config/BTEC/ASR/baseline-char.yaml` and `config/BTEC/AST/baseline-char.yaml`.
You'll need to modify the `data`, `model`, `data_prefix` and `vocab_prefix` parameters. Also, you should set the right `name`  for the `encoders` and `decoders` parameters (it should be the same as the source and target extensions).

A very important parameter for ASR and AST is the `max_len` parameters (specific to each encoder and decoder). It defines the maximum length of the input and output sequences. Training time and memory usage depend on this limit. Because audio sequences are very long (1 frame every 10 ms), training can take a lot of memory.

The following command may come in handy:
~~~
filename=data/BTEC/train.feats41
python3 -c "from translate.utils import read_binary_features as read; l = [len(v[0]) for v in read('${filename}')]; print(sorted(l)[len(l)*90//100-1])"
~~~

It prints the minimum number of audio frames to cover 90% of the training corpus. Use the `scripts/stats.py` to obtain length statistics on the text side of the corpus.
