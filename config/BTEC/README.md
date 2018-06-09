

# Speech processing

## Install Yaafe

~~~
sudo apt-get install cmake cmake-curses-gui libargtable2-0 libargtable2-dev
libsndfile1 libsndfile1-dev libmpg123-0 libmpg123-dev libfftw3-3 libfftw3-dev
liblapack-dev libhdf5-serial-dev gcc-4.8 g++-4.8

wget https://sourceforge.net/projects/yaafe/files/yaafe-v0.64.tgz/download -O yaafe-v0.64.tgz

tar xzf yaafe-v0.64.tgz
cd yaafe-v0.64

# fix bug in the official release
cat src_cpp/yaafe-core/Ports.h | sed "s/\tpush_back/\tthis->push_back/g" > src_cpp/yaafe-core/Ports.h.fixed
mv src_cpp/yaafe-core/Ports.h.fixed src_cpp/yaafe-core/Ports.h

mkdir build
cd build
export CC=/usr/bin/gcc-4.8
export CXX=/usr/bin/g++-4.8
cmake ..
make
sudo make install

echo "export PYTHONPATH=/usr/local/python_packages/:\$PYTHONPATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/lib/:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export YAAFE_PATH=/usr/local/yaafe_extensions" >> ~/.bashrc
~~~

## Configuration files

Examples of configuration files for ASR and AST are: `config/BTEC/ASR.yaml` and `config/BTEC/AST.yaml`.
You'll need to modify the `data`, `model`, `data_prefix` and `vocab_prefix` parameters. Also, you should set the right `name`  for the `encoders` and `decoders` parameters (it should be the same as the source and target extensions).

A very important parameter for ASR and AST is the `max_len` parameters (specific to each encoder and decoder). It defines the maximum length of the input and output sequences. Training time and memory usage depend on this limit. Because audio sequences are very long (1 frame every 10 ms), training can take a lot of memory.

