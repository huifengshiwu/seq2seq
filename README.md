# seq2seq
Attention-based sequence to sequence learning

## Dependencies

* [TensorFlow 1.2+ for Python 3](https://www.tensorflow.org/get_started/os_setup.html)
* YAML and Matplotlib modules for Python 3: `sudo apt-get install python3-yaml python3-matplotlib`
* A recent NVIDIA GPU

## How to use


Train a model (CONFIG is a YAML configuration file, such as `config/default.yaml`):

    ./seq2seq.sh CONFIG --train -v 


Translate text using an existing model:

    ./seq2seq.sh CONFIG --decode FILE_TO_TRANSLATE --output OUTPUT_FILE
or for interactive decoding:

    ./seq2seq.sh CONFIG --decode

#### Example English&rarr;French model
This is the same model and dataset as [Bahdanau et al. 2015](https://arxiv.org/abs/1409.0473).

    config/WMT14/download.sh    # download WMT14 data into raw_data/WMT14
    config/WMT14/prepare.sh     # preprocess the data, and copy the files to data/WMT14
    ./seq2seq.sh config/WMT14/RNNsearch.yaml --train -v   # train a baseline model on this data

You should get similar BLEU scores as these (our models took about 5 days to train each on a single Titan X).

| Model     | Dev   | +beam | Test  | +beam | Steps | Time |
|:----------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
| RNNsearch | 23.57 | 25.48 | 26.86 | 29.21 | 690k  | 120h |
| RNNsearch_Adam | 24.64 | 26.26 | 27.56 | 29.87 | 480k | 124h |
| RNNsearch_BPE  | 28.78 | 30.01 | 32.70 | 34.63 | 440k  | 108h |

`RNNsearch_Adam` is the exact same model with the Adam optimizer (instead of AdaDelta).
`RNNsearch_BPE` translates subword units instead of words.  

You can download these models from [here](https://drive.google.com/file/d/1x_MoU13NXVtu1iY1bY7IKLkVodXDueeg/view?usp=sharing). To use the `RNNsearch_BPE` model, just extract the archive into the `seq2seq/models` folder, and run:

     ./seq2seq.sh models/WMT14/RNNsearch_BPE/config.yaml --decode -v
     
Here is the full pipeline if you want to translate non-tokenized text (with the BPE-based model):

     cat YOUR_DATA | \
     scripts/moses/tokenizer.perl -l en | \
     scripts/moses/normalize-punctuation.perl  -l en | \
     scripts/moses/escape-special-chars.perl | \
     scripts/bpe/apply_bpe.py -c models/WMT14/data/bpe.joint | \
     ./seq2seq.sh models/WMT14/RNNsearch_BPE/config.yaml --decode -v --beam-size 8 | \
     scripts/moses/detokenizer.perl -l fr 

#### Example German&rarr;English model
This is the same dataset as [Ranzato et al. 2015](https://arxiv.org/abs/1511.06732).

    config/IWSLT14/prepare.sh
    ./seq2seq.sh config/IWSLT14/BPE.yaml --train -v

| Model | Dev   | +beam | Test  | +beam | Steps | Time |
|:------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
| BPE to BPE | 32.74 | 33.99 | 29.79 | 31.20 | 300k  | 31h  |
| BPE to char |32.81 | 34.29 | 30.55 | 32.19 | 792k  | 148h |

The models are available for download [here](https://drive.google.com/file/d/1b4B-72wbLlej1TPcS9ckCXMZaJVMF2Wc/view?usp=sharing).
The pre-processed data can be downloaded [here](https://drive.google.com/open?id=1otTRVm1kre1b2PzfRPWuGl7V1POk8V4-).

## A note on BLEU scores

The BLEU scores that are reported are computed with a reimplementation of `multi-bleu.perl` on tokenized hypotheses and references.
We either get already pre-processed data from sources against which we compare, or perform our
own tokenization with Moses' `tokenizer.perl`

- For WMT14 En-Fr, it seems like early results were obtained with the same evaluation as ours (actually, we download tokenized data from the [same source](http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/) as [Bahdanau et al.](https://arxiv.org/abs/1409.0473)) However, for
more recent ones (e.g., [ConvS2S](https://arxiv.org/abs/1705.03122), [Transformer](https://arxiv.org/abs/1706.03762)) it is not clear.
- For IWSLT14 De-En, we use the same pre-processing as the [first paper](https://arxiv.org/abs/1511.06732) on this task (https://github.com/facebookresearch/MIXER). It seems like
follow-up works have been using the same pre-processing and evaluation (Moses tokenization, lowercasing and `multi-bleu.perl`)

**As said by [Post 2018](https://arxiv.org/abs/1804.08771), to be really comparable, the evaluation should be performed on *detokenized* text, against
unprocessed references, using a standard tool like Moses' [`mteval-13a.pl`](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v13a.pl) or [`sacrebleu`](https://github.com/awslabs/sockeye/tree/master/contrib/sacrebleu).**

To facilitate replication of our results, and comparison against other models, we provide
the outputs of our models. Anyone is welcome to detokenize this data and evaluate it with their own metric.

## Audio pre-processing
If you want to use the toolkit for Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST), then you'll need to pre-process your audio files accordingly.
This [README](https://github.com/eske/seq2seq/tree/master/config/BTEC) details how it can be done. You'll need to install the **Yaafe** library, and use `scripts/speech/extract-audio-features.py` to extract MFCCs from a set of wav files.

In our work, we (mistakenly) extracted 40 MFCCs from the audio files. This is non-optimal, and 13 MFCCs would probably do just as fine (if not better).
Probably better results can be obtained by extracting *40 log-mel filterbank features*.

We invite anyone who would want to train new models on Augmented LibriSpeech, to do a new preprocessing of the data
with the `scripts/speech/extract-new.py` script.

## Pretrained-models and data 

## Features
* **YAML configuration files**
* **Beam-search decoder**
* **Ensemble decoding**
* **Multiple encoders**
* **Hierarchical encoder**
* **Bidirectional encoder**
* **Local attention model**
* **Convolutional attention model**
* **Detailed logging**
* **Periodic BLEU evaluation**
* **Periodic checkpoints**
* **Multi-task training:** train on several tasks at once (e.g. French->English and German->English MT)
* **Subwords training and decoding**
* **Input binary features instead of text**
* **Pre-processing script:** we provide a fully-featured Python script for data pre-processing (vocabulary creation, lowercasing, tokenizing, splitting, etc.)
* **Dynamic RNNs:** we use symbolic loops instead of statically unrolled RNNs. This means that we don't mean to manually configure bucket sizes, and that model creation is much faster.

## Credits

* This project is based on [TensorFlow's reference implementation](https://www.tensorflow.org/tutorials/seq2seq)
* We include some of the pre-processing scripts from [Moses](http://www.statmt.org/moses/)
* The scripts for subword units come from [github.com/rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt)
