
## Augmented LibriSpeech

The raw corpus can be downloaded [here](https://persyval-platform.univ-grenoble-alpes.fr/DS91/detaildataset). It consists in an automatic alignment of the [LibriSpeech ASR corpus](http://www.openslr.org/12/) (English audio with transcriptions), with [Project Gutenberg](https://www.gutenberg.org/), which distributes public domain e-books in many languages.
The scripts that were used for the alignment are freely available [here](https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus).

The pre-processed corpus (with MFCCs) is available [here](https://drive.google.com/open?id=15ZwzXe_FEx-K7yn6ZVksrUc0QWV072Xt). If you want to use it to train new models, you should extract it as `data/LibriSpeech`. Then, you can train a new model using the configuration files inside `config/LibriSpeech`. For example:

    ./seq2seq.sh config/LibriSpeech/AST.yaml --train -v --purge

If you want to do your own pre-processing, then you can use [this corpus](https://drive.google.com/open?id=1n6r-gkTPooK8oEWjllv1i5vO3ZWHkRNe). The audio files are grouped into tar archives for convenience. The `scripts/speech/extract.py` and `scripts/speech/extract-new.py` directly take this tar archive as input, and output a numpy binary file containing the extracted features. The text files are non-processed and should be tokenized and optionally lowercased before training.

## Trained models

You can download some pre-trained models on Augmented LibriSpeech [here](https://drive.google.com/open?id=1QUS7VjaaFouBX7HNAl05vzKLzlzkZvcY).
This archive should be extracted inside `models/`. Then, to decode the test set using a model, e.g., `AST.1`, do:
    
    ./seq2seq.sh models/LibriSpeech/AST.1/config.yaml --decode models/LibriSpeech/data/test.npz

The directory `models/LibriSpeech/eval-outputs` contains all the outputs by our our pre-trained models on the test and dev set. The `models/LibriSpeech/eval.log` file contains the commands that were used for the evaluation along with the obtained scores. Each model has a `config.yaml` file that can be used to use it or re-train it. The config files of the more important models are also available inside `config/LibriSpeech/`.
