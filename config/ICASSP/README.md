
## Augmented LibriSpeech

The raw corpus can be downloaded [here](https://persyval-platform.univ-grenoble-alpes.fr/DS91/detaildataset). It consists in an automatic alignment of the [LibriSpeech ASR corpus](http://www.openslr.org/12/) (English audio with transcriptions), with [Project Gutenberg](https://www.gutenberg.org/), which distributes public domain e-books in many languages.
The scripts that were used for the alignment are freely available [here](https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus).

The pre-processed corpus (with MFCCs) is available [here](https://drive.google.com/open?id=1UchOtgOXYEjXxeI8WbbOH-gPdveo25AS). If you want to use it to train new models, you should extract it inside `data/ICASSP/Books`. Then, you can train a new model using the configuration files inside `config/ICASSP/Books`. For example:

    ./seq2seq.sh config/ICASSP/Books/AST.yaml --train -v --purge

## Trained models

You can download some pre-trained models on Augmented LibriSpeech [here](https://drive.google.com/open?id=1H2gQ0c6CjD5CdaoJjMDapPeN3AQ_p2cq).
This archive should be extracted as `models/ICASSP/Books`. Then, to decode the test set using a model, e.g., `AST`, do:
    
    ./seq2seq.sh models/ICASSP/Books/AST/config.yaml --decode data/ICASSP/Books/test.feats41
