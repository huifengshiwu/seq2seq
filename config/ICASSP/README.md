
## Augmented LibriSpeech

The raw corpus is available [here](https://persyval-platform.univ-grenoble-alpes.fr/DS91/detaildataset).

The pre-processed corpus (with MFCCs) is available [here](https://drive.google.com/open?id=1UchOtgOXYEjXxeI8WbbOH-gPdveo25AS).

## Trained models

You can download some pre-trained models on Augmented LibriSpeech [here](https://drive.google.com/open?id=1H2gQ0c6CjD5CdaoJjMDapPeN3AQ_p2cq).
This archive should be extracted as `models/ICASSP/Books`. Then, to decode the test set using a model, e.g., `AST`, do:
    
    ./seq2seq.sh models/ICASSP/Books/AST/config.yaml --decode data/ICASSP/Books/test.feats41
