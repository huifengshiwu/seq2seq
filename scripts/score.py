#!/usr/bin/env python3

import argparse
import sys
import os
import re
from collections import OrderedDict

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)
tercom_path = os.path.join(script_dir, 'tercom.jar')

from translate.evaluation import corpus_bleu, corpus_ter, corpus_wer, corpus_cer, corpus_bleu1

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--bleu', action='store_true')
parser.add_argument('--ter', action='store_true')
parser.add_argument('--wer', action='store_true')
parser.add_argument('--cer', action='store_true')
parser.add_argument('--bleu1', action='store_true')
parser.add_argument('--all', '-a', action='store_true')
parser.add_argument('--max-size', type=int)
parser.add_argument('--no-punk', action='store_true')

parser.add_argument('--case-insensitive', '-i', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()

    if not any([args.all, args.wer, args.ter, args.bleu, args.cer, args.bleu1]):
        args.all = True

    if args.all:
        args.wer = args.ter = args.bleu = args.cer = args.bleu1 = True

    with open(args.source) as src_file, open(args.target) as trg_file:
        def remove_punk(sentence):
            sentence = re.sub(r'[,!;:?"]', '', sentence)
            return re.sub(r'\s+', ' ', sentence)

        if args.case_insensitive:
            hypotheses = [line.strip().lower() for line in src_file]
            references = [line.strip().lower() for line in trg_file]
        else:
            hypotheses = [line.strip() for line in src_file]
            references = [line.strip() for line in trg_file]

        if args.no_punk:
            hypotheses = [remove_punk(line) for line in hypotheses]
            references = [remove_punk(line) for line in references]

        if args.max_size is not None:
            hypotheses = hypotheses[:args.max_size]
            references = references[:args.max_size]

        if len(hypotheses) != len(references):
            sys.stderr.write('warning: source and target don\'t have the same length\n')
            size = min(len(hypotheses), len(references))
            hypotheses = hypotheses[:size]
            references = references[:size]

        scores = OrderedDict()
        if args.bleu:
            scores['bleu'], summary = corpus_bleu(hypotheses, references)
            try:
                scores['penalty'], scores['ratio'] = map(float, re.findall('\w+=(\d+.\d+)', summary))
            except ValueError:
                pass
        if args.wer:
            scores['wer'], _ = corpus_wer(hypotheses, references)
        if args.ter:
            try:  # java missing
                scores['ter'], _ = corpus_ter(hypotheses, references, tercom_path=tercom_path)
            except:
                scores['ter'] = 0
        if args.cer:
            scores['cer'], _ = corpus_cer(hypotheses, references)
        if args.bleu1:
            scores['bleu1'], _ = corpus_bleu1(hypotheses, references)

        print(' '.join('{}={:.2f}'.format(k, v) for k, v in scores.items()))
