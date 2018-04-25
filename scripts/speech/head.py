#!/usr/bin/env python3
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-n', type=int, default=10)

args = parser.parse_args()

with open(args.input, 'rb') as input_file, open(args.output, 'wb') as output_file:
    n, dim = np.load(input_file)
    n = min(args.n, n)
    np.save(output_file, (n, dim))
    for _ in range(n):
        feats = np.load(input_file)
        np.save(output_file, feats)

