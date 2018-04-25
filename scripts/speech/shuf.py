#!/usr/bin/env python3
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('--output')
parser.add_argument('-n', type=int, default=0)

parser.add_argument('--input-txt', nargs='*')
parser.add_argument('--output-txt', nargs='*')

args = parser.parse_args()

if not args.output:
    args.output = args.input
if args.input_txt and not args.output_txt:
    args.output_txt = args.input_txt

with open(args.input, 'rb') as input_file:
    n, dim = np.load(input_file)

    indices = list(range(n))
    random.shuffle(indices)

    if args.n > 0:
        indices = indices[:args.n]

    frames = []

    for _ in range(n):
        feats = np.load(input_file)
        frames.append(feats)

with open(args.output, 'wb') as output_file:
    np.save(output_file, (len(indices), dim))
    for index in indices:
        feats = frames[index]
        np.save(output_file, feats)

if args.input_txt and args.output_txt:
    lines = []
    for input_filename in args.input_txt:
        with open(input_filename) as input_file:
            lines.append(input_file.readlines())

    for lines_, output_filename in zip(lines, args.output_txt):
        with open(output_filename, 'w') as output_file:
            for index in indices:
                line = lines_[index]
                output_file.write(line)

