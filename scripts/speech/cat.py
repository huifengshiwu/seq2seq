#!/usr/bin/env python3
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+')
parser.add_argument('output')
parser.add_argument('-v', '--verbose', action='store_true')

args = parser.parse_args()

dim = None
n = 0
for filename in args.inputs:
    with open(filename, 'rb') as f:
        n_, dim_ = np.load(f)
        n += n_
        assert dim is None or dim_ == dim, 'incompatible dimensions {} != {}'.format(dim_, dim)
        dim = dim_

if args.verbose:
    print('count: {}, dim: {}'.format(n, dim))

with open(args.output, 'wb') as output_file:
    np.save(output_file, (n, dim))
    for filename in args.inputs:
        with open(filename, 'rb') as f:
            n_, _ = np.load(f)
            for _ in range(n_):
                feats = np.load(f)
                np.save(output_file, feats)

