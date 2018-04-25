#!/usr/bin/env python3
import argparse
import numpy as np
import struct

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')

args = parser.parse_args()

with open(args.input, 'rb') as infile, open(args.output, 'wb') as outfile:
    lines, dim = struct.unpack('ii', infile.read(8))
    np.save(outfile, (lines, dim))
    
    for _ in range(lines):
        x = infile.read(4)
        frames, = struct.unpack('i', x)
        n = frames * dim
        x = infile.read(4 * n)
        feats = struct.unpack('f' * n, x)
        feats = np.array(feats).reshape(frames, dim)
        np.save(outfile, feats.astype(np.float32))

