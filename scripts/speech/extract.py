#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import numpy as np
import yaafelib
import tarfile
import tempfile
import os
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='tar archive which contains all the wav files')
parser.add_argument('output', help='output file')
parser.add_argument('--derivatives', action='store_true')

args = parser.parse_args()

parameters = dict(
    step_size=160,  # corresponds to 10 ms (at 16 kHz)
    block_size=640,  # corresponds to 40 ms
    mfcc_coeffs=40,
    mfcc_filters=41  # more filters? (needs to be at least mfcc_coeffs+1, because first coeff is ignored)
)

# TODO: ensure that all input files use this rate
fp = yaafelib.FeaturePlan(sample_rate=16000)

mfcc_features = 'MFCC MelNbFilters={mfcc_filters} CepsNbCoeffs={mfcc_coeffs} ' \
                'blockSize={block_size} stepSize={step_size}'.format(**parameters)
energy_features = 'Energy blockSize={block_size} stepSize={step_size}'.format(**parameters)

fp.addFeature('mfcc: {}'.format(mfcc_features))
if args.derivatives:
    fp.addFeature('mfcc_d1: {} > Derivate DOrder=1'.format(mfcc_features))
    fp.addFeature('mfcc_d2: {} > Derivate DOrder=2'.format(mfcc_features))

fp.addFeature('energy: {}'.format(energy_features))
if args.derivatives:
    fp.addFeature('energy_d1: {} > Derivate DOrder=1'.format(energy_features))
    fp.addFeature('energy_d2: {} > Derivate DOrder=2'.format(energy_features))

if args.derivatives:
    keys = ['mfcc', 'mfcc_d1', 'mfcc_d2', 'energy', 'energy_d1', 'energy_d2']
else:
    keys = ['mfcc', 'energy']

df = fp.getDataFlow()
engine = yaafelib.Engine()
engine.load(df)
afp = yaafelib.AudioFileProcessor()

frame_counter = Counter()

outfile = open(args.output, 'wb')

total = 0
for filename in args.inputs:
    tar = tarfile.open(filename)
    total += len([f for f in tar if f.isfile()])

_, tmp_file = tempfile.mkstemp()

for j, filename in enumerate(args.inputs):
    tar = tarfile.open(filename)
    files = sorted([f for f in tar if f.isfile()], key=lambda f: f.name)

    for i, fileinfo in enumerate(files):
        file_ = tar.extractfile(fileinfo)
        with open(tmp_file, 'wb') as f:
            f.write(file_.read())

        afp.processFile(engine, tmp_file)
        feats = engine.readAllOutputs()
        feats = np.concatenate([feats[k] for k in keys], axis=1)
        frames, dim = feats.shape

        feats = feats.astype(np.float32)

        if frames == 0:
            print(frames, dim, fileinfo.name)
            raise Exception

        if i == 0 and j == 0:
            np.save(outfile, (total, dim))

        np.save(outfile, feats)

outfile.close()
os.remove(tmp_file)
