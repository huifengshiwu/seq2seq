#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import numpy as np
import tarfile
import yaafelib
import struct
import sys
import scipy.io.wavfile as wav
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

outfile = open(args.output, 'wb')

total = 0
for filename in args.inputs:
    tar = tarfile.open(filename)
    total += len([f for f in tar if f.isfile()])

for j, filename in enumerate(args.inputs):
    tar = tarfile.open(filename)
    files = sorted([f for f in tar if f.isfile()], key=lambda f: f.name)

    for i, fileinfo in enumerate(files):
        _, data = wav.read(tar.extractfile(fileinfo))
        data = data.astype(np.float64)
        data = np.expand_dims(data, axis=0)

        feats = engine.processAudio(data)
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
