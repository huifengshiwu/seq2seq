#!/usr/bin/env python3
import argparse
import numpy as np
import scipy.io.wavfile as wav
import tarfile
import sys
from python_speech_features import mfcc, delta, fbank

parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+')
parser.add_argument('output')

parser.add_argument('--mfcc', action='store_true')
parser.add_argument('--filters', type=int, default=40)
parser.add_argument('--energy', action='store_true')
parser.add_argument('--step-size', type=float, default=0.010)
parser.add_argument('--win-size', type=float, default=0.025)
parser.add_argument('--delta', action='store_true')
parser.add_argument('--delta-delta', action='store_true')
parser.add_argument('--window', default='hamming')
parser.add_argument('--nfft', type=int, default=512)
parser.add_argument('--low-freq', type=float, default=0)
parser.add_argument('--high-freq', type=float)
parser.add_argument('-v', '--verbose', action='store_true')

args = parser.parse_args()

if args.delta_delta:
    args.delta = True

if args.window.lower().startswith('ham'):
    winfunc = np.hamming
elif args.window.lower().startswith('han'):
    winfunc = np.hanning
else:
    winfunc = lambda x: np.ones((x,))

params = dict(
        winlen=args.win_size,
        winstep=args.step_size,
        nfilt=args.filters,
        preemph=0,
        winfunc=winfunc,
        lowfreq=args.low_freq,
        highfreq=args.high_freq,
        nfft=args.nfft)

outfile = open(args.output, 'wb')

total = 0
for filename in args.inputs:
    tar = tarfile.open(filename)
    files = [f for f in tar.getmembers() if f.isfile()]
    total += len(files)

dim = min(12, args.filters - 1) if args.mfcc else args.filters
if args.delta_delta:
    dim *= 3
elif args.delta:
    dim *= 2
if args.energy:
    dim += 1
if args.verbose:
    print('count: {}, dim: {}'.format(total, dim))

np.save(outfile, (total, dim))

i = 1
for filename in args.inputs:
    tar = tarfile.open(filename)
    files = [f for f in tar.getmembers() if f.isfile()]
    files = sorted(files, key=lambda f: f.name)

    for fileinfo in files:
        with tar.extractfile(fileinfo) as f:
            rate, data = wav.read(f)
            
            if args.mfcc:
                feats = mfcc(data, rate, ceplifter=0, **params)
                energy = feats[:,:1]
                feats = feats[:,1:]
            else:
                feats, energy = fbank(data, rate, **params)
                feats = np.log(feats)
                energy = np.expand_dims(np.log(energy), axis=1)

            if args.delta:
                d1 = delta(feats, 2)
                feats = np.concatenate([feats, d1], axis=1)
                if args.delta_delta:
                    d2 = delta(d1, 2)
                    feats = np.concatenate([feats, d2], axis=1)

            if args.energy:
                feats = np.concatenate([energy, feats], axis=1)
               
            np.save(outfile, feats)
        if args.verbose and i % 10 == 0:
            sys.stdout.write('\rfiles processed: {}'.format(i))
        i += 1

if args.verbose:
    print('\rfiles processed: {}'.format(i))

outfile.close()

