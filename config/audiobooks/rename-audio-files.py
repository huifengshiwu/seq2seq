#!/usr/bin/env python3

import argparse
import os
import shutil
import math
parser = argparse.ArgumentParser()

parser.add_argument('dir')
parser.add_argument('alignment')
parser.add_argument('dest')

args = parser.parse_args()

if not os.path.exists(args.dest):
    os.makedirs(args.dest)

with open(args.alignment) as f:
    lines = list(f)

    for i, line in enumerate(lines, 1):
        filename = line.split()[4] + '.wav'
        new_name = '{{:0{}d}}'.format(int(math.ceil(math.log(len(lines) + 1, 10)))).format(i) + '.wav'
        shutil.copy(os.path.join(args.dir, filename), os.path.join(args.dest, new_name))
