#!/usr/bin/env python

import sys
import re
import math
from collections import defaultdict

stats = defaultdict(list)

for line in sys.stdin:
    for s in re.findall(r'[^\s]*=\d+\.?\d*', line):
        key, value = s.split('=')
        stats[key].append(float(value))

keys = ['ter', 'bleu', 'bleu1', 'wer']
def sort_key(item):
    key, _ = item
    if key in keys:
        return keys.index(key)
    else:
        return len(keys)

new_stats = []
for key, values in sorted(stats.items(), key=sort_key):
    mean = sum(values) / len(values)
    stdev = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    new_stats.append((key, mean, stdev))

print('\n'.join('{:<7} {:6.2f} ({:.2f})'.format(*data) for data in new_stats))
