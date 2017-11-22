#!/usr/bin/env python3

# import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
import argparse

# font = {'family' : 'normal',
#         'size'   : 16}
# matplotlib.rc('font', **font)

# matplotlib.rcParams['axes.linewidth'] = 1
fontsize = None

parser = argparse.ArgumentParser()
parser.add_argument('eval_dir')

args = parser.parse_args()

eval_dir = args.eval_dir

ast1_file = os.path.join(eval_dir, 'AST.1.csv')
ast2_file = os.path.join(eval_dir, 'AST.2.csv')
ast3_file = os.path.join(eval_dir, 'AST.3.csv')

dest_file = os.path.join(eval_dir, 'AST.pdf')

strides = [2, 2, 4, 4]
# strides = [4, 4, 8, 8]
min_step = 0
max_step = 120000

def parse(filename):
    steps, scores = zip(*[line.split(',') for line in open(filename)])
    steps = [int(step) for step in steps]
    scores = [float(score) for score in scores]

    steps, scores = list(zip(*[(step, score) for step, score in zip(steps, scores) if step >= min_step
                               and (max_step == 0 or step <= max_step) ]))

    return steps, scores

def smooth(data, stride=1):
    if stride <= 1:
        return data
    steps, scores = data
    steps_ = []
    scores_ = []
    for i in range(len(steps) // stride):
        step = steps[i * stride:(i + 1) * stride][-1]
        score = scores[i * stride:(i + 1) * stride]
        score = sum(score) / len(score)
        steps_.append(step)
        scores_.append(score)
    return steps_, scores_

ast1_data = smooth(parse(ast1_file), strides[0])
ast2_data = smooth(parse(ast2_file), strides[1])
ast3_data = smooth(parse(ast3_file), strides[2])

fig, ax_left = plt.subplots()
fig.set_size_inches(4, 3)
ax_left.set_xlabel('steps', fontsize=fontsize)
ax_left.set_ylabel('BLEU', fontsize=fontsize)

#labels=['E2E all data', 'End-to-End', 'Pre-train', 'Multi-task']
#linestyles = [':', '--', '-', '-.']
labels=['End-to-End', 'Pre-train', 'Multi-task']
linestyles = [':', '--', '-']

ax_left.set_prop_cycle(None)
ax_left.plot(*ast1_data, linestyle=linestyles[0], c='black')
ax_left.plot(*ast2_data, linestyle=linestyles[1], c='black')
ax_left.plot(*ast3_data, linestyle=linestyles[2], c='black')

plt.xticks(np.arange(0, max_step+20000, 20000))
ax_left.set_xticklabels(['{}k'.format(i) for i in range(0, max_step//1000 + 20, 20)])

fig.tight_layout()
lines = [plt.plot([], [], c='black', linestyle=linestyle)[0] for linestyle in linestyles]
plt.legend(lines, labels, loc='lower right', fontsize=fontsize)

#plt.subplots_adjust(left=0.15, right=1.0, top=1.0, bottom=0.15)
fig.savefig(dest_file)
