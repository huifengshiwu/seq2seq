#!/usr/bin/env python3

from matplotlib import pyplot as plt
import os
import argparse

fontsize = None

parser = argparse.ArgumentParser()
parser.add_argument('eval_dir')

args = parser.parse_args()

eval_dir = args.eval_dir
asr_file = os.path.join(eval_dir, 'ASR.csv')
mt_file = os.path.join(eval_dir, 'MT.csv')
asr_multi_file = os.path.join(eval_dir, 'ASR.multitask.csv')
mt_multi_file = os.path.join(eval_dir, 'MT.multitask.csv')

dest_file = os.path.join(eval_dir, 'multitask.pdf')

strides = [2, 2, 4, 4]
# strides = [4, 4, 8, 8]
min_step = 6000
max_step = 0

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

asr_data = smooth(parse(asr_file), strides[0])
mt_data = smooth(parse(mt_file), strides[1])
asr_multi_data = smooth(parse(asr_multi_file), strides[2])
mt_multi_data = smooth(parse(mt_multi_file), strides[3])

fig, ax_left = plt.subplots()
fig.set_size_inches(4, 3)
ax_left.set_xlabel('steps', fontsize=fontsize)
ax_left.set_ylabel('WER (ASR)', fontsize=fontsize)
ax_right = ax_left.twinx()
ax_right.set_ylabel('BLEU (MT)', fontsize=fontsize)

labels=['ASR mono', 'ASR multi', 'MT mono', 'MT multi']
linestyles = [':', '--', '-.', '-']

ax_left.set_prop_cycle(None)
ax_left.plot(*asr_data, linestyle=linestyles[0], c='black')
ax_left.plot(*asr_multi_data, linestyle=linestyles[1], c='black')

ax_right.set_prop_cycle(None)
ax_right.plot(*mt_data, linestyle=linestyles[2], c='black')
ax_right.plot(*mt_multi_data, linestyle=linestyles[3], c='black')

fig.tight_layout()
#plt.legend(lines, labels, loc=args.legend_loc, framealpha=0.3)

lines = [plt.plot([], [], c='black', linestyle=linestyle)[0] for linestyle in linestyles]
plt.legend(lines, labels, loc='center right', fontsize=fontsize)

plt.savefig(dest_file)
