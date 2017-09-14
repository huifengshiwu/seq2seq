#!/usr/bin/env python3
import argparse
import re
import os
import sys
import itertools
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('log_files', nargs='+')
parser.add_argument('--output')
parser.add_argument('--max-steps', type=int, default=0)
parser.add_argument('--min-steps', type=int, default=0)
parser.add_argument('--labels', nargs='+')

parser.add_argument('--no-x', action='store_true', help='Run with no X server')

parser.add_argument('--txt', '--text', action='store_true')
parser.add_argument('--stride', type=int)
parser.add_argument('-n', type=int, default=15, dest='max_values')
parser.add_argument('--shortest', action='store_true')
parser.add_argument('--avg', action='store_true')
parser.add_argument('--best', action='store_true')

parser.add_argument('--plot', nargs='+', default=[])
parser.add_argument('--bleu', action='store_true')
parser.add_argument('--ter', action='store_true')
parser.add_argument('--dev-loss', action='store_true')
parser.add_argument('--train-loss', action='store_true')

args = parser.parse_args()


args.log_files = [os.path.join(log_file, 'log.txt') if os.path.isdir(log_file) else log_file
                  for log_file in args.log_files]

args.plot = [x.lower() for x in args.plot]

if args.bleu and 'bleu' not in args.plot:
    args.plot.append('bleu')
if args.ter and 'ter' not in args.plot:
    args.plot.append('ter')
if args.dev_loss and 'dev' not in args.plot:
    args.plot.append('dev')
if args.train_loss and 'train' not in args.plot:
    args.plot.append('train')

if not args.plot:
    args.plot = ['bleu'] if args.txt else ['dev', 'train']

args.bleu = 'bleu' in args.plot
args.ter = 'ter' in args.plot
args.dev = 'dev' in args.plot
args.dev_loss = 'dev' in args.plot
args.train_loss = 'train' in args.plot

if not args.txt:
    try:
        import matplotlib
        if args.no_x:
            matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except ImportError:
        sys.stderr.write('failed to import matplotlib: reverting to txt mode\n')
        args.txt = True

labels = None
if args.labels:
    if len(args.labels) != len(args.log_files):
        raise Exception('error: wrong number of labels')
    labels = args.labels

if not labels:
    dirnames = [os.path.basename(os.path.dirname(log_file)) for log_file in args.log_files]
    if all(dirnames) and len(set(dirnames)) == len(dirnames):
        labels = dirnames

if not labels:
    filenames = [os.path.basename(log_file) for log_file in args.log_files]
    if all(filenames) and len(set(filenames)) == len(filenames):
        labels = filenames

labels = labels or ['model {}'.format(i) for i in range(1, len(args.log_files) + 1)]
data = OrderedDict()
for name in args.plot:
    data[name] = []

for log_file in args.log_files:
    current_step = 0

    dev_perplexities = []
    train_perplexities = []
    bleu_scores = []
    ter_scores = []

    with open(log_file) as f:
        for line in f:
            m = re.search('step (\d+)', line)
            if m:
                current_step = int(m.group(1))

            m = re.search(r'eval: loss (-?\d+.\d+)', line)
            if m and not any(step == current_step for step, _ in dev_perplexities):
                perplexity = float(m.group(1))
                dev_perplexities.append((current_step, perplexity))
                continue

            m = re.search(r'loss (-?\d+.\d+)$', line)
            if m and not any(step == current_step for step, _ in train_perplexities):
                perplexity = float(m.group(1))
                train_perplexities.append((current_step, perplexity))

            m = re.search(r'bleu=(\d+\.\d+)', line)
            m = m or re.search(r'score=(\d+\.\d+)', line)
            if m and not any(step == current_step for step, _ in bleu_scores):
                bleu_score = float(m.group(1))
                bleu_scores.append((current_step, bleu_score))

            m = re.search(r'ter=(\d+\.\d+)', line)
            m = m or re.search(r'score=(\d+\.\d+)', line)
            if m and not any(step == current_step for step, _ in ter_scores):
                ter_score = float(m.group(1))
                ter_scores.append((current_step, ter_score))

    if 'ter' in data:
        data['ter'].append(ter_scores)
    if 'bleu' in data:
        data['bleu'].append(bleu_scores)
    if 'dev' in data:
        data['dev'].append(dev_perplexities)
    if 'train' in data:
        data['train'].append(train_perplexities)

metric_labels = {
    'bleu': 'BLEU',
    'ter': 'TER',
    'dev': 'dev loss',
    'train': 'train loss'
}

def boldify(text):
    return '\033[1m' + text + '\033[0m'

if args.txt:
    # data = list(zip(*data))

    l = max(len(metric_labels[name]) for name in data)
    l = max(l, max(map(len, labels)))
    fmt = '{{:<{}}}'.format(l)

    i = 0
    for name, values in data.items():
        if i > 0:
            print()
        i += 1

        steps = [set([step for step, value in values_]) for values_ in values]
        if args.shortest:
            steps = sorted(list(set.intersection(*steps)))
        else:
            steps = sorted(list(set.union(*steps)))

        steps = [step for step in steps if step >= args.min_steps]
        steps = [step for step in steps if args.max_steps == 0 or step <= args.max_steps]

        if args.stride:
            if args.min_steps:  # we want to include the first value
                steps = steps[::args.stride]
            else:
                steps = steps[args.stride - 1::args.stride]

        if args.max_values:
            steps = steps[:args.max_values]

        steps_ = set(steps)
        
        print(fmt.format(metric_labels[name]), ''.join('{:>7}'.format(step) for step in steps))
        for model_label, values_ in zip(labels, values):
            values__ = []
            for min_step, max_step in zip([-1] + steps, steps):
                a = [value for step, value in values_ if min_step < step <= max_step]
                if not a:
                    a = None
                elif args.best and name == 'bleu':
                    a = max(a)
                elif args.best:
                    a = min(a)
                elif args.avg:
                    a = sum(a) / len(a)
                else:
                    a = a[-1]
                values__.append(a)
            values_ = values__

            try:
                best_value = max(filter(None, values_)) if name == 'bleu' else min(filter(None, values_))
            except ValueError:
                best_value = 0
            s = ['{:>7}'.format('') if x is None else '{:>7.2f}'.format(x) for x in values_]
            s = [boldify(y) if x == best_value else y for x, y in zip(values_, s)]
            print(fmt.format(model_label), ''.join(s))
else:
    linestyles = [':', '--', '-.']

    assert not (args.ter and args.bleu) or not args.dev_loss and not args.train_loss

    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()

    axes = [ax_left, ax_right, ax_right]
    axes = axes[:len(args.plot)]
    ax_left.set_xlabel('steps')

    ax_left.set_ylabel(metric_labels[args.plot[0]])
    if len(axes) > 1:
        label = ', '.join(metric_labels[name] for name in args.plot[1:])
        label = label.replace('dev loss, train loss', 'dev/train loss')
        ax_right.set_ylabel(label)

    for i, (name, values) in enumerate(data.items()):
        ax = axes[i]
        ax.set_prop_cycle(None)
        linestyle = linestyles[i]

        for values_ in values:
            values_ = [(step, value) for step, value in values_
                       if step >= args.min_steps and (args.max_steps == 0 or step <= args.max_steps)]

            ax.plot(*zip(*values_), linestyle=linestyle)

    colors = [line.get_color() for line in ax_left.get_lines()]

    lines = [plt.plot([], [], c=color)[0] for color in colors]
    lines += [plt.plot([], [], c='black', linestyle=linestyle)[0] for linestyle in linestyles[:len(args.plot)]]
    labels += [metric_labels[name] for name in data]

    fig.tight_layout()
    plt.legend(lines, labels, loc='best', shadow=True)

    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()
