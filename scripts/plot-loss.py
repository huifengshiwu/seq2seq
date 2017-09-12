#!/usr/bin/env python3
import argparse
import re
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('log_files', nargs='+')
parser.add_argument('--output')
parser.add_argument('--max-steps', type=int, default=0)
parser.add_argument('--min-steps', type=int, default=0)
parser.add_argument('--labels', nargs='+')

parser.add_argument('--no-x', action='store_true', help='Run with no X server')

parser.add_argument('--txt', '--text', action='store_true')
parser.add_argument('--stride', type=int)
parser.add_argument('-n', '--max-values', type=int, default=15)
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
data = []

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

            # if 0 < args.max_steps < current_step:
            #     break
            # if current_step < args.min_steps:
            #     continue

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

    data.append((bleu_scores, ter_scores, dev_perplexities, train_perplexities))


score_names = ['bleu', 'ter', 'dev', 'train']
score_labels = ['BLEU', 'TER', 'dev loss', 'train loss']
linestyles = [':', ':', '--', '--']

def boldify(text):
    return '\033[1m' + text + '\033[0m'

if args.txt:
    data = list(zip(*data))

    l = max(len(label) for name, label in zip(score_names, score_labels) if name in args.plot)
    l = max(l, max(map(len, labels)))
    fmt = '{{:<{}}}'.format(l)

    i = 0
    for score_name, score_label, values in zip(score_names, score_labels, data):
        if score_name not in args.plot or not values or not any(values):
            continue

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
        
        print(fmt.format(score_label), ''.join('{:>7}'.format(step) for step in steps))
        for model_label, values_ in zip(labels, values):
            values__ = []
            for min_step, max_step in zip([-1] + steps, steps):
                a = [value for step, value in values_ if min_step < step <= max_step]
                if not a:
                    a = None
                elif args.best and score_name == 'bleu':
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
                best_value = max(filter(None, values_)) if score_name == 'bleu' else min(filter(None, values_))
            except ValueError:
                best_value = 0
            s = ['{:>7}'.format('') if x is None else '{:>7.2f}'.format(x) for x in values_]
            s = [boldify(y) if x == best_value else y for x, y in zip(values_, s)]
            print(fmt.format(model_label), ''.join(s))
else:
    for label, data_ in zip(labels, data):
        data_ = [[(step, value) for step, value in data__ if step >= args.min_steps
                  and (args.max_steps == 0 or step <= args.max_steps)] for data__ in data_]

        for score_name, score_label, linestyle, values in zip(score_names, score_labels, linestyles, data_):
            if score_name in args.plot and values:
                plt.plot(*zip(*values), label=' '.join([label, score_label]), linestyle=linestyle)

    legend = plt.legend(loc='best', shadow=True)

    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()
