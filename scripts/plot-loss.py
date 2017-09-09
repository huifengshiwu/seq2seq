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
parser.add_argument('--plot', nargs='+')
parser.add_argument('--average', type=int, nargs='+')
parser.add_argument('--smooth', type=int)
parser.add_argument('--no-x', action='store_true', help='Run with no X server')
parser.add_argument('--txt', action='store_true')
parser.add_argument('--stride', type=int)
parser.add_argument('-n', '--max-values', type=int, default=15)
parser.add_argument('--shortest', action='store_true')

args = parser.parse_args()


args.log_files = [os.path.join(log_file, 'log.txt') if os.path.isdir(log_file) else log_file
                  for log_file in args.log_files]

if not args.plot:
    if args.txt:
        args.plot = ('bleu',)
    else:
        args.plot = ('dev', 'train')

if not args.txt:
    try:
        import matplotlib
        if args.no_x:
            matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except ImportError:
        sys.stderr.write('failed to import matplotlib: reverting to txt mode\n')
        args.txt = True

args.plot = [x.lower() for x in args.plot]

if args.average:
    assert sum(args.average) == len(args.log_files)

n = len(args.average) if args.average else len(args.log_files)

labels = None
if args.labels:
    if len(args.labels) != n:
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

labels = labels or ['model {}'.format(i) for i in range(1, n + 1)]
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

            if 0 < args.max_steps < current_step:
                break
            if current_step < args.min_steps:
                continue

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


if args.average:
    new_data = []

    i = 0
    for n in args.average:
        data_ = zip(*data[i:i + n])
        i += n

        def avg(data_):
            dicts = [dict(l) for l in data_]
            keys = set.intersection(*[set(d.keys()) for d in dicts])
            data_ = {k: (sum(d[k] for d in dicts) / n) for k in keys}
            data_ = sorted(list(data_.items()))

            if args.smooth is not None and args.smooth > 1:
                k = args.smooth
                data_ = [(data_[i*k][0], sum(x for _, x in data_[i*k:(i+1)*k]) / k) for i in range(len(data_) // k)]

            return data_

        new_data.append(list(map(avg, data_)))
    data = new_data


score_names = ['bleu', 'ter', 'dev', 'train']
score_labels = ['BLEU', 'TER', 'dev loss', 'train loss']
linestyles = [':', ':', '--', None]

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

        if args.stride:
            steps = steps[args.stride - 1::args.stride]
        if args.max_values:
            steps = steps[:args.max_values]

        steps_ = set(steps)
        
        print(fmt.format(score_label), ''.join('{:>7}'.format(step) for step in steps))
        for model_label, values_ in zip(labels, values):
            values_ = dict(values_)
            values_ = [values_.get(step) for step in steps]
            try:
                best_value = max(filter(None, values_)) if score_name == 'bleu' else min(filter(None, values_))
            except ValueError:
                best_value = 0
            s = ['{:7>}'.format('') if x is None else '{:>7.2f}'.format(x) for x in values_]
            s = [boldify(y) if x == best_value else y for x, y in zip(values_, s)]
            print(fmt.format(model_label), ''.join(s))
else:
    for label, data_ in zip(labels, data):
        for score_name, score_label, linestyle, values in zip(score_names, score_labels, linestyles, data_):
            if score_name in args.plot and values:
                plt.plot(*zip(*values), label=' '.join([label, score_label]), linestyle=linestyle)

    legend = plt.legend(loc='best', shadow=True)

    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()
