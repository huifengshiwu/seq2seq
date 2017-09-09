#!/usr/bin/env python3

import itertools
import argparse
import re
import os
import dateutil.parser

parser = argparse.ArgumentParser()
parser.add_argument('log_files', nargs='+')
parser.add_argument('--dev-prefix')
parser.add_argument('--score', default='bleu', choices=('ter', 'bleu', 'wer'))
parser.add_argument('--task-name')

def print_scores(log_file):
    with open(log_file) as log_file:
        scores = {}
        times = {}
        current_step = 0
        max_step = 0
        starting_time = None

        def read_time(line):
            m = re.match('../.. ..:..:..', line)
            if m:
                return dateutil.parser.parse(m.group(0))

        for line in log_file:
            if starting_time is None:
                starting_time = read_time(line)

            m = re.search('step (\d+)', line)
            if m:
                current_step = int(m.group(1))
                times.setdefault(current_step, read_time(line)) 
                max_step = max(max_step, current_step)
                continue

            if args.task_name is not None:
                if not re.search(args.task_name, line):
                    continue
            if args.dev_prefix is not None:
                if not re.search(args.task_name, line):
                    continue

            m = re.findall('(bleu|score|ter|wer|penalty|ratio)=(\d+.\d+)', line)
            if m:
                scores_ = {k: float(v) for k, v in m}
                scores.setdefault(current_step, scores_)

        def key(d):
            score = d.get(args.score.lower())
            if score is None:
                score = d.get('score')

            if args.score in ('ter', 'wer'):
                score = -score
            return score

        step, best = max(scores.items(), key=lambda p: key(p[1]))

        if 'score' in best:
            missing_key = next(k for k in ['bleu', 'ter', 'wer'] if k not in best)
            best[missing_key] = best.pop('score')

        keys = [args.score, 'bleu', 'ter', 'wer', 'penalty', 'ratio']
        best = sorted(best.items(), key=lambda p: keys.index(p[0]))

        total_time = (times[max_step] - starting_time).total_seconds() / 3600
        train_time = (times[step] - starting_time).total_seconds() / 3600

        print(' '.join(itertools.starmap('{}={:.2f}'.format, best)),
              'step={}/{}'.format(step, max_step),
              'hours={:.1f}/{:.1f}'.format(train_time, total_time))

if __name__ == '__main__':
    args = parser.parse_args()

    for log_file in args.log_files:
        if os.path.isdir(log_file):
            log_file = os.path.join(log_file, 'log.txt')
        print_scores(log_file)

