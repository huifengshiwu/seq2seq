#!/usr/bin/env python3
import argparse
from translate.evaluation import tercom_statistics
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('hyp_files', nargs='+')
parser.add_argument('ref_file')
parser.add_argument('--reverse', action='store_true')
parser.add_argument('--labels', nargs='+')
parser.add_argument('--legend-loc', default='upper right')
parser.add_argument('--bar-width', type=float, default=0.2)
parser.add_argument('--ymin', type=float, default=0.0)
parser.add_argument('--ymax', type=float, default=0.3)
parser.add_argument('--ops', nargs='+', default=['ins', 'del', 'sub', 'shift'])
parser.add_argument('--fig-size',  nargs=2, type=float)
parser.add_argument('--save')

parser.add_argument('--average', nargs='+', type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.ref_file) as f:
        references = [line.strip() for line in f]

    hypotheses = []
    for hyp_file in args.hyp_files:
        with open(hyp_file) as f:
            hypotheses.append([line.strip() for line in f])

    if args.reverse:
        scores = [tercom_statistics(references, hyp)[0] for hyp in hypotheses]
    else:
        scores = [tercom_statistics(hyp, references)[0] for hyp in hypotheses]

    N = len(args.average) if args.average else len(args.hyp_files)
    ind = np.arange(N)
    op_name_mapping = {'ins': 'Insertions', 'del': 'Deletions', 'sub': 'Substitutions', 'shift': 'Shifts'}

    ref_words = np.array([score["REF_WORDS"] for score in scores])
    bars = []
    legend = []

    bottom = np.zeros(N)
    
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']
    
    if args.fig_size:
        plt.figure(figsize=tuple(args.fig_size))
    
    for op, color in zip(args.ops, colors):
        scores_ = np.array([score[op.upper()] for score in scores]) / ref_words
        if args.average:
            new_scores_ = []
            j = 0
            for n in args.average:
                new_scores_.append(np.average(scores_[j:j+n]))
                j += n
            scores_ = np.array(new_scores_)

        bar = plt.bar(ind, scores_, args.bar_width, bottom=bottom, color=color, align='center')
        
        bars.append(bar)
        legend.append(op_name_mapping[op])
        bottom += scores_
        
    #plt.legend((p_ins[0], p_del[0], p_sub[0], p_shift[0])[::-1], ('Insertions', 'Deletions', 'Substitutions', 'Shifts')[::-1],
    #           loc='upper right')
    
    try:
        loc = float(args.legend_loc)
        plt.legend(bars[::-1], legend[::-1], bbox_to_anchor=[loc, 1], loc="upper center")
    except:
        plt.legend(bars[::-1], legend[::-1], loc=args.legend_loc)

    plt.ylabel('TER')

    if args.labels:
        plt.xticks(ind, args.labels)
    else:
        plt.xticks([])
        
    axes = plt.gca()
    axes.set_ylim([args.ymin, args.ymax])
    
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()

"""
N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, yerr=menStd)
p2 = plt.bar(ind, womenMeans, width,
	     bottom=menMeans, yerr=womenStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()

    import ipdb; ipdb.set_trace()
"""
