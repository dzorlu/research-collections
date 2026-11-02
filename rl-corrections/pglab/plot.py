"""
One figure per metric: one panel per (lag, noise, epochs), one line per method (mean over seeds).

  python -m pglab.plot runs/latest
"""
import collections
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pglab.train import Row

METRICS = ['reward', 'solved', 'w_zero', 'gap']


def main():
  out = sys.argv[1] if len(sys.argv) > 1 else 'runs/latest'
  rows = [Row.model_validate(r) for r in csv.DictReader(open(f'{out}/metrics.csv'))]
  # curves[(lag, noise, epochs)][method][step] -> rows over seeds
  curves = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
  for r in rows:
    curves[(r.lag, r.noise, r.epochs)][r.method][r.step].append(r)
  panels = sorted(curves)

  for metric in METRICS:
    cols = min(4, len(panels))
    rows_ = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(4 * cols, 3.2 * rows_), sharey=True, squeeze=False)
    for ax, panel in zip(axes.flat, panels):
      for method, by_step in curves[panel].items():
        steps = sorted(by_step)
        ax.plot(steps, [sum(getattr(r, metric) for r in by_step[s]) / len(by_step[s]) for s in steps], label=method)
      ax.set_title(f"lag={panel[0]} noise={panel[1]} epochs={panel[2]}", fontsize=9)
      ax.set_xlabel('step')
    axes[0][0].set_ylabel(metric)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{out}/{metric}.png', dpi=120)
    print(f"wrote {out}/{metric}.png")


if __name__ == '__main__':
  main()
