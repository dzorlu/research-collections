"""
Train the toy model on sort with each method, under lag, noise and K steps per rollout.

  python -m pglab.train --methods reinforce,ppo,cispo,tis --lag 0,4 --noise 0,0.5 --k 1,4
"""
import argparse
import collections
import copy
import csv
import itertools
import os

import torch
import torch.nn.functional as F

from pglab.methods import METHODS
from pglab.model import NN

S, V = 4, 10                     # digits per prompt, digit values 0-9


def make_prompts(P):
  prompt = torch.randint(0, V, (P, S))                             # [P, S]
  target = prompt.sort(-1).values                                  # [P, S]  the correct output
  # Each position sees the whole prompt plus which slot it fills.
  x = torch.cat([F.one_hot(prompt, V).flatten(1).unsqueeze(1).expand(P, S, S * V),   # [P, S, S*V]
                 torch.eye(S).expand(P, S, S)], -1)                                  # [P, S, S]
  return x.float(), target                                         # [P, S, I], [P, S]


def logp_of(model, y):
  return torch.log_softmax(model.acts['s2'], -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)   # [B, S]


@torch.no_grad()
def run(method, lag, noise, k, seed, steps, prompts, group, hidden, lr):
  torch.manual_seed(seed)
  model = NN(S * V + S, hidden, V)
  engine = copy.deepcopy(model)
  snapshots = collections.deque([copy.deepcopy(model.state_dict())], maxlen=lag + 1)
  rows = []
  for step in range(steps):
    # Rollout: the engine (weights from `lag` updates ago, noisy logits) samples `group` responses per prompt.
    x, target = make_prompts(prompts)
    x, target = x.repeat_interleave(group, 0), target.repeat_interleave(group, 0)   # [B, S, I], [B, S]
    engine.load_state_dict(snapshots[0])
    engine.forward(x)
    s2 = engine.acts['s2'] + noise * torch.randn_like(engine.acts['s2'])           # [B, S, V]
    y = torch.distributions.Categorical(logits=s2).sample()                        # [B, S]
    logp_engine = torch.log_softmax(s2, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [B, S]

    # Reward and GRPO advantage, one per response, copied to every token.
    R = (y == target).float().mean(-1)                                             # [B]
    Rg = R.reshape(prompts, group)                                                 # [P, G]
    A = ((Rg - Rg.mean(-1, keepdim=True)) / (Rg.std(-1, keepdim=True) + 1e-6)).reshape(-1, 1).expand(-1, S)  # [B, S]

    model.forward(x)
    logp_old = logp_of(model, y)                                                   # [B, S]
    for _ in range(k):
      model.forward(x)
      staleness_ratio = (logp_of(model, y) - logp_old).exp()                       # [B, S]
      mismatch_ratio = (logp_old - logp_engine).exp()                              # [B, S]
      w = METHODS[method](staleness_ratio, mismatch_ratio, A)                      # [B, S]
      model.backward(w, A, y)
      model.step(lr)
    snapshots.append(copy.deepcopy(model.state_dict()))
    if not (model.w1.isfinite().all() and model.w2.isfinite().all()):
      print(f"  {method} lag={lag} noise={noise} k={k} seed={seed}: diverged at step {step}")
      break

    rows.append(dict(method=method, lag=lag, noise=noise, k=k, seed=seed, step=step,
                     reward=R.mean().item(),
                     solved=(R == 1).float().mean().item(),
                     w_zero=(w == 0).float().mean().item(),
                     gap=(logp_old - logp_engine).abs().mean().item()))
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--methods', default='reinforce,ppo,cispo,tis')
  ap.add_argument('--lag', default='0')
  ap.add_argument('--noise', default='0')
  ap.add_argument('--k', default='1')
  ap.add_argument('--seeds', type=int, default=3)
  ap.add_argument('--steps', type=int, default=300)
  ap.add_argument('--prompts', type=int, default=64)
  ap.add_argument('--group', type=int, default=8)
  ap.add_argument('--hidden', type=int, default=128)
  ap.add_argument('--lr', type=float, default=1.0)
  ap.add_argument('--out', default='runs/latest')
  a = ap.parse_args()

  methods = a.methods.split(',')
  unknown = set(methods) - set(METHODS)
  assert not unknown, f"unknown methods {unknown}; registered: {list(METHODS)}"
  grid = itertools.product(methods, map(int, a.lag.split(',')), map(float, a.noise.split(',')),
                           map(int, a.k.split(',')), range(a.seeds))

  os.makedirs(a.out, exist_ok=True)
  with open(f'{a.out}/metrics.csv', 'w', newline='') as f:
    writer = None
    for method, lag, noise, k, seed in grid:
      rows = run(method, lag, noise, k, seed, a.steps, a.prompts, a.group, a.hidden, a.lr)
      writer = writer or csv.DictWriter(f, rows[0].keys())
      if f.tell() == 0:
        writer.writeheader()
      writer.writerows(rows)
      print(f"{method:10s} lag={lag} noise={noise} k={k} seed={seed}  reward {rows[-1]['reward']:.3f}")
  print(f"wrote {a.out}/metrics.csv")


if __name__ == '__main__':
  main()
