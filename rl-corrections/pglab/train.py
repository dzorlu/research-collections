"""
Train the toy model on sort with each method, under lag, noise and epochs (optimizer steps) per rollout.

  python -m pglab.train --methods reinforce,ppo,cispo,tis --lag 0,4 --noise 0,0.5 --epochs 1,4
"""
import collections
import copy
import csv
import itertools
import os

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pglab.methods import METHODS
from pglab.model import NN

S, V = 4, 10                     # digits per prompt, digit values 0-9


class Config(BaseSettings):
  """CLI flags. Lists take commas: --lag 0,8."""
  model_config = SettingsConfigDict(cli_parse_args=True, cli_prog_name='pglab.train', env_prefix='PGLAB_')

  methods: list[str] = ['reinforce', 'ppo', 'cispo', 'tis']
  lag: list[int] = [0]           # engine weights are this many updates old
  noise: list[float] = [0.0]     # std of the noise added to the engine's logits
  epochs: list[int] = [1]        # optimizer steps per rollout
  seeds: int = 3
  steps: int = 300               # rollouts per run
  prompts: int = 64
  group: int = 8                 # samples per prompt (GRPO group)
  hidden: int = 128
  lr: float = 1.0
  out: str = 'runs/latest'

  @field_validator('methods')
  @classmethod
  def registered(cls, v):
    unknown = set(v) - set(METHODS)
    assert not unknown, f"unknown methods {unknown}; registered: {list(METHODS)}"
    return v


class RunSpec(BaseModel):
  """One point of the sweep."""
  model_config = ConfigDict(frozen=True)
  method: str
  lag: int
  noise: float
  epochs: int
  seed: int


class Row(RunSpec):
  """One line of metrics.csv: one rollout of one run."""
  step: int
  reward: float                  # mean fraction of positions right
  solved: float                  # share of responses fully sorted
  w_zero: float                  # share of tokens with w = 0 at the last epoch
  gap: float                     # mean |logp_old - logp_engine|


class Rollout(BaseModel):
  """What the engine hands the trainer."""
  model_config = ConfigDict(arbitrary_types_allowed=True)
  x: torch.Tensor                # [B, S, I]
  y: torch.Tensor                # [B, S]  sampled tokens
  R: torch.Tensor                # [B]     reward
  A: torch.Tensor                # [B, S]  GRPO advantage
  logp_engine: torch.Tensor      # [B, S]


def make_prompts(P):
  prompt = torch.randint(0, V, (P, S))                             # [P, S]
  target = prompt.sort(-1).values                                  # [P, S]  the correct output
  # Each position sees the whole prompt plus which slot it fills.
  x = torch.cat([F.one_hot(prompt, V).flatten(1).unsqueeze(1).expand(P, S, S * V),   # [P, S, S*V]
                 torch.eye(S).expand(P, S, S)], -1)                                  # [P, S, S]
  return x.float(), target                                         # [P, S, I], [P, S]


def logp_of(model, y):
  return torch.log_softmax(model.acts['s2'], -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)   # [B, S]


def rollout(engine, spec, cfg):
  # The engine (weights from `lag` updates ago, noisy logits) samples `group` responses per prompt.
  x, target = make_prompts(cfg.prompts)
  x, target = x.repeat_interleave(cfg.group, 0), target.repeat_interleave(cfg.group, 0)   # [B, S, I], [B, S]
  engine.forward(x)
  s2 = engine.acts['s2'] + spec.noise * torch.randn_like(engine.acts['s2'])         # [B, S, V]
  y = torch.distributions.Categorical(logits=s2).sample()                          # [B, S]
  logp_engine = torch.log_softmax(s2, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [B, S]

  # Reward and GRPO advantage, one per response, copied to every token.
  R = (y == target).float().mean(-1)                                               # [B]
  Rg = R.reshape(cfg.prompts, cfg.group)                                           # [P, G]
  A = ((Rg - Rg.mean(-1, keepdim=True)) / (Rg.std(-1, keepdim=True) + 1e-6)).reshape(-1, 1).expand(-1, S)  # [B, S]
  return Rollout(x=x, y=y, R=R, A=A, logp_engine=logp_engine)


@torch.no_grad()
def run(spec: RunSpec, cfg: Config) -> list[Row]:
  torch.manual_seed(spec.seed)
  model = NN(S * V + S, cfg.hidden, V)
  engine = copy.deepcopy(model)
  snapshots = collections.deque([copy.deepcopy(model.state_dict())], maxlen=spec.lag + 1)
  rows = []
  for step in range(cfg.steps):
    engine.load_state_dict(snapshots[0])
    ro = rollout(engine, spec, cfg)

    model.forward(ro.x)
    logp_old = logp_of(model, ro.y)                                                # [B, S]
    for _ in range(spec.epochs):
      model.forward(ro.x)
      staleness_ratio = (logp_of(model, ro.y) - logp_old).exp()                    # [B, S]
      mismatch_ratio = (logp_old - ro.logp_engine).exp()                           # [B, S]
      w = METHODS[spec.method](staleness_ratio, mismatch_ratio, ro.A)              # [B, S]
      model.backward(w, ro.A, ro.y)
      model.step(cfg.lr)
    snapshots.append(copy.deepcopy(model.state_dict()))
    if not (model.w1.isfinite().all() and model.w2.isfinite().all()):
      print(f"  {spec}: diverged at step {step}")
      break

    rows.append(Row(**spec.model_dump(), step=step,
                    reward=ro.R.mean().item(),
                    solved=(ro.R == 1).float().mean().item(),
                    w_zero=(w == 0).float().mean().item(),
                    gap=(logp_old - ro.logp_engine).abs().mean().item()))
  return rows


def main():
  cfg = Config()
  os.makedirs(cfg.out, exist_ok=True)
  with open(f'{cfg.out}/config.json', 'w') as f:
    f.write(cfg.model_dump_json(indent=2))
  with open(f'{cfg.out}/metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, Row.model_fields)
    writer.writeheader()
    for method, lag, noise, epochs, seed in itertools.product(cfg.methods, cfg.lag, cfg.noise, cfg.epochs, range(cfg.seeds)):
      spec = RunSpec(method=method, lag=lag, noise=noise, epochs=epochs, seed=seed)
      rows = run(spec, cfg)
      writer.writerows(r.model_dump() for r in rows)
      print(f"{spec}  reward {rows[-1].reward:.3f}")
  print(f"wrote {cfg.out}/metrics.csv")


if __name__ == '__main__':
  main()
