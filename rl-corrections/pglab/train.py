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
  hidden: list[int] = [128]      # one entry per hidden layer: --hidden 128,128
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
  """
    P random prompts of S digits, the sorted answer, and the network input for each output position.
    returns x: [P, S, I], target: [P, S]
  """
  prompt = torch.randint(0, V, (P, S))                 # [P, S]  e.g. [3, 1, 4, 1]
  target = prompt.sort(-1).values                      # [P, S]  e.g. [1, 1, 3, 4]

  # Every output position sees the whole prompt ...
  prompt_onehot = F.one_hot(prompt, V)                 # [P, S, V]
  prompt_flat = prompt_onehot.flatten(1)               # [P, S*V]
  prompt_per_slot = prompt_flat.unsqueeze(1)           # [P, 1, S*V]
  prompt_per_slot = prompt_per_slot.expand(P, S, S * V)   # [P, S, S*V]  same prompt copied to each slot

  # ... plus which slot it fills.
  slot = torch.eye(S)                                  # [S, S]  row s = one-hot of slot s
  slot = slot.expand(P, S, S)                          # [P, S, S]

  x = torch.cat([prompt_per_slot, slot], -1).float()   # [P, S, I]  I = S*V + S
  return x, target


def logp_of_tokens(logits, y):
  """
    Log-prob of each written token.
    logits: [B, S, V], y: [B, S] -> [B, S]
  """
  logp = torch.log_softmax(logits, -1)                 # [B, S, V]  every token
  index = y.unsqueeze(-1)                              # [B, S, 1]  gather needs the same number of dims
  logp_y = logp.gather(-1, index)                      # [B, S, 1]  logp_y[b, s, 0] = logp[b, s, y[b, s]]
  return logp_y.squeeze(-1)                            # [B, S]


def grpo_advantage(R, prompts, group):
  """
    Each response's reward compared with the other responses to the same prompt.
    R: [B] with B = prompts * group, responses of one prompt next to each other -> [B]
  """
  R = R.reshape(prompts, group)                        # [P, G]
  mean = R.mean(-1, keepdim=True)                      # [P, 1]
  std = R.std(-1, keepdim=True)                        # [P, 1]
  A = (R - mean) / (std + 1e-6)                        # [P, G]  0 when every response got the same reward
  return A.reshape(-1)                                 # [B]


def rollout(engine, spec, cfg):
  """
    The engine (weights from `lag` updates ago, noisy logits) writes `group` responses per prompt.
  """
  x, target = make_prompts(cfg.prompts)                # [P, S, I], [P, S]
  x = x.repeat_interleave(cfg.group, 0)                # [B, S, I]  B = P * G, each prompt G times in a row
  target = target.repeat_interleave(cfg.group, 0)      # [B, S]

  # Sample from the engine: its logits differ from the trainer's by lag and noise.
  engine.forward(x)
  noise = spec.noise * torch.randn_like(engine.logits)  # [B, S, V]
  engine_logits = engine.logits + noise                # [B, S, V]
  y = torch.distributions.Categorical(logits=engine_logits).sample()   # [B, S]
  logp_engine = logp_of_tokens(engine_logits, y)       # [B, S]

  # Reward: fraction of positions sorted correctly. Advantage: one per response, copied to every token.
  correct = (y == target).float()                      # [B, S]
  R = correct.mean(-1)                                 # [B]
  A = grpo_advantage(R, cfg.prompts, cfg.group)        # [B]
  A = A.unsqueeze(-1).expand(-1, S)                    # [B, S]
  return Rollout(x=x, y=y, R=R, A=A, logp_engine=logp_engine)


@torch.no_grad()
def run(spec: RunSpec, cfg: Config) -> list[Row]:
  torch.manual_seed(spec.seed)
  model = NN(S * V + S, cfg.hidden, V)
  engine = copy.deepcopy(model)

  # The engine always loads the oldest snapshot: weights `lag` updates behind the trainer.
  snapshots = collections.deque([copy.deepcopy(model.state_dict())], maxlen=spec.lag + 1)

  rows = []
  for step in range(cfg.steps):
    # 1. Rollout with the lagged, noisy engine.
    engine.load_state_dict(snapshots[0])
    ro = rollout(engine, spec, cfg)

    # 2. The trainer's log-probs of the same tokens before any update: fixed for this rollout.
    model.forward(ro.x)
    logp_old = logp_of_tokens(model.logits, ro.y)                     # [B, S]
    mismatch_ratio = (logp_old - ro.logp_engine).exp()                # [B, S]  p_old / p_engine

    # 3. `epochs` optimizer steps on this rollout. Only the w line depends on the method.
    for _ in range(spec.epochs):
      model.forward(ro.x)
      logp_now = logp_of_tokens(model.logits, ro.y)                   # [B, S]
      staleness_ratio = (logp_now - logp_old).exp()                   # [B, S]  p_now / p_old, 1 at the first epoch
      w = METHODS[spec.method](staleness_ratio, mismatch_ratio, ro.A)  # [B, S]
      model.backward(w, ro.A, ro.y)
      model.step(cfg.lr)

    snapshots.append(copy.deepcopy(model.state_dict()))
    if not all(W.isfinite().all() for W in model.weights):
      print(f"  {spec}: diverged at step {step}")
      break

    # 4. Metrics for this rollout.
    rows.append(Row(
        **spec.model_dump(),
        step=step,
        reward=ro.R.mean().item(),
        solved=(ro.R == 1).float().mean().item(),
        w_zero=(w == 0).float().mean().item(),
        gap=(logp_old - ro.logp_engine).abs().mean().item(),
    ))
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
