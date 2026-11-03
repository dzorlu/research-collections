"""
Figures for bias_variance.md. One position, three tokens, b = 0.
We track one number: the batch's estimate of token 2's gradient, dlogits[:, 2].sum().

  python bias_variance_figs.py
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
p = torch.tensor([0.5, 0.3, 0.2])        # [V]  model
R_of = torch.tensor([0.0, 0.0, 1.0])     # [V]  reward if token k is written
TRUE = -p[2] * (1 - p[2])                # ()   true gradient on token 2's logit: -0.16
B = 16
N_BATCHES = 20000
ENGINES = {'close': torch.tensor([0.6, 0.3, 0.1]), 'far': torch.tensor([0.69, 0.30, 0.01])}

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
})


def weights(method, p_engine):
  """Per-token weight w[k] if token k is written, A > 0 on token 2 (b = 0). Returns [V]."""
  r = p / p_engine                                     # [V]
  moved = (p - p_engine).abs()                         # [V]
  if method == 'REINFORCE':
    return torch.ones(3)
  if method == 'IS':
    return r
  if method == 'TIS (cap 2)':
    return r.clamp(max=2.0)
  if method == 'PPO (ratio to engine)':
    return torch.where(r > 1.28, 0.0, r)               # A > 0 only matters on token 2
  if method == 'DPPO (delta 0.15)':
    return torch.where((r > 1) & (moved > 0.15), 0.0, r)
  raise ValueError(method)


def row(w, b=0.0):
  """Token 2's dlogits entry if token k is written, one sample. Returns [V]."""
  A = R_of - b                                         # [V]
  is_2 = torch.tensor([0.0, 0.0, 1.0])                 # [V]  onehot(k)[2] for k = 0, 1, 2
  return w * A * (p[2] - is_2)                         # [V]


def exact(w, p_engine, b=0.0):
  """Bias and per-sample variance of token 2's estimate, from the exact row distribution."""
  x = row(w, b)                                        # [V]
  mean = (p_engine * x).sum()
  var = (p_engine * (x - mean) ** 2).sum()
  bias = mean - (p * row(torch.ones(3), b)).sum()      # target: the model's own average row
  return bias.item(), var.item()


def batches(w, p_engine, b=0.0):
  """N_BATCHES batch estimates of token 2's gradient. Returns [N_BATCHES]."""
  y = torch.multinomial(p_engine, N_BATCHES * B, replacement=True).reshape(N_BATCHES, B)   # [N, B]
  return row(w, b)[y].mean(-1)                         # [N]


def hist(ax, est, color, lo, hi):
  """Share of batches at each estimate (the estimates are discrete), true value dashed, average as a triangle."""
  values, counts = est.unique(return_counts=True)                       # [n], [n]
  ax.bar(values.numpy(), (counts / len(est)).numpy(), width=(hi - lo) / 120, color=color)
  ax.axvline(TRUE, color=INK, linestyle='--', linewidth=1.2)
  ax.plot(est.mean().item(), 0, marker='^', color=INK, markersize=9, clip_on=False, zorder=5)
  ax.set_xlim(lo, hi)
  ax.set_ylabel('share of batches', fontsize=8)


# Figure 1: the baseline moves the spread, not the center (model samples itself).
fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True, sharey=True)
for ax, b, color in zip(axes, [0.0, 0.2], [BLUE, ORANGE]):
  est = batches(torch.ones(3), p, b)
  bias, var = exact(torch.ones(3), p, b)
  hist(ax, est, color, -0.5, 0.1)
  ax.set_title(f'b = {b}:  bias {bias:+.3f},  variance per sample {var:.3f}', fontsize=10, loc='left', color=INK)
axes[1].set_xlabel(f"batch estimate of token 2's gradient (B = {B}, {N_BATCHES:,} batches)\ndashed: true gradient -0.16;  triangle: average over batches")
fig.tight_layout()
fig.savefig('figs/bv_baseline.png', dpi=150)

# Figure 2: three weights, far engine. Where the estimates land.
methods = ['REINFORCE', 'IS', 'TIS (cap 2)']
fig, axes = plt.subplots(3, 1, figsize=(7, 5.8), sharex=True, sharey=True)
for ax, m, color in zip(axes, methods, [BLUE, ORANGE, AQUA]):
  w = weights(m, ENGINES['far'])
  bias, var = exact(w, ENGINES['far'])
  hist(ax, batches(w, ENGINES['far']), color, -2.2, 0.2)
  ax.set_title(f'{m}:  bias {bias:+.3f},  variance per sample {var:.3f}', fontsize=10, loc='left', color=INK)
axes[2].set_xlabel(f"batch estimate of token 2's gradient, far engine (B = {B}, {N_BATCHES:,} batches)\ndashed: true gradient -0.16;  triangle: average over batches")
fig.tight_layout()
fig.savefig('figs/bv_weights_far.png', dpi=150)

# Figure 3: error = bias^2 + variance / B, per method, close and far engine.
all_methods = ['REINFORCE', 'IS', 'TIS (cap 2)', 'PPO (ratio to engine)', 'DPPO (delta 0.15)']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (name, p_engine) in zip(axes, ENGINES.items()):
  b2, v = zip(*[(bias ** 2, var / B) for bias, var in (exact(weights(m, p_engine), p_engine) for m in all_methods)])
  x = range(len(all_methods))
  ax.bar(x, b2, color=BLUE, width=0.6, label='bias²')
  ax.bar(x, v, bottom=b2, color=ORANGE, width=0.6, label=f'variance / B', edgecolor='#fcfcfb', linewidth=2)
  ax.set_xticks(list(x), [m.replace(' (', '\n(') for m in all_methods], fontsize=8)
  ax.set_title(f'{name} engine  (p_engine = {[round(v, 2) for v in p_engine.tolist()]})', fontsize=10, loc='left')
  ax.grid(axis='x', visible=False)
axes[0].set_ylabel(f'squared error of the batch estimate (B = {B})')
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig('figs/bv_methods.png', dpi=150)

# Figure 4: TV and chi-squared as the engine makes token 2 rarer: p_engine = [0.7 - e, 0.3, e].
e = torch.logspace(torch.log10(torch.tensor(0.2)), torch.log10(torch.tensor(0.005)), 60)   # [60]
p_engines = torch.stack([0.7 - e, torch.full_like(e, 0.3), e], -1)                         # [60, V]
tv = 0.5 * (p - p_engines).abs().sum(-1)                                                    # [60]
chi2 = (p ** 2 / p_engines).sum(-1) - 1                                                     # [60]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
for ax, y, color, name in zip(axes, [tv, chi2], [BLUE, ORANGE], ['TV', 'chi-squared']):
  ax.plot(e.numpy(), y.numpy(), color=color, linewidth=2)
  ax.set_xscale('log')
  ax.set_title(name, fontsize=10, loc='left')
  ax.set_xlabel("engine's probability of token 2 (model: 0.2)")
  for label, pe in ENGINES.items():
    exact_y = 0.5 * (p - pe).abs().sum() if name == 'TV' else (p ** 2 / pe).sum() - 1
    ax.plot(pe[2].item(), exact_y.item(), 'o', color=color, markersize=7, markeredgecolor='#fcfcfb', markeredgewidth=2)
    ax.annotate(f' {label}: {exact_y.item():.2f}', (pe[2].item(), exact_y.item()), fontsize=9, color=INK,
                xytext=(4, 4), textcoords='offset points')
axes[0].invert_xaxis()
fig.tight_layout()
fig.savefig('figs/bv_distance.png', dpi=150)

# Figure 5: the cap trade, far engine. Error vs cap for two batch sizes.
caps = torch.logspace(0, torch.log10(torch.tensor(20.0)), 80)   # [80]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
r = p / ENGINES['far']
for ax, batch in zip(axes, [16, 256]):
  stats = [exact(r.clamp(max=c.item()), ENGINES['far']) for c in caps]
  b2 = torch.tensor([s[0] ** 2 for s in stats])
  v = torch.tensor([s[1] / batch for s in stats])
  ax.plot(caps.numpy(), b2.numpy(), color=BLUE, linewidth=2, label='bias²')
  ax.plot(caps.numpy(), v.numpy(), color=ORANGE, linewidth=2, label='variance / B')
  ax.plot(caps.numpy(), (b2 + v).numpy(), color=AQUA, linewidth=2, label='error')
  best = (b2 + v).argmin()
  ax.annotate(f'best cap {caps[best].item():.1f}', (caps[best].item(), (b2 + v)[best].item()), fontsize=9,
              xytext=(0, 18), textcoords='offset points', ha='center')
  ax.set_xscale('log')
  ax.set_title(f'B = {batch}', fontsize=10, loc='left')
  ax.set_xlabel('cap on w (20 = no cap)')
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig('figs/bv_cap.png', dpi=150)

# Numbers used in the text.
for name, p_engine in ENGINES.items():
  for m in all_methods:
    w = weights(m, p_engine)
    bias, var = exact(w, p_engine)
    print(f'{name:5s} {m:22s} w2={w[2].item():6.2f}  bias {bias:+.4f}  var {var:.4f}  error@16 {bias**2 + var/16:.4f}')
for b in [0.0, 0.2]:
  print('on-policy b', b, exact(torch.ones(3), p, b))
