"""
Figures for bias_variance.md. One position, three tokens, b = 0 unless stated.
Every bias, variance and error is for the full dlogits vector [V]. Figures 1 and 2 draw one coordinate of the
batch estimate, token 2's, because a distribution of vectors is hard to draw.

  python bias_variance_figs.py
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
V = 3
p = torch.tensor([0.5, 0.3, 0.2])        # [V]  model
reward = torch.tensor([0.0, 0.0, 1.0])   # [V]  reward if token k is written
E = torch.eye(V)                         # [V, V]  row k = e_k, one-hot of token k
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
    return torch.ones(V)
  if method == 'IS':
    return r
  if method == 'TIS (cap 2)':
    return r.clamp(max=2.0)
  if method == 'PPO (ratio to engine)':
    return torch.where(r > 1.28, 0.0, r)               # A > 0 only matters on token 2
  if method == 'DPPO (delta 0.15)':
    return torch.where((r > 1) & (moved > 0.15), 0.0, r)
  raise ValueError(method)


def rows(w, b=0.0):
  """The dlogits row if token k is written, for each k. Returns [V, V]: rows[k] = w[k] * A[k] * (p - e_k)."""
  A = reward - b                                       # [V]
  return (w * A)[:, None] * (p - E)                    # [V, V]


G = (p[:, None] * rows(torch.ones(V))).sum(0)          # [V]  true gradient: (0.10, 0.06, -0.16)
TRUE2 = G[2].item()                                    # ()   its token 2 coordinate, -0.16, for the coordinate plots


def exact(w, p_engine, b=0.0):
  """Squared bias and per-sample variance of the batch estimate, from the exact row distribution."""
  x = rows(w, b)                                       # [V, V]
  mean = (p_engine[:, None] * x).sum(0)                # [V]
  var = (p_engine * ((x - mean) ** 2).sum(-1)).sum()   # ()   E||x - mean||^2
  bias2 = ((mean - G) ** 2).sum()                      # ()   ||mean - G||^2
  return bias2.item(), var.item()


def batches(w, p_engine, b=0.0):
  """N_BATCHES batch estimates, token 2's coordinate. Returns [N_BATCHES]."""
  y = torch.multinomial(p_engine, N_BATCHES * B, replacement=True).reshape(N_BATCHES, B)   # [N, B]
  return rows(w, b)[y][..., 2].mean(-1)                # [N]


def hist(ax, est, color, lo, hi):
  """Share of batches at each estimate (the estimates are discrete), true value dashed, average as a triangle."""
  values, counts = est.unique(return_counts=True)                       # [n], [n]
  ax.bar(values.numpy(), (counts / len(est)).numpy(), width=(hi - lo) / 120, color=color)
  ax.axvline(TRUE2, color=INK, linestyle='--', linewidth=1.2)
  ax.plot(est.mean().item(), 0, marker='^', color=INK, markersize=9, clip_on=False, zorder=5)
  ax.set_xlim(lo, hi)
  ax.set_ylabel('share of batches', fontsize=8)


# Figure 1: model vs close engine writing the tokens, w = 1. Same scatter, different center.
fig, ax = plt.subplots(figsize=(7, 3.4))
lo, hi = -0.5, 0.1
width = (hi - lo) / 120
for (name, p_sampler), color, shift in zip([('model, q = 0.2', p), ('close engine, q = 0.1', ENGINES['close'])], [BLUE, ORANGE], [-0.5, 0.5]):
  est = batches(torch.ones(V), p_sampler)                              # [N_BATCHES]
  values, counts = est.unique(return_counts=True)                      # [n], [n]
  ax.bar(values.numpy() + shift * width, (counts / len(est)).numpy(), width=width, color=color, label=name)
  ax.plot(est.mean().item(), 0, marker='^', color=color, markersize=10, markeredgecolor=INK, clip_on=False, zorder=5)
ax.axvline(TRUE2, color=INK, linestyle='--', linewidth=1.2)
ax.set_xlim(lo, hi)
ax.set_ylabel('share of batches', fontsize=8)
ax.set_xlabel(f"token 2's coordinate of the batch estimate (B = {B}, {N_BATCHES:,} batches each)\ndashed: true value -0.16;  triangles: average over batches")
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig('figs/bv_bias_scatter.png', dpi=150)

# Figure 2: ||bias||^2 and variance / B as the sampler's probability of token 2, q, varies. w = 1, b = 0.
v = p - E[2]                                                            # [V]  the y = 2 row
qs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, batch in zip(axes, [4, 16]):
  bias2 = [(v.norm() ** 2 * (q - p[2]) ** 2).item() for q in qs]
  var_b = [(v.norm() ** 2 * q * (1 - q) / batch).item() for q in qs]
  x = range(len(qs))
  ax.bar(x, bias2, color=BLUE, width=0.6, label='||bias||²')
  ax.bar(x, var_b, bottom=bias2, color=ORANGE, width=0.6, label='variance / B', edgecolor='#fcfcfb', linewidth=2)
  ax.set_xticks(list(x), [f'{q:g}' for q in qs])
  for q, name in [(0.1, 'engine'), (0.2, 'model')]:
    i = qs.index(q)
    ax.annotate(name, (i, bias2[i] + var_b[i]), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, color=INK)
  ax.set_title(f'B = {batch}', fontsize=10, loc='left')
  ax.set_xlabel("q: the sampler's probability of token 2 (model: 0.2)")
  ax.grid(axis='x', visible=False)
axes[0].set_ylabel('squared error of the batch estimate')
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig('figs/bv_q.png', dpi=150)

# Figure 3: the baseline moves the spread, not the center (model samples itself).
fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True, sharey=True)
for ax, b, color in zip(axes, [0.0, 0.2], [BLUE, ORANGE]):
  est = batches(torch.ones(V), p, b)
  bias2, var = exact(torch.ones(V), p, b)
  hist(ax, est, color, -0.5, 0.1)
  ax.set_title(f'b = {b}:  ||bias||² {bias2:.3f},  variance per sample {var:.3f}', fontsize=10, loc='left', color=INK)
axes[1].set_xlabel(f"token 2's coordinate of the batch estimate (B = {B}, {N_BATCHES:,} batches)\ndashed: true value -0.16;  triangle: average over batches")
fig.tight_layout()
fig.savefig('figs/bv_baseline.png', dpi=150)

# Figure 2: three weights, far engine. Left: individual batches. Right: what averaging many batches converges to.
methods = ['REINFORCE', 'IS', 'TIS (cap 2)']
colors = [BLUE, ORANGE, AQUA]
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={'width_ratios': [1, 1.3]})
n_show = 300
for i, (m, color) in enumerate(zip(methods, colors)):
  w = weights(m, ENGINES['far'])
  est = batches(w, ENGINES['far'])                                   # [N_BATCHES]
  jitter = (torch.rand(n_show) - 0.5) * 0.5                           # [n_show]
  ax_l.scatter(est[:n_show].numpy(), (i + jitter).numpy(), s=10, color=color, alpha=0.5, linewidths=0)
  running = est.cumsum(0) / torch.arange(1, len(est) + 1)            # [N_BATCHES]
  ax_r.plot(torch.arange(1, len(est) + 1).numpy(), running.numpy(), color=color, linewidth=2, label=f'{m}: ends at {running[-1].item():.3f}')
for ax in (ax_l, ax_r):
  ax.grid(axis='y' if ax is ax_r else 'x')
ax_l.axvline(TRUE2, color=INK, linestyle='--', linewidth=1.2)
ax_l.set_yticks(range(3), methods)
ax_l.invert_yaxis()
ax_l.grid(axis='y', visible=False)
ax_l.set_xlabel(f"token 2's coordinate, one batch of {B}  (dashed: true -0.16)")
ax_l.set_title(f'{n_show} single batches', fontsize=10, loc='left')
ax_r.axhline(TRUE2, color=INK, linestyle='--', linewidth=1.2)
ax_r.set_xscale('log')
ax_r.set_xlim(1, N_BATCHES)
ax_r.legend(frameon=False, loc='lower right', fontsize=9)
ax_r.annotate('true -0.16', (1.2, TRUE2), xytext=(0, 4), textcoords='offset points', fontsize=9, color=INK)
ax_r.set_ylim(-0.6, 0.05)
ax_r.set_xlabel('number of batches averaged')
ax_r.set_ylabel("average estimate, token 2's coordinate")
ax_r.set_title('averaging more batches: where each method ends up', fontsize=10, loc='left')
fig.tight_layout()
fig.savefig('figs/bv_weights_far.png', dpi=150)

# Figure 3: error = ||bias||^2 + variance / B, per method, close and far engine.
all_methods = ['REINFORCE', 'IS', 'TIS (cap 2)', 'PPO (ratio to engine)', 'DPPO (delta 0.15)']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (name, p_engine) in zip(axes, ENGINES.items()):
  b2, v = zip(*[(bias2, var / B) for bias2, var in (exact(weights(m, p_engine), p_engine) for m in all_methods)])
  x = range(len(all_methods))
  ax.bar(x, b2, color=BLUE, width=0.6, label='||bias||²')
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
  b2 = torch.tensor([s[0] for s in stats])
  v = torch.tensor([s[1] / batch for s in stats])
  ax.plot(caps.numpy(), b2.numpy(), color=BLUE, linewidth=2, label='||bias||²')
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
torch.set_printoptions(precision=3, sci_mode=False)
print('G', G)
for name, p_engine in ENGINES.items():
  for m in all_methods:
    w = weights(m, p_engine)
    bias2, var = exact(w, p_engine)
    mean = (p_engine[:, None] * rows(w)).sum(0)
    print(f'{name:5s} {m:22s} w2={w[2].item():6.2f}  bias {(mean - G).tolist()}  ||bias||² {bias2:.4f}  var {var:.4f}  error@16 {bias2 + var/16:.4f}')
for b in [0.0, 0.2]:
  print('on-policy b', b, 'rows', rows(torch.ones(V), b), 'bias², var', exact(torch.ones(V), p, b))
far_is = exact(weights('IS', ENGINES['far']), ENGINES['far'])
p0 = 0.99 ** 16
print(f'far IS: P(no token 2 in 16) {p0:.3f}, P(exactly one) {16 * 0.01 * 0.99 ** 15:.3f}')
