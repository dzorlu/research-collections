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
  if method == 'PPO (eps 0.2)':
    return torch.where(r > 1.2, 0.0, r)                # A > 0 only matters on token 2
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
  values, counts = (est * 1e4).round().div(1e4).unique(return_counts=True)   # [n], [n]  rounded: the estimates are discrete
  width = 0.8 * values.diff().min().item() if len(values) > 1 else (hi - lo) / 120   # fill the gap between values
  ax.bar(values.numpy(), (counts / len(est)).numpy(), width=width, color=color)
  ax.axvline(TRUE2, color=INK, linestyle='--', linewidth=1.2)
  ax.plot(est.mean().item(), 0, marker='^', color=INK, markersize=9, clip_on=False, zorder=5)
  ax.set_xlim(lo, hi)
  ax.set_ylabel('share of batches', fontsize=8)


# Figure 1: model vs close engine writing the tokens, w = 1. Same scatter, different center.
fig, ax = plt.subplots(figsize=(7, 3.4))
lo, hi = -0.5, 0.1
width = 0.05 * 0.4                                                      # estimates are multiples of 0.05; two bars per slot
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

# Figure 4: three weights, far engine. The exact distribution of one batch estimate (token 2's coordinate).
# A batch of B rows from the far engine holds k token-2 rows with k ~ Binomial(B, 0.01); its estimate is w_2 * k / B * v.
methods = ['REINFORCE', 'IS', 'TIS (cap 2)']
colors = [BLUE, ORANGE, AQUA]
q = ENGINES['far'][2].item()
k = torch.arange(0, 4)                                                            # [4]  token-2 rows in the batch
pk = torch.distributions.Binomial(B, torch.tensor(q)).log_prob(k.float()).exp()   # [4]  P(k)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
for ax, m, color in zip(axes, methods, colors):
  w2 = weights(m, ENGINES['far'])[2].item()
  est = w2 * k / B * rows(torch.ones(V))[2, 2]                                    # [4]  token 2's coordinate of the estimate
  mean = (pk * est).sum().item()
  ax.bar(est.numpy(), pk.numpy(), width=0.045 * (est[-1] - est[0]).abs().item(), color=color)
  for e, pr in zip(est.tolist(), pk.tolist()):
    if pr < 0.005: continue
    ax.annotate(f'{pr:.0%}', (e, pr), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, color=MUTED)
  ax.axvline(TRUE2, color=INK, linestyle='--', linewidth=1.2)
  ax.plot(mean, 0, marker='^', color=INK, markersize=9, clip_on=False, zorder=5)
  ax.set_title(f'{m}, w_2 = {w2:g}', fontsize=10, loc='left')
  ax.set_xlabel(f"token 2's coordinate of one batch of {B}")
  ax.grid(axis='x', visible=False)
  ax.set_ylim(0, 1)
axes[0].set_ylabel('probability of the batch')
fig.tight_layout()
fig.savefig('figs/bv_weights_far.png', dpi=150)
print('far engine, B=16: P(k token-2 rows)', [f'{v:.3f}' for v in pk.tolist()])

all_methods = ['REINFORCE', 'IS', 'TIS (cap 2)', 'PPO (eps 0.2)', 'DPPO (delta 0.15)']

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
  print(f'cap trade B={batch}: best cap {caps[best].item():.1f}, error there {(b2 + v)[best].item():.4f}, '
        f'error at cap 2 {(b2 + v)[(caps - 2).abs().argmin()].item():.4f}, at cap 4 {(b2 + v)[(caps - 4).abs().argmin()].item():.4f}, uncapped {(b2 + v)[-1].item():.4f}')
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
