"""Reference solution for 02_rl_loop.ipynb: every cell, with the TODO cells filled in."""

import time
import torch

V = 10


class Policy(torch.nn.Module):
    def __init__(self, d=32, h=128):
        super().__init__()
        self.ea = torch.nn.Embedding(10, d)
        self.eb = torch.nn.Embedding(10, d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * d, h), torch.nn.ReLU(), torch.nn.Linear(h, V))

    def forward(self, a, b):                                           # a, b (N,) int
        return self.mlp(torch.cat([self.ea(a), self.eb(b)], -1))       # (N, V) logits


def sample_prompts(n, gen=None):
    return torch.randint(10, (n,), generator=gen), torch.randint(10, (n,), generator=gen)   # (n,), (n,)


def grpo_advantages(rewards, G, eps=1e-6):
    r = rewards.view(-1, G)                                              # (N, G)
    A = (r - r.mean(-1, keepdim=True)) / (r.std(-1, correction=0, keepdim=True) + eps)
    return A.reshape(-1)                                                 # (N*G,)


def policy_loss(logp_new, logp_old, A, mask, mode, eps_lo=0.2, eps_hi=0.28, cispo_max=4.0, norm="token"):
    real = mask.bool()                                                   # (B, T)
    mask = mask.to(logp_new.dtype)                                       # (B, T)
    diff = torch.where(real, logp_new - logp_old, torch.zeros_like(logp_new))   # (B, T), garbage-safe
    if mode == "gspo":
        s = torch.exp((diff * mask).sum(-1) / mask.sum(-1))              # (B,)
        l = -torch.minimum(s * A, s.clamp(1 - eps_lo, 1 + eps_hi) * A)   # (B,)
        return l.mean()
    A_t = A.view(-1, 1)                                                  # (B, 1)
    r = torch.exp(diff)                                                  # (B, T)
    if mode == "reinforce":
        l = -A_t * logp_new
    elif mode == "ppo":
        l = -torch.minimum(r * A_t, r.clamp(1 - eps_lo, 1 + eps_hi) * A_t)
    elif mode == "cispo":
        l = -r.clamp(0, cispo_max).detach() * A_t * logp_new
    else:
        raise ValueError(mode)
    l = torch.where(real, l, torch.zeros_like(l))                        # (B, T)
    if norm == "token":
        return l.sum() / mask.sum()
    return (l.sum(-1) / mask.sum(-1)).mean()


def reward_fn(a, b, out):
    return (out == (a + b) % 10).float()                                 # (N,)


def rollout(policy, a, b, G):
    a_rep, b_rep = a.repeat_interleave(G), b.repeat_interleave(G)       # (N*G,)
    with torch.no_grad():
        logp_all = torch.log_softmax(policy(a_rep, b_rep), -1)           # (N*G, V)
        out = torch.multinomial(logp_all.exp(), 1).squeeze(-1)           # (N*G,)
        logp_old = logp_all.gather(-1, out.view(-1, 1)).squeeze(-1)      # (N*G,)
    return a_rep, b_rep, out, logp_old


def train(policy, reward, iters=600, n_prompts=64, G=16, ppo_epochs=2, lr=1e-3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = {"reward": [], "true_acc": []}
    for _ in range(iters):
        a, b = sample_prompts(n_prompts, gen)
        a_rep, b_rep, out, logp_old = rollout(policy, a, b, G)
        R = reward(a_rep, b_rep, out)                                    # (N*G,)
        A = grpo_advantages(R, G)                                        # (N*G,)
        for _ in range(ppo_epochs):
            logp = torch.log_softmax(policy(a_rep, b_rep), -1).gather(-1, out.view(-1, 1))   # (N*G, 1)
            loss = policy_loss(logp, logp_old.view(-1, 1), A, torch.ones_like(logp), "ppo")
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist["reward"].append(R.mean().item())
        hist["true_acc"].append(reward_fn(a_rep, b_rep, out).mean().item())
    return hist


r = reward_fn(torch.tensor([1, 9, 5]), torch.tensor([2, 3, 5]), torch.tensor([3, 2, 1]))
assert r.dtype.is_floating_point and r.tolist() == [1.0, 1.0, 0.0], r
print("ok")


torch.manual_seed(0)
policy = Policy()
a, b = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
a_rep, b_rep, out, logp_old = rollout(policy, a, b, G=4)
assert a_rep.tolist() == [1] * 4 + [2] * 4 + [3] * 4, "the G samples of one prompt must be next to each other"
assert b_rep.tolist() == [4] * 4 + [5] * 4 + [6] * 4
assert out.shape == logp_old.shape == (12,)
assert not logp_old.requires_grad, "logp_old must not carry grad"
want = torch.log_softmax(policy(a_rep, b_rep), -1).gather(-1, out.view(-1, 1)).squeeze(-1)
assert torch.allclose(logp_old, want, atol=1e-5), "logp_old is not the log-prob of the sampled answers"

# sampling, not argmax: empirical frequencies match the policy's probabilities
_, _, out_many, _ = rollout(policy, torch.tensor([7]), torch.tensor([8]), G=20000)
freq = torch.bincount(out_many, minlength=10).float() / 20000
probs = torch.softmax(policy(torch.tensor([7]), torch.tensor([8])), -1)[0]
assert (freq - probs).abs().max() < 0.02, (freq, probs)
print("ok")


torch.manual_seed(0)
policy = Policy()
t0 = time.time()
hist = train(policy, reward_fn)
last = sum(hist["reward"][-20:]) / 20
print(f"iterations: {len(hist['reward'])}   reward first 10: {sum(hist['reward'][:10]) / 10:.2f}   "
      f"last 20: {last:.2f}   time {time.time() - t0:.0f}s")
assert hist["true_acc"] == hist["reward"], "with the true reward, true_acc must equal reward"
assert last > 0.9, "reward did not reach 0.9"
print("ok")


def output_share(policy, n=2000, seed=1):
    a, b = sample_prompts(n, torch.Generator().manual_seed(seed))
    _, _, out, _ = rollout(policy, a, b, 1)                              # (n,)
    return torch.bincount(out, minlength=10).float() / n                 # (10,)


def hacked_reward(a, b, out):
    return ((out == (a + b) % 10) | (out == 0)).float()


torch.manual_seed(0)
policy_h = Policy()
hist_h = train(policy_h, hacked_reward)
r_h = sum(hist_h["reward"][-20:]) / 20
acc_h = sum(hist_h["true_acc"][-20:]) / 20
share_h, share_ok = output_share(policy_h), output_share(policy)
print(f"hacked run: reward {r_h:.2f}   true_acc {acc_h:.2f}")
print("answer share, hacked policy:", [round(x, 2) for x in share_h.tolist()])
print("answer share, honest policy:", [round(x, 2) for x in share_ok.tolist()])
assert r_h > 0.9 and acc_h < 0.5, "expected high reward with low true accuracy"
assert abs(share_ok.sum().item() - 1) < 1e-5 and share_h.shape == (10,)
assert share_h[0] > 0.5 and share_ok.max() < 0.3, "output_share should show the collapse onto 0"
print("ok")
