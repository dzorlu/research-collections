"""Reproduces every number in mopd-reverse-kl.md.

Run: python3 reverse_kl_example.py
"""
import numpy as np

TOKENS = ["A", "B", "C"]
P_T = np.array([0.6, 0.3, 0.1])  # teacher
P_S = np.array([0.2, 0.5, 0.3])  # student


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def kl(p, q):
    return float((p * np.log(p / q)).sum())


def kl_grad_logits(p_s, p_t):
    """Exact gradient of KL(student || teacher) w.r.t. the student's logits."""
    return p_s * (np.log(p_s / p_t) - kl(p_s, p_t))


def main():
    student_surprise = -np.log(P_S)
    teacher_surprise = -np.log(P_T)
    adv = np.log(P_T) - np.log(P_S)
    opd_loss = -adv * np.log(P_S)

    print("Table 2: per token")
    for i, t in enumerate(TOKENS):
        print(f"  {t}: -log pS={student_surprise[i]:.3f}  -log pT={teacher_surprise[i]:.3f}"
              f"  A={adv[i]:+.3f}  -A*log pS={opd_loss[i]:+.3f}")
    for name, w in [("teacher samples (SFT)", P_T), ("student samples (OPD)", P_S)]:
        cols = [float((w * c).sum()) for c in (student_surprise, teacher_surprise, adv, opd_loss)]
        print(f"  average over {name}: " + "  ".join(f"{c:+.3f}" for c in cols))
    print(f"forward KL(T||S) = {kl(P_T, P_S):.3f}, reverse KL(S||T) = {kl(P_S, P_T):.3f}")

    # Expected gradient of the policy-gradient loss -sg(A) * log pS(y), y ~ pS.
    # d log pS(y) / d logits = onehot(y) - pS.
    pg_grad = sum(P_S[y] * -adv[y] * (np.eye(3)[y] - P_S) for y in range(3))
    print("gradient wrt logits: exact KL", np.round(kl_grad_logits(P_S, P_T), 3),
          " policy-gradient loss", np.round(pg_grad, 3))
    # Expected gradient of -A alone (no stop-gradient): sum_y pS(y) * grad log pS(y) = 0.
    print("expected gradient of -A alone", np.round(sum(P_S[y] * (np.eye(3)[y] - P_S) for y in range(3)), 3))

    z = np.log(P_S) - 1.0 * kl_grad_logits(P_S, P_T)
    print("after one step (lr 1): student", np.round(softmax(z), 3), " reverse KL", round(kl(softmax(z), P_T), 3))

    two_mode = np.array([0.49, 0.02, 0.49])
    for name, s in [("one mode", np.array([0.96, 0.02, 0.02])), ("spread", np.array([0.34, 0.32, 0.34]))]:
        print(f"mode seeking, {name}: reverse KL {kl(s, two_mode):.3f}, forward KL {kl(two_mode, s):.3f}")

    print(f"clipping example: A = {np.log(1e-6 / 0.5):.1f}; smoothed teacher (alpha 0.2): "
          f"{np.log((0.2 * 1e-6 + 0.8 * 0.5) / 0.5):.3f}")

    p, top = np.array([0.3, 0.15, 0.55]), [0, 1]
    naive = sum(p[v] * np.log(p[v] / P_T[v]) for v in top)
    corrected = sum(p[v] * np.log(p[v] / P_T[v]) - p[v] + P_T[v] for v in top)
    print(f"top-2 loss at student (0.3, 0.15, 0.55): naive {naive:.2f}, corrected {corrected:.2f}")


if __name__ == "__main__":
    main()
