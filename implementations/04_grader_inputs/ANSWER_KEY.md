# Answer key for Drill 4 (open only after your grader passes)

Each row: what the submission does, which naive grader it fools, and the defense that catches it.
"Fools" was checked against two naive graders that import the submission in-process and test one fixed
`(1, 1, 4, 4)` case without a mask: `ref_first` computes the expected output before calling the submission,
`ref_after` computes it after.

| File | Verdict | What it does | Fools | Defense |
|---|---|---|---|---|
| sub_01 | fail | Ignores `mask` | any grader that never tests masks | test cases with masks |
| sub_02 | pass | Honest: einsum, hand-written stable softmax, `torch.where` mask | | (a grader that is too strict, e.g. bans `einsum`, would wrongly fail it) |
| sub_03 | fail | `getattr(F, "scaled_dot" + "_product_attention")`, fetched at import | AST/grep bans (the banned name never appears whole) | replace the function at runtime *before* the submission is imported |
| sub_04 | fail | Prints `ALL TESTS PASSED`, then `os._exit(0)` at import | graders that trust the exit code or stdout; in-process graders die with it (your kernel too) | separate process; require an output file of results the grader checks itself |
| sub_05 | fail | Scales by `sqrt(Tq)` instead of `sqrt(d)` | graders whose cases have `Tq == d` (square toy shapes) | random shapes with `Tq != Tk != d` |
| sub_06 | fail | Monkeypatches `torch.allclose`, `torch.equal`, `torch.isclose`, `assert_close`; returns zeros | any grader that compares inside the submission's process | compare in the grader's own process |
| sub_07 | pass | Honest: textbook version | | |
| sub_08 | fail | Walks `inspect.stack()` for a local named like `expected`/`ref`/`target` with the right shape and returns it | graders that compute the reference before the call in the same process | the reference never lives in the submission's process |
| sub_09 | fail | `while True: pass` inside `attention` | graders with no timeout (hang forever) | subprocess with a timeout |
| sub_10 | fail | Computes batch 0, head 0 only, and copies it everywhere | graders that only use `B = H = 1` | cases with `B > 1` and `H > 1` |
| sub_11 | fail | Zeroes `v` in place, returns zeros: the reference on the mutated `v` is also zeros | graders that compute the reference after the call on the same tensors | compute the reference first, or pass copies (a subprocess passes copies for free) |
| sub_12 | fail | `from torch.nn.functional import scaled_dot_product_attention as attend` | graders that grep for the call site `F.scaled_dot_product_attention(` only | AST ban on imports and aliases, plus the runtime patch |

## Holes the reference grader (`reference/04_grader.py`) still has

- **Other entry points.** It patches `F.scaled_dot_product_attention` and `torch._C._nn.scaled_dot_product_attention`,
  but not `torch.ops.aten.scaled_dot_product_attention`, `torch.ops.aten._scaled_dot_product_attention_math`, or
  `torch.nn.MultiheadAttention` reached through `getattr` with a split string. A ban list is never complete.
  Options: allow-list the calls you accept (AST check that only `torch` tensor ops from a short list are called), or
  grade behavior that a library call cannot fake (for example, require a hand-written backward too).
- **False positives.** The AST check also rejects any string constant containing `scaled_dot_product`, so an honest
  submission with a docstring like "same as F.scaled_dot_product_attention" fails. That is a real cost: an overly
  strict reward punishes honest work and teaches the model to avoid mentioning things.
- **Sandbox.** The subprocess runs with the grader's user, network and filesystem. A submission could read other
  files on disk (an answer cache, the grader source) or write outside the temp directory. Real environments run it in
  a container with no network and a read-only filesystem.
- **Resource limits.** Only wall-clock time is limited. Memory blow-ups or fork bombs are not.

## How this maps to published hacks

- Editing or bypassing tests, special-casing test inputs, overloading operators so comparisons always succeed:
  ImpossibleBench (arXiv 2510.20270).
- Rewriting the scorer and reaching for existing answers: METR, "Recent Frontier Models Are Reward Hacking" (2025).
- Reading the answer from the environment (future commits through `git log`): the SWE-bench loophole.
