# Quality checks for RL environments

## Summary

A reward is a measurement of "was the task done". Environment quality is the four properties any
measurement needs:

1. **Correct work scores 1.** Fails when the environment doesn't run under training limits, the task
   is unsolvable or ambiguous, the gold answer is wrong, or the verifier demands things the task never
   said.
2. **Non-work scores 0.** Fails when the verifier checks less than the task asks, the answer can be
   guessed, found leaked, or recalled from memory, or the agent can reach the verifier by a path other
   than doing the task (editing tests, flattering the judge).
3. **Same work, same score.** Fails with flaky tests, inconsistent judges, timeouts.
4. **Still true under optimization.** The policy gets stronger every step and finds what static checks
   missed; judges and human preferences drift over a multi-week run.

Every check below protects one of these. The checks come in three kinds, by who does the judging.

---

## Deterministic checks — code, no model judgment

Cheap, run first, most bad tasks die here.

**1. Environment works.** Build the task image; run the reference solution three separate times
inside the same CPU, memory, and time limits the RL sandbox uses; all three must agree. Catches tasks
that pass on a developer's machine but time out or OOM in training (a correct solution would then
score 0) and flaky setups.

**2. Verifier discriminates.** Two controls on every verifier: the reference ("oracle") solution must
score 1.0 on five runs; a no-op (unchanged environment, no actions) must score 0.0 on five runs. SWE
adds fail-to-pass (tests fail before the reference patch, pass after) and pass-to-pass (previously
passing tests still pass). Math: the gold answer must parse into a SymPy object, otherwise grading
silently falls back to string matching. Competitive coding: a minimum number of test cases, and the
output-comparison script must honor what the prompt says about case, whitespace, and precision.

**3. Structural guarantees.** Hacks that are made impossible rather than detected:
- the agent runs in an untrusted sandbox; grading runs in a fresh container from a pinned image; only
  the diff or output files cross between them, so nothing the agent did to its own box (shadowed
  `pytest`, edited `conftest.py`, installed packages) reaches the verifier;
- network leaves the sandbox through a policy proxy with mirrors and per-sandbox logs;
- verifier guards reject edits to test-selection config and import hooks;
- answers are regex-extracted and length-capped before any judge sees them (search: max(150,
  4.5 × gold length) chars; math: 200 chars), which removes hedging ("3 or 4 or 5") mechanically;
- the online judge is fail-closed: if it errors after retries the rollout fails, so an unreviewed pass
  never becomes positive reward;
- serving instances verify transferred weights before adopting them.

**4. Leak scans.** Strip post-fix git history from repo images; rebuild compiled artifacts from the
modified repo so stale copies of removed code don't survive; grep images for the removed code or
tests; for web tasks, blocklist URLs of known answer dumps and fuzzy-match question+answer against
pages the model fetched.

**5. Statistical filters computed from rollouts.**
- *Pass-rate band*: drop tasks that are all-fail or all-pass for the current model (no gradient).
- *Strong-vs-weak gap*: sample a strong and a weak model; keep a question only if the strong model
  wins by a margin (web: ≥ 12.5 points, or weak = 0). A weak model beating a strong one usually means
  the gold answer is wrong.
- *Memorized*: a search question the model answers correctly with no tools is dropped.
- *Too easy*: dropped if the initial checkpoint solves it at a low thinking budget.
- *Guessable by statistics*: drop questions whose gold answer is unusually frequent across the pool
  (0, 1, π).
- *Dedup and decontamination*: character 7-gram MinHash; 0.95 similarity within training data, 0.80
  against evaluation sets; SWE also matches patches to held-out evals.
- *During training*: discard rollouts past their task deadline; keep a group only if enough finished;
  discard groups where every rollout got the same reward.

---

## Agentic checks — a model judges

Each is a prompt-plus-checklist or an agent with tools. They are themselves models and can be wrong,
which is why the human checks below exist.

**1. Task soundness.** A judge is given the task prompt (and for SWE the repository) with the
reference solution withheld, and runs a checklist:
- references something the agent can't see — a figure, an attachment, an earlier discussion;
- contradicts itself or is missing an assumption;
- can be read two ways;
- demands an answer format the grader can't handle;
- answer guessable from the form of the question — yes/no, multiple choice, a quantity with few
  possible values, a definition to recite.

Any flag fails the task; repairable prompts are rewritten and re-judged. The reference is withheld
because a judge who sees the answer rationalizes the task from it.

**2. Specification ↔ verifier alignment.** The prompt and the verifier must demand the same set of
things.
- *Ambiguity judge* — verifier demands more than the prompt states. It looks for test requirements
  not justified by the visible spec and, where practical, writes an alternative implementation that
  satisfies the spec and shows the verifier rejecting it.
- *Verifier-strength judge* — verifier demands less than the prompt states. It lists the requirements
  from the prompt, maps each to a test, then writes a deliberately incomplete solution to see if it
  passes.
A separate knob, task *specificity* (how much the prompt spells out vs leaves to inference), is a
generation setting for diversity, not a filter; a low-specificity task still has to pass check 1.

**3. Gold verification.** The gold answer is not trusted. Three checks:
- *Blind re-derivation*: a strong model is given the question only — never the gold — and solves it
  with tools (code execution, search). Several such solvers run independently. The question is
  accepted only if they agree with each other and with the gold. If they can't solve it or disagree,
  the question is dropped: the cause can't be told apart (wrong gold, ambiguous question, solver too
  weak), so it isn't admitted. Conservative — some hard valid questions are lost.
- *Review with the gold*: a separate model sees question + gold and judges whether the gold is valid.
  Web tasks also sample many rollouts, collect answers that recur, and ask a model whether a common
  alternative is the better answer or whether the question is contradictory (neither fits) or
  underspecified (both fit).
- *Rewrite*: questions are edited to pin units, precision, and constants (9.8 vs 9.81) so correct
  reasoning always lands on the gold instead of being graded on rounding.

Wrong golds that slip through are caught statistically by the strong-vs-weak gap (deterministic 5): a
weak model out-scoring a strong one usually means the key is wrong.

**4. Adversarial attack.** Two attackers, one instruction — get reward 1.0 without doing the task —
differing only in what they see:

```text
                 sees                            produces              scored by
black-box  ───►  task + environment         ───► deliverable ──┐
                 + written hack taxonomy                       ├──►  rubric judge  (criteria met?)
white-box  ───►  task + environment         ───► deliverable ──┘     goal judge    (job actually done?)
                 + rubric + grading spec
                                                 rubric PASS & goal FAIL  →  task is exploitable
                                                   black-box got through  →  fault is in the task
                                                   white-box got through  →  fault is in the grader
```

*Black-box attacker*
- sees only what the policy will see: task text and environment;
- works through a written checklist of known hacks: hard-code the expected output, edit tests or
  `conftest.py`, shadow binaries on PATH, dig for leaked answers in git history or build artifacts,
  hedge between answers, take the easiest reading of an ambiguous prompt, ask the simulated user;
- a success means the fault is in the task or environment — the policy would find the same path.

*White-box attacker* (rubric-graded tasks)
- additionally sees the rubric, criterion weights, and grading spec;
- does not try to solve the task; writes to the rubric's letter — a "Risk analysis" heading with one
  filler sentence, a chart that exists but shows nothing, text addressed to the judge;
- a success means a criterion is loose — it used knowledge the policy doesn't have.

*Afterwards*
- each attacker deliverable is frozen and scored by the rubric judge and by an independent goal judge;
- rubric pass + goal fail → task exploitable; removed until repaired and re-validated;
- hacks found later during RL are added to the taxonomy so every future task is tested for them.

**5. Solve inspection.** A judge examines rollouts that *passed* (a failed rollout can't be a
successful hack). This runs at three times:
- *Curation*: a strong model attempts each task k times; successes go to a judge for hacks and for
  environment problems (missing tools) that interfered with the intended path.
- *Online*: a fixed judge model, served separately and never updated with the policy, reviews every
  pass during training and can veto it. It sees the final deliverable (and, for rubric tasks, the
  trajectory) and looks for unsupported provenance, criterion gaming, and text aimed at the evaluator.
  A vetoed rollout gets correctness 0 but keeps its length cost — the same reward as a failed attempt
  of equal length. Chain-of-thought is monitored for hacking intent but not optimized against.
- *Offline, during the run*: on groups already scored and trained on, reading the logs rather than
  changing rewards. Reward-integrity judges on passes (did it exploit a test gap?), task-health judges
  on all-fail groups (hard, underspecified, broken environment, or grader failure?), and a scan for
  near-identical patches that received different grades. Output is task decisions — remove, repair,
  fix the grader — not reward changes.

**6. RL derisking.** A short training run on the candidate tasks with the exact production reward
setup — rubric judge, threshold, integrity judge, length penalty. Check 5 looks at individual
rollouts; this looks at what the *policy converges to* under optimization pressure: exploits that
random sampling never hits, reward noise that destabilizes groups, tasks that saturate in a few steps.
Findings feed back into checks 1–4; the task re-enters only after re-validation.

**7. Synthetic authoring gate.** When a model writes the task, the task is judged before it is used:
is it professionally meaningful, hard enough, does the rubric cover it, is the ground truth protected
from the solver. Because author and judge share a model family, accepted tasks also pass a majority
vote from a different model family. The authoring agent gets ten revision rounds, then the task is
abandoned.

**8. Leak adversary and simulated-user audit.** An agent is told to find and use anything leaked to
pass without solving — the only test for leaks nobody thought to scan for. The simulated user's hidden
notes and real rollouts are checked so it can never be talked into disclosing the answer.

---

## Human checks

Every agentic check is a model that can drift or be gamed. Two human processes anchor them.

**1. Judge calibration.** Annotators sample judge decisions to measure whether each automated check
does what it was meant to; check definitions are expanded when a systemic flaw appears (e.g. tasks
with "black-box" binaries the agent could simply reverse-engineer).

**2. Fresh labels for judge-rewarded RL.** Where the reward is itself a model (conversational
quality), humans label the *current* policy's outputs every two weeks — 200–300 prompts, five
annotators, ground truth = mean of the three most agreeing — and both the proxy arena judge and the
generative reward model are recalibrated to them. The relative weight humans put on attributes shifts
as the policy changes, so a judge calibrated once goes stale. Beam's first recipe collapsed when the
policy exploited its reward model; the final recipe's predicted 56.5% human win rate matched the
measured 56.3%.

Other anti-gaming designs in the same section: a ternary reward (correct 1 / abstain 0.5 / wrong 0)
after a five-level confidence reward was gamed through wording; a double-gated instruction-following
reward (Python verifier *and* judge, because verifier-only training taught the model to satisfy the
verifier and skip the request); two graders that penalize padding (follow-ups, self-validation) and
withholding (clarifying instead of answering), weighted by their precision against human labels.

---

---

## Reference

Beam Model Technical Report, Reflection AI, Oct 2026 — https://reflection.ai/documents/beam-model-tech-report
(§4.1 task curation by domain, §4.2.3 alignment-teacher RL, §4.4 and §6.4–6.6 infrastructure).
