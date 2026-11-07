# Quality checks for RL environments

How a frontier lab keeps RL rewards honest, worked through from the Beam Model Technical Report
(Reflection AI, Oct 2026, https://reflection.ai/documents/beam-model-tech-report; §4.1 task curation
by domain, §4.2.3 alignment-teacher RL, §4.4 and §6.4–6.6 infrastructure). Cross-references to
MAI-Thinking-1 (Microsoft AI, https://microsoft.ai/pdf/mai-thinking-1.pdf) where Beam borrows from it.

**The one property every check serves:** reward = 1 exactly when the task was genuinely done.
Each check removes one way that fails: the task is broken, the grader is wrong, the agent cheats, or
the training loop drifts. Checks are ordered cheapest first; most bad tasks die at A1–A2.

---

## A. Before a task enters the pool

### A1. Environment works
- Image builds; task format valid (Harbor); Dockerfile layered for cache reuse.
- Reference solution run 3 independent times **under the training resource limits** (CPU, memory,
  timeouts of the rollout sandbox). Inconsistent outcomes → drop. Catches tasks that pass on a dev box
  but OOM or time out in training (a correct solution would score 0), and flaky tests.
- Task re-run through the *training* grading path (extract submission → apply in a separate verifier
  env) to confirm it works under production isolation.

### A2. Verifier discriminates (positive and negative controls)
- Oracle (reference) solution → 1.0 on 5/5 runs.
- No-op (environment unchanged / no actions taken) → 0.0 on 5/5 runs.
- SWE: tests fail before the reference patch and pass after (fail-to-pass); previously passing tests
  still pass (pass-to-pass).
- Competitive coding: minimum number of test cases; output-comparison script matches the prompt
  (case sensitivity, whitespace, numeric precision); reference passes in *their* sandbox; no prompts
  that promise partial scoring.
- Math: gold must resolve to a SymPy object, else grading silently degrades to string match → drop.
  LaTeX vs numeric answers routed to the right resolver (false positives); normalizations added for
  ASCII `sqrt(13)`, European interval markers, stylistic commands (false negatives).

### A3. Solvable from the solver's view
A judge reads the task statement and flags:
- references to things the agent cannot see (figure, attachment, earlier sub-question, discussion);
- self-contradiction, missing assumptions, two valid readings;
- format demands that contradict the training format; spurious characters; non-English.

SWE stricter pass, with the **reference solution withheld** from the judge (a judge that sees the
answer rationalizes the tests from it), 3 independent calls per criterion, any negative rejects:
- every tested requirement is determinable from the prompt plus visible repository;
- the tests enforce the requested behavior;
- trivial or incomplete implementations cannot pass.

Repairable statements are rewritten and re-assessed; unresolved defects → excluded.

### A4. Specification ↔ verifier alignment (under- and over-specification)
Two axes that are easy to conflate.

**Axis 1 — what the task *says* vs what the verifier *checks*.** This is the alignment check. The
task text can be perfectly detailed and still fail either way:
- *Verifier checks more than the task says* (Beam: "underspecified", "unstated constraints",
  "assumes a particular implementation"). There are several valid ways to do the job and the verifier
  silently accepts one. A correct solution scores 0 → reward noise, and the agent learns the grader's
  taste instead of the task.
- *Verifier checks less than the task says* (Beam: "too weak", "incomplete coverage"). A partial
  solution scores 1 → the policy learns to stop early.

**Axis 2 — how much the task *tells* the agent** (Beam's "specificity knob", set during synthetic
generation for diversity, not a filter in itself):
- *Low specificity*: requirements left to be inferred from the environment or common sense. Harder,
  and allowed — as long as the omitted requirements are genuinely inferable. If they are not, it
  collapses into axis 1 case 1 and is caught by A3 (determinable from the solver's view).
- *High specificity*: every requirement and edge case spelled out. Easier; at the extreme the prompt
  leaks the solution or reduces the task to step-following. Caught by the "statement does not leak the
  solution" static judge below and the answer-in-the-question screen in A10.

So: alignment = prompt and verifier must demand the same set of things; specificity = how much of
that set the prompt states outright vs leaves to inference. "Too specific makes it easy" is axis 2,
not an alignment failure.

How axis 1 is checked — static judges on SWE artifacts:
- the statement describes an actionable change without leaking the solution;
- tests agree with the stated requirements, assert meaningful behavior, and **permit equivalent
  correct implementations**;
- the reference fix is relevant, has no hard-coded answers, and does not modify test files;
- the verifier command does not manufacture success, suppress failures, or substitute superficial
  checks for behavioral tests.

Agentic audits (inspect the repo, trace the verifier, execute code):
- **Ambiguity judge** — catches *verifier checks more than the task says*. Looks for requirements
  enforced by tests but not justified by the visible spec. Where practical, writes an alternative
  implementation consistent with the spec and shows the verifier rejecting it.
- **Verifier-strength judge** — catches *verifier checks less than the task says*. Records the
  independently testable requirements before seeing hidden tests, maps each to verification logic,
  then writes incomplete or incorrect implementations to find gaps. Bypasses must be discoverable from
  solver-visible information.

Terminal verifier principles: alignment (no unstated constraints, no assumed implementation or file
format), comprehensive coverage (every requirement tested, else partial solutions are rewarded),
robustness (test inputs and expected outputs unmodifiable by the agent).

General tool use: drop tasks where the verifier demands things the instruction never stated, the
reward is unreasonable given the request, tools are insufficient, or criteria are vague or unscorable.

### A5. Gold answer is right
- A model reviews the gold; a model re-derives it with tools; several independent solvers must agree.
- Questions rewritten to pin units, precision, and constants (9.8 vs 9.81) so correct reasoning
  always lands on the gold.
- **Strong-vs-weak gap**: if a weaker model scores higher than a stronger one, the gold is probably
  wrong. Also used as a difficulty signal.
- Web search: sample many rollouts and extract answers that recur; drop the question if a model
  prefers the alternative, or judges the question contradictory (neither answer fits) or
  underspecified (both fit).
- Science Q&A: rewrite until the reference is the *unique* correct answer; solve several times with the
  training judge; a final diagnosis step separates wrong reference / ambiguous question / judge noise /
  genuine difficulty.

### A6. Not guessable, not leaked, not memorized
A task whose answer can be produced without doing the work teaches the policy to guess, which stops
chain-of-thought from scaling. Beam treats this as a reward hack to avoid completely.

Guessability — how it is checked:
- *Judge over the question text*: yes/no, multiple choice, wording that hands out a shortlist,
  asked quantities with few possible values (rank of a matrix, finite probability masses),
  definition-recall questions, answers visible in the prompt.
- *Pool statistics*: drop questions whose gold answer is unusually frequent across the pool (0, 1, π).
- *Empirical*: tool-use tasks must score 0 on the untouched environment; generated math problems must
  be self-solved by the generator and revised if a one- or two-line shortcut works.

Leakage — how it is checked:
- *Static scans*: repo images scrubbed of post-fix commit history; derived artifacts rebuilt from the
  modified repo; images grepped for surviving copies of removed code or tests; prompts inspected for
  implementation details.
- *Verifier guards* reject unauthorized changes to test-selection config and import hooks.
- *Web*: lexical and judge checks of prompt vs answer; URL blocklists for known answer dumps
  (datasets, papers, blogs) plus a dynamic blocklist that fuzzy-matches question+answer against pages
  the model actually fetched during rollouts.
- *Adversarial agent* explicitly tries to recover leaked artifacts and pass with them.
- *Simulated-user audit*: the persona's hidden notes and real rollouts are checked so it can never
  disclose the expected answer.
- *Terminal rule-based check*: no privileged information present in the agent sandbox.

Memorization — how it is checked:
- Run the model on the search question **without tools**; a correct answer → drop.

### A7. Adversarial attack on the task and grader
- **Black-box attacker**: sees only what the solver sees (task + environment). Given a written
  taxonomy of reward hacks and told to obtain reward 1.0 with an insufficient or incorrect solution.
  Tests the task itself: ambiguity, missing information, unintended shortcuts.
- **White-box attacker**: additionally sees the rubric and grading spec. Tests the grader: criteria
  that are coarse, redundant, or keyword-satisfiable.
- **Rubric judge vs goal judge** on the attacker's frozen deliverable: the rubric judge checks the
  stated criteria; the goal judge independently asks whether the professional objective was achieved.
  Disagreement flags an underspecified or exploitable task.
- Rubric static checks first: coverage, atomicity, observability, non-duplication, appropriate
  weighting; rubrics may include penalty criteria for undesirable outputs.
- Hacked tasks are **removed**, not patched.

### A8. Rollout validation (empirical solvability) + pass-check on solves
Sample k attempts per task from a strong model; no training.
- Zero successes in k → drop (or inspect: repeated failure is a practical filter, not proof the task
  is broken).
- **Successful rollouts go to a judge** for reward hacking and for environment issues (missing tools)
  that interfered with the intended path. Only passes are inspected; a failed rollout cannot be a
  successful hack.
- The pass-rate band gives difficulty for *this* model.

### A9. RL derisking
A short RL run on the candidate task set with the **exact production reward setup** (rubric judge,
threshold, integrity judge, length penalty). The only check that applies optimization pressure. Catches:
- exploits the policy converges to that random sampling never hits;
- reward noise that destabilizes groups;
- tasks that saturate within a few steps.

Failures feed back into A3–A7: tasks revised or removed, rubric criteria and grader checks updated,
task re-evaluated before re-entry.

### A10. Difficulty and dedup
- Drop what the initial checkpoint solves frequently at a low thinking budget; drop what a reference
  model solves reliably.
- Web: keep if strong − weak solve rate ≥ 12.5 points, or weak = 0.
- "Wrong-reason" screen: tasks that look easy because the answer leaked, or hard because of missing
  context, chat-template formatting errors, impossible questions (which induce hacking), inaccessible
  sites, over-strict memory/time limits.
- Model-specific: solvable without chain-of-thought → low quality for this model.
- Dedup: character-7-gram MinHash-LSH with exact Jaccard verification; 0.95 within training data,
  0.80 against evaluation sets. SWE additionally excludes task IDs and *patch matches* tied to held-out
  evals; competitive coding matches canonical problem IDs; terminal uses taxonomy tags plus embedding
  similarity to drop templated near-duplicates.

### A11. Synthetic authoring gates
- Terminal / knowledge-work authoring agent builds the task in a sandbox and gets 10 rounds to pass
  the gate (static checks + judged criteria), then the task is abandoned.
- Knowledge work: 8 deterministic validators + 9 judged criteria (professional meaningfulness,
  difficulty, rubric coverage, ground-truth protection). Because author and in-loop judge share a model
  family, accepted tasks also pass an independent majority vote from a **different model family**
  (3 votes over 5 defect classes) — retains ~42%.
- Expert math / physics: adversarial generate–solve–refine loop; independent solvers expose
  ambiguities, errors, and shortcuts; physics checked for consistency with the source paper and for
  sufficiency of the given information.

---

## B. Per rollout, during training

### B1. Isolation: untrusted vs trusted
- The agent runs in an **untrusted** sandbox: anything it touched is presumed compromised (binaries,
  test files, env vars, Python path).
- Grading runs in a fresh **trusted** container built from a digest-pinned image; only the deliverable
  (a diff, or output files) crosses over as inert data.
- Beam abandoned the legacy Harbor format (agent and verifier in one runtime) because it was hackable.
- Network: a per-sandbox enforcer re-originates only policy-allowed traffic; package-registry mirrors;
  per-sandbox request logs; TLS interception via an installed CA. Supporting containers are mounted
  read-only; no registry credentials exist inside the sandbox.

### B2. Answer hardening before any judge
- Regex extraction of the final answer ("The answer is: …", `\boxed{}`); unextractable → 0.
- Length caps: search `max(150, 4.5 · len(gold))` characters; math 200 characters (or 3× gold).
  Kills hedging ("3 or 4 or 5") mechanically before a judge can be fooled by it.
- Judge prompts carry few-shot examples on handling ambiguity and rejecting hedges.
- Competitive coding: binary reward, isolated grading sandbox, time limits; compile errors, runtime
  failures, wrong output, timeouts all score 0.
- Tool use: reward 1 only if every criterion passes; meta-tool rejects unknown tool names; verifiers
  may use an expected environment state-hash.

### B3. Online judges on passes
- **Independent judge service**: a fixed model, served separately, never updated with the policy.
  Reviews rollouts that passed automated checks against versioned criteria and can veto.
  **Fail-closed**: if the judge call errors after retries, the rollout fails — an unreviewed pass never
  becomes a positive reward.
- SWE: the judge sees the final patch and the verifier; confirmed hacks get zero reward.
- Knowledge work **integrity judge** sees artifact *and* trajectory. Flags: unsupported provenance
  (citing sources or computations never performed), surface-level criterion gaming (writing to the
  rubric's keywords), evaluator manipulation (text aimed at the judge).
- Vetoed rollout: correctness 0, **length cost kept** → same reward as a failed attempt of equal length.
- Chain-of-thought is monitored for hacking intent but never optimized against.
- Rollouts that exhaust the effort budget get no correctness reward.

### B4. Deadlines and staleness
- Task-dependent rollout deadlines; rollouts exceeding them are discarded; a group is kept only when
  enough rollouts completed.
- Policy version recorded per token; staleness (avg ~5 versions behind) is monitored, not hard-bounded.
  (MAI-Thinking-1 by contrast discards rollouts more than 8 inference updates stale.)

---

## C. Per prompt group

- Groups with saturated rewards (all identical) are discarded by the trainer — no gradient.
- **Task-health judges** on all-fail groups: genuine difficulty, underspecified prompt, broken
  environment, or grading failure?
- **Reward-integrity judges** on passes: exploited a test gap? Do hidden tests assume unstated things?
- Near-identical patches receiving different grades → grader bug, not agent behavior.

---

## D. Over the run

- Rubrics updated as new hacks appear; tasks on which hacks or breakage are observed are pulled from
  the mixture until repaired and re-validated; harder task sets introduced as pass rates rise.
- Offline audits of artifacts and trajectories feed findings back into task, rubric, and online judge.
- Training–inference integrity: nucleus masks and routed experts replayed in the trainer; FP32 LM head
  and router; router weights and load-balancing biases frozen during RL. Per-token KL between
  sampling and training policies monitored (median 2.2e-4 nats; 8.7e-5 right after a restart is the
  irreducible floor).
- Stability telemetry: gradient norm, response length per mixture phase, sampled-token NLL split by
  reasoning vs answer tokens.

---

## E. Alignment-teacher RL (judge-rewarded environments)

Here the "verifier" is itself a model and can be Goodharted; the defenses are different.

Knowledge calibration:
- Question filters: time-sensitive, ambiguous, unanswerable from source, false-premise → drop.
- 16 samples per question scored with the RL judge; drop if all receive the same reward; bin the rest
  by mean reward into five bands, sample equally, require abstentions in the low bands.
- Ternary reward vs search-verified gold: correct 1.0 / abstain 0.5 / wrong 0.0.
- A five-way reward (confident correct > hedged correct > abstain > …) was **abandoned**: the policy
  learned to game the judge's inference of confidence from wording. Judge prompt tightened to penalize
  multiple and empty answers and to score hedged commitments as abstentions.

Instruction following:
- Filter malformed instructions, unreliable verifiers, incompatible request–instruction pairs, tasks
  trivial for the starting policy.
- **Double-gated** reward: a response must pass the Python verifier **and** a judge that it genuinely
  addressed the request — "verifier-only training makes the model circumvent the underlying task."
- Rubric tasks (~22 rubrics each): all-or-nothing beat averaging partial credit.
- Multi-turn conversations checked for satisfiability and consistency.

Faithfulness and long context:
- Claim-level faithfulness judge; because judges degrade as input context grows, SQL-based rule
  rewards are used instead where possible. The SQL pipeline rejects unsafe SQL, idle stages, unstable
  cut-offs, and leaked answers, and runs a **reverse check**: a second model rebuilds the SQL from the
  question alone; both must agree on real and perturbed data.

Helpfulness (RLAIF):
- **Fresh human labels every two weeks** on the *current* policy's outputs (200–300 prompts, five
  annotators, ground truth = mean of the three most agreeing). Framed as the main defense against
  Goodharting: only fresh labels on the trained policy's outputs detect judge–human divergence.
- Proxy arena judge and generative reward model recalibrated to those labels, because attribute
  weights drift as the policy changes; the proxy clears the human inter-rater agreement ceiling.
- Anchor responses filtered by claim-level factuality and temporal checks (2.6% of prompts dropped).
- Position bias cancelled by fixing comparison order per prompt within a GRPO group; pairwise verdicts
  over pointwise scores (which compressed the reward range and saturated).
- **Two anti-gaming graders** multiplied onto the preference reward: unnecessary content (follow-up
  questions, self-validation) and withheld content (clarifying instead of answering, unwarranted
  refusal). Each flag's weight set from its measured precision against human labels.
- Linear length penalty on the final answer to counter the judge's preference for long responses.
- Evidence: recipe v1 "collapsed as the policy exploited its reward model"; the final recipe's
  arena-predicted 56.5% win rate vs GLM 5.2 matched the 56.3% measured with human raters.

---

## F. Infrastructure guarantees underneath

- Serving instances pull and verify transferred weights before adopting a version; exhausted
  verification retries fail the instance rather than serve a corrupt policy.
- Rollouts interrupted by faults are preserved and retried, not scored as zeros.
- The judge service scales independently and never participates in policy updates.
- Vendor task validation is routed to clusters separate from training and evaluation workloads.

---

## Recurring moves

1. **Controls on every grader** — the known-good solution must pass, doing nothing must fail.
2. **Judge from the solver's seat** — hide the reference when judging whether a task is fair.
3. **Gate with two independent signals** when one is gameable — verifier + judge; rubric judge +
   goal judge; proxy judge + fresh human labels.
4. **Integrity is monitored state, not a one-time gate** — online veto, group triage, derisking runs,
   human recalibration keep catching what a stronger policy finds later.

## Cheapest high-leverage subset for a small harness

- A2 controls (oracle 5/5, no-op 5/5) run under the cheapest node's resource caps, three trials.
- A3/A4 judge with the reference withheld; A6 empirical no-op check on the untouched environment.
- A7 one black-box adversarial pass per new task with a written hack taxonomy; drop, don't patch.
- A10 strong-vs-weak gap as a combined difficulty and wrong-gold filter; MinHash dedup first.
- C group-level consistency checks on rollout logs already paid for.
- B3 online judge on passes as the backstop (one model call per *pass*, prompt = deliverable + rubric).
