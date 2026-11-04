# Interview drills (one day)

Four notebooks. Each has instructions, stub cells marked `# TODO`, and test cells that print `ok`.
Blank page, no AI; docs are fine. Run the tests often.

| Notebook | Time | What you write |
|---|---|---|
| `01_policy_loss.ipynb` | 25 min | GRPO advantages; REINFORCE, PPO, CISPO, GSPO losses with masks and two normalizations |
| `02_rl_loop.ipynb` | 45 + 10 min | reward, sampling and a GRPO + PPO training loop on a toy task; then a hacked reward and a label-free check that catches it |
| `03_attention.ipynb` | 25 min + bonus | causal multi-head attention with a KV cache (incl. chunked prefill); attention backward by hand; online softmax (bonus) |
| `04_grader.ipynb` | 45 min | a grader for "write attention by hand" that survives 12 submissions, some of which cheat |

Order: 1, 2, 3, 4. Out of time? Skip 3 before 4.

What each drill taught: [LEARNINGS.md](LEARNINGS.md). Worked versions: `*_solved.ipynb`.

## Running

torch is in the `pglab` conda env (torch 2.15), which has no Jupyter kernel yet:

```bash
conda activate pglab
pip install ipykernel
python -m ipykernel install --user --name pglab
jupyter lab        # pick the "pglab" kernel
```

Or open the notebooks in VS Code and choose the `pglab` interpreter.

## Don't peek

- `solutions/`: reference solutions, one script per notebook. Every test passes against them
  (`cd interview-drills && python solutions/02_rl_loop.py`).
- `04_submissions/sub_*.py` and `04_submissions/ANSWER_KEY.md`: open only after your grader passes.
