"""Reference solution for 04_grader.ipynb: every cell, with the TODO cells filled in."""

import ast
import math
import os
import subprocess
import sys
import tempfile

import torch

BANNED = ("scaled_dot_product", "multi_head_attention", "multiheadattention", "flash_attn")

RUNNER = r"""
import importlib.util, sys
import torch, torch.nn.functional as F

def _banned(*a, **k):
    raise RuntimeError("banned")

# patch before the submission is imported, so it cannot grab the real function
F.scaled_dot_product_attention = _banned
torch._C._nn.scaled_dot_product_attention = _banned

sub_path, in_path, out_path = sys.argv[1:4]
cases = torch.load(in_path, weights_only=True)
spec = importlib.util.spec_from_file_location("submission", sub_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
outs = []
for c in cases:
    o = mod.attention(c["q"], c["k"], c["v"], c["mask"])
    outs.append(o.detach().clone() if isinstance(o, torch.Tensor) else None)
torch.save(outs, out_path)
"""


def static_ok(src):
    """Ban attention shortcuts by name anywhere in the syntax tree (names, attributes, imports, strings)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        words = []
        if isinstance(node, ast.Name):
            words.append(node.id)
        elif isinstance(node, ast.Attribute):
            words.append(node.attr)
        elif isinstance(node, ast.alias):
            words += [node.name, node.asname or ""]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            words.append(node.value)
        if any(b in w.lower() for w in words for b in BANNED):
            return False
    return True


def make_cases(seed, n=12):
    """Fresh random cases every call: varied B, H, Tq != Tk, d, half with a mask, float64."""
    g = torch.Generator().manual_seed(seed)
    ri = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=g))
    cases = []
    for i in range(n):
        B, H, Tq, Tk, d = ri(1, 3), ri(1, 4), ri(1, 9), ri(1, 9), ri(2, 16)
        if i == 0:
            B, H, Tq, Tk, d = 3, 2, 5, 7, 4                              # always one multi-batch, non-square case
        q = torch.randn(B, H, Tq, d, generator=g, dtype=torch.float64)
        k = torch.randn(B, H, Tk, d, generator=g, dtype=torch.float64)
        v = torch.randn(B, H, Tk, d, generator=g, dtype=torch.float64)
        mask = None
        if i % 2 == 0:
            mask = torch.rand(B, 1, Tq, Tk, generator=g) < 0.6
            mask[..., 0] |= ~mask.any(-1)                               # every query sees at least one key
        cases.append({"q": q, "k": k, "v": v, "mask": mask})
    return cases


def reference_attention(q, k, v, mask):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, -1) @ v


def grade(path, timeout=10):
    with open(path) as f:
        if not static_ok(f.read()):
            return False
    cases = make_cases(int.from_bytes(os.urandom(4), "little"))
    expected = [reference_attention(c["q"], c["k"], c["v"], c["mask"]) for c in cases]   # computed here, before and apart from the submission
    with tempfile.TemporaryDirectory() as td:
        runner, inp, out = (os.path.join(td, n) for n in ("runner.py", "in.pt", "out.pt"))
        with open(runner, "w") as f:
            f.write(RUNNER)
        torch.save(cases, inp)
        try:
            proc = subprocess.run([sys.executable, "-I", runner, os.path.abspath(path), inp, out],
                                  cwd=td, timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            return False
        if proc.returncode != 0 or not os.path.exists(out):              # exit code 0 alone proves nothing
            return False
        try:
            got = torch.load(out, weights_only=True)
        except Exception:
            return False
    if not isinstance(got, list) or len(got) != len(cases):
        return False
    for o, e in zip(got, expected):                                      # compare here, in the grader process
        if not isinstance(o, torch.Tensor) or o.shape != e.shape or o.dtype != e.dtype:
            return False
        if not torch.isfinite(o).all() or not torch.allclose(o, e, atol=1e-9, rtol=1e-7):
            return False
    return True


import glob
import os
import time

SUB_DIR = "04_grader_inputs"
SHOULD_PASS = {"sub_02.py", "sub_07.py"}                              # every other submission must fail

results = {}
for path in sorted(glob.glob(os.path.join(SUB_DIR, "sub_*.py"))):
    name = os.path.basename(path)
    t0 = time.time()
    results[name] = bool(grade(path, timeout=10))
    dt = time.time() - t0
    want = name in SHOULD_PASS
    flag = "" if results[name] == want else "   <-- WRONG"
    print(f"{name}: {'PASS' if results[name] else 'FAIL'}   (should {'PASS' if want else 'FAIL'}, {dt:.1f}s){flag}")
    assert dt < 30, f"{name} took {dt:.0f}s: the grader must not hang"

wrong = sorted(n for n in results if results[n] != (n in SHOULD_PASS))
assert len(results) == 12, "expected 12 submissions"
assert not wrong, f"wrong verdicts: {wrong}"
print("ok")


# run it again: an honest submission must pass every time, on fresh random cases
for _ in range(3):
    assert grade(os.path.join(SUB_DIR, "sub_07.py")) and grade(os.path.join(SUB_DIR, "sub_02.py"))
print("ok")
