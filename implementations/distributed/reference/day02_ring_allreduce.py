"""Day 2: ring all-reduce from point-to-point sends.

Implement ring_all_reduce(t): sum `t` across all ranks, in place, using only dist.isend / dist.irecv /
dist.send / dist.recv. Every collective (all_reduce, broadcast, reduce, all_gather, ...) is banned.

The ring: rank r sends to (r + 1) % N and receives from (r - 1) % N. Split t into N chunks.
  Phase 1, reduce-scatter (N - 1 steps): pass chunks around the ring, adding as you go. Afterwards each rank holds
           the full sum of one chunk.
  Phase 2, all-gather (N - 1 steps): pass the finished chunks around so every rank ends with all of them.

Yardsticks
  correctness: every rank ends with the sum, for a size not divisible by N.
  efficiency:  each rank sends at most 2 (N - 1) chunks, i.e. about 2 (N - 1) / N of the tensor, independent of N.
               Sending everything to one rank and back costs that rank (N - 1) full tensors.

Run: python day02_ring_allreduce.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # reference only

import math

import torch
import torch.distributed as dist

from _harness import launch


def ring_all_reduce(t):
    """t: 1-D tensor (n,). Sum across ranks in place and return t."""
    # >>> TODO
    rank, N = dist.get_rank(), dist.get_world_size()
    chunks = list(t.tensor_split(N))                        # N views into t
    right, left = (rank + 1) % N, (rank - 1) % N
    for step in range(N - 1):                               # reduce-scatter
        send_i, recv_i = (rank - step) % N, (rank - step - 1) % N
        buf = torch.empty_like(chunks[recv_i])
        req = dist.isend(chunks[send_i], right)
        dist.recv(buf, left)
        req.wait()
        chunks[recv_i] += buf
    for step in range(N - 1):                               # all-gather
        send_i, recv_i = (rank - step + 1) % N, (rank - step) % N
        buf = torch.empty_like(chunks[recv_i])
        req = dist.isend(chunks[send_i], right)
        dist.recv(buf, left)
        req.wait()
        chunks[recv_i].copy_(buf)
    return t
    # <<< TODO


# ----------------------------------------------------------------------------- tests (don't edit)
N_ELEMS = 1003


def _run(rank, world_size):
    g = torch.Generator().manual_seed(rank)
    t = torch.randn(N_ELEMS, generator=g)
    return ring_all_reduce(t)


BANNED = ("all_reduce", "broadcast", "reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter",
          "reduce_scatter_tensor", "gather", "scatter", "all_to_all")

if __name__ == "__main__":
    for N in (2, 4):
        out = launch(_run, N, banned=BANNED)
        expected = sum(torch.randn(N_ELEMS, generator=torch.Generator().manual_seed(r), dtype=torch.float64)
                       for r in range(N))
        for r in range(N):
            assert torch.allclose(out[r]["result"], expected), f"N={N} rank {r}: wrong sum"
        limit = 2 * (N - 1) * math.ceil(N_ELEMS / N) * 8
        sent = [sum(o["bytes"].get(op, 0) for op in ("send", "isend")) for o in out]
        print(f"N={N}: correct; bytes sent per rank {sent}, limit {limit} "
              f"({2 * (N - 1) / N:.2f} x the {N_ELEMS * 8}-byte tensor)")
        assert max(sent) <= limit, f"N={N}: a rank sent {max(sent)} bytes, more than {limit}"
    print("ok")
