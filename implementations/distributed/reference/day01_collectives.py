"""Day 1: what every collective does, and point-to-point sends.

Each function below runs on every rank at once (SPMD). Use one dist.* call per function (two for ring_pass).
For each collective, first write down in `predict()` the output shape and how many bytes this rank sends, then
implement it. Tests run on 4 ranks with float64 tensors.

  all_reduce          every rank ends with the elementwise sum of all ranks' tensors     (n,)   -> (n,)
  all_gather          every rank ends with every rank's tensor, stacked                    (n,)   -> (N, n)
  reduce_scatter      sum across ranks, then rank r keeps the r-th chunk of the sum        (N*c,) -> (c,)
  broadcast           every rank ends with the source rank's tensor                         (n,)   -> (n,)
  all_to_all          rank r sends its j-th chunk to rank j; a transpose across ranks       (N*c,) -> (N*c,)
  send / recv         rank r sends to rank (r + 1) % N and receives from (r - 1) % N

Point-to-point: a blocking dist.send waits until the peer posts the matching recv. If every rank calls send first,
nobody reaches recv and the ring deadlocks. Post the send asynchronously (dist.isend), then recv, then wait.

Gotcha: all_gather_into_tensor and reduce_scatter_tensor take flat buffers on gloo, (N*n,) not (N, n); reshape
afterwards.

Run: python day01_collectives.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # reference only

import torch
import torch.distributed as dist

from _harness import launch


def all_reduce_mean(t):
    """t (n,). Return the mean over ranks, (n,)."""
    # >>> TODO
    out = t.clone()
    dist.all_reduce(out)
    return out / dist.get_world_size()
    # <<< TODO


def gather_all(t):
    """t (n,). Return every rank's tensor stacked in rank order, (N, n)."""
    # >>> TODO
    N = dist.get_world_size()
    out = torch.empty(N * t.numel(), dtype=t.dtype)                     # (N*n,) flat: gloo wants one flat buffer
    dist.all_gather_into_tensor(out, t)
    return out.view(N, -1)                                              # (N, n)
    # <<< TODO


def shard_of_sum(t):
    """t (N*c,). Sum t across ranks and return this rank's chunk of the sum, (c,)."""
    # >>> TODO
    N = dist.get_world_size()
    out = torch.empty(t.numel() // N, dtype=t.dtype)
    dist.reduce_scatter_tensor(out, t)
    return out
    # <<< TODO


def from_rank(t, src):
    """t (n,). Return the tensor held by rank `src`, on every rank, (n,)."""
    # >>> TODO
    out = t.clone()
    dist.broadcast(out, src)
    return out
    # <<< TODO


def transpose(t):
    """t (N*c,) seen as N chunks of c. Rank r returns [chunk r of rank 0, chunk r of rank 1, ...], (N*c,)."""
    # >>> TODO
    out = torch.empty_like(t)
    dist.all_to_all_single(out, t)
    return out
    # <<< TODO


def ring_pass(t):
    """t (n,). Send t to rank (r + 1) % N, return the tensor received from rank (r - 1) % N. Must not deadlock."""
    # >>> TODO
    r, N = dist.get_rank(), dist.get_world_size()
    out = torch.empty_like(t)
    req = dist.isend(t, (r + 1) % N)
    dist.recv(out, (r - 1) % N)
    req.wait()
    return out
    # <<< TODO


def predict(N, n, c, bytes_per_elem):
    """Write these BEFORE implementing. Bytes = the logical tensor this rank hands to the call (what the harness logs).

    Returns {name: (output_shape, bytes_logged)} for the inputs used in the tests:
      all_reduce_mean, gather_all, from_rank: input (n,);  shard_of_sum, transpose: input (N*c,);  ring_pass: (n,).
    """
    # >>> TODO
    return {
        "all_reduce_mean": ((n,), n * bytes_per_elem),
        "gather_all": ((N, n), n * bytes_per_elem),
        "shard_of_sum": ((c,), N * c * bytes_per_elem),
        "from_rank": ((n,), n * bytes_per_elem),
        "transpose": ((N * c,), N * c * bytes_per_elem),
        "ring_pass": ((n,), n * bytes_per_elem),
    }
    # <<< TODO


# ----------------------------------------------------------------------------- tests (don't edit)
N_RANKS, n, c = 4, 5, 3
OPS = {"all_reduce_mean": "all_reduce", "gather_all": "all_gather_into_tensor",
       "shard_of_sum": "reduce_scatter_tensor", "from_rank": "broadcast",
       "transpose": "all_to_all_single", "ring_pass": "isend"}


def _data(rank, size):
    return torch.arange(size, dtype=torch.float64) + 100 * rank           # rank r holds [100r, 100r + 1, ...]


def _run(rank, world_size, name):
    fn = globals()[name]
    if name in ("shard_of_sum", "transpose"):
        return fn(_data(rank, world_size * c))
    if name == "from_rank":
        return fn(_data(rank, n), 2)
    return fn(_data(rank, n))


def _expected(name, rank, N):
    xs = [_data(r, n) for r in range(N)]
    if name == "all_reduce_mean":
        return sum(xs) / N
    if name == "gather_all":
        return torch.stack(xs)
    if name == "shard_of_sum":
        return sum(_data(r, N * c) for r in range(N))[rank * c:(rank + 1) * c]
    if name == "from_rank":
        return xs[2]
    if name == "transpose":
        return torch.cat([_data(r, N * c)[rank * c:(rank + 1) * c] for r in range(N)])
    if name == "ring_pass":
        return xs[(rank - 1) % N]


if __name__ == "__main__":
    pred = predict(N_RANKS, n, c, 8)
    for name, op in OPS.items():
        out = launch(_run, N_RANKS, name)
        for r in range(N_RANKS):
            got = out[r]["result"]
            assert torch.allclose(got, _expected(name, r, N_RANKS)), f"{name}: rank {r} wrong result\n{got}"
            shape, nbytes = pred[name]
            assert tuple(got.shape) == tuple(shape), f"{name}: predicted shape {shape}, got {tuple(got.shape)}"
            logged = out[r]["bytes"].get(op, 0)
            assert logged == nbytes, f"{name}: predicted {nbytes} bytes, harness logged {logged} for dist.{op}"
            calls = sum(out[r]["calls"].values())
            limit = 3 if name == "ring_pass" else 1                           # isend + recv (+ nothing else)
            assert calls <= limit, f"{name}: {calls} collective calls, expected at most {limit}"
        print(f"{name:16s} ok   output {tuple(got.shape)}, {nbytes} bytes per rank")
    print("ok")
