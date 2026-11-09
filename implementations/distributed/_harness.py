"""Run a function on N CPU processes (gloo) and log every collective call.

launch(fn, world_size, *args, banned=()) runs fn(rank, world_size, *args) on each rank and returns, per rank,
a dict {"result": ..., "calls": {op: count}, "bytes": {op: bytes sent}, "async": {op: count}}.

Call collectives as dist.<op>(...) (not `from torch.distributed import ...`) so the log sees them.
Bytes are what this rank puts on the wire for point-to-point ops, and the logical tensor size for collectives.
"""
import functools
import io
import os
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LOG = {"calls": {}, "bytes": {}, "async": {}}

_nb = lambda t: t.numel() * t.element_size()
_SIZE = {
    "send": lambda tensor, *a, **k: _nb(tensor),
    "isend": lambda tensor, *a, **k: _nb(tensor),
    "recv": lambda *a, **k: 0,
    "irecv": lambda *a, **k: 0,
    "all_reduce": lambda tensor, *a, **k: _nb(tensor),
    "broadcast": lambda tensor, *a, **k: _nb(tensor),
    "reduce": lambda tensor, *a, **k: _nb(tensor),
    "all_gather": lambda tensor_list, tensor, *a, **k: _nb(tensor),
    "all_gather_into_tensor": lambda out, inp, *a, **k: _nb(inp),
    "reduce_scatter": lambda out, inp_list, *a, **k: sum(_nb(t) for t in inp_list),
    "reduce_scatter_tensor": lambda out, inp, *a, **k: _nb(inp),
    "gather": lambda tensor, *a, **k: _nb(tensor),
    "scatter": lambda tensor, *a, **k: _nb(tensor),
    "all_to_all": lambda out_list, in_list, *a, **k: sum(_nb(t) for t in in_list),
    "all_to_all_single": lambda out, inp, *a, **k: _nb(inp),
}


def _install(banned):
    for name, size in _SIZE.items():
        fn = getattr(dist, name)

        @functools.wraps(fn)
        def logged(*a, _fn=fn, _name=name, _size=size, **k):
            if _name in banned:
                raise RuntimeError(f"dist.{_name} is not allowed in this drill")
            LOG["calls"][_name] = LOG["calls"].get(_name, 0) + 1
            LOG["bytes"][_name] = LOG["bytes"].get(_name, 0) + _size(*a, **k)
            if k.get("async_op"):
                LOG["async"][_name] = LOG["async"].get(_name, 0) + 1
            return _fn(*a, **k)

        setattr(dist, name, logged)


def reset_log():
    for d in LOG.values():
        d.clear()


def _worker(rank, world_size, init_file, fn, args, banned, queue):
    try:
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
        _install(set(banned))
        torch.set_default_dtype(torch.float64)
        result = fn(rank, world_size, *args)
        buf = io.BytesIO()                                   # plain bytes, not shared-memory tensors
        torch.save({"result": result, "calls": dict(LOG["calls"]), "bytes": dict(LOG["bytes"]),
                    "async": dict(LOG["async"])}, buf)
        queue.put((rank, None, buf.getvalue()))
    except Exception:
        queue.put((rank, traceback.format_exc(), None))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(fn, world_size, *args, banned=(), timeout=60):
    init_file = os.path.join(tempfile.mkdtemp(), "init")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world_size, init_file, fn, args, banned, queue))
             for r in range(world_size)]
    for p in procs:
        p.start()
    out = {}
    try:
        for _ in range(world_size):
            rank, err, res = queue.get(timeout=timeout)
            if err:
                raise RuntimeError(f"rank {rank} failed:\n{err}")
            out[rank] = torch.load(io.BytesIO(res), weights_only=False)
    except Exception as e:
        if type(e).__name__ == "Empty":
            raise RuntimeError(f"timed out after {timeout}s: a hang usually means ranks called collectives "
                               f"in different orders, or a send has no matching recv") from None
        raise
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    return [out[r] for r in range(world_size)]
