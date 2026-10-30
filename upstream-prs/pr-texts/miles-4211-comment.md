# Comment draft for radixark/miles#4211 (not posted)

#4211 is upstream's own native LoRA for Muse Spark with merged rollout weights. It covers our Muse LoRA branch, so we do not raise ours. This comment shares what we saw at full scale. Post only with Deniz's go.

---

We ran the same design on our fork at full scale, and the results may help review.

Setup: Muse Spark Open, LoRA rank 32, alpha 32 on attention and routed experts (the shared `GroupedExpertsSpec` form, as here), 4 nodes × 8 B300, colocated, R3 and TIS on, lr 1e-5. 12 training steps; train reward went from 0.584 to 0.818.

1. **Merging on the adapter's device.** Your `weight.to(update.device).float()` matches our fix. For a pinned 1 GiB bf16 tensor on a B300 it took the merge from 1,475 ms to 21 ms. Our first full merged sync went from 610 s to 173 s.
2. **The embedding hook is required.** On an earlier branch without it, with full recompute and pipeline parallelism, stage 0 (layers 0-14) changed 0 of 120 adapter tensors over 3 steps. With the hook, every adapted tensor in all 62 layers changed on the engines after the first merged sync. `muse-spark-open` has the hook (`lora.py:106-112`). A test that a stage-0 adapter receives a gradient would guard it.
3. **GPU memory.** After each step the last micro-batch's recompute tensors stay alive: 25 GiB per GPU on stage 0. The holes they leave fragmented memory, and step 1 hit an OOM with 49-77 GB of free but unusable splits. `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` on the trainer held for 12 steps, with at most 7.75 GiB of fragments. We have not found what keeps those tensors alive.
4. **Resume.** A native LoRA resume dropped the saved Adam state and stepped the LR schedule twice; fixed in radixark/miles#4523.
5. **CPU-safe import.** On `002ec9cf35`, importing `miles.rollout` pulls in Megatron, so a CPU-only process fails at import. Fixed in radixark/miles#4524, into `muse-spark-open`.
