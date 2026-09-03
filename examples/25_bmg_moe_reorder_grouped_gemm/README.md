<!--
Copyright (C) 2026 Intel Corporation, All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# 25 - BMG MoE reorder + grouped GEMM

Standalone BMG/SYCL port of the **Inkling routed-expert MoE dispatch pipeline**. It mirrors,
one-for-one, the ops the model runs in sglang:

- `python/sglang/srt/models/inkling_common/moe.py` - `run_moe_preprocess`, `moe_tp_forward`
- `python/sglang/kernels/ops/moe/inkling_moe.py` - every kernel reproduced below

Every stage is timed individually via SYCL profiling events and every stage has a CPU reference.

## Stages

| Stage | sglang kernel | This example |
| --- | --- | --- |
| 1 | `fused_moe_preprocess` / `_fused_moe_preprocess_kernel` | `launch_fused_preprocess` - one work-group does the packed-key sort, `src2dst`, expert offsets, counts, block offsets and block schedule |
| 2 | `torch.sort(int16, stable=True)` + `get_src2dst` | `launch_sort_histogram` / `_block_scan` / `_expert_base` / `_scatter` + `launch_get_src2dst` |
| 3 | `compute_grouped_gemm_metadata` | `launch_expert_offsets`, `launch_expert_counts`, `launch_memset_block_metadata`, `launch_block_metadata` |
| 4 | `pre_reorder` / `_pre_reorder_kernel` | `launch_pre_reorder` |
| 5a | `grouped_gemm_triton` (w13) | `launch_grouped_gemm` |
| 5b | `silu_and_mul` | `launch_silu_and_mul` |
| 5c | `grouped_gemm_triton` (w2) | `launch_grouped_gemm` |
| 6 | `post_reorder` / `_post_reorder_kernel` | `launch_post_reorder` |

No stage is dropped.

## Performance status

**Performance-tuned (the genuinely Inkling-specific machinery):** stages 1, 2, 3, 4, 5b and 6.
The preprocess is a single launch, the sort is an atomic-free stable counting sort, and the
reorder kernels use `sycl::vec<uint32_t,4>` copies when `hidden % 8 == 0`.

**Correctness-only:** the grouped GEMM itself (5a / 5c). It reproduces the triton kernel's
schedule decode, padding-block early exit, tail masking and sequential fp32 accumulation order
exactly, but the inner loop is a plain SLM-staged scalar kernel - it is **not** DPAS-tuned. Its
reported `TOPS` is a floor, not a target, and it dominates the wall clock at prefill row counts.
A follow-up should swap the inner GEMM for a CUTLASS collective while keeping the surrounding
schedule contract, which is what this example pins down.

Measured on an Arc Pro B60 (`--suite=perf --iterations=10`, `E=256`, `top_k=6`, `H=1536`,
`I_p=384`), which makes the split concrete:

| stage | `T = 1` (decode) | `T = 16384` (prefill) |
| --- | --- | --- |
| preprocess (all of stages 1-3) | 0.014 ms | 0.186 ms |
| `4_pre_reorder` | 0.0017 ms | 0.883 ms @ **399 GB/s** |
| `5a_grouped_gemm1` | 0.436 ms @ 0.03 TOPS | 265.7 ms @ 0.87 TOPS |
| `5b_silu_and_mul` | 0.0012 ms | 0.933 ms @ 243 GB/s |
| `5c_grouped_gemm2` | 0.220 ms @ 0.03 TOPS | 135.2 ms @ 0.86 TOPS |
| `6_post_reorder` | 0.0079 ms | 1.535 ms @ 230 GB/s |
| pipeline | 0.681 ms | 404.4 ms |

`pre_reorder` sits at the part's real ~400 GB/s DRAM ceiling, so the reorder side is done. The two
GEMMs are 99% of the prefill wall clock at under 1 TOPS, which is the whole of the remaining
headroom. The decode GEMM number is additionally limited by `BLOCK_M = 16` against ~3.4 rows per
expert - a 4.7x row-padding waste that the triton kernel has too, and that the schedule contract
requires.

The card is shared with other jobs, so expect a few percent of run-to-run noise.

## Contract details established from the sglang source

These were read out of the source rather than assumed:

- **`pre_reorder` folds nothing.** It is a pure gather/replicate - the triton kernel casts to
  fp32 and stores back into the input dtype, a bitwise copy for bf16. There is no per-token
  input scale and no top-k weight. The top-k weight is applied at the very end, in
  `post_reorder`. (`silu_and_mul` *can* fold `topk_weights`, but Inkling's `moe_tp_forward`
  calls `activation()` without them.)
- **Sort key** = the expert id cast to `int16`, sorted stably; the values are the flat slot
  positions `t * top_k + k`. The fused path gets stability by packing
  `(expert_id << 12) | position` into one key, which caps that path at `n <= 4096`; the model
  gates it far below, at `n <= 1024`.
- **`src2dst` is the inverse permutation**, indexed by *source* slot:
  `src2dst[reorder_ids[dst]] = dst`.
- **Index dtypes**: `topk_ids` `int32` -> `int16` for the sort; `reorder_ids` `int64`;
  `src2dst` / `num_tokens_per_expert` / `expert_token_offs` / `expert_block_offs` /
  `expert_block_schedule` all `int32`. The fused path keeps `reorder_topk_ids` in `int32` while
  the sort path leaves it `int16`; both dtypes are exercised.
- **Weight layout** is `b[E, N, K]` and the GEMM is `A x B^T` per expert:
  `c[m, n] = sum_k a[m, k] * b[expert, n, k]`.
- **Schedule packing**: `expert_block_schedule[s] = (block_id << 16) | expert_id`, with `-1` in
  the padding slots, and `max_num_blocks = cdiv(n, BLOCK_M) + E - 1`.
- **Block-config selection rule** (`select_grouped_gemm_block_m`, with `n = T * top_k`):

  | condition | BLOCK_M | BLOCK_N | BLOCK_K |
  | --- | --- | --- | --- |
  | `n <= 6144` (`GROUPED_GEMM_SMALL_M_MAX`) | 16 | 128 | 128 |
  | `n > 6144` | 128 | 256 | 64 |

  `BLOCK_M` is load-bearing: `expert_block_schedule` is built for it and the GEMM decodes it
  back out of the schedule. `BLOCK_N` / `BLOCK_K` are pure perf knobs. Both configs are covered.
- **Preprocess path split**: `n <= 1024` (`FUSED_PREPROCESS_WIN_TOKENS`) takes the fused
  single-work-group path and *skips stage 3 entirely*; `n > 1024` takes sort + stage 3. The
  example follows the same rule under `--preprocess=auto`.
- **Interleaving**: `inference_moe_w13_interleaved` defaults to `True`
  (`sglang/srt/configs/inkling.py`), so `silu_and_mul` reads `[g0, u0, g1, u1, ...]` by default.
- **Biases**: Inkling's `_forward_routed` passes `w13_bias` / `w2_bias` as `None`, so
  `apply_grouped_bias` never fires. It is not modelled.

## Verification

Under `--verify=1`:

- metadata (`src2dst`, `reorder_topk_ids`, `reorder_ids`, counts, token offsets, block offsets,
  block schedule) is checked **exactly and in full** on every case against a
  `std::stable_sort`-based host reference;
- `pre_reorder` is checked **bitwise**;
- `silu_and_mul` and `post_reorder` are checked against host references computed from the device
  intermediates, so a GEMM inaccuracy cannot mask a reorder bug;
- both grouped GEMMs are checked against a host reference on a sampled set of rows and columns
  (full coverage on the tiny `quick` shapes);
- when both preprocess paths are legal, they are cross-checked against each other and must agree
  **bit-identically**.

Verification is *sampled by row/column* on the large shapes: at `T = 16384`, `H = 6144` a full
check would pull well over a GiB per buffer back to the host for every case.

Weights and activations are generated by a **counter-based hash evaluated identically on host and
device**. This has two purposes: the multi-GiB weight tensors never need a host staging copy (the
CPU reference recomputes `B` from the linear index), and every buffer holds genuinely random data
- Xe memory compression would otherwise inflate every reported `GB/s` against constant data.

## Options

```
--suite=quick|inkling|perf     built-in suite (default quick)
--shape=T=..,topk=..,E=..,H=..,I=..[,verify_gemm=0|1]
--dtype=bf16                   bf16 only (the routed path is bf16)
--preprocess=auto|fused|sort   force a preprocess path (default auto = the model's rule)
--interleaved=0|1              w13 gate/up interleaving (default 1)
--iterations=<int>             timed pipeline iterations (default 5)
--warmup=<int>                 minimum warmup iterations (default 2)
--verify=0|1                   run the CPU references (default 1)
--benchmark=0|1                per-stage event timing (default 1)
--perf-threshold-scale=<f>     scale every perf gate (default 1.0)
--mem-budget-gb=<f>            skip cases needing more device memory (default 12)
```

The warmup loop also keeps running until roughly 2.5 s of wall time has elapsed, because BMG
ramps 1200 -> 2400 MHz over about 2 s and a handful of launches on a small shape would otherwise
measure the ramp instead of the kernel.

### Suites

- `quick` - tiny shapes chosen for the branchy edges: single token, empty experts, tail blocks, a
  `hidden` that is not a multiple of 8 (non-vectorized copy path), `n = 1024` / `1026` (the
  fused/sort boundary) and `n = 6144` / `6150` (the `BLOCK_M` 16 -> 128 boundary).
- `inkling` - `E = 256`, `top_k = 6`, `H` in `{768, 1536, 6144}`, `I_p = I / P` for
  `I` in `{384, 3072}` and `P` in `{1, 2, 4, 8}`, and `T` in `{1, 3, 9, 144, 512, 4096, 16384}`
  (`T = 3` and `9` are the two shipped `draft_token_num` values, `16384` is
  `max_prefill_tokens`). Plus
  `T = 170 / 171` and `T = 1024 / 1025`, which straddle the two config boundaries at production
  `E` and `top_k` - the main `T` grid steps over both.
- `perf` - `H` in `{1536, 6144}`, `I_p` in `{384, 3072}`, `T` in `{1, 9, 144, 4096, 16384}`,
  verification off.

### Device-memory budget

`I_p = 3072` with `H = 6144` and `E = 256` needs about 29 GiB of weights (19.3 GiB `w13` +
9.7 GiB `w2`), which does not fit on a B60 - this is exactly why the model shards the routed
experts over tensor parallel. Such cases print `skip=OVER_MEM_BUDGET` rather than failing; raise
`--mem-budget-gb` to attempt them on a larger part. A device allocation that fails anyway is
reported as `skip=DEVICE_ALLOC_FAILED`, so one oversized case cannot take a whole suite down.

## Perf gates

All `target_gbps` / `target_tops` values are **`0.0` (report-only)**, following
`examples/17_bmg_relative_attention_backend`. Nothing in this pipeline has a calibrated target
yet, and the grouped GEMM is explicitly untuned, so any number here would be a guess that flakes
CI. Numbers are printed per stage on every run.

## Building and running

```bash
ninja -C <build-dir> 25_bmg_moe_reorder_grouped_gemm

./examples/25_bmg_moe_reorder_grouped_gemm/25_bmg_moe_reorder_grouped_gemm --suite=quick   --verify=1
./examples/25_bmg_moe_reorder_grouped_gemm/25_bmg_moe_reorder_grouped_gemm --suite=inkling --verify=1
./examples/25_bmg_moe_reorder_grouped_gemm/25_bmg_moe_reorder_grouped_gemm --suite=perf    --verify=0
```

All GPU work on the shared BMG hosts must go through the container wrapper
(`enter-sglang-syk.sh --gpu <n> -- ...`); bare-metal runs hang.
