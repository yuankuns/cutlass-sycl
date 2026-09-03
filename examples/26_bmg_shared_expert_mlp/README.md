# 26_bmg_shared_expert_mlp

Inkling shared-expert / dense MLP (`InklingBatchDenseMLP`) for CUTLASS SYCL on
BMG. This is the "sink" MLP that every token runs in addition to the routed
experts: `n_shared_experts = 2` dense SwiGLU MLPs, weighted by the router's
shared gammas and summed. The model dispatches between two implementations of
the same math and this example implements **both**.

Source of truth:
`sglang/python/sglang/srt/models/inkling_common/dense_mlp.py`
(`InklingBatchDenseMLP.forward` and `_forward_bf16_linearized`), with the
activation from `sglang/python/sglang/kernels/ops/moe/inkling_moe.py`
(`silu_and_mul_triton`).

## The two paths

Notation follows the model's docstring: `t` tokens, `s = n_shared_experts`,
`d = d_model`, `f = shared_d_mlp` (per TP partition).
`w13_weight` is `[s, 2f, d]` with gate/up **interleaved** on the `2f` axis
(`inference_moe_w13_interleaved=True`, so row `2j` is `gate_j` and row `2j+1`
is `up_j` — `swiglu()` slices `[..., ::2]` and `[..., 1::2]`).
`w2_weight` is `[s, d, f]`. `gammas` is `[t, s]`, bf16.

**1. bmm path** (`forward`, the default):

```
x_std  = x_td.unsqueeze(0).expand(s, -1, -1).contiguous()   # [s, t, d]
y_st2f = bmm(x_std, w13_weight.mT)                          # [s, t, 2f]
y_stf  = silu_and_mul(y_st2f, gammas.mT)                    # [s, t, f]
z_std  = bmm(y_stf, w2_weight.mT)                           # [s, t, d]
out_td = z_std.float().sum(dim=0).to(bf16)                  # [t, d]
```

Every `torch.bmm` is bf16 in / bf16 out with an fp32 accumulator, so `y`, `act`
and `z` are **rounded to bf16 between stages**. Only the expert-axis reduction
(`_sum_dim0`) accumulates in fp32 before a single bf16 cast — that is the
"match TorchTitan's accumulation precision" comment in the model.

**2. linearized bf16 path** (`_forward_bf16_linearized`, enabled by
`linearized_bf16` when the sink serves bf16 and w13 is interleaved). The expert
axis is folded into the weights:

```
w13_lin = w13_weight.view(s * 2f, d)                 # a pure reshape, same memory
w2_lin  = w2_weight.transpose(1, 2).reshape(s * f, d)
y       = mm(x_td, w13_lin.T).view(t, s, 2f)
act     = silu_and_mul(y, gammas)                    # [t, s, f]
out_td  = mm(act.reshape(t, s * f), w2_lin)          # [t, d]
```

The expert sum now happens **inside the second GEMM's fp32 accumulator**, so it
never rounds `z` to bf16. That is a real (small) numerical difference from the
bmm path, so each path is checked against its own CPU reference and the example
also prints `path_delta max_abs` between the two.

**The gamma fold.** `silu_and_mul_triton` multiplies the SwiGLU product by the
per-`(token, expert)` gamma in fp32 before the bf16 store:

```
act[s, t, j] = bf16( silu(f32(y[s,t,2j])) * f32(y[s,t,2j+1]) * f32(gamma[t,s]) )
```

Writing `e` for the expert index and `n` for the token index, the bmm path passes
`gammas.mT.reshape(-1)` (expert-major activation rows, `row = e*t + n`) and the
linearized path passes `gammas.reshape(-1)`
(token-major rows, `row = n*s + e`). Both index the same `gammas[n, e]`
value; only the row ordering of the flattened activation differs. The example
runs one SwiGLU kernel with a `row_major_s` flag and exercises both orderings.

Out of scope: the NVFP4 serving strategy (`_forward_fp4`, a different code
path), and the trailing `symm_mem_all_reduce` of the `[t, d]` output (see
example 19 for the all-reduce variants).

## Implementation notes

The two GEMMs use oneMKL (`oneapi::mkl::blas::row_major::gemm` and
`gemm_batch`) rather than a hand-rolled CUTLASS collective: MKL exposes exactly
the bf16 x bf16 -> bf16 with fp32 compute type that `torch.mm` / `torch.bmm`
use, plus the strided batched form the bmm path needs, so both paths reproduce
the model's rounding seams without re-deriving a tile config per shape.
Example 20 (`20_bmg_dflash_cache_path`) sets the precedent for linking oneMKL
from an example here. Everything else — the SwiGLU + gamma fold, the fp32
expert reduction, and the `expand(...).contiguous()` replication of `x` — is
plain SYCL. No ESIMD.

`--replicate-x=1` (the default) materializes `x.unsqueeze(0).expand(s, -1,
-1).contiguous()` the way the model does. `--replicate-x=0` instead passes a
stride-0 `A` to `gemm_batch`, which is mathematically identical; the difference
prices the model's `.contiguous()`.

Inputs are random (never constant/zero — Xe memory compression would inflate
the bandwidth numbers) and scaled by fan-in so intermediates and outputs have
std ~1. Without that scaling the bf16 outputs collapse toward zero and an
absolute-tolerance check would pass a kernel that is silently wrong.
Verification uses `abs_err <= 4e-2 + 2e-2 * |expected|`; bf16 ULP distance is
reported for informational purposes only where `|expected| >= 1`, since a
near-cancellation output sits many bf16 exponents below the tensor scale and
its ULP distance says nothing.

## Shapes

`n_shared_experts = 2` always. The shared sink is column-parallel on `w13` and
row-parallel on `w2`, so tensor parallelism shards only `shared_d_mlp`
(`f -> f/P`); `d_model` is replicated and the `[t, d]` output is all-reduced.
The suites therefore sweep `f/P` for `P in {1, 2, 4, 8}` at a replicated `d`.

- `d_model`: 768 (checkpoint), 1536 (config defaults), 6144 (production)
- `shared_d_mlp`: 384 (`intermediate_size`, checkpoint) and 3072
  (`dense_intermediate_size`, production)
- tokens: 1 (decode), 9 (`draft_token_num`, MTP verify), 144 (small prefill
  chunk), 4096 and 16384 (`max_prefill_tokens`) large prefill

`--suite=quick` is the CI suite: tiny shapes with awkward tails (odd `f` to
exercise the interleaved gate/up pair indexing and the 256-wide tile tail, odd
`d` for the GEMM leading dimensions, and `s = 1` / `s = 3` to catch expert-axis
mix-ups) plus the two cheapest real shapes. `--suite=inkling` walks the real
shape x TP x token grid with the CPU reference enabled wherever it stays under
a few seconds. `--suite=perf` is the timing sweep.

## Roofline

The layer is `6*s*t*d*f` FLOPs against `6*s*d*f` bf16 weight bytes, so
arithmetic intensity is roughly `t` FLOP/B: decode (`t = 1..9`) is
weight-bandwidth bound and prefill (`t >= 4096`) is DPAS bound. The benchmark
reports both effective TFLOP/s and estimated effective GB/s per case, and only
one of the two is the meaningful number for a given band.

Perf gates are **report-only (`0.0`)** in every case, following
`17_bmg_relative_attention_backend`. The measurements below were taken on a
shared B60, so a hard gate calibrated from them would flake.

## Measured (Intel Arc Pro B60, `--gpu 6`, `--suite=perf --iterations=20 --warmup=10`)

| case | bmm ms | bmm TFLOP/s | bmm GB/s | lin ms | lin TFLOP/s | lin GB/s |
|---|---|---|---|---|---|---|
| prod d6144 f3072 tp1 t=1     |  1.002 |  0.23 | 226 |  0.640 |  0.35 | 354 |
| prod d6144 f3072 tp1 t=9     |  1.027 |  1.98 | 222 |  0.854 |  2.39 | 266 |
| prod d6144 f3072 tp1 t=4096  | 12.277 | 75.56 |  84 | 11.607 | 79.93 |  54 |
| prod d6144 f3072 tp1 t=16384 | 50.61  | 73.33 |  68 | 47.42  | 78.25 |  39 |
| prod d6144 f/8=384 t=1       |  0.124 |  0.23 | 230 |  0.088 |  0.32 | 324 |
| prod d6144 f/8=384 t=4096    |  2.987 | 38.82 | 191 |  1.830 | 63.37 |  91 |
| prod d6144 f/8=384 t=16384   | 12.076 | 38.41 | 182 |  5.978 | 77.60 |  97 |
| cfg d1536 f384 tp1 t=1       |  0.029 |  0.24 | 245 |  0.021 |  0.35 | 347 |
| cfg d1536 f384 tp1 t=4096    |  0.723 | 40.08 | 236 |  0.430 | 67.40 | 163 |
| cfg d1536 f384 tp1 t=16384   |  2.900 | 40.00 | 228 |  1.658 | 69.94 | 156 |
| cfg d1536 f/8=48 t=16384     |  1.784 |  8.13 | 293 |  0.457 | 31.75 | 264 |

(The `prod tp1 t=16384` row is the median of three dedicated repeats; the
in-suite reading for that one case swung by ~15% because the card is shared.)

Observations:

- Decode (`t = 1..9`) is weight-bandwidth bound as predicted: the production
  TP1 shape reads `6*s*d*f = 226 MB` of weights and the linearized path hits 354 GB/s,
  close to the ~400 GB/s real DRAM ceiling on this part. `est_GBps` is the
  number to read in this band; `TFLOPs` is the number to read at `t >= 4096`.
- The linearized path is **faster than the bmm path at every shape measured**,
  by 1.06-4.0x, and the margin grows with TP sharding (4.0x at config TP8
  `t = 16384`, 2.0x at production TP8 `t = 16384`, 1.06x at production TP1
  `t = 4096`). Folding the expert axis into one GEMM gives MKL a single large
  `n = s*2f` / `k = s*f` problem instead of two small batched ones, and it also
  removes the `z` round-trip and the reduction launch. The more TP shrinks `f`,
  the worse the two batched GEMMs get and the more the fold pays.
- Per-stage breakdown at production TP1 `t = 4096` (`--breakdown=1`, each stage
  serialized behind its own wait): bmm =
  `replicate 0.52 ms, gemm1 7.15 ms, swiglu 0.63 ms, gemm2 3.41 ms,
  reduce 0.64 ms`. GEMM1 is 2x GEMM2 because it is `2f` wide, and the two
  non-GEMM stages the bmm path adds (replicate + reduce) are ~9% of it.
- The model's `x.expand(s, -1, -1).contiguous()` is *not* on the critical path
  at that shape, even though the serialized breakdown prices it at 0.52 ms:
  e2e bmm time with `--replicate-x=0` (stride-0 batched GEMM) is 11.83-12.30 ms
  against 12.25-12.29 ms with `--replicate-x=1`, i.e. equal within run-to-run
  noise. Do not "optimize" the replication away on the strength of the
  breakdown number alone.
- Measurement caveat worth recording: an earlier pass of this table on `--gpu 7`
  (which additionally has a degraded 5 GT/s PCIe link) reported the bmm path
  *winning* production TP1 `t = 16384` by 20%. Three dedicated repeats on
  `--gpu 6` put the linearized path ahead by 6% with the bmm number stable to
  0.04%, so the inversion was contention on the shared card, not an MKL tiling
  cliff. Single readings on a shared B60 are not trustworthy at 50 ms scale.

## Typical commands

```bash
WS=/workspace/cutlass-sycl
BIN=$WS/build-batch/examples/26_bmg_shared_expert_mlp/26_bmg_shared_expert_mlp
docker exec sglang-syk bash -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; ninja -C $WS/build-batch 26_bmg_shared_expert_mlp"
docker exec sglang-syk bash -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; $BIN --suite=quick --iterations=5 --verify=1"
docker exec sglang-syk bash -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; $BIN --suite=inkling --iterations=10 --verify=1"
docker exec sglang-syk bash -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; $BIN --suite=perf --iterations=20 --warmup=10 --verify=0"
docker exec sglang-syk bash -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; $BIN --shape=t=4096,d=6144,f=3072 --verify=0 --breakdown=1"
```
