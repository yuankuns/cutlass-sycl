# 24_bmg_moe_silu_and_mul

Inkling MoE / dense-MLP `silu_and_mul` (SwiGLU) activation for CUTLASS SYCL on
BMG. This is the activation between the two grouped GEMMs of the routed MoE and
between the two linears of the dense / shared-expert MLP; no other example in
this tree covered it.

## Semantics (the point of the example)

Ported from `python/sglang/kernels/ops/moe/inkling_moe.py`
(`silu_and_mul_interleaved_kernel`, `silu_and_mul_non_interleaved_kernel`) and
`python/sglang/srt/models/inkling_common/dense_mlp.py` (`swiglu`,
`swiglu_contiguous`, `InklingSwiglu`):

```
gateup[M, 2N] -> out[M, N]

interleaved (model default, inference_moe_w13_interleaved=True):
    gate = gateup[m, 2n],  up = gateup[m, 2n + 1]
non-interleaved / contiguous ([gate || up], the --enable-lora layout):
    gate = gateup[m, n],   up = gateup[m, N + n]

float g = float(gate), u = float(up);            // widen first
float v = (g * sigmoid(g)) * u;                  // fp32 silu-and-mul
if (has_topk_weights) v = v * float(weight[m]);  // routing weight LAST
out[m, n] = Element(v);                          // ONE rounding cast
```

Three details are load-bearing and are the reason this example exists:

1. **Interleaved gate/up.** Inkling sets `inference_moe_w13_interleaved=True`, so
   the fused `w13` output stores gate at even columns and up at odd columns, not
   as two contiguous halves. Both layouts are implemented and selectable
   (`--layout=`, or `interleaved=0|1` in `--shape=`).
2. **fp32 math, routing weight last, one rounding cast.** The upstream kernel is
   documented as a bitwise-identical port of the Helion kernels it replaced:
   fp32 math, `gate * sigmoid(gate) * up`, weight scale last, one rounding cast
   at the store. The stock sglang `silu_and_mul_kernel`
   (`ops/elementwise/elementwise.py`) instead rounds the SiLU result before
   multiplying by `up` ("cast down before mul to better match training") -- that
   is a *different* kernel and is not what Inkling runs.
3. **`InklingSwiglu` adds nothing.** It is only a flag-driven dispatch between
   `swiglu` (interleaved) and `swiglu_contiguous`; there is no gamma, alpha,
   limit or clamp term (the shared-expert `MoeRunnerConfig` passes
   `gemm1_alpha=None`, `gemm1_clamp_limit=None`). So the dense path is exactly
   the routed path with `has_topk_weights=false`, and the shared-expert path
   (`InklingBatchDenseMLP._swiglu`) is the routed path with the gate *gammas*
   passed in the routing-weight slot.

Because a cast-per-multiply kernel differs from the correct one by only about
one output ulp, a plain 1-ulp tolerance cannot tell them apart. `--verify=1`
therefore also builds the cast-per-multiply reference and requires the
single-cast reference to be the strictly better bit-exact match. On this kernel
the single-cast reference matches >99.99% of outputs bit-for-bit (the residue is
the host/device `exp()` differing by an ulp) while the cast-per-multiply variant
matches only ~64-70%, so the check is decisive.

## Shapes

`M = T * top_k` with `top_k = num_experts_per_tok = 6` and
`T in {1, 9 (draft_token_num), 144, 4096, 16384 (max_prefill_tokens)}`;
`N = I / P` for `I in {384 (checkpoint routed), 3072 (production routed and
dense_intermediate_size)}` and TP `P in {1, 2, 4, 8}`. The dense path runs `M = T`
rows with no routing weight, and the shared-expert path runs
`M = T * n_shared_experts (2)` with weights. dtype is bf16 (and fp16 for
coverage) in and out with fp32 math, matching the model.

`--suite=inkling` enumerates all of the above; `--suite=quick` is
correctness-shaped (odd widths, vector tails, both layouts, weights on and off);
`--suite=perf` is the gated prefill bands.

## Kernel shape

One work-item produces 8 contiguous outputs (`--elems-per-item`, auto-selected:
8 when `N % 8 == 0`, else 4 or 1; the scalar bounds-checked tail path is reached
by pinning a smaller-than-`N` divisor, which the `pin_epi*` quick cases do). The
interleaved path reads the 16 gate/up elements as 4x `uint64` and de-interleaves
in registers -- an element-strided gather would halve the effective read
bandwidth, which is the same conclusion the upstream Triton kernel reached with
`tl.split`.

Work-groups are 256 items split (rows x columns) by the row width, since columns
are the contiguous direction. The split is the largest power-of-two divisor of
the column count when that is at least a full subgroup (16), else the columns
rounded up to a power of two. Both halves of that rule were measured: taking a
non-divisor at `N=3072` pads a third of the launch away for nothing, while
narrowing `N/P = 48` from 8 columns to its exact divisor 2 costs 13% at T=16384
despite launching 25% fewer work-items -- coalescing beats item count.

Arithmetic intensity is 3 bytes moved per output element for ~5 FP32 ops
(~0.8 FLOP/B): purely memory bound, so the benchmark reports effective GB/s
against the ~400 GB/s BMG DRAM ceiling and never TOPS. Inputs are filled with
random data (large cases tile a 1 Mi-element random block) because Xe memory
compression reports fictitious bandwidth on constant or zero buffers.

## Measured performance

Intel Arc Pro B60 (card 5, selected by the container wrapper's `--gpu 5`, which
sets `ZE_AFFINITY_MASK`; the binary itself always takes the default SYCL GPU),
shared with another job so a few percent of noise, bf16,
`--suite=perf --verify=0 --iterations=50 --warmup=20`:

All 19 gated cases, interleaved (`routed`) and non-interleaved (`contig`):

| case | M x N | avg_ms | GB/s | gate |
| --- | --- | --- | --- | --- |
| perf_routed_i384_tp1_t4096 | 24576 x 384 | 0.1474 | 385 | 300 |
| perf_contig_i384_tp1_t4096 | 24576 x 384 | 0.1470 | 386 | 300 |
| perf_routed_i384_tp8_t4096 | 24576 x 48 | 0.0132 | 544 | 420 |
| perf_contig_i384_tp8_t4096 | 24576 x 48 | 0.0130 | 554 | 420 |
| perf_routed_i3072_tp1_t4096 | 24576 x 3072 | 1.1520 | 393 | 300 |
| perf_contig_i3072_tp1_t4096 | 24576 x 3072 | 1.1525 | 393 | 300 |
| perf_routed_i3072_tp8_t4096 | 24576 x 384 | 0.1465 | 387 | 300 |
| perf_contig_i3072_tp8_t4096 | 24576 x 384 | 0.1461 | 388 | 300 |
| perf_routed_i384_tp1_t16384 | 98304 x 384 | 0.5809 | 391 | 300 |
| perf_contig_i384_tp1_t16384 | 98304 x 384 | 0.5794 | 392 | 300 |
| perf_routed_i384_tp8_t16384 | 98304 x 48 | 0.0690 | 416 | 350 |
| perf_contig_i384_tp8_t16384 | 98304 x 48 | 0.0689 | 416 | 350 |
| perf_routed_i3072_tp1_t16384 | 98304 x 3072 | 4.6050 | 394 | 300 |
| perf_contig_i3072_tp1_t16384 | 98304 x 3072 | 4.6089 | 393 | 300 |
| perf_routed_i3072_tp8_t16384 | 98304 x 384 | 0.5811 | 390 | 300 |
| perf_contig_i3072_tp8_t16384 | 98304 x 384 | 0.5795 | 392 | 300 |
| perf_dense_i3072_tp1_t4096 | 4096 x 3072 | 0.1934 | 390 | 300 |
| perf_dense_i3072_tp1_t16384 | 16384 x 3072 | 0.7689 | 393 | 300 |

(`perf_routed/contig_i3072_tp8_*` is the same M x N as `i384_tp1_*`, kept because
the two configurations reach it from different `I` and TP.)

The `N/P = 48` cases exceed the DRAM ceiling because their working set is L2
resident (7 MB at T=4096, 29 MB at T=16384). The interleaved and contiguous
layouts measure within noise of each other, so the vectorized de-interleave
costs nothing. fp16 matches bf16 within noise. The whole suite also passes at the
default `--iterations=20 --warmup=5`, so the gates do not depend on the long
timing loop.

Gates: 300 GB/s for DRAM-resident cases, 420 GB/s for the L2-resident
`N/P = 48` T=4096 band, 350 GB/s for the partly-resident T=16384 band -- each
~15% below the measured floor. Decode / small-`T` cases are launch-latency bound
and are report-only (`0.0`), never a guessed number.
`--perf-threshold-scale=0` disables all gates.

## Typical commands

All GPU work goes through the BMG container wrapper (`--gpu N` selects the card
by setting `ZE_AFFINITY_MASK`; the binary itself has no `--gpu` flag). With
`WRAP` the wrapper and `BUILD` the configured build directory *as seen inside the
container* (`/data2/syk` is mounted at `/workspace`):

```bash
WRAP="/data2/syk/.codex/skills/bmg-gpu-container/scripts/enter-sglang-syk.sh --gpu 5 --"
BUILD=/workspace/cutlass-sycl/build-batch
BIN=$BUILD/examples/24_bmg_moe_silu_and_mul/24_bmg_moe_silu_and_mul

$WRAP ninja -C $BUILD -j 32 24_bmg_moe_silu_and_mul
$WRAP $BIN --suite=quick --iterations=5 --verify=1
$WRAP $BIN --suite=inkling --verify=1
$WRAP $BIN --suite=perf --dtype=bf16 --verify=0 --iterations=50 --warmup=20
```

Single shape, e.g. the production routed T=144 point in the contiguous layout:

```bash
$WRAP $BIN --shape=m=864,n=3072,interleaved=0,weights=1 --verify=1
```
