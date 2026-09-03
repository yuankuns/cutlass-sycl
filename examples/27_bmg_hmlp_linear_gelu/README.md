<!--
 Copyright (C) 2026 Intel Corporation, All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
-->

# 27_bmg_hmlp_linear_gelu — Inkling HMLP per-layer Linear + RMSNorm + GELU

`examples/22_bmg_hmlp` covers only the `fold_timespace_to_depth` permutation of the
Inkling vision patch encoder. The compute that runs *after* every fold was
untested; this example covers it.

Mirrored source (read-only ground truth):

- `python/sglang/srt/models/inkling_common/hmlp.py` — `plan_out_scales`,
  `HMLPPatchEncoder.__init__` / `.forward`
- `python/sglang/srt/models/inkling_common/norm.py` — `RMSNorm(hidden, eps=1e-6)`
  (weighted, `F.rms_norm`)
- `python/sglang/srt/configs/inkling.py` — `InklingVisionConfig` defaults
- checkpoint `config.json` (`thinkingmachines/Inkling`) — the shipped
  `vision_config`

## Semantics under test

Per encoder layer, in this order:

1. `Linear(start_c * shuffle_mult, end_c, bias=False)` — activations and weights
   bf16 (or fp16), **fp32 accumulate**, result rounded back to the activation
   dtype, exactly as torch does for a bf16 `nn.Linear`.
2. `RMSNorm(end_c, eps=1e-6)` — *post*-linear, weighted, fp32 reduction.
3. `F.gelu(...)` — the **exact** (erf) variant, `approximate='none'`.

The **last** layer maps to `decoder_dmodel` and has **no** RMSNorm and **no**
GELU. When `use_vision_norm` is set, a single `RMSNorm(decoder_dmodel)` runs
after it (`final_norm`).

`shuffle_mult` for a layer is `Π (end/start)` over `(t, h, w)` of the two
adjacent scales — the depth growth the fold contributes.

## Widths are derived, not hard-coded

`plan_out_scales` is transcribed into the example (prime-factor scale ladder,
`round_up` to 64, log-spaced ideal reduction, and an exhaustive
`linear_sum_assignment` standing in for SciPy's). It was cross-checked against
the Python original for both shipped configurations; both have a *unique*
optimum, so the C++ assignment cannot drift from SciPy's tie-breaking.

Shipped vision config, verbatim from the checkpoint's `config.json`
(`"vision_encoder_type": "hmlp"`, `temporal_patch_size=2, patch_size=40,
n_channels=3, n_layers=4, decoder_dmodel=768, use_vision_norm=true`) — *not* the
`InklingVisionConfig` dataclass defaults, which are covered separately below.
Unfiltered ladder:

| scale (t,h,w) | channels |
|---|---|
| 1,1,1 | 3 |
| 1,5,5 | 128 |
| 1,10,10 | 320 |
| 1,20,20 | 1216 |
| 1,40,40 | 4800 |
| 2,40,40 | 9600 |

The assignment drops the `1216` scale, giving four layers:

| layer | in_features | out_features | RMSNorm | GELU | rows/patch |
|---|---|---|---|---|---|
| 0 | 75 (3 × 25) | 128 | yes | yes | 128 |
| 1 | 512 (128 × 4) | 320 | yes | yes | 32 |
| 2 | 5120 (320 × 16) | 4800 | yes | yes | 2 |
| 3 | 9600 (4800 × 2) | `decoder_dmodel` | no | no | 1 |

plus `final_norm = RMSNorm(decoder_dmodel)`.

Default `InklingVisionConfig` (`patch_size=16, temporal_patch_size=1,
n_layers=1`) collapses to a single layer `768 → decoder_dmodel` with no
in-layer norm/GELU (it is the last layer) and, by default, no final norm.

`decoder_dmodel` is covered at both shipped values, **768** and **6144**. The
whole HMLP tower is replicated on every rank (it is not tensor-parallel
sharded), so TP=1/2/4/8 all see exactly these shapes — no per-rank cases are
needed.

## Implementation choices

- **oneMKL for the Linear**, not a CUTLASS collective GEMM. Reasons: (a) it is
  the convention in this example family — of 14…23 only `20_bmg_dflash` needs a
  GEMM and it uses oneMKL; (b) `row_major::gemm` with `transb=trans` consumes
  the weight in the exact `nn.Linear` `[out_features, in_features]` layout, so
  no host-side transpose is invented that the model does not do; (c) the
  `(bfloat16, bfloat16, bfloat16, float)` overload gives bf16 in/out with fp32
  accumulate, matching torch; (d) `in_features = 75` on layer 0 would need
  padding to satisfy the Xe 2D-block-copy alignment a CUTLASS mainloop wants,
  and this example is not the place to test a GEMM it does not own.
- **Plain SYCL for RMSNorm (+ fused GELU)**: one work-group per row,
  `sycl::reduce_over_group` over an fp32 partial sum, `sycl::rsqrt`,
  `sycl::erf`. No `sycl::ext::intel::esimd`.
- The norm is a **separate launch**, not a GEMM epilogue: an RMSNorm needs the
  complete row, i.e. all K, so it cannot be folded into a tile epilogue without
  a second pass or a split-K reduction. Fusing it is a possible follow-up, not a
  requirement here.
- Each stage is verified against a CPU reference computed from **that stage's
  own device-side input**, so bf16 error does not compound down the 4-layer
  chain and the tolerance stays tight and meaningful.

## Tolerance

Both sides accumulate in fp32 and round to the activation dtype, but oneMKL's
accumulation order (and its k-blocking) differs from the scalar host loop, so a
value can land a few dtype ulps away. The accepted band is

```
|got - ref| <= 2·ulp_rel·|ref| + 1e-3·max|ref over the block|
```

(`ulp_rel` = 2^-8 for bf16, 2^-11 for fp16). The absolute term is what carries
the long-K stages, where up to ~4 ulps of deviation is observed, and it also
covers outputs near zero — a cancelling dot product or `GELU(x<0)` gives values
where a 1e-5 error is thousands of ulps but numerically irrelevant. It is scaled
by the block's own largest magnitude rather than being a magic constant, so the
band is "0.8% relative, or 0.1% of the block's largest value, whichever is
larger". A NaN or Inf in the device output fails explicitly (every comparison
against NaN is false, so it would otherwise report as a pass).

A structural error — dropped norm weight, missing GELU, GELU applied on the last
layer, wrong width — deviates by tens of percent and is caught. Two limits worth
stating:

- `max_ulps` is printed for information only; it is a true ulp distance (the
  sign-magnitude encoding is mapped to a monotone signed key, so a pair
  straddling zero is not reported as ~32768 apart), but ulps near zero are
  minuscule, so a five-digit figure alongside a `max_abs` of one ulp of the
  block's scale is expected rather than alarming.
- At bf16 output resolution the example **cannot** discriminate exact-erf GELU
  from the tanh approximation: the two differ by at most ~3e-4 absolute, below
  half a bf16 ulp near |x| ≈ 2. The exact form is used because that is what
  `F.gelu` defaults to; separating the variants would need an fp32 output path.

## Perf gates

All per-case `target_tops` / `target_gbps` are **0.0 (report-only)**. The wall
time of every case is dominated by the third-party oneMKL GEMM, so a gate here
would police oneMKL's tile selection rather than anything this example owns, and
would flake across oneMKL versions. The numbers are printed for tracking. Note
also that the reported GB/s is *effective* traffic (it counts the weight once
per iteration); on the small-row cases the weight is L2-resident across
iterations, so the figure can exceed the ~400 GB/s DRAM ceiling.

## Measured (Intel Arc Pro B60, `--suite=perf --iterations=50 --warmup=20`, bf16)

The GPU was shared with other jobs during these runs, so expect a few percent of
noise (the two `l2_rows4096` entries, which are the same shape under different
`decoder_dmodel`, spread by up to 30% across runs).

| case | shape | ms | TOPS | eff. GB/s |
|---|---|---|---|---|
| l0 rows16384 | 75 → 128, norm+GELU | 0.179 | 1.76 | 108 |
| l1 rows16384 | 512 → 320, norm+GELU | 0.291 | 18.5 | 203 |
| l2 rows16384 | 5120 → 4800, norm+GELU | 10.09 | 79.8 | 84 |
| l3 rows16384 (d=768) | 9600 → 768 + final norm | 2.86 | 84.5 | 150 |
| l3 rows16384 (d=6144) | 9600 → 6144 + final norm | 23.86 | 81.0 | 52 |
| full stack, 128 patches, d=768 | 4 stages + final norm | 0.507 | 31.8 | 225 |
| full stack, 128 patches, d=6144 | 4 stages + final norm | 1.049 | 28.0 | 212 |
| default p16 rows16384 | 768 → 768 | 0.240 | 80.6 | 215 |

The wide-K layers reach ~80 TOPS; layer 0 (K=75, N=128) is launch- and
norm-bound, not GEMM-bound, which is why its TOPS figure is small.

## Running

```
# quick smoke (this is what ctest runs)
./27_bmg_hmlp_linear_gelu --suite=quick --verify=1

# the shipped model shapes
./27_bmg_hmlp_linear_gelu --suite=inkling --verify=1

# perf sweep (large row counts, verification off)
./27_bmg_hmlp_linear_gelu --suite=perf --verify=0 --iterations=50

# single ad-hoc shape
./27_bmg_hmlp_linear_gelu --shape=patch_size=40,temporal_patch_size=2,n_layers=4,\
n_channels=3,decoder_dmodel=768,use_vision_norm=1,layer=2,rows=4096 --dtype=bf16
```

Other flags: `--dtype={bf16,fp16}`, `--iterations`, `--warmup`, `--benchmark`,
`--perf-threshold-scale`. Exit code 0 = pass, 2 = verification failure.

`--shape` notes: `layer=<i>` runs exactly that layer (and, when it is the last
layer and `use_vision_norm` is set, the final RMSNorm with it); `layer` omitted
runs the whole chained stack, which is sized in `patches` — a `rows=` given for a
full stack is taken as the *first* stage's row count and must be a whole number
of patches, otherwise it is rejected rather than silently ignored.

Give the timing loop enough iterations to clear the ~2 s BMG clock ramp
(1200 → 2400 MHz) before trusting a number, and note that all buffers are
initialized with **random** data so Xe memory compression does not inflate the
bandwidth figures.
