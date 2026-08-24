# FMHA Decode KV Cache

This example ports the paged Split-K full-attention decode kernel used by the
Gemma4 TP=4 XPU path.  It is intentionally separate from the cached-KV
prefill examples because its dispatch and reduction path are different.

Default production shape:

```text
B=1, q_len=1, Hq/Hkv=8/1, head_dim=512, page_size=64, context=4097
dtype=BF16, scale=1.0, non-contiguous paged KV cache
```

The default `--num_kv_splits=0` uses an occupancy-oriented heuristic. This
matches the SGL kernel binding default, `num_splits=0`.

For one attention layer, QK plus PV performs
`4 * 8 * 512 * context` FLOPs.  At context 4097 this is 67.1 MFLOPs.  The
minimum unique K/V read volume is about 8.4 MB, giving an optimistic arithmetic
intensity of 8 FLOP/byte.  This is memory/latency bound; assess it using
per-layer latency and effective KV bandwidth rather than a 50 TOPS target.

## Optimization history (2026-08-08)

The kernel as ported ran at 7.87 GB/s of logical KV traffic. Three defects were
found and fixed, taking it to ~311-370 GB/s. Each was verified against the BF16
reference at contexts `1,63,64,65,4097,4608,5120`.

1. **Split-K was silently disabled.** The kernel collapsed all splits onto split
   0 whenever the windowed K-block count was below a hard-coded 128. At
   page_size 64 that covers every context under 8192 tokens, so the whole 8.4 MB
   KV read ran on one work-group -- 4 of ~1280 available sub-groups. The
   threshold is now the `min_blocks_for_split` parameter, defaulting to 2.
   Worth 7.9 -> 39.7 GB/s.

2. **The O accumulator filled the entire register file.** `SubgroupLayoutPV`
   splits the k dimension, not v, so every sub-group held the full
   `8 x 512` float accumulator: 256 registers per lane at `grf_size<256>`. IGC
   spilled 17,728 bytes per thread, and the resulting scratch traffic (203 spill
   loads and 264 spill stores per iteration, against ~27 real KV block loads)
   dominated the kernel. Tiling the output head dim across work-groups
   (`GEMMA4_DECODE_VTILE_O`, default 256) shrinks the accumulator and raises
   occupancy. Worth 103 -> 349 GB/s on the mainloop alone.

3. **The Split-K reduction was latency-bound.** Its grid was
   `(seq, heads, batch)` -- 8 work-groups for a batch-1 decode -- and each thread
   walked every split serially with a whole `heads x head_dim` stride between
   consecutive loads. It moves only a few hundred KB yet cost 30 us, more than
   the mainloop. Tiling `head_size_vo` across work-groups and dividing the split
   loop across the sub-groups of each work-group took it to 5.4 us.

Three plausible fixes were measured and **refuted**; do not re-try them without
new evidence:

- Giving V a prefetch distance (it is prefetched immediately before the GEMM2
  that consumes it, unlike K which is a block ahead): 102.9 -> 100.6 GB/s.
- Deepening the K prefetch to 2, 3, 4, 6 or 8 blocks: flat at 99-102 GB/s.
  Both prefetch results are explained by fix 2: the mainloop is already within a
  few percent of the DRAM roof, so prefetching was never the limit.
- **Fusing the reduction into the FMHA launch**, tried two ways and reverted (it
  is not in the tree; read this before attempting it again):
  1. *Last arrival reduces the whole v tile.* 231 us versus 26 us. The cost is
     serialization, not synchronization: one work-group does the tile's
     `head_group_q * vtile` = 2048 elements as ~1408 dependent scalar gathers per
     thread while the other 21 splits sit idle, where the separate dispatch
     spreads that same work over 256 work-groups. Fences and codegen were
     eliminated as causes: `relaxed` atomics measured identically (231 us), the
     ISA was within 5 instructions of the fast build, and there was no spill.
  2. *Every split reduces a slice, waiting on an arrival counter.* This restores
     the parallelism but **deadlocks**: 22 splits are launched onto 20 Xe cores,
     so they are not all co-resident and a spinning work-group starves the
     producer it is waiting for. Any fused variant needs a handoff that cannot
     block on a not-yet-scheduled producer.

The `fmha=` bandwidth printed at the production contexts reaches 458-545 GB/s,
above the board's ~440 GB/s peak, for two compounding reasons. First, an 8 MB KV
buffer re-read thousands of times in a timing loop becomes largely L2-resident.
Second, the reported figure divides by the *unique* KV bytes (8.39 MB at
ctx=4097), but `kHeadDim / kVTileO` v-tile groups each re-read all of K to
recompute Q*K, so the real traffic is `2 * K + V` = 12.59 MB. Always take a
DRAM-honest figure from a context too large to cache: at `--seq_len_kv=262144`
(268 MB) the kernel sustains **432 GB/s, or 98% of peak**.

That redundant K read looks like an obvious target, and `GEMMA4_DECODE_VTILE_O`
512 removes it entirely (one v-tile group, so K is read once). It is **much
worse**: 88 GB/s against 251, because a 512-wide tile puts the O accumulator back
in the spill regime of fix 2. The K re-read is the cheaper of the two costs since
it hits L2, so 256 is the measured optimum. 64 and 128 are also worse (155 and
216 GB/s) -- too little work per work-group.

### What now limits the production contexts

At ctx=4097 the layer is ~28 us: mainloop 18.3 us, reduction 5.8 us, and the
rest inter-launch gap. An empty two-kernel dispatch measures 9.4 us (6.0 + 3.6),
and the reduction costs 4.6-4.8 us even at 4 splits where it has almost nothing
to do -- so it is ~80% launch latency, moving 180 KB at an apparent 37 GB/s.
Fixed launch overhead is therefore about a third of the layer, and it, not
bandwidth, is what holds ctx=4097 to ~301 GB/s while the mainloop runs at the
DRAM roof. Further gains at this shape have to come from
removing a dispatch (see the refuted fusion above, which needs a cheaper
handoff than device-scope fences) rather than from the memory path.

Build and run in the required BMG container:

```bash
/data2/syk/.codex/skills/bmg-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu 0 -- bash -lc \
  'cmake --build /workspace/worktrees/cutlass-sycl/fmha-cri/build/fmha_decode_origin_main \
    --target 15_fmha_decode_kvcache -j2'

/data2/syk/.codex/skills/bmg-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu 0 -- bash -lc \
  '/workspace/worktrees/cutlass-sycl/fmha-cri/examples/15_fmha_decode_kvcache/bench_gemma4_full_decode.sh \
    /workspace/worktrees/cutlass-sycl/fmha-cri/build/fmha_decode_origin_main/examples/15_fmha_decode_kvcache/15_fmha_decode_kvcache'
```

The executable validates its BF16 output against a CPU FP32 attention
reference, rounded to BF16 before comparison.  Run boundary coverage after a
build:

```bash
for length in 1 63 64 65 4097 5120; do
  /workspace/worktrees/cutlass-sycl/fmha-cri/build/fmha_decode_origin_main/examples/15_fmha_decode_kvcache/15_fmha_decode_kvcache \
    --seq_len_kv="${length}" --warmup=1 --iterations=2 --verify=1
done
```

The performance script measures the exact production full-decode
specialization at the observed context range `4097, 4608, 5120`. It fixes
`q_len=1`, `Hq/Hkv=8/1`, `D=512`, BF16, `page_size=64`,
`softmax_scale=1.0`, non-contiguous paging, no window mask, and automatic
Split-K selection. It disables CPU reference verification while timing.

This dispatch takes ~30 us, so the timer's noise floor is a real hazard: at 100
iterations the reported bandwidth swings by tens of GB/s between identical runs,
and single runs ranked configurations wrongly during this work. The script
defaults to 100 warmup and 500 timed iterations; use interleaved repeats
(alternating the configurations under test) before believing any A/B difference.

Validation on 2026-08-08 passed the BF16 reference at contexts
`1,63,64,65,4097,4608,5120`. With 200 warmups and 2000 device-timed iterations,
medians of 5 runs:

Medians of 18 runs per context (200 warmup, 2000 timed iterations each):

| Context | Splits | Per-layer latency | Logical KV bandwidth | % of 440 peak | vs. as-ported |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4097 | 22 | 0.0279 ms | 301 GB/s | 68% | 38x |
| 4608 | 24 | 0.0260 ms | 363 GB/s | 82% | 46x |
| 5120 | 27 | 0.0280 ms | 374 GB/s | 85% | 48x |
| 262144 | 64 | 1.2419 ms | 432 GB/s | 98% | n/a |

Report medians, not best-of-n: individual runs at these contexts range over
roughly 300-410 GB/s, so a single run can overstate the result by a third.

The last row is the DRAM-honest measurement -- the only one whose working set is
too large to sit in L2 -- and it shows the memory path itself is essentially
saturated. The production contexts fall short of it because of fixed launch
overhead, quantified below, not because of the memory path.

## Relative attention

`--rel_extent N` adds Inkling's relative-attention bias to the scores.  The bias
is logically `bias[token, head, rel]` with `rel = row_kv - col_kv`, defined for
`rel` in `[0, rel_extent)` and zero outside -- a band, not a rectangle.  The
producing kernel (the `r x proj` einsum) writes it **sheared**: for each query
token it right-aligns that token's band into a `page_size`-aligned column window,
so the mainloop reads a plain rectangle and needs no gather.  The contract for
that surface -- its column count, its window origin, and why the widening to a
whole K tile is what keeps it legal for the block 2D copy atom -- lives in
`sycl/kernels/flash_attention_v2/collective/fmha_relative_bias.hpp`, which is
byte-identical to example 14's copy, so both mainloops consume one definition of
the arithmetic instead of two.

Two things differ from the prefill consumer, and both follow from the tile shape:

- A prefill M tile holds consecutive query tokens, so `row_kv` drifts across the
  tile and the surface has to be padded by a whole Q tile to cover that drift.
  A decode M tile holds the *query heads of one token*, so every row shares
  `row_kv`: the drift term is zero, the band is a single column window for the
  whole tile, and no per-row KV position is needed.
- Because the rows are heads rather than tokens, decode walks the same
  `[total_q, heads_q, padded_cols]` buffer as `[total_q * heads_q, padded_cols]`.
  One `rel_bias_row_stride` therefore steps both from head to head and from a
  token's last head to the next token's first, where prefill needs a separate
  token stride and head stride.

The bias rides along in the multiply-add that applies the still-pending `Q*K`
scale, so it costs no extra pass over the scores.  Every Split-K split reads the
same surface rows -- the band is a property of the token, not of the K range --
and adds the bias before its own softmax, so the partial max and sum handed to
the reduction already include it.

`HasRelBias` is a template parameter, not a runtime null check: IGC allocates
registers across every inlined branch of the mainloop, so a bias load that is
present but disabled would still cost the plain decode path, which runs within a
few percent of the DRAM roof.  The example instantiates both kernels and picks
one from `--rel_extent`, so the `--rel_extent=0` instantiation contains no bias
load at all and its mainloop body is textually what it was before this feature.

The bias itself is too cheap to measure at these shapes.  Medians of 5
interleaved runs (200 warmup, 2000 timed iterations), per-layer latency:

| Context | `--rel_extent=0` | `128` | `1024` |
| ---: | ---: | ---: | ---: |
| 4097 | 0.0266 ms | 0.0253 ms | 0.0253 ms |
| 5120 | 0.0262 ms | 0.0284 ms | 0.0268 ms |

Individual runs spanned 0.0245-0.0339 ms, so every column here is inside the
noise floor of the others; this is a "no measurable cost", not a speedup.  Read
the note above about this dispatch's timer noise before drawing a finer
conclusion.

Padding and shearing stay the producer's job.  The consumer is given only the
window origin and the surface's row stride and never recomputes the column
count, so a stride that disagrees with `rel_extent` is rejected by
`can_implement` rather than silently reading shifted columns.  This example
stands in for the `r x proj` kernel: it draws the band, shears it on the host,
and uploads the result.

`ctest -R 15_fmha_decode_kvcache` covers one shape from each half of this
(`ctest_examples_15_fmha_decode_kvcache_relative` is the decisive one).  Full
validation on 2026-08-24, all passing the BF16 reference:

```bash
BIN=.../examples/15_fmha_decode_kvcache/15_fmha_decode_kvcache

# Realistic magnitudes.
for spec in "1 1" "63 64" "64 64" "65 64" "4097 128" "4097 4097" "5120 192"; do
  set -- $spec
  $BIN --seq_len_kv=$1 --rel_extent=$2 --warmup=1 --iterations=2 --verify=1
done

# Decisive: the bias alone shapes the softmax.
for spec in "64 64" "65 64" "4097 128" "4097 4097" "4097 1" "5120 192" "1 1"; do
  set -- $spec
  $BIN --seq_len_kv=$1 --rel_extent=$2 --softmax_scale=0 --rel_bias_range=8 \
       --warmup=1 --iterations=2 --verify=1
done
```

The second sweep is the one that proves the bias is applied, and it exists
because the first does not.  `O` is an average over thousands of random `V`
rows, so its magnitude is ~1e-2; at an O(1) bias the reweighting of the softmax
moves the output by less than the verification tolerance, and a kernel that
dropped the bias entirely would still pass.  Setting `--softmax_scale=0` makes
the bias the only thing shaping the softmax and `--rel_bias_range=8` makes it
concentrate, so the reference output becomes the single highest-bias token's `V`
-- roughly 0.5 in magnitude against 1e-2 for a uniform average.  Those runs
report `max_abs=0.000000` at most shapes, which a bias-dropping kernel could not
do.  Between them the sweeps cover `col_origin` of 0, 3968, 4032, 4928 and -64
(both signs, aligned and not), extents 1/64/128/192/4097, contexts
1/63/64/65/4097/5120, and 1, 22 and 27 splits.

This port exposes the runtime-varying context length and split count.  It
implements the production BF16 specialization only.  FP16 is deliberately
out of scope because the observed Gemma4 XPU dispatch instantiates BF16; an
FP16 variant must be added with separate numerical validation rather than
silently changing the production type.
