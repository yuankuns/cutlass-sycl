# FMHA Decode KV Cache - Performance Status and Reproduction

Status as of 2026-08-10, branch `fmha-cri`, on the BMG (Xe2, 20 Xe cores,
~440 GB/s peak) validation GPU of the `sglang-syk` host.

This document is written so another agent can reproduce every number below and
knows which avenues are already closed. If you are picking this work up, read
"Do not re-try" before writing any code.

## Shape under test

The Gemma4 TP=4 full-attention decode specialization, per the summary in
`~/modeltune/gemma4/reports/gemma4_optimization_plan.md`:

```text
B=1, q_len=1, Hq/Hkv=8/1, D=512, BF16, page_size=64,
softmax_scale=1.0, non-contiguous paged KV cache, window_left=-1,
context = 4097, 4608, 5120     (perf)
context = 1, 63, 64, 65, 4097, 4608, 5120   (correctness)
```

Runtime path: paged `XeFMHAFwdSplitKVKernel` + `ReduceSplitK`, auto Split-K.

## Headline numbers

Medians of 18 runs per context (200 warmup, 2000 timed iterations each):

| Context | Splits | Latency | Logical KV BW | % of 440 peak | vs. as-ported |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4097 | 22 | 27.9 us | **301 GB/s** | 68% | 38x |
| 4608 | 24 | 26.0 us | **363 GB/s** | 82% | 46x |
| 5120 | 27 | 28.0 us | **374 GB/s** | 85% | 48x |
| 262144 | 64 | 1.2419 ms | **432 GB/s** | **98%** | n/a |

As-ported baseline for comparison: 1.0663 / 1.1854 / 1.3435 ms, i.e. 7.87 /
7.96 / 7.80 GB/s.

**The 370 GB/s (80% of peak) goal is NOT met at all three production
contexts.** 5120 clears it (374), 4608 is just short (363), 4097 falls well
short (301). The memory path itself is saturated -- see "What limits it now".

Correctness passes at all 7 contexts (`failing_elements=0`).

### Read the metric carefully

Two reasons the printed `fmha=` sub-launch bandwidth (458-545 GB/s) exceeds the
440 GB/s board peak and must not be quoted as the result:

1. **L2 residency.** An 8 MB KV buffer re-read 2000 times in a timing loop is
   largely L2-resident. Only the ctx=262144 row (268 MB) is DRAM-honest.
2. **The denominator is unique bytes.** The tool divides by the *minimum unique*
   KV bytes (8.39 MB at ctx=4097), but `kHeadDim / kVTileO` = 2 v-tile groups
   each re-read all of K to recompute Q*K, so real traffic is `2*K + V` =
   12.59 MB.

Headline the end-to-end `Performance:` line, not `fmha=`. And report medians:
individual runs at these contexts span roughly 300-410 GB/s, so a single run can
overstate the result by a third.

## Reproduce

Everything must run in the `sglang-syk` container; bare-metal BMG runs may hang.
Long execs on this host get SIGKILLed for unrelated reasons, so run detached and
retry rather than trusting a truncated log.

```bash
ENTER=/data2/syk/.codex/skills/bmg-gpu-container/scripts/enter-sglang-syk.sh
BUILD=/workspace/worktrees/cutlass-sycl/fmha-cri/build/fmha_decode_origin_main

# build
$ENTER --gpu 0 -- bash -lc "cmake --build $BUILD --target 15_fmha_decode_kvcache -j8"

# correctness (must print failing_elements=0 for every context)
$ENTER --gpu 0 -- bash -lc '
B='"$BUILD"'/examples/15_fmha_decode_kvcache/15_fmha_decode_kvcache
for ctx in 1 63 64 65 4097 4608 5120; do
  echo -n "ctx=$ctx "; $B --seq_len_kv=$ctx --warmup=5 --iterations=10 --verify=1 \
    | grep -oE "failing_elements=[0-9]+"
done'

# performance at the plan's contexts
$ENTER --gpu 0 -- bash -lc \
  "bash /workspace/worktrees/cutlass-sycl/fmha-cri/examples/15_fmha_decode_kvcache/bench_gemma4_full_decode.sh \
     $BUILD/examples/15_fmha_decode_kvcache/15_fmha_decode_kvcache"

# DRAM-honest figure (working set too large for L2)
$ENTER --gpu 0 -- bash -lc \
  "$BUILD/examples/15_fmha_decode_kvcache/15_fmha_decode_kvcache --seq_len_kv=262144 \
     --warmup=20 --iterations=200 --verify=0"
```

The build dir `build/fmha_decode_origin_main` already exists and is configured
with icpx 2025.3. Its CMake cache uses container paths (`/workspace/...`), so
configure/build from inside the container only.

### Benchmarking discipline

This dispatch is ~28 us, close enough to the timer's noise floor that sloppy
measurement produces wrong *rankings*, not just noisy values. During this work,
100-iteration runs ranked split counts backwards. Rules that were needed to get
stable answers:

- 200 warmup / 2000 timed iterations minimum (the script's defaults).
- Medians of 5+ runs; never best-of-n.
- When A/B-ing two configs, **interleave** them (`for rep; for config`), so
  thermal or host drift cannot be mistaken for a difference.
- Pin the GPU with `--gpu 0`; concurrent jobs without an affinity mask all land
  on the first visible device and corrupt results.

## Tunables and their measured optima

| Knob | Where | Default | Notes |
| --- | --- | ---: | --- |
| `GEMMA4_DECODE_VTILE_O` | `15_fmha_decode_kvcache.cpp` | 256 | O-accumulator / occupancy trade. 64/128/512 measured worse (155/216/88 GB/s). |
| `GEMMA4_REDUCE_DIM_TILE` | `xe_tile_scheduler.hpp` | 16 | Must be `<= sub_group_size`; there is a `static_assert`. 4/8 slower; **32 silently corrupted 266 outputs** before the assert existed. |
| `--min_blocks_for_split` | CLI / `KernelArguments` | 2 | Was a hard-coded 128. See fix 1. |
| `--num_kv_splits` | CLI | 0 (auto) | 0 uses the occupancy heuristic; resolves to 22/24/27. |
| `--prefetch_depth` | CLI / mainloop `Arguments` | 1 | Refuted knob, kept only for reproducibility. |

## The three root causes that were fixed

Ordered by impact. Each was verified against the CPU FP32 reference (rounded to
BF16) at all 7 contexts.

### 1. Split-K was silently disabled (7.9 -> 39.7 GB/s)

`XeFMHAFwdSplitKVKernel` collapsed every split onto split 0 whenever
`windowed_k_blocks < 128`. The threshold's unit is K blocks and 1 block = 64
tokens here, so "short sequence" meant "under 8192 tokens" -- which is *every*
production decode context. The entire 8.4 MB KV read ran on one work-group (4 of
~1280 subgroups).

It was silent because the split count is still computed, printed, and gridded;
splits 1..N just `continue`, and the epilogue writes sentinel stats
(`exp_sums=1, max_logits=0`) that make `ReduceSplitK` a pass-through, so results
stay correct. Nothing in the logs looks wrong.

This code is **upstream**, not a porting artifact: it comes from sgl-kernel-xpu
commit `8ab487b "fmha (#349)"` (2026-07-28) and is present in
`examples/14_fmha_prefill_kvcache` and every sgl-kernel-xpu worktree. The
original comment says the intent was to avoid split-reduce precision loss on
short sequences. **The production path likely has the same problem, but that has
not been measured** -- only this example was changed and validated. Before
pushing a fix upstream, express the threshold in tokens or available parallelism
rather than a fixed block count, and re-run precision regressions across
head_dim / window configurations (the BF16 result here is specific to
Hq/Hkv=8/1, D=512).

### 2. The O accumulator filled the whole register file (103 -> 349 GB/s)

The dominant cost, and invisible without an ISA dump. `SubgroupLayoutPV` splits
the k dimension rather than v, so every subgroup held the *entire* `8 x 512`
float accumulator = 256 registers per lane at `grf_size<256>`. IGC spilled
**17,728 bytes per thread**; the scratch traffic (203 spill loads + 264 spill
stores per iteration, against only ~27 real KV block loads) dominated everything.

Fixed by tiling the output head dim across work-groups (`GEMMA4_DECODE_VTILE_O`,
default 256), which shrinks the accumulator and raises occupancy. Each v-tile
group re-reads K to recompute Q*K, but the softmax statistics are v-independent,
so all groups write identical `exp_sums`/`max_logits` and the reduction is
unaffected.

Verified in the ISA, not inferred from timing: the fixed build has **zero**
`spilled -> Scratch` declarations and no `.spill size` line at all. To re-check:

```bash
$ENTER --gpu 0 -- bash -lc '
  mkdir -p /tmp/igc && cd '"$BUILD"' &&
  touch /workspace/worktrees/cutlass-sycl/fmha-cri/examples/15_fmha_decode_kvcache/15_fmha_decode_kvcache.cpp &&
  IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/tmp/igc \
    cmake --build . --target 15_fmha_decode_kvcache -j2 >/dev/null 2>&1
  grep -h "spill size" /tmp/igc/*_simd16_entry_*.asm | sort -u
  grep -c "spilled -> Scratch" /tmp/igc/*_simd16_entry_0001.asm'
```

### 3. The Split-K reduction was latency-bound (30 -> 5.8 us)

Its grid was `(seq, heads, batch)` = 8 work-groups for a batch-1 decode, and each
thread walked every split serially with a whole `heads x head_dim` stride between
consecutive loads. It moves only 180 KB yet cost 30 us -- more than the mainloop
at the time.

Fixed by tiling `head_size_vo` across work-groups (one subgroup per tile, one
output element per lane) and dividing the split loop across the subgroups of each
work-group, combining their partial sums through SLM.

## Do not re-try (refuted by measurement)

### V prefetch distance
V is prefetched immediately before the GEMM2 that consumes it, unlike K which is
a block ahead. Giving it a distance: 102.9 -> 100.6 GB/s. Note this *was* a win
on the prefill path, so the intuition does not transfer.

### Deeper K prefetch
Depths 2, 3, 4, 6, 8: flat at 99-102 GB/s. Both prefetch results follow from
fix 2 -- the mainloop is within a few percent of the DRAM roof, so prefetching
was never the limit.

### Removing the redundant K read
Real traffic is `2*K + V` = 12.59 MB, so eliminating K's second read looks like
an easy 33%. `GEMMA4_DECODE_VTILE_O=512` does exactly that (one v-tile group) and
is **much worse**: 88 vs 251 GB/s, because a 512-wide tile puts the accumulator
back in fix 2's spill regime. The K re-read is the cheaper cost since it hits L2.

### Fusing the reduction into the FMHA launch
Tried two ways, both reverted and **not in the tree**. This is the most tempting
remaining idea, so the failure modes matter:

1. *Last-arriving split reduces the whole v tile.* 231 us vs 26 us. The cause is
   **serialization, not synchronization**: one work-group does the tile's
   `head_group_q * vtile` = 2048 elements as ~1408 dependent scalar gathers per
   thread while the other 21 splits idle, where the separate dispatch spreads the
   same work over 256 work-groups. Fences and codegen were eliminated as
   suspects: `relaxed` atomics measured identically (231 us), the ISA was within
   5 instructions of the fast build, and there was no spill.
2. *Every split reduces a slice, spinning on an arrival counter.* Restores the
   parallelism but **deadlocks**: 22 splits are launched onto 20 Xe cores, so
   they are not all co-resident and a spinning work-group starves the producer it
   waits for.

Any fused variant needs a handoff that cannot block on a not-yet-scheduled
producer.

## What limits it now

At ctx=4097 the layer is ~28 us: mainloop 18.3 us, reduction 5.8 us, remainder
inter-launch gap. Supporting measurements:

- An empty two-kernel dispatch (ctx=64, 1 split) costs **9.4 us** (fmha 6.0,
  reduce 3.6).
- The reduction costs **4.6-4.8 us even at 4 splits**, where it has almost
  nothing to do -- i.e. ~80% pure launch latency, moving 180 KB at an apparent
  37 GB/s.
- The mainloop sustains 432 GB/s (98% of peak) at ctx=262144, where the working
  set defeats L2.

So fixed launch overhead is about a third of the layer, and *it*, not bandwidth,
is what holds ctx=4097 to 301 GB/s. Tuning the memory path further cannot close
the gap.

## If you want to push past 370 GB/s at ctx=4097

The only lever left is removing a dispatch, which needs a producer-consumer
handoff that never blocks an unscheduled work-group. Untried options:

- **Persistent kernel with grid <= Xe core count.** Makes all work-groups
  co-resident, which is exactly the precondition variant 2 above violated. Cost:
  the split count becomes a tuning parameter bounded by 20, and the tile
  scheduler needs a work-queue loop.
- **Batch the reduction across layers.** Gemma4 runs 10 full-decode layers per
  token; one reduce dispatch serving all of them amortizes the launch 10x.
  Requires the mainloop outputs of all layers to be live simultaneously, so it is
  a runtime-integration change, not a kernel change.
- **Amortize across the batch.** At B=1 there is no parallelism to find, but the
  fixed overhead is per-dispatch, not per-token, so larger batches dilute it for
  free. Worth quantifying before optimizing further for B=1.

Neither of the first two is a small change; both are larger than the tuning done
so far.
