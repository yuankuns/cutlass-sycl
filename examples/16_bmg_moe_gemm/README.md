# BMG W4A16 MoE GEMM

`16_bmg_moe_gemm` now directly runs the Xe2 W4A16 grouped-MoE GEMM from
`sgl-kernel-xpu`. Activations and outputs are BF16. The small accuracy case
uses packed signed INT4 weights with BF16 scales; GPT-OSS-120B performance
workloads use packed MXFP4 (E2M1) weights with E8M0 scales. GPT-OSS workload
accuracy mode compares the exact packed MXFP4/E8M0 representation against a
host reference using `atol=0.005` and `rtol=0.005`. Decode workloads compare
every output element; prefill workloads sample boundary and interior rows and
output channels for every active expert.

The files under `kernel/moe/xe20/w4a16` start from the W4A16 kernel sources of
`sgl-kernel-xpu` at `44060731` (PR 446). Upstream's policy menu, its
`select_w4a16_tile_m()` / `select_w4a16_policy_id()` scoring and its 4-bit
mainloop are all still here -- `--policy` and `--selector` still measure both
sides of that PR -- and on top of them the example carries the optimizations that
measure faster on Arc Pro B60, each behind a knob that prices it:

- **shape-specific work-group tiles**, from a registry of `BLK_M == SG_M`,
  `SG_N = 16` tiles (`--tile`, `--list-tiles`) picked by `select_workload_tile()`
  from `(avg_m, N, K)` alone: `64x256` for prefill GEMM1, `64x128` for prefill
  GEMM2 and the short-K bands, `8x64`/`16x64`/`32x64` for the decode band;
- **MXFP4 scale folding** into the E2M1-to-BF16 reorder, which ends in a multiply
  anyway, so the dequantize pass over the B fragment disappears;
- **a tuned (work-stealing chunk, prefetch distance) pair** per shape: a chunk of
  4 for the short-K GEMM2, none for GEMM1, and a prefetch distance of 2 against
  upstream's 6 (`--sched=<chunk>,<dist>`);
- **the per-k-tile work-group barrier only on the tiles that measure faster with
  it** (`w4a16_tile_wants_barrier()`); nothing in this mainloop is shared through
  SLM, so the barrier is purely a scheduling device;
- **carried group-scale pointers**, one 64-bit add per k-tile instead of the int
  add + widen + 64-bit add IGC needs to rebuild `Scales[offset + group_idx]`, and
  no group-scale prefetch;
- **the M-block skip**: an expert's last tile is partial, and on the l0 routing
  vector 23% of the dpas work at `BLK_M = 64` is on rows past the end of an
  expert. Only for a subgroup tile that holds more than one dpas M block -- at
  `SG_M = 8` there is nothing to skip and the predicate is pure cost;
- **the N-tail subgroup skip**, enabled per shape by `want_nskip()` rather than a
  flag, because it loses on GEMM1 and wins on GEMM2 (`--nskip=0|1` forces it);
- **tile-aligned A/B surfaces** for ragged expert tails, so an edge tile's 2D
  block load is filled by real memory (the next expert's rows) instead of crossing
  the surface's last row (`--row-extend=0` turns it off). This is the largest
  single win on the production shapes: 41% on the l0 GEMM1 (67.9 against 48.1
  TOPS), 22% on its GEMM2.

## Measured

GPT-OSS-120B routing vectors on Arc Pro B60, TOPS, median of three repetitions of
500 iterations after 1500 warmups -- ~3 s of warmup device time, so every number
is at a saturated clock -- against the pre-PR-446 reference build these
optimizations come from, measured identically:

| workload | GEMM1 | (reference) | GEMM2 | (reference) |
| -------- | ----- | ----------- | ----- | ----------- |
| TP=4 prefill l0 | 67.9 | 67.6 | 60.4 | 60.3 |
| TP=4 prefill l14 | 72.1 | 71.8 | 64.5 | 64.6 |
| TP=4 prefill l35 | 74.8 | 74.6 | 68.5 | 68.5 |
| TP=8 prefill l0 | 69.7 | 69.5 | 49.9 | 50.0 |
| TP=8 prefill l14 | 74.3 | 73.3 | 57.3 | 57.5 |
| TP=8 prefill l35 | 76.7 | 76.1 | 59.2 | 59.3 |

Shorter runs read ~1.5% higher on both builds -- at 300 iterations after 20
warmups the same two binaries measure 68.8/69.0 and 78.3/77.8 on the first and
last GEMM1 rows -- so compare only within one methodology.

The decode shapes are launch-bound rather than compute-bound (one token routed to
4 of 128 experts), so they are reported in device milliseconds per launch, median
of nine repetitions of 5000 iterations after 40000 warmup launches -- see the
warmup note below:

| workload | GEMM1 ms | (reference) | GEMM2 ms | (reference) |
| -------- | -------- | ----------- | -------- | ----------- |
| TP=4 decode | 0.035 | 0.040 | 0.026 | 0.027 |
| TP=8 decode | 0.035 | 0.035 | 0.021 | 0.022 |

The `8x64` tile the selector picks for that band runs it 7.5-14.3% faster than
the `16x128` this example used before (`--tile=16x128_1x8` measures the other
side).

## Tile selection

`--tile=<name>` picks one of the registry tiles above (`--list-tiles` prints
them); with neither flag `select_workload_tile()` chooses one. `--policy=<id>`
leaves the registry entirely and forces a tile of upstream's policy menu, at the
schedule upstream ships (prefetch distance 6, the work-group barrier on, no
work-stealing chunk, and none of the skips):

| id | policy | WG tile | subgroups | selected for |
| -- | ------ | ------- | --------- | ------------ |
| 0 | `w4a16_policy_m_8_n_64` | 8x64x32 | 1x4 | `avg_m <= 4` |
| 1 | `w4a16_policy_m_16_n_64` | 16x64x32 | 1x4 | `avg_m <= 8` |
| 2 | `w4a16_policy_m_32_n_64` | 32x64x32 | 1x4 | scored `tile_m <= 32` |
| 3 | `w4a16_policy_m_64_n_128` | 64x128x32 | 2x4 | scored `tile_m <= 64` |
| 4 | `w4a16_policy_m_128_n_128` | 128x128x32 | 4x4 | otherwise |
| 5 | `legacy_policy_m_128_n_256` | 128x256x32 | 4x8 | never (see below) |

Ids 0-4 are upstream's menu. Id 5 is the `<_128,_256,_32>` / 4x8 tile PR 446
replaced; it lives in the example rather than in the copied policy header so the
PR's "before" row stays measurable without adding anything upstream does not
have. `--selector=before` restores the thresholds PR 446 replaced
(`avg_m <= 4 / <= 8 / <= 128`, then the legacy tile), so a single binary can
measure both sides of the change:

```bash
./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm \
  --mode=perf --quant=mxfp4 --experts=4 --rows=512 --n=5760 --k=2880 \
  --selector=before --warmup=5 --iterations=30
```

`scripts/measure_446.sh` drives the three comparisons: `acc` (every GPT-OSS
workload on both selectors, plus one INT4 case per policy), `table` (PR 446's own
avg_m table, at the shape its description names -- GPT-OSS TP=1/EP=1, 32 experts,
N=5760, K=2880 -- plus the two shapes `bench_moe_w4a16_grouped_gemm.py` actually
runs, which the claim does not cover), and `wl` (the GPT-OSS-120B workloads,
before vs after).

`--quant=int4|mxfp4` selects the weight encoding for `--mode=perf` on an
explicit `--experts/--rows/--n/--k` shape; GPT-OSS workloads are always MXFP4.
Perf paths fill operands with device-side pseudo-random data, because Xe memory
compression makes a `memset` buffer read far faster than a real weight tensor,
and report the **median** of the per-iteration device-event samples.

Give every measured shape at least ~2 s of device time. The GPU ramps
1200 -> 2400 MHz over roughly two seconds (`act_freq` under
`/sys/class/drm/cardN/device/tile0/gt0/freq0`), so a short run measures the ramp:
one of these configurations reads 50.0 TFLOP/s at `--iterations=30` and 57.8 at
`--iterations=200`.

That budget is in *device time*, not iterations, which is what makes the decode
shapes awkward: at ~0.03 ms per launch, `--warmup=20` is 0.6 ms of ramp and the
same configuration then reads anywhere from 0.035 to 0.079 ms. Warm those up in
launches (`--warmup=40000 --iterations=5000`, i.e. `scripts/moe_decode.sh`) before
believing a decode delta -- at `--warmup=20` this build reads 6% *slower* than the
reference on TP=8 decode GEMM1 and 7% faster on TP=4, and both are the ramp.

`scripts/upstream_446_bench.py` cross-checks this example against upstream
sgl-kernel-xpu itself, by driving `bench_moe_w4a16_grouped_gemm.py`'s own
`_prepare_inputs` / `_run_mxfp4_fused` / FLOP convention / `do_bench` median from
a 4406073 build at an arbitrary shape (upstream's shape list is hard-coded and
cannot express PR 446's own claimed configuration). Across 24 points on three
shapes the two agree to within 0.3%, except at E=4 where the kernels are 0.2-1.3
ms and this example runs 0.2-3.2% faster for want of PyTorch dispatch overhead.

Build and run through the BMG container:

```bash
/data2/syk/.codex/skills/bmg-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu 0 --workdir /data2/syk/worktrees/cutlass-sycl/fmha-cri -- \
  bash -lc 'CC=icx CXX=icpx cmake -S . -B build/gpt-oss-moe-icpx -G Ninja \
    -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 && \
    cmake --build build/gpt-oss-moe-icpx --target 16_bmg_moe_gemm -j'
```

Run the CPU-reference accuracy test:

```bash
./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm \
  --mode=accuracy --experts=8 --rows=8 --n=256 --k=256
```

Run a device-event timing baseline:

```bash
./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm \
  --mode=perf --experts=8 --rows=128 --n=1024 --k=1024 \
  --warmup=10 --iterations=100
```

The real GPT-OSS-120B TP=4 / TP=8 routing vectors from the previous benchmark
are available without restoring any of its optimization paths:

```bash
./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm \
  --list-workloads
./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm \
  --mode=perf --workload=gpt-oss-120b-prefill-l14-gemm1 \
  --warmup=10 --iterations=100
```

Run the exact GPT-OSS-120B TP=4 decode geometry and routing vector against the
host reference:

```bash
./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm \
  --mode=accuracy --workload=gpt-oss-120b-decode-gemm1
./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm \
  --mode=accuracy --workload=gpt-oss-120b-decode-gemm2
```
