# BMG W4A16 MoE GEMM baseline

`16_bmg_moe_gemm` now directly runs the Xe2 W4A16 grouped-MoE GEMM from
`sgl-kernel-xpu`. Activations and outputs are BF16. The small accuracy case
uses packed signed INT4 weights with BF16 scales; GPT-OSS-120B performance
workloads use packed MXFP4 (E2M1) weights with E8M0 scales. GPT-OSS workload
accuracy mode compares the exact packed MXFP4/E8M0 representation against a
host reference using `atol=0.005` and `rtol=0.005`. Decode workloads compare
every output element; prefill workloads sample boundary and interior rows and
output channels for every active expert.

The example intentionally contains no local scheduling or kernel optimization:
no BF16 surrogate, fused activation, M bucketing, tile overrides, skew
heuristics, diagnostic variants, or compiler tuning flags. The files under
`kernel/moe/xe20/w4a16` are the W4A16 kernel sources copied unchanged from
`sgl-kernel-xpu` at `44060731` (PR 446), and the host picks a tile with that
commit's `select_w4a16_tile_m()` / `select_w4a16_policy_id()` scoring, ported
verbatim.

## Tile selection

`--policy=<id>` forces a work-group tile instead of letting the selector choose:

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
the fastest shape in the table reads 50 TFLOP/s at `--iterations=30` and 57.8 at
`--iterations=200`.

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
