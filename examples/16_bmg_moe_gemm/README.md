# BMG W4A16 MoE GEMM baseline

`16_bmg_moe_gemm` now directly runs the Xe2 W4A16 grouped-MoE GEMM from
`sgl-kernel-xpu`. Activations and outputs are BF16. The small accuracy case
uses packed signed INT4 weights with BF16 scales; GPT-OSS-120B performance
workloads use packed MXFP4 (E2M1) weights with E8M0 scales. GPT-OSS workload
accuracy mode compares the exact packed MXFP4/E8M0 representation against a
host reference.

The example intentionally contains no local scheduling or kernel optimization:
no BF16 surrogate, fused activation, M bucketing, tile overrides, skew
heuristics, diagnostic variants, or compiler tuning flags. The files under
`kernel/moe/xe20/w4a16` are the W4A16 kernel sources copied unchanged from
`sgl-kernel-xpu`.

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
