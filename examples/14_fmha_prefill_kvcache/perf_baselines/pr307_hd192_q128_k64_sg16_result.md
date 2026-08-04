# PR307 HD192 q128/k64/sg16 Result

Date: 2026-07-25

Optimization: change paged `head_dim=192` standalone prefill from q256/k64
with 32 subgroups to q128/k64 with 16 subgroups.

The baseline and optimized binaries were built from the same current source.
The baseline explicitly restores the previous q256/k64/sg32 defaults through
CMake cache variables; only the three paged hd192 tile parameters differ.

Environment:

- Container: `sglang-syk`
- Device selector: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
- GPU affinity: `ZE_AFFINITY_MASK=0`
- Device: `Intel(R) Arc(TM) Pro B60 Graphics`
- Timing: SYCL event profiling, `kernel_avg_ms`

Build:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; CC=icx CXX=icpx ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake -S . -B build/fmha_prefill_kvcache_pr307_hd192_q256_k64_sg32_current -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=bmg -DFMHA_STANDALONE_HEAD_DIMS=192 -DFMHA_STANDALONE_PAGED_HD192_TILE_Q=256 -DFMHA_STANDALONE_PAGED_HD192_TILE_KV=64 -DFMHA_STANDALONE_PAGED_HD192_NUM_SG=32'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake --build build/fmha_prefill_kvcache_pr307_hd192_q256_k64_sg32_current --target fmha_prefill_kvcache -j 3'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; CC=icx CXX=icpx ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake -S . -B build/fmha_prefill_kvcache_pr307_hd192_q128_k64_sg16 -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=bmg -DFMHA_STANDALONE_HEAD_DIMS=192'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake --build build/fmha_prefill_kvcache_pr307_hd192_q128_k64_sg16 --target fmha_prefill_kvcache -j 1'
```

Shape sweep:

- Paged bf16 prefill, no AppendKV
- `head_dim=192`, `head_dim_v=192`
- `batch={1,8,16}`
- `heads_q=16`
- `heads_kv={4,8}`
- `causal={0,1}`
- `seqlen_q=128`
- `seqlen_k=4096`
- `page_size=128`
- `warmup=100`
- `iters=500`
- `verify=0`

The baseline and optimized binaries were run back-to-back for each shape on
the same GPU.

Summary:

| configs | current baseline mean ms | optimized mean ms | latency change | mean per-config speedup | worst | best |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 2.544234 | 1.561724 | -38.62% | +56.81% | +40.57% | +75.11% |

Roofline:

- Logical work for the batch-1 shape is 6.442 GFLOP.
- Ideal arithmetic intensity is about 455 FLOP/B for `heads_kv=4` and
  241 FLOP/B for `heads_kv=8`.
- Accounting for per-query-head K/V reads gives about 124 FLOP/B, below the
  143 FLOP/B balance point implied by 50 TOPS and 350 GB/s.
- The q256 baseline also leaves half of its subgroups without valid Q rows at
  `seqlen_q=128`; q128 removes that waste while preserving the 8-row subgroup
  MMA shape.

Accuracy:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ZE_AFFINITY_MASK=2 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build/fmha_prefill_kvcache_pr307_hd192_q128_k64_sg16 --output-on-failure -R "(coverage\\.paged_hd192|boundary\\.paged_hd192)"'
```

Result: 4/4 tests passed, covering the existing hd192 case and Q lengths
127, 128, and 129 around the new tile boundary.

Raw results:

```csv
head_dim,batch,heads_kv,causal,current_baseline_ms,optimized_ms,speedup
192,1,4,0,0.402052,0.275454,45.96%
192,1,4,1,0.400412,0.274811,45.70%
192,1,8,0,0.370917,0.263872,40.57%
192,1,8,1,0.369880,0.261866,41.25%
192,8,4,0,2.628520,1.547000,69.91%
192,8,4,1,2.622560,1.579820,66.00%
192,8,8,0,2.432580,1.550970,56.84%
192,8,8,1,2.433080,1.592260,52.81%
192,16,4,0,4.915200,2.806850,75.11%
192,16,4,1,4.894780,2.869230,70.60%
192,16,8,0,4.531100,2.823560,60.47%
192,16,8,1,4.529730,2.894990,56.47%
```
