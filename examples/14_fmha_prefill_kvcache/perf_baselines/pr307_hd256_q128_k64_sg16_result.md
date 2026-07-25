# PR307 HD256 q128/k64/sg16 Result

Date: 2026-07-24

Optimization: change paged `head_dim=256` standalone prefill from q256/k64
with 32 subgroups to q128/k64 with 16 subgroups.

Baseline: `9394b36ccb55e9a89bbb9c05e400c43f7e6f4e82`

Baseline record: `examples/14_fmha_prefill_kvcache/perf_baselines/pr307_hd256_9394b36c.md`

Environment:

- Container: `sglang-syk`
- Device selector: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
- GPU affinity: `ZE_AFFINITY_MASK=0`
- Device: `Intel(R) Arc(TM) Pro B60 Graphics`
- Timing: SYCL event profiling, `kernel_avg_ms`

Build:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; CC=icx CXX=icpx ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake -S . -B build/fmha_prefill_kvcache_pr307_hd256_optimized_default -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=bmg -DFMHA_STANDALONE_HEAD_DIMS=256'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake --build build/fmha_prefill_kvcache_pr307_hd256_optimized_default --target fmha_prefill_kvcache -j 8'
```

Shape sweep:

- Paged prefill, no AppendKV
- `head_dim=256`, `head_dim_v=256`
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

Summary:

| configs | baseline mean ms | optimized mean ms | latency change | mean per-config speedup | best | worst |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 2.861642 | 1.979989 | -30.81% | +47.22% | +61.18% | +38.66% |

Accuracy:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build/fmha_prefill_kvcache_pr307_hd256_optimized_default --output-on-failure -R "(boundary\\.paged_hd256_q_tile_plus1|chunk\\.paged_hd256)"'
```

Result: 21/21 tests passed.

Raw results:

```csv
version,head_dim,batch,heads_kv,causal,kernel_avg_ms,host_avg_ms,estimated_tflops
optimized,256,1,4,0,0.257191,0.257904,33.399
optimized,256,1,4,1,0.258972,0.259698,33.1694
optimized,256,1,8,0,0.273338,0.274219,31.4261
optimized,256,1,8,1,0.276452,0.277345,31.072
optimized,256,8,4,0,1.9684,1.97202,34.9113
optimized,256,8,4,1,2.01542,2.01912,34.0969
optimized,256,8,8,0,2.01041,2.01419,34.1819
optimized,256,8,8,1,2.05038,2.05419,33.5155
optimized,256,16,4,0,3.5909,3.5972,38.2742
optimized,256,16,4,1,3.67969,3.68619,37.3506
optimized,256,16,8,0,3.65922,3.6656,37.5597
optimized,256,16,8,1,3.71949,3.72596,36.951
```
