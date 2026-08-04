# PR307 HD96 Multi-Tile Result

Date: 2026-07-25

Baseline commit: `02bc0628d3e9187ac7cc5632c78b719345dd5133`

Optimization:

- paged: q128/k64 changes from 8 to 16 subgroups
- non-paged `seqlen_q <= 32`: q32/k64/4 subgroups
- non-paged `32 < seqlen_q < 512`: q128/k64/16 subgroups
- non-paged `seqlen_q >= 512`: preserves q256/k64/16 subgroups

Environment:

- Container: `sglang-syk`
- Device selector: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
- GPU affinity: `ZE_AFFINITY_MASK=0`
- Device: `Intel(R) Arc(TM) Pro B60 Graphics`
- Dtype: bf16 input/output with fp32 accumulation
- Timing: SYCL event profiling, `kernel_avg_ms`

Build:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; CC=icx CXX=icpx ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake -S . -B build/fmha_prefill_kvcache_hd96_baseline_02bc -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=bmg -DFMHA_STANDALONE_HEAD_DIMS=96'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake --build build/fmha_prefill_kvcache_hd96_baseline_02bc --target fmha_prefill_kvcache -j 3'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; CC=icx CXX=icpx ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake -S . -B build/fmha_prefill_kvcache_hd96_optimized -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=bmg -DFMHA_STANDALONE_HEAD_DIMS=96'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ONEAPI_DEVICE_SELECTOR=level_zero:gpu cmake --build build/fmha_prefill_kvcache_hd96_optimized --target fmha_prefill_kvcache -j 2'
```

The baseline binary was built before the optimization edits, with the original
paged q128/k64/sg8 and non-paged q256/k64/sg16 defaults.

Production q128 sweep:

- non-paged bf16 prefill, no AppendKV
- `head_dim=head_dim_v=96`
- `batch={1,8,16}`
- `heads_q=16`, `heads_kv={4,8}`
- `causal={0,1}`
- `seqlen_q=128`, `seqlen_k=4096`
- `warmup=100`, `iters=500`, `verify=0`

Benchmark:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; BASE=build/fmha_prefill_kvcache_hd96_baseline_02bc/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache; OPT=build/fmha_prefill_kvcache_hd96_optimized/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache; common="--seqlen-q 128 --seqlen-k 4096 --heads-q 16 --head-dim 96 --head-dim-v 96 --paged 0 --warmup 100 --iters 500 --verify 0"; ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu "$BASE" --batch 1 --heads-kv 4 --causal 1 $common >/dev/null; ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu "$OPT" --batch 1 --heads-kv 4 --causal 1 $common >/dev/null; for batch in 1 8 16; do for hkv in 4 8; do for causal in 0 1; do for version in baseline optimized; do if [ "$version" = baseline ]; then bin=$BASE; else bin=$OPT; fi; printf "%s,%s,%s,%s," "$version" "$batch" "$hkv" "$causal"; ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu "$bin" --batch "$batch" --heads-kv "$hkv" --causal "$causal" $common | sed -n "s/^profile: kernel_avg_ms=\\([^ ]*\\).*/\\1/p"; done; done; done; done'
```

Summary:

| configs | baseline mean ms | optimized mean ms | latency change | mean per-config speedup | worst | best |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.895784 | 0.629284 | -29.75% | +49.54% | +35.71% | +69.47% |

The worst measured configuration exceeds the required 20% speedup.

Model and dispatch checks use the median of three back-to-back runs with 100
warmups and 1000 measured iterations:

| shape | baseline ms | optimized ms | speedup |
|---|---:|---:|---:|
| DeepSeek OCR2, non-paged q32/k64, local, h8 | 0.006817 | 0.003052 | +123.34% |
| Flux2 Klein 9B, non-paged q32/k64, h12 | 0.004842 | 0.002643 | +83.21% |
| non-paged q128/k4096, h16/hkv4 | 0.181839 | 0.104377 | +74.21% |
| non-paged q512/k4096, h16/hkv4 | 0.228661 | 0.230293 | -0.71% |
| paged q128/k4096, h16/hkv4 | 0.186185 | 0.112636 | +65.30% |

The q512 case selects the preserved large tile and remains within measurement
noise of baseline.

Roofline:

- The q128/k4096 batch-1 shape performs 3.221 GFLOP.
- Ideal bf16 Q/K/V/O traffic is 7.078 MB, or 455 FLOP/B.
- Accounting for K/V reads by each query head gives 25.952 MB, or 124 FLOP/B.
- The B60 balance point from 50 TOPS and 350 GB/s is 143 FLOP/B, so this path is
  memory-bound under the observed per-query-head traffic.
- At 0.112346 ms, optimized effective bandwidth on that traffic estimate is
  about 231 GB/s, up from about 137 GB/s for baseline.
- The q32/k64 model shape is only about 21 FLOP/B and was additionally dominated
  by the mostly empty q256 tile.

Accuracy:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/worktrees/cutlass-sycl/fmha-kvcache-chunk-prefill-efd0f6d8; ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build/fmha_prefill_kvcache_hd96_optimized --output-on-failure -R "(stretch\\.(deepseek_ocr2|flux2_klein_9b)|coverage\\.local_window_paged|boundary\\.(nonpaged|paged)_hd96|append\\.(paged|nopaged)\\.hd96)"'
```

Result: 17/17 tests passed. This covers q31/32/33, q127/128/129,
q511/512/513, paged q127/128/129, both model shapes, local masking, and
paged/non-paged AppendKV.

Raw q128 sweep:

```csv
batch,heads_kv,causal,baseline_ms,optimized_ms,speedup
1,4,0,0.189873,0.112346,69.01%
1,4,1,0.188077,0.110977,69.47%
1,8,0,0.192797,0.122497,57.39%
1,8,1,0.192160,0.114395,67.98%
8,4,0,0.885614,0.613427,44.37%
8,4,1,0.889085,0.584263,52.17%
8,8,0,0.916011,0.670229,36.67%
8,8,1,0.920099,0.677611,35.79%
16,4,0,1.573190,1.089610,44.38%
16,4,1,1.578650,1.095410,44.11%
16,8,0,1.610090,1.171480,37.44%
16,8,1,1.613760,1.189160,35.71%
```
