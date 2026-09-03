# 19 BMG Comm / AR + SConv

Standalone CUTLASS SYCL examples for Inkling `06_comm_ar_sconv`.

Targets:

- `19_bmg_all_reduce_variants`: two-shot, full one-shot, push one-shot, and direct staged all-reduce semantics.
- `19_bmg_ar_fused_decode`: fused decode all-reduce, SConv cache update, residual add, and RMSNorm.
- `19_bmg_ar_scattered_sconv`: reduce-scatter/all-gather equivalent fused with causal SConv and local cache update.
- `19_bmg_xpu_collective_mapping`: XPU collective mapping contract for direct reduce and shard/gather round trip.

`19_bmg_ar_scattered_sconv` additionally mirrors the optional modes of the real
`inkling_ar_scattered_sconv` kernel, each with a CPU reference under `--verify=1`:

- `full_update`: write the replicated full-width `[slots, W-1, hidden]` conv-state
  cache instead of the column shard, with this rank's columns at `cache_col0 = rank * Hc`.
- prefix-cache `track`: scatter selected conv rows into a tracking slot, either from the
  reduced stream (`extend`) or from the post-update conv window (`track_from_cache`, decode).
- fused add + RMSNorm tail: after the scattered sconv and all-gather, add the residual
  in place and write the normed hidden to `norm_out`.

`--suite=inkling` sweeps those modes over `world in {1,2,4,8}` x `hidden in {768, 1536, 6144}`
with `W = 4`. The new `perf` cases for these modes are report-only (`min_GBps = 0.0`)
because those bands have not been calibrated on this part yet.

These examples model multi-rank communication with multiple buffers on one BMG device. CUDA-only `multimem.ld_reduce/st` and in-kernel cross-GPU barriers are not available in this standalone SYCL environment, so staged-buffer event ordering is used as the XPU fallback boundary.

Build and run inside the required container:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; ninja -C /workspace/cutlass-sycl/build-syk 19_bmg_all_reduce_variants 19_bmg_ar_fused_decode 19_bmg_ar_scattered_sconv 19_bmg_xpu_collective_mapping'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; ctest --test-dir /workspace/cutlass-sycl/build-syk -R "19_bmg_(all_reduce_variants|ar_fused_decode|ar_scattered_sconv|xpu_collective_mapping)" --output-on-failure'
```

The `perf` suites enforce per-case `min_GBps` thresholds in addition to
correctness checks. Use `--perf-threshold-scale=0` only when collecting new
baseline numbers without gating.

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /workspace/cutlass-sycl/build-syk/examples/19_bmg_comm_ar_sconv; for exe in 19_bmg_all_reduce_variants 19_bmg_ar_fused_decode 19_bmg_ar_scattered_sconv 19_bmg_xpu_collective_mapping; do ./$exe --suite=perf --iterations=3 --verify=1 || exit 1; done'
```
