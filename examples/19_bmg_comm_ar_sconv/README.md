# 19 BMG Comm / AR + SConv

Standalone CUTLASS SYCL examples for Inkling `06_comm_ar_sconv`.

Targets:

- `19_bmg_all_reduce_variants`: two-shot, full one-shot, push one-shot, and direct staged all-reduce semantics.
- `19_bmg_ar_fused_decode`: the two fused all-reduce + SConv + add-RMSNorm kernels. `inkling_ar_sconv_norm` (decode: one row per sequence, cache shift-update) and `inkling_ar_sconv_norm_verify` (target verify: `draft_token_num` rows per sequence, causal conv along the draft-token axis, cache read-only, per-position windows written to `inter_out`). Suites: `quick`, `stress`, `inkling` (shipped Inkling verify shapes), `perf`.
- `19_bmg_ar_scattered_sconv`: reduce-scatter/all-gather equivalent fused with causal SConv and local cache update.
- `19_bmg_xpu_collective_mapping`: XPU collective mapping contract for direct reduce and shard/gather round trip.

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
