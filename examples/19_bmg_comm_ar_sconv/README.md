# 19 BMG Comm / AR + SConv

Standalone CUTLASS SYCL examples for Inkling `06_comm_ar_sconv`.

Targets:

- `19_bmg_all_reduce_variants`: two-shot, full one-shot, push one-shot, and direct staged all-reduce semantics.
- `19_bmg_ar_fused_decode`: fused decode all-reduce, SConv cache update, residual add, and RMSNorm.
- `19_bmg_ar_scattered_sconv`: reduce-scatter/all-gather equivalent fused with causal SConv and local cache update.
- `19_bmg_xpu_collective_mapping`: XPU collective mapping contract for direct reduce and shard/gather round trip.

These examples model multi-rank communication with multiple buffers on one BMG device. CUDA-only `multimem.ld_reduce/st` and in-kernel cross-GPU barriers are not available in this standalone SYCL environment, so staged-buffer event ordering is used as the XPU fallback boundary.

Build and run inside the required container:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; ninja -C /workspace/cutlass-sycl/build-syk 19_bmg_all_reduce_variants 19_bmg_ar_fused_decode 19_bmg_ar_scattered_sconv 19_bmg_xpu_collective_mapping'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; ctest --test-dir /workspace/cutlass-sycl/build-syk -R "19_bmg_(all_reduce_variants|ar_fused_decode|ar_scattered_sconv|xpu_collective_mapping)" --output-on-failure'
```
