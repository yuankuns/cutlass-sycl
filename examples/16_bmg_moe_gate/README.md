# Inkling MoE Gate Kernels

CUTLASS SYCL example implementations for the Inkling gate path.

Top-k renorm math contract:

- Input logits are fp32 rows with 258 logical columns: 256 routed experts and 2 shared experts.
- Ranking score for routed expert `e` is `sigmoid(logits[row, e]) + bias[e]`.
- Select top-6 routed experts. Exact ties select the lower expert id.
- Normalize the 6 selected routed sigmoid weights plus both shared sigmoid weights by `route_scale * global_scale[0]`.
- Non-packed output writes fp32 routed weights, int32 routed expert ids, and fp32 shared weights.
- Packed output writes int32 routed entries with high 16 bits as expert id and low 16 bits as bf16-rounded weight, plus fp32 shared weights.

Gate GEMV math contract:

- Input `x` is bf16 `[tokens, 6144]`.
- Input `weight` is bf16 `[>=258, 6144]`; rows `0..257` are the logical gate experts and rows `258..263` may be padded.
- Output `logits` is fp32 `[tokens, 264]`; the kernel writes only columns `0..257` as `x @ weight[:258].T` with fp32 accumulation and leaves padding columns `258..263` untouched.
- The default launch computes 1 expert per workgroup on BMG; `--experts-per-wg=1|2|4` and `--subgroup=16|32` are available for local tuning.

Fused packed output math contract:

- Input `x`, `weight`, `bias`, and `global_scale` match the split GEMV plus top-k contracts.
- The fused path supports `tokens <= 64`, writes fp32 scratch logits to a caller-provided `[tokens, 264]` workspace, and completes top-k/renorm in the same launch through a device-scope ticket.
- Non-packed fused output writes fp32 routed weights, int32 routed expert ids, and fp32 shared weights.
- Packed fused output writes int32 routed entries with high 16 bits as expert id and low 16 bits as bf16-rounded weight, plus fp32 shared weights.
- The ticket is reset by the kernel after the fused epilogue, so repeated in-order launches can reuse the same `int32[1]` ticket initialized to zero.

Top-k roofline and performance notes:

- Per token, the large-stream payload is 258 fp32 logits read plus 56 B of non-packed output or 32 B of packed output. Bias is 1 KiB and is reused by all rows, so it is effectively cache-resident for production token counts.
- This is the MoE gate postprocess, not the expert GEMM. It performs 258 sigmoid operations plus top-6 subgroup reductions per token, so a plain FLOP/byte roofline is misleading: `exp` throughput and subgroup shuffles are a significant part of the limit.
- The practical optimization target is reducing redundant logits traffic, keeping occupancy high enough to hide sigmoid latency, and avoiding extra subgroup communication. The kernel uses one 32-lane subgroup per token row, loads each routed logit once, keeps the selected sigmoid values in registers on lane 0 for the compact epilogue, and shares the same code path for packed and non-packed modes.
- The default launch policy uses two token rows per workgroup through 16k tokens and one row per workgroup for larger streams. Explicit `--rows-per-wg=1|2|4|8` remains available for local tuning.

Gate GEMV roofline and performance notes:

- Per call, useful work is `2 * tokens * 258 * 6144` FLOPs.
- With the default 1 expert/workgroup, estimated global traffic is roughly `258*6144*2` bytes of weight plus `258*tokens*6144*2` bytes of repeated `x` reads plus fp32 output stores. Operation intensity is about `0.50 FLOP/B` at `tokens=1` and `0.80 FLOP/B` at `tokens=4`, so the production decode kernel is memory-bandwidth bound. The relevant target is 350 GB/s effective read/write bandwidth rather than 50 TOPS.
- The production `tokens<=4` path skips weight SLM staging: one workgroup owns one expert row, reads that weight row once, accumulates up to four token logits in registers, and uses SLM only to fold subgroup partials. The default dispatch uses 32-lane subgroups on BMG; 16-lane subgroups remain available as an experiment switch. Larger token counts use the staged-weight fallback.
- bf16 is implemented because the production gate linear consumes bf16 activations and bf16 padded weights. fp16 is out of scope for this Inkling-specific gate GEMV until a production fp16 gate path exists.

Fused GEMV roofline and performance notes:

- The fused path is still dominated by the same bf16 GEMV work, `2 * tokens * 258 * 6144` FLOPs. It removes the split launch boundary but still uses a workspace because top-k needs all expert logits after all expert workgroups finish.
- Effective traffic is the GEMV traffic plus one fp32 workspace write and one fp32 workspace read for the 258 logical logits, plus the final packed or non-packed outputs. This remains memory-bandwidth bound for the supported `tokens <= 64` range.
- The fused epilogue reuses the standalone top-k row helper, so non-packed and packed modes share selection, tie-break, sigmoid, renorm, and store behavior.
- A short BMG sweep showed `--experts-per-wg=1` is the best default across `tokens=1..64`; `2` and `4` remain available for local experiments. The fused epilogue requires 32-lane subgroups.

Build and run inside the `sglang-syk` container:

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
cmake --build /workspace/cutlass-sycl/build-syk --target 16_bmg_moe_gate_topk_renorm -j
/workspace/cutlass-sycl/build-syk/examples/16_bmg_moe_gate/16_bmg_moe_gate_topk_renorm --suite=full --iterations=100
cmake --build /workspace/cutlass-sycl/build-syk --target 16_bmg_moe_gate_gemv -j
/workspace/cutlass-sycl/build-syk/examples/16_bmg_moe_gate/16_bmg_moe_gate_gemv --suite=full --iterations=100
cmake --build /workspace/cutlass-sycl/build-syk --target 16_bmg_moe_gate_gemv_fused -j
/workspace/cutlass-sycl/build-syk/examples/16_bmg_moe_gate/16_bmg_moe_gate_gemv_fused --suite=full --iterations=100
```

The top-k implementation is specialized to the Inkling fp32-logit gate. fp16/bf16 input support is out of scope because this kernel consumes the fp32 gate output; packed weights are bf16 only because that is the FlashInfer packed routed-MoE contract.
