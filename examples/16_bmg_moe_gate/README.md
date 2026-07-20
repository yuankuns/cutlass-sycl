# Inkling MoE Gate Top-K Renorm

CUTLASS SYCL example implementation for Inkling gate top-k renorm.

Math contract:

- Input logits are fp32 rows with 258 logical columns: 256 routed experts and 2 shared experts.
- Ranking score for routed expert `e` is `sigmoid(logits[row, e]) + bias[e]`.
- Select top-6 routed experts. Exact ties select the lower expert id.
- Normalize the 6 selected routed sigmoid weights plus both shared sigmoid weights by `route_scale * global_scale[0]`.
- Non-packed output writes fp32 routed weights, int32 routed expert ids, and fp32 shared weights.
- Packed output writes int32 routed entries with high 16 bits as expert id and low 16 bits as bf16-rounded weight, plus fp32 shared weights.

Roofline and performance notes:

- Per token, the large-stream payload is 258 fp32 logits read plus 56 B of non-packed output or 32 B of packed output. Bias is 1 KiB and is reused by all rows, so it is effectively cache-resident for production token counts.
- This is the MoE gate postprocess, not the expert GEMM. It performs 258 sigmoid operations plus top-6 subgroup reductions per token, so a plain FLOP/byte roofline is misleading: `exp` throughput and subgroup shuffles are a significant part of the limit.
- The practical optimization target is reducing redundant logits traffic, keeping occupancy high enough to hide sigmoid latency, and avoiding extra subgroup communication. The kernel uses one 32-lane subgroup per token row, loads each routed logit once, keeps the selected sigmoid values in registers on lane 0 for the compact epilogue, and shares the same code path for packed and non-packed modes.
- The default launch policy is mode-aware: non-packed uses one subgroup per workgroup, while packed uses two for smaller token counts and one for larger token counts. Explicit `--rows-per-wg=1|2|4|8` remains available for local tuning.

Build and run inside the `sglang-syk` container:

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
cmake --build /workspace/cutlass-sycl/build-syk --target 16_bmg_moe_gate_topk_renorm -j
/workspace/cutlass-sycl/build-syk/examples/16_bmg_moe_gate/16_bmg_moe_gate_topk_renorm --suite=full --iterations=100
```

The implementation is specialized to the Inkling fp32-logit gate. fp16/bf16 input support is out of scope because this kernel consumes the fp32 gate output; packed weights are bf16 only because that is the FlashInfer packed routed-MoE contract.
