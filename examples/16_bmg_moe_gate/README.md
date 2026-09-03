# Inkling MoE Gate Kernels

CUTLASS SYCL example implementations for the Inkling gate path.

Top-k renorm math contract:

- Input logits are fp32 rows with 258 logical columns: 256 routed experts and 2 shared experts.
- Ranking score for routed expert `e` is `act(logits[row, :])[e] + bias[e]`, where `act` is `sigmoid`
  (elementwise) for `gate_activation="sigmoid"` and `softmax(dim=-1)` over all 258 columns for
  `gate_activation="softmax"` (`InklingGate.forward`, `sglang/srt/models/inkling_common/moe.py`).
- Select top-6 routed experts. Exact ties select the lower expert id, matching `gate_topk`'s
  `indx_to_key(idx) = N_PAD - idx` packed into the sort key's low bits.
- With `norm_after_topk=true` the 6 selected plus the 2 shared columns are renormalized in the
  numerically stable form sglang uses, `exp(lp - logsumexp(lp))` with a max shift, where
  `lp = logsigmoid(logit) = min(logit, 0) - log1p(exp(-|logit|))` for the sigmoid gate and `lp = logit`
  for the softmax gate (`_logsigmoid_normalize` / `_inkling_compute_logsigmoid_norm` /
  `_renorm_topk_logits`). This is mathematically identical to the cheap ratio form
  `sigmoid(l_k) / sum_j sigmoid(l_j)` that the shipped fused CUDA/triton gate kernels use
  (`sglang/kernels/ops/moe/sigmoid_gate_topk_renorm.py`), but bit-comparable against the torch and
  triton reference paths.
- With `norm_after_topk=false` the weights are just the activated score gathered at the selected
  indices; sglang returns `shared_gammas=None` in that mode, so the two shared slots have no
  model-side counterpart and this example writes zeros there to keep the contract explicit.
- Weights are then multiplied by `route_scale` and, when `use_global_scale=true`, by `global_scale[0]`.
  `bias == nullptr` models `use_gate_bias=false` and `global_scale == nullptr` models
  `use_global_scale=false`.
- Non-packed output writes fp32 routed weights, int32 routed expert ids, and fp32 shared weights.
- Packed output writes int32 routed entries with high 16 bits as expert id and low 16 bits as bf16-rounded weight, plus fp32 shared weights.

Config coverage: `gate_activation` (`sigmoid`/`softmax`), `norm_after_topk`, `use_gate_bias`,
`use_global_scale` and `route_scale` are all runtime knobs. The suite tables cover both activations,
both `norm_after_topk` values, bias-absent, global-scale-absent, `route_scale=8.0` (shipped
checkpoint) and `route_scale=1.0` (`InklingModelConfig` default). The corresponding CLI overrides on
`16_bmg_moe_gate_topk_renorm` and `16_bmg_moe_gate_gemv_fused` are `--activation=sigmoid|softmax`,
`--norm-after-topk=0|1`, `--use-bias=0|1`, `--use-global-scale=0|1` and `--route-scale=<float>`.

Scope note: `InklingGate.forward` only takes its fused gate path when the gate is
`gate_activation="sigmoid"` **and** `norm_after_topk` **and** `use_gate_bias` **and**
`use_global_scale`; and outside sigmoid-with-global-scale (absent `shared_expert_sink`) it returns
`shared_gammas=None` even for `norm_after_topk=true`. The softmax, `norm_after_topk=false`,
bias-absent and global-scale-absent cases here therefore cover config combinations the model supports
but the production fused kernel never sees, which is deliberate: they pin this example's contract to
the torch/triton reference path rather than to the fused fast path alone.

Gate GEMV math contract:

- Input `x` is bf16 `[tokens, hidden]`, where `hidden` is a **runtime** argument
  (`GateGemvParams::hidden`, CLI `--hidden=<int>`). The gate's contraction width is
  `InklingModelConfig.hidden_size`: 1536 is the config default and 6144 the production checkpoint.
  6144 is also the width sglang's `_INKLING_GATE_GEMV_HIDDEN` GEMV shortcut is specialized for, so it
  is kept as a compile-time instantiation (`kGateHiddenSpecialized`, dispatched by
  `launch_gate_gemv_hidden`) while every other width runs the generic path with a runtime trip count.
  The suites also run a non-model width (1540) purely to exercise the generic path at a size that is
  not a multiple of the maximum warps-per-token contraction slice count.
- Input `weight` is bf16 `[>=258, hidden]`; rows `0..257` are the logical gate experts and rows `258..263` may be padded.
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
- Cost of the stable-form parity: the numerically stable `exp(lp - logsumexp(lp))` renorm runs ~24
  precise `exp`/`log`/`log1p` calls on lane 0 per row, where the cheap ratio form ran 8 divides. On B60
  that measures `prefill16k` at 0.151 ms / 118 GB/s versus 0.117 ms / 152 GB/s for the ratio form
  (1.29x). Two things were measured while closing that gap and are worth not re-litigating:
  carrying the selected raw logit through the top-k butterfly instead of re-reading the six selected
  logits on lane 0 costs a further 1.15x and spills ~25 registers in the fused kernel, and hoisting the
  `use_gate_bias` null test out of the per-lane loop is worth 1.12x. Switching the three math wrappers
  to `sycl::native::*` recovers only 0.005 ms and was rejected because it defeats the point of the
  stable form.

Gate GEMV roofline and performance notes:

- Per call, useful work is `2 * tokens * 258 * hidden` FLOPs.
- With the default 1 expert/workgroup, estimated global traffic is roughly `258*hidden*2` bytes of weight plus `258*tokens*hidden*2` bytes of repeated `x` reads plus fp32 output stores. Operation intensity is about `0.50 FLOP/B` at `tokens=1` and `0.80 FLOP/B` at `tokens=4` and is independent of `hidden`, so the production decode kernel is memory-bandwidth bound at every width. The relevant target is 350 GB/s effective read/write bandwidth rather than 50 TOPS. The narrower 1536 width moves proportionally less traffic per launch, so at `tokens<=4` it sits closer to the fixed launch overhead and reports lower effective GB/s.
- The production `tokens<=4` path skips weight SLM staging: one workgroup owns one expert row, reads that weight row once, accumulates up to four token logits in registers, and uses SLM only to fold subgroup partials. The default dispatch uses 32-lane subgroups on BMG; 16-lane subgroups remain available as an experiment switch. Larger token counts use the staged-weight fallback.
- bf16 is implemented because the production gate linear consumes bf16 activations and bf16 padded weights. fp16 is out of scope for this Inkling-specific gate GEMV until a production fp16 gate path exists.

Fused GEMV roofline and performance notes:

- The fused path is still dominated by the same bf16 GEMV work, `2 * tokens * 258 * hidden` FLOPs. It removes the split launch boundary but still uses a workspace because top-k needs all expert logits after all expert workgroups finish.
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
