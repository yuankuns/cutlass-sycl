# FMHA Prefill With KV Cache Standalone

This directory builds a C++/SYCL-only executable for the extracted SGL XPU FMHA
prefill-with-KV-cache kernel through the sycl-tla examples CMake flow. It does not
use Torch, Python, or the installed `sgl_kernel` extension.

The CMake file generates the same split paged, non-paged, and FP8 prefill
translation units from `src/sycl/kernels/flash_attention_v2/*.cpp.in`. The FMHA
kernel headers needed by this example live under this directory, so the target
does not depend on an external SGL checkout.

## Build

```bash
cmake -S . -B build/fmha_prefill_kvcache \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DDPCPP_SYCL_TARGET=bmg
cmake --build build/fmha_prefill_kvcache --target fmha_prefill_kvcache -j
```

To build only selected head dimensions while iterating:

```bash
cmake -S . -B build/fmha_prefill_kvcache \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DDPCPP_SYCL_TARGET=bmg \
  -DFMHA_STANDALONE_HEAD_DIMS=64
cmake --build build/fmha_prefill_kvcache --target fmha_prefill_kvcache -j
```

Paged `head_dim=128` keeps the q128/k64 base tile for smaller requests and,
by default, enables a q256/k32 path for model-sized requests with
`seqlen_q >= 512`.  The large-shape path can be disabled with
`-DFMHA_STANDALONE_PAGED_HD128_LARGE_TILE=OFF`; its threshold and tile sizes are
exposed as `FMHA_STANDALONE_PAGED_HD128_LARGE_TILE_*` CMake cache variables.
Paged `head_dim=192` defaults to a q128/k64 tile with 16 subgroups, matching
common `seqlen_q=128` prefill requests without a half-empty q256 work-group.
Paged `head_dim=256` defaults to a q128/k64 tile with 16 subgroups so common
`seqlen_q=128` prefill requests fill the Q tile instead of launching a half-full
q256 work-group.

Tests are enabled by default.  They are registered through CTest and remain
C++/SYCL-only:

```bash
ctest --test-dir build/fmha_prefill_kvcache -N
ctest --test-dir build/fmha_prefill_kvcache --output-on-failure -L main
ctest --test-dir build/fmha_prefill_kvcache --output-on-failure -L coverage
```

The test library lives in `tests/model_cases.cmake`.  It contains model-named
regression shapes for Gemma, Qwen, Flux, DeepSeek-OCR, Nemotron, TTS, and image
workloads, plus supplemental coverage for sink, local masking, MQA/GQA,
`head_dim_v != head_dim`, and all standalone head-dim families.

For accuracy plus kernel-time output, run the executable directly through the
shape sweep script instead of CTest:

```bash
examples/14_fmha_prefill_kvcache/scripts/run_model_shapes.sh \
  build/fmha_prefill_kvcache/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache
```

The script prints `verify:` and `profile:` for every shape.  `profile:` includes
`kernel_avg_ms` from SYCL event profiling and `host_avg_ms` around the launch
loop.  By default it runs the tile-boundary accuracy baseline plus large-seqlen
performance cases, with 5 warmup launches and 5 measured prefill launches per
case:

```bash
examples/14_fmha_prefill_kvcache/scripts/run_model_shapes.sh \
  build/fmha_prefill_kvcache/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache
```

The `tile` suite sweeps query and KV lengths around the kernel tile boundaries:
exact multiples, non-multiples, `+/-1`, `+/-2`, half tile, half tile + 1, page
tails, and multi-batch offsets.  The `perf` suite uses larger sequence lengths
and skips the CPU reference so kernel timing is not dominated by host work.  It
contains both `perf.model.*` cases that preserve model-like head layouts and
`perf.saturate.*` cases that intentionally increase `batch * heads` and
sequence length to expose BMG throughput limits.

To run only the performance cases:

```bash
examples/14_fmha_prefill_kvcache/scripts/run_model_shapes.sh --perf-only \
  build/fmha_prefill_kvcache/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache
```

`--suite` and `FMHA_PREFILL_SUITE` remain available as filters when needed;
accepted values include `tile`, `perf`, `model`, and `all`.

## Run

```bash
./build/fmha_prefill_kvcache/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache \
  --batch 4 --seqlen-q 128 --seqlen-k 1024 \
  --heads-q 16 --heads-kv 4 --head-dim 64 --head-dim-v 64 \
  --paged 1 --page-size 64 --causal 1 \
  --warmup 10 --iters 100 --verify 1
```

The program prints:

- shape and mode
- correctness summary: max absolute error, max relative error, bad element count
- profiling summary: average device kernel latency, host-side launch-loop
  latency, and estimated dense-attention TFLOP/s

## Current Scope

- dtype: bf16 inputs/outputs, CPU float reference
- paged KV head dims: 64, 96, 128, 192, 256, 512
- non-paged KV head dims: 64, 72, 80, 96, 128, 192
- optional causal/local masking
- optional softmax sink for the paged head_dim=64 path
- boundary sweeps for Q/KV tile tails and page-size tails
- large-seqlen performance runs with CPU reference disabled

This is intentionally a kernel extraction harness.  It does not exercise fused KV-cache update,
RoPE, qv, descale tensors, or decode split-KV paths.
