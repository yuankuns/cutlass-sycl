# FMHA Prefill KV Cache Tests

The test library is CTest-based and stays C++/SYCL-only.  It registers
model-named scenarios plus supplemental kernel-coverage cases against the
`fmha_prefill_kvcache` executable.

The model cases use scaled-down sequence lengths and head counts where needed;
they are regression shapes that preserve attention traits rather than full model
memory footprints.  Each test verifies output against the CPU reference and also
prints a profiling line.

CTest hides stdout for passing tests by default.  Use the shell sweep script when
you need accuracy and kernel time in normal output:

```bash
examples/14_fmha_prefill_kvcache/scripts/run_model_shapes.sh \
  /tmp/fmha_prefill_kvcache_build/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache
```

The script defaults to the 59-case tile-boundary accuracy baseline plus
large-seqlen performance cases, with 5 warmup launches and 5 measured prefill
launches per case.  Perf output includes `perf.model.*` model-like cases and
`perf.saturate.*` throughput-oriented cases with larger batch/head parallelism.
Use `--perf-only` to run only performance cases:

```bash
examples/14_fmha_prefill_kvcache/scripts/run_model_shapes.sh --perf-only \
  /tmp/fmha_prefill_kvcache_build/examples/14_fmha_prefill_kvcache/fmha_prefill_kvcache
```

## Run

```bash
ctest --test-dir /tmp/fmha_prefill_kvcache_build -N
ctest --test-dir /tmp/fmha_prefill_kvcache_build --output-on-failure -L main
ctest --test-dir /tmp/fmha_prefill_kvcache_build --output-on-failure -L stretch
ctest --test-dir /tmp/fmha_prefill_kvcache_build --output-on-failure -L coverage
ctest --test-dir /tmp/fmha_prefill_kvcache_build --output-on-failure -L boundary
```

Useful labels:

- `main`: Gemma-4-26B, Gemma-4-31B, Qwen3-32B, Qwen3-30B-A3B, Flux.2-dev
- `stretch`: the stretch-goal model list
- `coverage`: extra head-dim and feature coverage
- `boundary`: exact tile multiples, non-multiples, +/-1, +/-2, half-tile,
  page-tail, and multi-batch cases
- `relative`: relative-attention single/multi-Q-tile, Q/K tail, multi-batch,
  MQA/GQA, extent boundary, non-causal, and 4K production-length cases
- `paged`, `nonpaged`, `causal`, `noncausal`, `sink`

If `FMHA_STANDALONE_HEAD_DIMS` is restricted at configure time, tests requiring
missing head dimensions are skipped during CMake configuration.
