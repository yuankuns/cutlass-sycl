# 23_bmg_mel_embedding_sum

Inkling audio mel-bin embedding lookup plus `sum(dim=1)` for CUTLASS SYCL on
BMG. The kernel keeps `n_mel_bins`, `mel_vocab_size`, `hidden`, token count, and
chunk size runtime-variable. RMSNorm is intentionally not fused because Inkling
already uses the existing XPU RMSNorm after this stage when `use_audio_norm` is
enabled.

Roofline summary: production bf16/fp16 with `n_mel_bins=80` performs roughly 80
FP32 additions per output element while streaming 80 embedding elements and one
output element. This is about `80 / (80*2 + 2) = 0.49 FLOP/B`, so the useful
target is sustained memory bandwidth, not TOPS. The benchmark reports estimated
traffic including embedding reads, output writes, and amortized feature reads.
The `perf` suite includes both production `mel_vocab_size=16` cases and larger
`mel_vocab_size=256` cases so cache-hot table reuse is not the only bandwidth
signal.

Performance gates follow the workspace BMG guidance for memory-bound kernels:
production/config `mel_vocab_size=16` cases target 350 GB/s effective bandwidth.
The `mel_vocab_size=256` cases intentionally defeat cache-hot table reuse and
use a separate 200 GB/s random-row DRAM stress floor.

Typical commands:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cmake --build /workspace/cutlass-sycl/build-syk --target 23_bmg_mel_embedding_sum -j'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; /workspace/cutlass-sycl/build-syk/examples/23_bmg_mel_embedding_sum/23_bmg_mel_embedding_sum --suite=quick --iterations=5 --verify=1'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; /workspace/cutlass-sycl/build-syk/examples/23_bmg_mel_embedding_sum/23_bmg_mel_embedding_sum --suite=perf --dtype=bf16 --verify=0 --iterations=20'
```
