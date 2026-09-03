# 23_bmg_mel_embedding_sum

Inkling audio mel-bin embedding lookup plus `sum(dim=1)` for CUTLASS SYCL on
BMG. The kernel keeps `n_mel_bins`, `mel_vocab_size`, `hidden`, token count, and
chunk size runtime-variable. RMSNorm is intentionally not fused because Inkling
already uses the existing XPU RMSNorm after this stage when `use_audio_norm` is
enabled.

## Model correspondence

The op mirrors `InklingAudio.forward`
(`sglang/python/sglang/srt/models/inkling.py:964-985`): the encoder is
`nn.Embedding(n_mel_bins * mel_vocab_size, decoder_dmodel)` (lines 957-959) and
the gather index is `arange(n_mel_bins) * mel_vocab_size + audio_features`
(lines 971-974), followed by `.sum(axis=1)` (line 979). This example uses the
same index expression, the same `int32` feature dtype, and accumulates the mel
axis in increasing order in FP32 before a single rounding to the output dtype,
matching PyTorch's FP32 accumulation for a bf16/fp16 reduction.

`n_mel_bins=80` and `mel_vocab_size=16` are the `InklingAudioConfig` defaults
(`sglang/python/sglang/srt/configs/inkling.py:263-264`) and agree with the
feature extractor's `n_mels=80` / `num_dmel_bins=16`
(`sglang/python/sglang/srt/multimodal/inkling/feature_extraction.py:25-26`),
whose `_dmel_bins()` returns an `int32 [T, n_mels]` tensor of bin indices in
`[0, num_dmel_bins)`.

Three `decoder_dmodel` widths are covered, all at the same chunk-boundary token
bands (`T` in `{1, 9, 511, 512, 513, 1025}`):

| Band prefix | `decoder_dmodel` | Source |
| --- | --- | --- |
| `ckpt_h768` | 768 | `thinkingmachines/Inkling` `config.json` `audio_config.decoder_dmodel` (HF snapshot `85b071f87d9bf5ff16a213a2d825faeed3af2cbf`) |
| `cfg_h1536` | 1536 | text `hidden_size` default |
| `prod_h6144` | 6144 | production `hidden_size` |

The audio tower is a plain `nn.Embedding` and is not tensor-parallel sharded, so
every rank runs the full lookup and reduction; one set of bands therefore covers
TP=1/2/4/8. `InklingAudioConfig.decoder_dmodel` defaults to `None`
(`configs/inkling.py:262`) and is always supplied by the checkpoint, so the 768
width is only visible in the checkpoint `config.json`, not in sglang's defaults.

At `decoder_dmodel=768` the auto `--channels-per-item` heuristic still selects
the vectorized 8-channel path even though it leaves lanes idle (632 GB/s versus
607 at 4, 475 at 2, 381 at 1), but only when `hidden % 8 == 0` (so the vec8 fast
path is actually reachable) and `tokens >= 64` (below that the single-work-group
launch shape loses to the narrower tiles).

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
use a separate 200 GB/s random-row DRAM stress floor. The `decoder_dmodel=768`
chunk-boundary bands (`perf_ckpt_h768_t1` .. `t1025`) are report-only (target
`0.0`) because at those token counts the per-launch floor dominates rather than
bandwidth; the large-`T` 768 bands share the 350 GB/s target and measure
469-632 GB/s (bf16) / 556-587 GB/s (fp16) on B60 (`t18432` is the fastest point
at 632/583, `t2048` the slowest bf16 point at 469). Do not tighten the shared
target past the `t2048` number.

Because the embedding table at `mel_vocab_size=16` is only
`80*16*decoder_dmodel*2 B` (1.9 MB at 768), it is L2-resident, so the reported
effective bandwidth legitimately exceeds the part's DRAM peak; the number is an
effective-traffic rate, not a DRAM rate.

Typical commands:

```bash
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cmake --build /workspace/cutlass-sycl/build-syk --target 23_bmg_mel_embedding_sum -j'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; /workspace/cutlass-sycl/build-syk/examples/23_bmg_mel_embedding_sum/23_bmg_mel_embedding_sum --suite=quick --iterations=5 --verify=1'
docker exec sglang-syk bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; /workspace/cutlass-sycl/build-syk/examples/23_bmg_mel_embedding_sum/23_bmg_mel_embedding_sum --suite=perf --dtype=bf16 --verify=0 --iterations=20'
```
