#!/usr/bin/env bash
set -euo pipefail

binary="${1:-${PWD}/15_fmha_decode_kvcache}"
contexts="${GEMMA4_DECODE_CONTEXTS:-4097 4608 5120}"
# This shape's dispatch is ~26 us, close enough to the timer's noise floor that
# short runs swing the reported bandwidth by tens of GB/s and can rank two
# configurations backwards. 200/2000 is what the numbers in README.md use; even
# then, report medians of several runs and interleave the configurations when
# comparing.
warmup="${WARMUP:-200}"
iterations="${ITERATIONS:-2000}"

# num_kv_splits=0 uses the example's occupancy heuristic.
splits="${SPLITS:-0}"

for context in $contexts; do
  "$binary" \
    --seq_len_kv="$context" \
    --num_kv_splits="$splits" \
    --softmax_scale=1.0 \
    --warmup="$warmup" \
    --iterations="$iterations" \
    --verify=0
done
