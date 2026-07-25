#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BIN="/tmp/fmha_prefill_kvcache_full_build/fmha_prefill_kvcache"

BIN="${FMHA_PREFILL_BIN:-$DEFAULT_BIN}"
WARMUP="${FMHA_PREFILL_WARMUP:-5}"
ITERS="${FMHA_PREFILL_ITERS:-5}"
ATOL="${FMHA_PREFILL_ATOL:-0.08}"
RTOL="${FMHA_PREFILL_RTOL:-0.08}"
FILTER="${FMHA_PREFILL_FILTER:-}"
SUITE="${FMHA_PREFILL_SUITE:-tile,chunk,perf}"
PAGED_HD128_TILE_Q="${FMHA_PREFILL_PAGED_HD128_TILE_Q:-128}"
PAGED_HD128_TILE_KV="${FMHA_PREFILL_PAGED_HD128_TILE_KV:-64}"
PAGED_HD192_TILE_Q="${FMHA_PREFILL_PAGED_HD192_TILE_Q:-128}"
PAGED_HD256_TILE_Q="${FMHA_PREFILL_PAGED_HD256_TILE_Q:-128}"
NP_HD96_SMALL_MAX_Q="${FMHA_PREFILL_NP_HD96_SMALL_MAX_Q:-32}"
NP_HD96_TILE_Q="${FMHA_PREFILL_NP_HD96_TILE_Q:-128}"
NP_HD96_LARGE_MIN_Q="${FMHA_PREFILL_NP_HD96_LARGE_MIN_Q:-512}"

usage() {
  cat <<EOF
Usage: $0 [--perf-only] [--suite NAME[,NAME...]] [fmha_prefill_kvcache_binary]

Default suite: tile,chunk,perf
Common examples:
  $0 /path/to/fmha_prefill_kvcache
  $0 --perf-only /path/to/fmha_prefill_kvcache
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf-only)
      SUITE="perf"
      shift
      ;;
    --suite)
      if [[ $# -lt 2 ]]; then
        echo "error: --suite requires a value" >&2
        exit 2
      fi
      SUITE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      BIN="$1"
      shift
      ;;
  esac
done

if [[ ! -x "$BIN" ]]; then
  echo "error: executable not found or not executable: $BIN" >&2
  echo "build first, or set FMHA_PREFILL_BIN=/path/to/fmha_prefill_kvcache" >&2
  exit 2
fi

ACCURACY_ARGS=(
  --warmup "$WARMUP"
  --iters "$ITERS"
  --verify 1
  --atol "$ATOL"
  --rtol "$RTOL"
)

PERF_ARGS=(
  --warmup "$WARMUP"
  --iters "$ITERS"
  --verify 0
  --atol "$ATOL"
  --rtol "$RTOL"
)

total=0
passed=0
failed=0

suite_enabled() {
  local section="$1"
  case ",$SUITE," in
    *,all,*|*,"$section",*) return 0 ;;
    *,model,*)
      if [[ "$section" == "main" || "$section" == "stretch" ]]; then
        return 0
      fi
      ;;
  esac
  return 1
}

run_case() {
  local name="$1"
  shift

  if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
    return 0
  fi

  total=$((total + 1))
  echo "== $name =="

  local output
  output="$("$BIN" "$@" "${ACCURACY_ARGS[@]}" 2>&1)"
  local status=$?

  if [[ $status -eq 0 ]]; then
    echo "$output" | awk '/^device:|^shape:|^verify:|^profile:/ {print}'
    passed=$((passed + 1))
  else
    echo "$output"
    echo "FAILED: $name (exit=$status)" >&2
    failed=$((failed + 1))
  fi
  echo
}

run_perf_case() {
  local name="$1"
  shift

  if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
    return 0
  fi

  total=$((total + 1))
  echo "== $name =="

  local output
  output="$("$BIN" "$@" "${PERF_ARGS[@]}" 2>&1)"
  local status=$?

  if [[ $status -eq 0 ]]; then
    echo "$output" | awk '/^device:|^shape:/ {print}'
    echo "verify: skipped (large performance case; covered by tile accuracy baseline)"
    echo "$output" | awk '/^profile:/ {print}'
    passed=$((passed + 1))
  else
    echo "$output"
    echo "FAILED: $name (exit=$status)" >&2
    failed=$((failed + 1))
  fi
  echo
}

run_tile_boundary_family() {
  local prefix="$1"
  local paged="$2"
  local page_size="$3"
  local batch="$4"
  local heads_q="$5"
  local heads_kv="$6"
  local head_dim="$7"
  local head_dim_v="$8"
  local q_tile="$9"
  local kv_tile="${10}"

  local q_half=$((q_tile / 2))
  local q_1p5=$(((q_tile * 3) / 2))
  local kv_half=$((kv_tile / 2))
  local base_k=$((q_tile * 2))
  local base_q="$q_half"

  local common=(
    --batch "$batch"
    --heads-q "$heads_q"
    --heads-kv "$heads_kv"
    --head-dim "$head_dim"
    --head-dim-v "$head_dim_v"
    --paged "$paged"
    --page-size "$page_size"
  )

  run_case "$prefix.q_half_minus1" \
    "${common[@]}" --seqlen-q "$((q_half - 1))" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_half" \
    "${common[@]}" --seqlen-q "$q_half" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_half_plus1" \
    "${common[@]}" --seqlen-q "$((q_half + 1))" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_tile_minus2" \
    "${common[@]}" --seqlen-q "$((q_tile - 2))" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_tile_minus1" \
    "${common[@]}" --seqlen-q "$((q_tile - 1))" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_tile" \
    "${common[@]}" --seqlen-q "$q_tile" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_tile_plus1" \
    "${common[@]}" --seqlen-q "$((q_tile + 1))" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_tile_plus2" \
    "${common[@]}" --seqlen-q "$((q_tile + 2))" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_1p5tile" \
    "${common[@]}" --seqlen-q "$q_1p5" --seqlen-k "$base_k" --causal 1
  run_case "$prefix.q_1p5tile_plus1" \
    "${common[@]}" --seqlen-q "$((q_1p5 + 1))" --seqlen-k "$base_k" --causal 1

  run_case "$prefix.k_half_minus1" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_half - 1))" --causal 0
  run_case "$prefix.k_half" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$kv_half" --causal 0
  run_case "$prefix.k_half_plus1" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_half + 1))" --causal 0
  run_case "$prefix.k_tile_minus2" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile - 2))" --causal 0
  run_case "$prefix.k_tile_minus1" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile - 1))" --causal 0
  run_case "$prefix.k_tile" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$kv_tile" --causal 0
  run_case "$prefix.k_tile_plus1" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile + 1))" --causal 0
  run_case "$prefix.k_tile_plus2" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile + 2))" --causal 0
  run_case "$prefix.k_1p5tile" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$(((kv_tile * 3) / 2))" --causal 0
  run_case "$prefix.k_1p5tile_plus1" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$(((kv_tile * 3) / 2 + 1))" --causal 0
  run_case "$prefix.k_2tile_minus2" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile * 2 - 2))" --causal 0
  run_case "$prefix.k_2tile_minus1" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile * 2 - 1))" --causal 0
  run_case "$prefix.k_2tile" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile * 2))" --causal 0
  run_case "$prefix.k_2tile_plus1" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile * 2 + 1))" --causal 0
  run_case "$prefix.k_2tile_plus2" \
    "${common[@]}" --seqlen-q "$base_q" --seqlen-k "$((kv_tile * 2 + 2))" --causal 0
}

run_batch_boundary_cases() {
  run_case "tile.batch2.paged_hd128_q_tile_minus1_k_4tile_minus1" \
    --batch 2 --seqlen-q 127 --seqlen-k 255 --heads-q 4 --heads-kv 1 \
    --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

  run_case "tile.batch3.paged_hd128_q_tile_plus1_k_4tile_plus1" \
    --batch 3 --seqlen-q 129 --seqlen-k 257 --heads-q 4 --heads-kv 1 \
    --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

  run_case "tile.batch4.nonpaged_hd128_q_half_plus1_k_2tile_plus1" \
    --batch 4 --seqlen-q 129 --seqlen-k 65 --heads-q 2 --heads-kv 1 \
    --head-dim 128 --head-dim-v 128 --paged 0 --causal 0
}

run_chunk_family() {
  local prefix="$1"
  local paged="$2"
  local page_size="$3"
  local head_dim="$4"
  local heads_q="$5"
  local heads_kv="$6"
  local tile_q="$7"

  local tile_minus1=$((tile_q - 1))
  local tile_plus1=$((tile_q + 1))
  local past_values=(0 64 129 "$tile_minus1" "$tile_q" "$tile_plus1")
  local q_values=(1 32 128 "$tile_minus1")
  local seen_past=","
  local seen_q=","
  local unique_past=()
  local unique_q=()

  local value
  for value in "${past_values[@]}"; do
    if [[ "$seen_past" != *",$value,"* ]]; then
      unique_past+=("$value")
      seen_past+="$value,"
    fi
  done
  for value in "${q_values[@]}"; do
    if [[ "$seen_q" != *",$value,"* ]]; then
      unique_q+=("$value")
      seen_q+="$value,"
    fi
  done

  local common=(
    --batch 1
    --heads-q "$heads_q"
    --heads-kv "$heads_kv"
    --head-dim "$head_dim"
    --head-dim-v "$head_dim"
    --paged "$paged"
    --page-size "$page_size"
    --causal 1
  )

  local past q_len k_len
  for past in "${unique_past[@]}"; do
    for q_len in "${unique_q[@]}"; do
      k_len=$((past + q_len))
      run_case "chunk.${prefix}.past${past}_q${q_len}" \
        "${common[@]}" --seqlen-q "$q_len" --seqlen-k "$k_len" --past-kv "$past"
    done
  done
}

# Main goal
if suite_enabled main; then
run_case "main.gemma4_26b" \
  --batch 1 --seqlen-q 32 --seqlen-k 128 --heads-q 16 --heads-kv 8 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "main.gemma4_31b" \
  --batch 1 --seqlen-q 24 --seqlen-k 128 --heads-q 16 --heads-kv 4 \
  --head-dim 256 --head-dim-v 256 --paged 1 --page-size 64 --causal 1

run_case "main.qwen3_32b" \
  --batch 1 --seqlen-q 32 --seqlen-k 192 --heads-q 16 --heads-kv 4 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "main.qwen3_30b_a3b" \
  --batch 1 --seqlen-q 32 --seqlen-k 128 --heads-q 16 --heads-kv 2 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "main.flux2_dev" \
  --batch 1 --seqlen-q 32 --seqlen-k 32 --heads-q 8 --heads-kv 8 \
  --head-dim 128 --head-dim-v 128 --paged 0 --causal 0
fi

# Stretch goal
if suite_enabled stretch; then
run_case "stretch.qwen35_9b" \
  --batch 1 --seqlen-q 24 --seqlen-k 128 --heads-q 8 --heads-kv 2 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "stretch.qwen35_35b_a3b" \
  --batch 1 --seqlen-q 24 --seqlen-k 192 --heads-q 16 --heads-kv 2 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 \
  --causal 0 --window-left 96 --window-right 0

run_case "stretch.deepseek_ocr2" \
  --batch 1 --seqlen-q 32 --seqlen-k 64 --heads-q 8 --heads-kv 8 \
  --head-dim 96 --head-dim-v 96 --paged 0 \
  --causal 0 --window-left 48 --window-right 16

run_case "stretch.nemotron3_nano_30b_a3b" \
  --batch 1 --seqlen-q 32 --seqlen-k 128 --heads-q 16 --heads-kv 4 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "stretch.flux2_klein_4b" \
  --batch 1 --seqlen-q 32 --seqlen-k 32 --heads-q 8 --heads-kv 8 \
  --head-dim 64 --head-dim-v 64 --paged 0 --causal 0

run_case "stretch.flux2_klein_9b" \
  --batch 1 --seqlen-q 32 --seqlen-k 64 --heads-q 12 --heads-kv 12 \
  --head-dim 96 --head-dim-v 96 --paged 0 --causal 0

run_case "stretch.qwen3_tts" \
  --batch 1 --seqlen-q 24 --seqlen-k 128 --heads-q 8 --heads-kv 2 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "stretch.qwen36_35b_a3b" \
  --batch 1 --seqlen-q 32 --seqlen-k 256 --heads-q 16 --heads-kv 2 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_case "stretch.z_image_turbo" \
  --batch 1 --seqlen-q 32 --seqlen-k 32 --heads-q 8 --heads-kv 8 \
  --head-dim 64 --head-dim-v 64 --paged 0 --causal 0
fi

# Supplemental coverage
if suite_enabled coverage; then
run_case "coverage.paged_hd192" \
  --batch 1 --seqlen-q 16 --seqlen-k 128 --heads-q 4 --heads-kv 1 \
  --head-dim 192 --head-dim-v 192 --paged 1 --page-size 64 --causal 1

run_case "coverage.paged_hd512_mqa" \
  --batch 1 --seqlen-q 8 --seqlen-k 64 --heads-q 4 --heads-kv 1 \
  --head-dim 512 --head-dim-v 512 --paged 1 --page-size 64 --causal 1

run_case "coverage.nonpaged_hd72" \
  --batch 1 --seqlen-q 16 --seqlen-k 64 --heads-q 4 --heads-kv 2 \
  --head-dim 72 --head-dim-v 72 --paged 0 --causal 1

run_case "coverage.dv128_from_d64" \
  --batch 1 --seqlen-q 16 --seqlen-k 64 --heads-q 4 --heads-kv 1 \
  --head-dim 64 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "coverage.sink_paged_hd64" \
  --batch 1 --seqlen-q 16 --seqlen-k 64 --heads-q 4 --heads-kv 1 \
  --head-dim 64 --head-dim-v 64 --paged 1 --page-size 64 --causal 1 --sink 1

run_case "coverage.local_window_paged" \
  --batch 1 --seqlen-q 24 --seqlen-k 128 --heads-q 4 --heads-kv 1 \
  --head-dim 96 --head-dim-v 96 --paged 1 --page-size 64 \
  --causal 0 --window-left 64 --window-right 8
fi

# Tile-boundary sweep.  These cases intentionally vary one axis around the
# underlying Q/KV tile size: exact multiples, non-multiples, +/-1, +/-2,
# half-tile, and half-tile+1 tails.
if suite_enabled tile; then
run_tile_boundary_family "tile.paged_hd128_q${PAGED_HD128_TILE_Q}_k${PAGED_HD128_TILE_KV}" \
  1 64 1 4 1 128 128 "$PAGED_HD128_TILE_Q" "$PAGED_HD128_TILE_KV"

run_tile_boundary_family "tile.nonpaged_hd128_q256_k32" \
  0 64 1 2 1 128 128 256 32

run_case "tile.paged_hd192_q${PAGED_HD192_TILE_Q}_minus1" \
  --batch 1 --seqlen-q "$((PAGED_HD192_TILE_Q - 1))" --seqlen-k 512 --heads-q 2 --heads-kv 1 \
  --head-dim 192 --head-dim-v 192 --paged 1 --page-size 64 --causal 1
run_case "tile.paged_hd192_q${PAGED_HD192_TILE_Q}" \
  --batch 1 --seqlen-q "$PAGED_HD192_TILE_Q" --seqlen-k 512 --heads-q 2 --heads-kv 1 \
  --head-dim 192 --head-dim-v 192 --paged 1 --page-size 64 --causal 1
run_case "tile.paged_hd192_q${PAGED_HD192_TILE_Q}_plus1" \
  --batch 1 --seqlen-q "$((PAGED_HD192_TILE_Q + 1))" --seqlen-k 512 --heads-q 2 --heads-kv 1 \
  --head-dim 192 --head-dim-v 192 --paged 1 --page-size 64 --causal 1

run_case "tile.paged_hd256_q${PAGED_HD256_TILE_Q}_minus1" \
  --batch 1 --seqlen-q "$((PAGED_HD256_TILE_Q - 1))" --seqlen-k 512 --heads-q 2 --heads-kv 1 \
  --head-dim 256 --head-dim-v 256 --paged 1 --page-size 64 --causal 1
run_case "tile.paged_hd256_q${PAGED_HD256_TILE_Q}" \
  --batch 1 --seqlen-q "$PAGED_HD256_TILE_Q" --seqlen-k 512 --heads-q 2 --heads-kv 1 \
  --head-dim 256 --head-dim-v 256 --paged 1 --page-size 64 --causal 1
run_case "tile.paged_hd256_q${PAGED_HD256_TILE_Q}_plus1" \
  --batch 1 --seqlen-q "$((PAGED_HD256_TILE_Q + 1))" --seqlen-k 512 --heads-q 2 --heads-kv 1 \
  --head-dim 256 --head-dim-v 256 --paged 1 --page-size 64 --causal 1

for q_len in \
  "$((NP_HD96_SMALL_MAX_Q - 1))" "$NP_HD96_SMALL_MAX_Q" "$((NP_HD96_SMALL_MAX_Q + 1))" \
  "$((NP_HD96_TILE_Q - 1))" "$NP_HD96_TILE_Q" "$((NP_HD96_TILE_Q + 1))" \
  "$((NP_HD96_LARGE_MIN_Q - 1))" "$NP_HD96_LARGE_MIN_Q" "$((NP_HD96_LARGE_MIN_Q + 1))"; do
  run_case "tile.nonpaged_hd96_q${q_len}" \
    --batch 1 --seqlen-q "$q_len" --seqlen-k 64 --heads-q 2 --heads-kv 1 \
    --head-dim 96 --head-dim-v 96 --paged 0 --causal 0
done

for q_len in 127 128 129; do
  run_case "tile.paged_hd96_q${q_len}" \
    --batch 1 --seqlen-q "$q_len" --seqlen-k 256 --heads-q 2 --heads-kv 1 \
    --head-dim 96 --head-dim-v 96 --paged 1 --page-size 64 --causal 1
done

run_case "tile.page128_k_page_minus1" \
  --batch 1 --seqlen-q 64 --seqlen-k 127 --heads-q 4 --heads-kv 1 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 0
run_case "tile.page128_k_page" \
  --batch 1 --seqlen-q 64 --seqlen-k 128 --heads-q 4 --heads-kv 1 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 0
run_case "tile.page128_k_page_plus1" \
  --batch 1 --seqlen-q 64 --seqlen-k 129 --heads-q 4 --heads-kv 1 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 0

run_batch_boundary_cases
fi

# Chunk-prefill sweep. These cases keep CPU reference sizes modest while
# covering paged/non-paged cache, multiple head dimensions, and heterogeneous
# per-batch Q/KV lengths.
if suite_enabled chunk; then
run_chunk_family "paged_hd64" 1 64 64 4 1 128
run_chunk_family "paged_hd128" 1 64 128 4 1 "$PAGED_HD128_TILE_Q"
run_chunk_family "paged_hd256" 1 64 256 2 1 "$PAGED_HD256_TILE_Q"
run_chunk_family "nonpaged_hd128" 0 64 128 2 1 256

run_case "chunk.hetero_paged_hd128_lists" \
  --batch 3 --seqlen-q 128 --seqlen-k 257 --seqlen-q-list 1,32,127 --past-kv-list 0,64,130 \
  --heads-q 4 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "chunk.hetero_nonpaged_hd128_lists" \
  --batch 3 --seqlen-q 128 --seqlen-k 257 --seqlen-q-list 1,32,255 --past-kv-list 0,64,130 \
  --heads-q 2 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 0 --causal 1

# Generalized chunk-prefill coverage derived from vLLM/SGLang test patterns:
# mixed query/context lengths, decode-like q=1, page/block tails, GQA/MQA,
# local windows, and multi-request batches.
run_case "chunk.generalized_paged_batch2_q1024_k1024" \
  --batch 2 --seqlen-q 1024 --seqlen-k 1024 --seqlen-q-list 1024,1024 --past-kv-list 0,0 \
  --heads-q 2 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_case "chunk.generalized_paged_mixed_decode_extend" \
  --batch 6 --seqlen-q 255 --seqlen-k 1024 --seqlen-q-list 1,16,33,64,128,255 \
  --past-kv-list 1023,17,64,0,129,257 \
  --heads-q 4 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "chunk.generalized_paged_page128_tails" \
  --batch 4 --seqlen-q 129 --seqlen-k 384 --seqlen-q-list 1,31,127,129 \
  --past-kv-list 127,128,129,255 \
  --heads-q 4 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_case "chunk.generalized_paged_random_page_table" \
  --batch 4 --seqlen-q 96 --seqlen-k 320 --seqlen-q-list 1,32,64,96 \
  --past-kv-list 63,64,129,224 \
  --heads-q 4 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 \
  --page-table-random 1 --causal 1

run_case "chunk.generalized_paged_gqa_mixed_prefix" \
  --batch 4 --seqlen-q 128 --seqlen-k 320 --seqlen-q-list 17,64,96,128 \
  --past-kv-list 15,96,127,192 \
  --heads-q 8 --heads-kv 2 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "chunk.generalized_paged_sglang_prefix_lens" \
  --batch 4 --seqlen-q 64 --seqlen-k 128 --seqlen-q-list 64,64,64,64 \
  --past-kv-list 16,32,48,64 \
  --heads-q 4 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_case "chunk.generalized_paged_local_window_offsets" \
  --batch 3 --seqlen-q 10 --seqlen-k 17 --seqlen-q-list 4,10,5 --past-kv-list 2,7,4 \
  --heads-q 4 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 \
  --causal 0 --window-left 4 --window-right 0

run_case "chunk.generalized_nonpaged_mixed_prefix" \
  --batch 4 --seqlen-q 129 --seqlen-k 193 --seqlen-q-list 1,17,64,129 \
  --past-kv-list 16,32,48,64 \
  --heads-q 2 --heads-kv 1 --head-dim 128 --head-dim-v 128 --paged 0 --causal 1
fi

# Large-seqlen performance checks.  These skip the CPU reference intentionally:
# the tile suite is the accuracy baseline, while these cases measure steady
# kernel latency with warmup and five measured launches by default.
if suite_enabled perf; then
run_perf_case "perf.model.gemma4_26b.paged_sq512_sk4096" \
  --batch 1 --seqlen-q 512 --seqlen-k 4096 --heads-q 16 --heads-kv 8 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_perf_case "perf.model.gemma4_31b.paged_hd256_sq512_sk4096" \
  --batch 1 --seqlen-q 512 --seqlen-k 4096 --heads-q 16 --heads-kv 4 \
  --head-dim 256 --head-dim-v 256 --paged 1 --page-size 64 --causal 1

run_perf_case "perf.model.qwen3_32b.paged_sq1024_sk8192" \
  --batch 1 --seqlen-q 1024 --seqlen-k 8192 --heads-q 16 --heads-kv 4 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.model.qwen3_30b_a3b.paged_tail_sq512_sk8193" \
  --batch 1 --seqlen-q 512 --seqlen-k 8193 --heads-q 16 --heads-kv 2 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_perf_case "perf.model.flux2_dev.nonpaged_sq1024_sk1024" \
  --batch 1 --seqlen-q 1024 --seqlen-k 1024 --heads-q 8 --heads-kv 8 \
  --head-dim 128 --head-dim-v 128 --paged 0 --causal 0

run_perf_case "perf.model.deepseek_ocr2.nonpaged_hd96_sq32_sk64" \
  --batch 1 --seqlen-q 32 --seqlen-k 64 --heads-q 8 --heads-kv 8 \
  --head-dim 96 --head-dim-v 96 --paged 0 --causal 0 --window-left 48 --window-right 16

run_perf_case "perf.model.flux2_klein_9b.nonpaged_hd96_sq32_sk64" \
  --batch 1 --seqlen-q 32 --seqlen-k 64 --heads-q 12 --heads-kv 12 \
  --head-dim 96 --head-dim-v 96 --paged 0 --causal 0

run_perf_case "perf.saturate.batch4_h32_hd128_sq1024_sk8192" \
  --batch 4 --seqlen-q 1024 --seqlen-k 8192 --heads-q 32 --heads-kv 8 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.saturate.batch1_h32_hkv1_hd128_sq4096_sk8192" \
  --batch 1 --seqlen-q 4096 --seqlen-k 8192 --heads-q 32 --heads-kv 1 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.saturate.batch1_h64_hkv1_hd64_sq4096_sk8192" \
  --batch 1 --seqlen-q 4096 --seqlen-k 8192 --heads-q 64 --heads-kv 1 \
  --head-dim 64 --head-dim-v 64 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.saturate.batch2_h32_hd128_sq4096_sk8192" \
  --batch 2 --seqlen-q 4096 --seqlen-k 8192 --heads-q 32 --heads-kv 8 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.saturate.batch2_tail_h8_hd128_sq512_sk4097" \
  --batch 2 --seqlen-q 512 --seqlen-k 4097 --heads-q 8 --heads-kv 2 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 64 --causal 1

run_perf_case "perf.chunk.qwen3_32b.paged_sq512_past7680_sk8192" \
  --batch 1 --seqlen-q 512 --seqlen-k 8192 --past-kv 7680 --heads-q 16 --heads-kv 4 \
  --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

# Decode-like means q_len=1 with a long KV cache, not kv_len=1.
run_perf_case "perf.chunk.mixed_decode_prefill.paged_q1_q512_past8191_7680_sk8192" \
  --batch 2 --seqlen-q 512 --seqlen-k 8192 --seqlen-q-list 1,512 --past-kv-list 8191,7680 \
  --heads-q 16 --heads-kv 4 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1
fi

if suite_enabled perf || suite_enabled append; then
run_perf_case "perf.append.paged.hd64_sq512_old8191_new1_sk8192" \
  --batch 1 --seqlen-q 512 --seqlen-k 8192 --cache-seqlens-old 8191 --k-new-seqlens 1 \
  --heads-q 16 --heads-kv 4 --head-dim 64 --head-dim-v 64 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.append.paged.hd128_sq512_old8191_new1_sk8192" \
  --batch 1 --seqlen-q 512 --seqlen-k 8192 --cache-seqlens-old 8191 --k-new-seqlens 1 \
  --heads-q 16 --heads-kv 4 --head-dim 128 --head-dim-v 128 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.append.paged.hd256_sq512_old8191_new1_sk8192" \
  --batch 1 --seqlen-q 512 --seqlen-k 8192 --cache-seqlens-old 8191 --k-new-seqlens 1 \
  --heads-q 16 --heads-kv 4 --head-dim 256 --head-dim-v 256 --paged 1 --page-size 128 --causal 1

run_perf_case "perf.append.nopaged.hd128_sq512_old8191_new1_sk8192" \
  --batch 1 --seqlen-q 512 --seqlen-k 8192 --cache-seqlens-old 8191 --k-new-seqlens 1 \
  --heads-q 16 --heads-kv 4 --head-dim 128 --head-dim-v 128 --paged 0 --causal 1
fi

echo "summary: passed=$passed failed=$failed total=$total"
if [[ $failed -ne 0 ]]; then
  exit 1
fi
