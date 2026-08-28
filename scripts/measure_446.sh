#!/usr/bin/env bash
# Reproduce PR 446's claims and the accuracy gate for example 16.
#
#   scripts/measure_446.sh acc     -- accuracy: 16 gpt-oss workloads + INT4 cases
#   scripts/measure_446.sh table   -- PR 446's avg_m sweep, before vs after
#   scripts/measure_446.sh wl      -- the example's gpt-oss workloads, before vs after
#
# Run inside the BMG container from the worktree root.
set -u

BIN=${BIN:-./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm}
# The GPU ramps 1200 -> 2400 MHz over ~2 s (act_freq under
# /sys/class/drm/cardN/device/tile0/gt0/freq0), so a short run measures the
# ramp, not the kernel: the fastest shape here reads 50 TFLOP/s over 30
# iterations and 57.8 over 200. Keep every shape above ~2 s of device time --
# upstream's do_bench(warmup=50, rep=200) counts *milliseconds*, not
# iterations, and so is never this short.
WARMUP=${WARMUP:-20}
ITERS=${ITERS:-200}

phase=${1:-acc}

case "$phase" in
acc)
  fail=0
  for wl in $("$BIN" --list-workloads | cut -d: -f1); do
    for sel in after before; do
      out=$("$BIN" --mode=accuracy --workload="$wl" --selector=$sel 2>&1)
      rc=$?
      printf '%-6s %s\n' "$([ $rc -eq 0 ] && echo PASS || echo FAIL)" "$out"
      [ $rc -eq 0 ] || fail=1
    done
  done
  # INT4 path, one case per policy height.
  for rows in 4 8 32 64 128 129; do
    out=$("$BIN" --mode=accuracy --experts=8 --rows=$rows --n=256 --k=256 2>&1)
    rc=$?
    printf '%-6s %s\n' "$([ $rc -eq 0 ] && echo PASS || echo FAIL)" "$out"
    [ $rc -eq 0 ] || fail=1
  done
  echo "accuracy: $([ $fail -eq 0 ] && echo ALL PASS || echo FAILURES)"
  exit $fail
  ;;
table)
  # PR 446's own table, at the shape its description names: "Arc Pro B60, MXFP4
  # bf16 activations, gemm1 TFLOP/s, GPT-OSS TP=1/EP=1, 32 experts, N=5760,
  # K=2880".
  #   avg_m |  64    96   128   129   160   192   256   512
  #   before| 48.5  49.3  49.2  24.8  34.8  47.0  66.0  66.5
  #   after | 60.0  49.4  66.4  38.1  53.2  64.1  67.4  68.1
  # The two extra shapes are the ones bench_moe_w4a16_grouped_gemm.py actually
  # runs -- GPT-OSS gemm1 on one rank of 8 (4 local experts) and a DeepSeek-V4
  # EP=8 rank -- which the claim does not cover.
  for shape in "32 5760 2880 pr446-gpt-oss-tp1" "4 5760 2880 gpt-oss-tp8-gemm1" "32 4096 4096 dsv4-gemm1"; do
    set -- $shape
    experts=$1 n=$2 k=$3 name=$4
    for m in 64 96 128 129 160 192 256 512; do
      for sel in before after; do
        "$BIN" --mode=perf --quant=mxfp4 --experts="$experts" --rows=$m --n="$n" --k="$k" \
          --selector=$sel --warmup="$WARMUP" --iterations="$ITERS" \
          | sed "s/^/$name avg_m=$m selector=$sel /"
      done
    done
  done
  ;;
wl)
  for wl in $("$BIN" --list-workloads | cut -d: -f1); do
    for sel in before after; do
      "$BIN" --mode=perf --workload="$wl" --selector=$sel \
        --warmup="$WARMUP" --iterations="$ITERS" | sed "s/^/selector=$sel /"
    done
  done
  ;;
*)
  echo "unknown phase: $phase" >&2
  exit 2
  ;;
esac
