#!/usr/bin/env bash
# Time example 16's four decode workloads at a saturated clock.
#
# The decode band is launch-bound (~0.03 ms/launch), so the usual --warmup=20
# is 0.6 ms of load and the run measures the GPU's 1200 -> 2400 MHz ramp rather
# than the kernel: one configuration reads anywhere from 0.035 to 0.079 ms
# across processes that way, and two sweeps of these four shapes disagreed on
# the *sign* of a change. The 2 s budget is device time, so warm up in launch
# counts (40000 launches ~= 2 s) and take a median of nine reps.
#
# Run inside the BMG container from the worktree root:
#
#   scripts/moe_decode.sh > /tmp/decode.log
#   scripts/median.py /tmp/decode.log --pivot
#
# Env: BIN, WARMUP, ITERS, REPS, and TILE to add a second column comparing one
# registry tile against whatever select_workload_tile() picks.
set -u

BIN=${BIN:-./build/gpt-oss-moe-icpx/examples/16_bmg_moe_gemm/16_bmg_moe_gemm}
WARMUP=${WARMUP:-40000}
ITERS=${ITERS:-5000}
REPS=${REPS:-9}
TILE=${TILE:-}

DECODE=(gpt-oss-120b-decode-gemm1 gpt-oss-120b-decode-gemm2
        gpt-oss-120b-tp8-decode-gemm1 gpt-oss-120b-tp8-decode-gemm2)

run() { # <label> <extra args...>
  local label=$1; shift
  echo -n "$label "
  "$BIN" --mode=perf --warmup="$WARMUP" --iterations="$ITERS" "$@" 2>&1 |
    tail -1 | grep -o 'device_ms=[0-9.]* TOPS=[0-9.]*' || echo FAIL
}

for rep in $(seq 1 "$REPS"); do
  for wl in "${DECODE[@]}"; do
    run "rep$rep $wl selected" --workload="$wl"
    [ -n "$TILE" ] && run "rep$rep $wl $TILE" --workload="$wl" --tile="$TILE"
  done
done
