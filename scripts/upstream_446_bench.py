"""Run upstream sgl-kernel-xpu's own W4A16 grouped-GEMM bench at PR 446's shape.

Uses the 4406073 worktree's benchmark module verbatim -- its _prepare_inputs
(uniform rows_per_expert), its _run_mxfp4_fused (one op call), its FLOP
convention (2*total_m*N*K) and its do_bench median -- so the only thing that
differs from `python benchmark/bench_moe_w4a16_grouped_gemm.py` is the shape
list, which is fixed there and cannot express PR 446's own claimed
configuration (32 experts, N=5760, K=2880).

  PYTHONPATH is not usable here (it breaks torch's import in this container);
  the sgl_kernel build under test is prepended to sys.path instead.
"""

import os
import sys

SGLK = os.environ.get("SGLK_TARGET", "/tmp/sglk446")
BENCH_DIR = os.environ.get(
    "BENCH_DIR", "/workspace/worktrees/sgl-kernel-xpu/pr446-4406073/benchmark"
)

sys.path.insert(0, SGLK)
sys.path.insert(1, BENCH_DIR)

import torch  # noqa: E402
import sgl_kernel  # noqa: E402,F401

print("sgl_kernel:", sgl_kernel.__file__, flush=True)

import bench_moe_w4a16_grouped_gemm as B  # noqa: E402

NUM_EXPERTS = int(os.environ.get("EXPERTS", 32))
GEMM_N = int(os.environ.get("N", 5760))
GEMM_K = int(os.environ.get("K", 2880))
AVG_MS = [int(x) for x in os.environ.get("AVG_MS", "64,96,128,129,160,192,256,512").split(",")]

try:
    import triton

    def median_ms(fn):
        ms, _, _ = triton.testing.do_bench(fn, warmup=50, rep=200, quantiles=[0.5, 0.2, 0.8])
        return ms

    timer = "triton.do_bench(warmup=50, rep=200, q=0.5)"
except ImportError:
    import statistics

    def median_ms(fn):
        for _ in range(20):
            fn()
        torch.xpu.synchronize()
        samples = []
        for _ in range(200):
            s, e = torch.xpu.Event(True), torch.xpu.Event(True)
            s.record()
            fn()
            e.record()
            torch.xpu.synchronize()
            samples.append(s.elapsed_time(e))
        return statistics.median(samples)

    timer = "xpu.Event median of 200"

print(f"[timer] {timer}", flush=True)
print(f"[shape] experts={NUM_EXPERTS} N={GEMM_N} K={GEMM_K}", flush=True)

for avg_m in AVG_MS:
    inputs = B._prepare_inputs(NUM_EXPERTS, avg_m, GEMM_N, GEMM_K, "mxfp4", "sgl")
    run = lambda: B._run_mxfp4_fused(inputs)  # noqa: E731
    for _ in range(5):
        run()
    torch.xpu.synchronize()
    ms = median_ms(run)
    flop = 2 * inputs["total_m"] * GEMM_N * GEMM_K
    print(f"upstream4406073 avg_m={avg_m} ms={ms:.4f} TFLOPS={flop / ms * 1e-9:.3f}", flush=True)
    del inputs
    torch.xpu.empty_cache()
