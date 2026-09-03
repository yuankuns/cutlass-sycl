/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling shared-expert / dense MLP (InklingBatchDenseMLP) for CUTLASS SYCL on BMG.
 *
 * Mirrors sglang/srt/models/inkling_common/dense_mlp.py::InklingBatchDenseMLP.
 * The layer runs `n_shared_experts` (S = 2) dense SwiGLU MLPs over every token
 * and sums them, weighted by the router's shared gammas. It dispatches between
 * two implementations of the same math; this example implements both.
 *
 * Notation (matching the docstring in dense_mlp.py):
 *   t = tokens, s = n_shared_experts, d = d_model (H), f = shared_d_mlp (I)
 *   w13_weight : [s, 2f, d]  gate/up interleaved on the 2f axis
 *                (inference_moe_w13_interleaved=True, so row 2j is gate_j and
 *                 row 2j+1 is up_j -- swiglu() slices [..., ::2] / [..., 1::2])
 *   w2_weight  : [s, d, f]
 *   gammas     : [t, s]      shared-expert gate weights, bf16
 *
 * (1) The bmm path (`forward`):
 *       x_std   = x_td.unsqueeze(0).expand(s, -1, -1).contiguous()   # [s, t, d]
 *       y_st2f  = bmm(x_std, w13_weight.mT)                          # [s, t, 2f]
 *       y_stf   = silu_and_mul(y_st2f, gammas.mT)                    # [s, t, f]
 *       z_std   = bmm(y_stf, w2_weight.mT)                           # [s, t, d]
 *       out_td  = z_std.float().sum(dim=0).to(bf16)                  # [t, d]
 *     Every torch.bmm here is bf16 in / bf16 out with an fp32 accumulator, so
 *     y, act and z are *rounded to bf16* between stages. Only the expert-axis
 *     reduction (_sum_dim0) is done in fp32 before the final bf16 cast, which
 *     is the "match TorchTitan's accumulation precision" comment in the model.
 *
 * (2) The linearized bf16 path (`_forward_bf16_linearized`), enabled when
 *     linearized_bf16 and inference_moe_w13_interleaved and the sink serves
 *     bf16. The expert axis is folded into the weight matrices:
 *       w13_lin = w13_weight.view(s * 2f, d)         # a pure reshape
 *       w2_lin  = w2_weight.transpose(1, 2).reshape(s * f, d)
 *       y       = mm(x_td, w13_lin.T).view(t, s, 2f)
 *       act     = silu_and_mul(y, gammas)            # [t, s, f]
 *       out_td  = mm(act.reshape(t, s * f), w2_lin)  # [t, d]
 *     The expert sum now happens *inside* the second GEMM's fp32 accumulator,
 *     so it never rounds z to bf16. That is a real (small) numerical difference
 *     from the bmm path, and each path is checked against its own reference.
 *
 * The gamma fold: silu_and_mul_triton multiplies the SwiGLU product by the
 * per-(token, expert) gamma in fp32 before the bf16 store, i.e.
 *   act[s, t, j] = bf16( silu(f32(y[s,t,2j])) * f32(y[s,t,2j+1]) * f32(gamma[t,s]) )
 * The bmm path passes gammas.mT.reshape(-1) (expert-major rows) and the
 * linearized path passes gammas.reshape(-1) (token-major rows); both index the
 * same gammas[t, s] value, only the row ordering of the flattened activation
 * differs. Both orderings are exercised here.
 *
 * Roofline: the whole layer is 6*s*t*d*f FLOPs against 6*s*d*f weight bytes
 * (bf16), so arithmetic intensity is ~t FLOP/B. Decode (t=1..9) is firmly
 * weight-bandwidth bound and prefill (t>=4096) is DPAS bound; the benchmark
 * therefore reports both effective TFLOP/s and effective GB/s per case.
 *
 * GEMMs use oneMKL (oneapi::mkl::blas::row_major::gemm / gemm_batch) rather
 * than a hand-rolled CUTLASS collective: MKL exposes exactly the bf16 x bf16 ->
 * bf16 with fp32 compute type that torch.bmm/torch.mm use, plus the strided
 * batched form the bmm path needs, so the numerics of both paths match the
 * model without re-deriving a tile config per shape. Example 20
 * (20_bmg_dflash_cache_path) sets the precedent for linking oneMKL here.
 * Everything else (SwiGLU + gamma fold, the fp32 expert reduction, the
 * expand+contiguous replication of x) is plain SYCL -- no ESIMD.
 **************************************************************************************************/

#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

#include "cute/util/compat.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cutlass::examples::bmg_shared_expert_mlp {

using Bf16 = oneapi::mkl::bfloat16;

constexpr int kThreads = 256;
constexpr double kBytesPerGB = 1.0e9;
constexpr double kOpsPerTOP = 1.0e12;

// ---------------------------------------------------------------------------
// bf16 helpers. sycl::ext::oneapi::bfloat16 converts on host and device, so a
// single definition serves both the kernels and the CPU reference.
// ---------------------------------------------------------------------------

inline float to_f32(Bf16 v) {
  return static_cast<float>(v);
}

inline Bf16 to_bf16(float v) {
  return static_cast<Bf16>(v);
}

inline uint16_t bf16_bits(Bf16 v) {
  uint16_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

// Monotonic ordering of bf16 bit patterns, so |a - b| counts ULPs.
inline int ordered_bf16(uint16_t raw) {
  int value = static_cast<int>(raw);
  return (value & 0x8000) ? (0x8000 - value) : (value + 0x8000);
}

inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Options / cases
// ---------------------------------------------------------------------------

enum class Path {
  kBoth,
  kBmm,
  kLinearized
};

inline char const* path_text(Path path) {
  switch (path) {
    case Path::kBoth:
      return "both";
    case Path::kBmm:
      return "bmm";
    case Path::kLinearized:
      return "linearized";
  }
  return "unknown";
}

inline bool parse_path(std::string const& text, Path& path) {
  if (text == "both" || text == "all") {
    path = Path::kBoth;
    return true;
  }
  if (text == "bmm" || text == "batched") {
    path = Path::kBmm;
    return true;
  }
  if (text == "linearized" || text == "lin" || text == "linearized_bf16") {
    path = Path::kLinearized;
    return true;
  }
  return false;
}

struct Options {
  std::string suite = "quick";
  std::string shape;
  std::string dtype = "bf16";
  Path path = Path::kBoth;
  int iterations = 20;
  int warmup = 5;
  bool verify = true;
  bool benchmark = true;
  // The model's expand(...).contiguous() really materializes [s, t, d]. Keep it
  // by default so the example mirrors the layer; --replicate-x=0 measures the
  // stride-0 batched-GEMM alternative instead.
  bool replicate_x = true;
  bool breakdown = false;
  double perf_threshold_scale = 1.0;
};

struct Case {
  std::string name;
  int tokens = 1;         // t
  int hidden = 1536;      // d = d_model
  int inter = 384;        // f = shared_d_mlp per TP partition (already / P)
  int experts = 2;        // s = n_shared_experts
  int tp = 1;             // reporting only; inter is the sharded value
  // Perf gates are report-only (0.0). B60 numbers measured for this example are
  // recorded in the README; the GPU here is shared, so hard gates would flake.
  double target_tops = 0.0;
  double target_gbps = 0.0;
  bool allow_verify = true;
};

// ---------------------------------------------------------------------------
// Device buffer
// ---------------------------------------------------------------------------

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
    ptr = sycl::malloc_device<T>(std::max<std::size_t>(n, 1), q);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : queue(other.queue), ptr(other.ptr), count(other.count) {
    other.queue = nullptr;
    other.ptr = nullptr;
    other.count = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      queue = other.queue;
      ptr = other.ptr;
      count = other.count;
      other.queue = nullptr;
      other.ptr = nullptr;
      other.count = 0;
    }
    return *this;
  }

  ~DeviceBuffer() {
    reset();
  }

  void reset() {
    if (ptr != nullptr) {
      sycl::free(ptr, *queue);
    }
    ptr = nullptr;
    queue = nullptr;
    count = 0;
  }

  T* get() const {
    return ptr;
  }

  void copy_from(std::vector<T> const& host) {
    if (host.size() > count) {
      throw std::runtime_error("copy_from exceeds device buffer");
    }
    if (!host.empty()) {
      queue->memcpy(ptr, host.data(), sizeof(T) * host.size()).wait();
    }
  }

  void copy_to(std::vector<T>& host) const {
    if (host.size() > count) {
      throw std::runtime_error("copy_to exceeds device buffer");
    }
    if (!host.empty()) {
      queue->memcpy(host.data(), ptr, sizeof(T) * host.size()).wait();
    }
  }
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// silu_and_mul_triton with the shared-expert gamma folded in, fp32 math and a
// bf16 store. `rows` activation rows are (s, t) pairs; row_major_s selects the
// flattening the caller uses:
//   row_major_s = 1 -> row = s * tokens + t   (bmm path, gammas.mT.reshape(-1))
//   row_major_s = 0 -> row = t * experts + s  (linearized path, gammas.reshape(-1))
struct SwigluParams {
  Bf16 const* __restrict__ y = nullptr;       // [rows, 2 * inter]
  Bf16 const* __restrict__ gammas = nullptr;  // [tokens, experts]
  Bf16* __restrict__ act = nullptr;           // [rows, inter]
  int rows = 0;
  int inter = 0;
  int tokens = 0;
  int experts = 0;
  int row_major_s = 1;
};

class SwigluKernel;

inline sycl::event launch_swiglu(sycl::queue& queue, SwigluParams const& params) {
  int tiles = ceil_div(params.inter, kThreads);
  sycl::range<2> global(static_cast<std::size_t>(params.rows),
                        static_cast<std::size_t>(tiles) * kThreads);
  sycl::range<2> local(1, kThreads);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SwigluKernel>(
        sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
          int row = static_cast<int>(item.get_global_id(0));
          int j = static_cast<int>(item.get_global_id(1));
          if (j >= params.inter) {
            return;
          }
          int s = params.row_major_s ? row / params.tokens : row % params.experts;
          int t = params.row_major_s ? row % params.tokens : row / params.experts;
          float gamma = to_f32(params.gammas[static_cast<int64_t>(t) * params.experts + s]);

          int64_t y_base = static_cast<int64_t>(row) * (2 * params.inter) + 2 * j;
          float gate = to_f32(params.y[y_base]);
          float up = to_f32(params.y[y_base + 1]);
          float silu = gate / (1.0f + sycl::exp(-gate));
          params.act[static_cast<int64_t>(row) * params.inter + j] =
              to_bf16(silu * up * gamma);
        });
  });
}

// _sum_dim0: out[t, d] = bf16(sum_s f32(z[s, t, d])). Expert-axis reduction of
// the bmm path, accumulated in fp32 before the single bf16 rounding.
struct ExpertSumParams {
  Bf16 const* __restrict__ z = nullptr;  // [experts, tokens, hidden]
  Bf16* __restrict__ out = nullptr;      // [tokens, hidden]
  int tokens = 0;
  int hidden = 0;
  int experts = 0;
};

class ExpertSumKernel;

inline sycl::event launch_expert_sum(sycl::queue& queue, ExpertSumParams const& params) {
  int tiles = ceil_div(params.hidden, kThreads);
  sycl::range<2> global(static_cast<std::size_t>(params.tokens),
                        static_cast<std::size_t>(tiles) * kThreads);
  sycl::range<2> local(1, kThreads);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<ExpertSumKernel>(
        sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
          int t = static_cast<int>(item.get_global_id(0));
          int h = static_cast<int>(item.get_global_id(1));
          if (h >= params.hidden) {
            return;
          }
          float accum = 0.0f;
          int64_t stride = static_cast<int64_t>(params.tokens) * params.hidden;
          int64_t offset = static_cast<int64_t>(t) * params.hidden + h;
          for (int s = 0; s < params.experts; ++s) {
            accum += to_f32(params.z[s * stride + offset]);
          }
          params.out[offset] = to_bf16(accum);
        });
  });
}

// ---------------------------------------------------------------------------
// Path launchers
// ---------------------------------------------------------------------------

struct DeviceTensors {
  Bf16* x = nullptr;        // [t, d]
  Bf16* xs = nullptr;       // [s, t, d] replicated x (bmm path)
  Bf16* w13 = nullptr;      // [s, 2f, d] == w13_lin [s*2f, d]
  Bf16* w2 = nullptr;       // [s, d, f]
  Bf16* w2_lin = nullptr;   // [s*f, d]
  Bf16* gammas = nullptr;   // [t, s]
  Bf16* y = nullptr;        // [rows, 2f]
  Bf16* act = nullptr;      // [rows, f]
  Bf16* z = nullptr;        // [s, t, d] (bmm path only)
  Bf16* out = nullptr;      // [t, d]
};

struct StageTimes {
  double replicate_ms = 0.0;
  double gemm1_ms = 0.0;
  double swiglu_ms = 0.0;
  double gemm2_ms = 0.0;
  double reduce_ms = 0.0;
};

// The bmm path. Returns after enqueueing; the caller waits.
inline void enqueue_bmm_path(
    sycl::queue& queue,
    Case const& cfg,
    DeviceTensors const& d,
    bool replicate_x) {
  int64_t t = cfg.tokens;
  int64_t s = cfg.experts;
  int64_t dm = cfg.hidden;
  int64_t f = cfg.inter;

  Bf16 const* a1 = d.x;
  int64_t stride_a1 = 0;
  if (replicate_x) {
    // x.unsqueeze(0).expand(s, -1, -1).contiguous()
    for (int e = 0; e < cfg.experts; ++e) {
      queue.memcpy(d.xs + e * t * dm, d.x, sizeof(Bf16) * static_cast<std::size_t>(t * dm));
    }
    a1 = d.xs;
    stride_a1 = t * dm;
  }

  // y_st2f = bmm(x_std, w13_weight.mT): [s, t, d] x [s, d, 2f] -> [s, t, 2f]
  oneapi::mkl::blas::row_major::gemm_batch(
      queue,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans,
      t, 2 * f, dm,
      1.0f,
      a1, dm, stride_a1,
      d.w13, dm, 2 * f * dm,
      0.0f,
      d.y, 2 * f, t * 2 * f,
      s);

  SwigluParams swiglu;
  swiglu.y = d.y;
  swiglu.gammas = d.gammas;
  swiglu.act = d.act;
  swiglu.rows = static_cast<int>(s * t);
  swiglu.inter = cfg.inter;
  swiglu.tokens = cfg.tokens;
  swiglu.experts = cfg.experts;
  swiglu.row_major_s = 1;
  launch_swiglu(queue, swiglu);

  // z_std = bmm(y_stf, w2_weight.mT): [s, t, f] x [s, f, d] -> [s, t, d]
  oneapi::mkl::blas::row_major::gemm_batch(
      queue,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans,
      t, dm, f,
      1.0f,
      d.act, f, t * f,
      d.w2, f, dm * f,
      0.0f,
      d.z, dm, t * dm,
      s);

  ExpertSumParams sum;
  sum.z = d.z;
  sum.out = d.out;
  sum.tokens = cfg.tokens;
  sum.hidden = cfg.hidden;
  sum.experts = cfg.experts;
  launch_expert_sum(queue, sum);
}

// The linearized bf16 path.
inline void enqueue_linearized_path(
    sycl::queue& queue,
    Case const& cfg,
    DeviceTensors const& d) {
  int64_t t = cfg.tokens;
  int64_t s = cfg.experts;
  int64_t dm = cfg.hidden;
  int64_t f = cfg.inter;

  // y = mm(x_td, w13_lin.T): [t, d] x [d, s*2f] -> [t, s*2f]
  oneapi::mkl::blas::row_major::gemm(
      queue,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans,
      t, s * 2 * f, dm,
      1.0f,
      d.x, dm,
      d.w13, dm,
      0.0f,
      d.y, s * 2 * f);

  SwigluParams swiglu;
  swiglu.y = d.y;
  swiglu.gammas = d.gammas;
  swiglu.act = d.act;
  swiglu.rows = static_cast<int>(t * s);
  swiglu.inter = cfg.inter;
  swiglu.tokens = cfg.tokens;
  swiglu.experts = cfg.experts;
  swiglu.row_major_s = 0;
  launch_swiglu(queue, swiglu);

  // out = mm(act.reshape(t, s*f), w2_lin): [t, s*f] x [s*f, d] -> [t, d].
  // The expert sum lives in this GEMM's fp32 accumulator.
  oneapi::mkl::blas::row_major::gemm(
      queue,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      t, dm, s * f,
      1.0f,
      d.act, s * f,
      d.w2_lin, dm,
      0.0f,
      d.out, dm);
}

// Per-stage timing (serialized: one wait per stage), used by --breakdown.
inline StageTimes time_bmm_stages(
    sycl::queue& queue,
    Case const& cfg,
    DeviceTensors const& d,
    bool replicate_x) {
  int64_t t = cfg.tokens;
  int64_t s = cfg.experts;
  int64_t dm = cfg.hidden;
  int64_t f = cfg.inter;
  StageTimes times;
  GPU_Clock clock;

  Bf16 const* a1 = d.x;
  int64_t stride_a1 = 0;
  if (replicate_x) {
    clock.start();
    for (int e = 0; e < cfg.experts; ++e) {
      queue.memcpy(d.xs + e * t * dm, d.x, sizeof(Bf16) * static_cast<std::size_t>(t * dm));
    }
    queue.wait();
    times.replicate_ms = clock.milliseconds();
    a1 = d.xs;
    stride_a1 = t * dm;
  }

  clock.start();
  oneapi::mkl::blas::row_major::gemm_batch(
      queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
      t, 2 * f, dm, 1.0f, a1, dm, stride_a1, d.w13, dm, 2 * f * dm, 0.0f,
      d.y, 2 * f, t * 2 * f, s);
  queue.wait();
  times.gemm1_ms = clock.milliseconds();

  SwigluParams swiglu;
  swiglu.y = d.y;
  swiglu.gammas = d.gammas;
  swiglu.act = d.act;
  swiglu.rows = static_cast<int>(s * t);
  swiglu.inter = cfg.inter;
  swiglu.tokens = cfg.tokens;
  swiglu.experts = cfg.experts;
  swiglu.row_major_s = 1;
  clock.start();
  launch_swiglu(queue, swiglu);
  queue.wait();
  times.swiglu_ms = clock.milliseconds();

  clock.start();
  oneapi::mkl::blas::row_major::gemm_batch(
      queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
      t, dm, f, 1.0f, d.act, f, t * f, d.w2, f, dm * f, 0.0f,
      d.z, dm, t * dm, s);
  queue.wait();
  times.gemm2_ms = clock.milliseconds();

  ExpertSumParams sum;
  sum.z = d.z;
  sum.out = d.out;
  sum.tokens = cfg.tokens;
  sum.hidden = cfg.hidden;
  sum.experts = cfg.experts;
  clock.start();
  launch_expert_sum(queue, sum);
  queue.wait();
  times.reduce_ms = clock.milliseconds();
  return times;
}

inline StageTimes time_linearized_stages(
    sycl::queue& queue,
    Case const& cfg,
    DeviceTensors const& d) {
  int64_t t = cfg.tokens;
  int64_t s = cfg.experts;
  int64_t dm = cfg.hidden;
  int64_t f = cfg.inter;
  StageTimes times;
  GPU_Clock clock;

  clock.start();
  oneapi::mkl::blas::row_major::gemm(
      queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
      t, s * 2 * f, dm, 1.0f, d.x, dm, d.w13, dm, 0.0f, d.y, s * 2 * f);
  queue.wait();
  times.gemm1_ms = clock.milliseconds();

  SwigluParams swiglu;
  swiglu.y = d.y;
  swiglu.gammas = d.gammas;
  swiglu.act = d.act;
  swiglu.rows = static_cast<int>(t * s);
  swiglu.inter = cfg.inter;
  swiglu.tokens = cfg.tokens;
  swiglu.experts = cfg.experts;
  swiglu.row_major_s = 0;
  clock.start();
  launch_swiglu(queue, swiglu);
  queue.wait();
  times.swiglu_ms = clock.milliseconds();

  clock.start();
  oneapi::mkl::blas::row_major::gemm(
      queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      t, dm, s * f, 1.0f, d.act, s * f, d.w2_lin, dm, 0.0f, d.out, dm);
  queue.wait();
  times.gemm2_ms = clock.milliseconds();
  return times;
}

// ---------------------------------------------------------------------------
// Host data / reference
// ---------------------------------------------------------------------------

// Scale the random inputs so intermediates and outputs are O(1). Without this
// the bf16 outputs collapse toward zero and an absolute-tolerance check would
// pass a kernel that is silently wrong.
inline float fan_in_scale(int fan_in, float target_std) {
  // x, w ~ U(-a, a) => std(sum of `fan_in` products) = sqrt(fan_in) * a^2 / 3.
  double a = std::sqrt(3.0 * target_std / std::sqrt(static_cast<double>(fan_in)));
  return static_cast<float>(a);
}

inline std::vector<Bf16> make_random(std::size_t count, float amplitude, uint32_t seed) {
  std::vector<Bf16> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-amplitude, amplitude);
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = to_bf16(dist(gen));
  }
  return data;
}

// Router gammas are sigmoid gate weights, so strictly positive and O(1).
inline std::vector<Bf16> make_gammas(std::size_t count, uint32_t seed) {
  std::vector<Bf16> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.05f, 0.95f);
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = to_bf16(dist(gen));
  }
  return data;
}

// w2_lin[s * f + j, d] = w2[s, d, j]  (w2_weight.transpose(1, 2) flattened).
inline std::vector<Bf16> make_w2_lin(Case const& cfg, std::vector<Bf16> const& w2) {
  std::size_t f = static_cast<std::size_t>(cfg.inter);
  std::size_t dm = static_cast<std::size_t>(cfg.hidden);
  std::vector<Bf16> lin(static_cast<std::size_t>(cfg.experts) * f * dm);
  for (int s = 0; s < cfg.experts; ++s) {
    for (std::size_t h = 0; h < dm; ++h) {
      for (std::size_t j = 0; j < f; ++j) {
        lin[((static_cast<std::size_t>(s) * f) + j) * dm + h] =
            w2[(static_cast<std::size_t>(s) * dm + h) * f + j];
      }
    }
  }
  return lin;
}

struct ReferenceOutputs {
  std::vector<Bf16> bmm;         // [t, d]
  std::vector<Bf16> linearized;  // [t, d]
};

// CPU reference for both paths. Shares y/act, which are identical, and then
// reproduces each path's second GEMM + reduction seam exactly:
//   bmm        : z rounded to bf16 per expert, then fp32 sum over s
//   linearized : one fp32 accumulator over the whole s*f contraction
inline ReferenceOutputs reference_shared_expert_mlp(
    Case const& cfg,
    std::vector<Bf16> const& x,
    std::vector<Bf16> const& w13,
    std::vector<Bf16> const& w2,
    std::vector<Bf16> const& gammas) {
  std::size_t t = static_cast<std::size_t>(cfg.tokens);
  std::size_t s = static_cast<std::size_t>(cfg.experts);
  std::size_t dm = static_cast<std::size_t>(cfg.hidden);
  std::size_t f = static_cast<std::size_t>(cfg.inter);

  // act[s][t][f], bf16-rounded exactly like the model's intermediates.
  std::vector<Bf16> act(s * t * f);
  std::vector<float> row(2 * f);
  for (std::size_t e = 0; e < s; ++e) {
    for (std::size_t token = 0; token < t; ++token) {
      Bf16 const* x_row = x.data() + token * dm;
      for (std::size_t r = 0; r < 2 * f; ++r) {
        Bf16 const* w_row = w13.data() + (e * 2 * f + r) * dm;
        float accum = 0.0f;
        for (std::size_t h = 0; h < dm; ++h) {
          accum += to_f32(x_row[h]) * to_f32(w_row[h]);
        }
        row[r] = to_f32(to_bf16(accum));  // torch.bmm returns bf16
      }
      float gamma = to_f32(gammas[token * s + e]);
      for (std::size_t j = 0; j < f; ++j) {
        float gate = row[2 * j];
        float up = row[2 * j + 1];
        float silu = gate / (1.0f + std::exp(-gate));
        act[(e * t + token) * f + j] = to_bf16(silu * up * gamma);
      }
    }
  }

  ReferenceOutputs out;
  out.bmm.assign(t * dm, to_bf16(0.0f));
  out.linearized.assign(t * dm, to_bf16(0.0f));
  for (std::size_t token = 0; token < t; ++token) {
    for (std::size_t h = 0; h < dm; ++h) {
      float sum_bmm = 0.0f;
      float sum_lin = 0.0f;
      for (std::size_t e = 0; e < s; ++e) {
        Bf16 const* act_row = act.data() + (e * t + token) * f;
        Bf16 const* w2_row = w2.data() + (e * dm + h) * f;
        float accum = 0.0f;
        for (std::size_t j = 0; j < f; ++j) {
          accum += to_f32(act_row[j]) * to_f32(w2_row[j]);
        }
        // bmm path: z is a bf16 tensor before the fp32 sum over experts.
        sum_bmm += to_f32(to_bf16(accum));
        // linearized path: no intermediate rounding, one fp32 accumulator.
        sum_lin += accum;
      }
      out.bmm[token * dm + h] = to_bf16(sum_bmm);
      out.linearized[token * dm + h] = to_bf16(sum_lin);
    }
  }
  return out;
}

struct VerifyResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  int max_ulps = 0;
  std::size_t index = 0;
  uint16_t got_bits = 0;
  uint16_t expected_bits = 0;
};

// bf16 storage with fp32 accumulation: MKL's contraction order differs from the
// reference's, so allow a few bf16 ULPs / a small relative error.
constexpr double kAtol = 4.0e-2;
constexpr double kRtol = 2.0e-2;
// Inputs are scaled so the output std is ~1 (see fan_in_scale).
constexpr double kUlpMinMagnitude = 1.0;

inline VerifyResult compare(std::vector<Bf16> const& got, std::vector<Bf16> const& expected) {
  if (got.size() != expected.size()) {
    throw std::invalid_argument("compare size mismatch");
  }
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double g = to_f32(got[i]);
    double e = to_f32(expected[i]);
    double abs_err = std::abs(g - e);
    double rel_err = abs_err / std::max(1.0e-6, std::abs(e));
    if (std::abs(e) >= kUlpMinMagnitude) {
      // ULP distance is only informative away from zero: near-cancellation
      // outputs sit many bf16 exponents below the tensor's scale, so a 1e-2
      // absolute difference there is thousands of ULPs and says nothing.
      int ulps = std::abs(ordered_bf16(bf16_bits(got[i])) - ordered_bf16(bf16_bits(expected[i])));
      result.max_ulps = std::max(result.max_ulps, ulps);
    }
    // Track the two maxima independently: the pass predicate mixes an absolute
    // and a relative term, so a failure driven by the relative term at a small
    // output would be invisible if max_rel were only sampled at the max-abs
    // element.
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);
    if (abs_err > kAtol + kRtol * std::abs(e) && result.passed) {
      result.passed = false;
      result.index = i;
      result.got_bits = bf16_bits(got[i]);
      result.expected_bits = bf16_bits(expected[i]);
    }
  }
  return result;
}

inline double max_abs_diff(std::vector<Bf16> const& a, std::vector<Bf16> const& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(to_f32(a[i])) - to_f32(b[i])));
  }
  return worst;
}

// ---------------------------------------------------------------------------
// Roofline accounting
// ---------------------------------------------------------------------------

inline double case_flops(Case const& cfg) {
  // gemm1: 2 * s*t*(2f)*d, gemm2: 2 * s*t*d*f  ->  6 * s*t*d*f
  return 6.0 * cfg.experts * cfg.tokens * cfg.hidden * cfg.inter;
}

// The single-threaded CPU reference runs at roughly 0.5 G MAC/s on this host, so
// an unbounded shape can look like a 15-minute hang with no output. Both suites
// and custom --shape cases gate verification on this budget.
constexpr double kMaxReferenceOps = 2.5e9;

// Estimated DRAM traffic assuming the intermediates round-trip through memory
// (they are far larger than L2 at prefill shapes) and the weights are read once.
inline double case_bytes(Case const& cfg, Path path, bool replicate_x) {
  double s = cfg.experts;
  double t = cfg.tokens;
  double dm = cfg.hidden;
  double f = cfg.inter;
  double elem = 2.0;
  double weights = (s * 2.0 * f * dm + s * dm * f) * elem;
  double gammas = t * s * elem;
  double out = t * dm * elem;
  double y = 2.0 * s * t * 2.0 * f * elem;  // written by gemm1, read by swiglu
  double act = 2.0 * s * t * f * elem;      // written by swiglu, read by gemm2
  if (path == Path::kBmm) {
    double x = replicate_x ? (t * dm * elem + 2.0 * s * t * dm * elem) : (s * t * dm * elem);
    double z = 2.0 * s * t * dm * elem;  // written by gemm2, read by the reduction
    return weights + gammas + x + y + act + z + out;
  }
  return weights + gammas + t * dm * elem + y + act + out;
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

// Small shapes with awkward tails: odd `inter` exercises the interleaved
// gate/up pair indexing and the kThreads tile tail, odd `hidden` exercises the
// GEMM leading dimensions, and experts=1/3 catches expert-axis mix-ups even
// though the model always ships n_shared_experts=2.
inline std::vector<Case> quick_suite() {
  return {
      {"tiny_t1", 1, 16, 3, 2, 1, 0.0, 0.0, true},
      {"tiny_odd_tail", 3, 13, 7, 2, 1, 0.0, 0.0, true},
      {"inter_tile_tail_257", 5, 32, 257, 2, 1, 0.0, 0.0, true},
      {"experts1", 4, 24, 5, 1, 1, 0.0, 0.0, true},
      {"experts3", 4, 24, 5, 3, 1, 0.0, 0.0, true},
      {"ckpt_h768_i384_t9", 9, 768, 384, 2, 1, 0.0, 0.0, true},
      {"cfg_h1536_i384_t9", 9, 1536, 384, 2, 1, 0.0, 0.0, true},
  };
}

// Real Inkling shared-expert shapes. n_shared_experts=2 always. The shared
// sink is column-parallel on w13 / row-parallel on w2, so tensor parallelism
// shards only `shared_d_mlp` (f -> f/P) while d_model is replicated and the
// [t, d] output is all-reduced (symm_mem_all_reduce, out of scope here).
// Hidden sizes: 768 (checkpoint), 1536 (config defaults), 6144 (production).
// shared_d_mlp: 384 (intermediate_size checkpoint) and 3072
// (dense_intermediate_size, production). Token counts follow the model's
// bands: T=1 decode, T=9 draft_token_num (MTP verify), T=144 a small prefill
// chunk, and T=4096 / T=16384 (max_prefill_tokens) large prefill.
//
// Verification is enabled where the O(s*t*d*f) CPU reference stays under a few
// seconds; the large-T rows are perf-shaped and run with verify=SKIP.
inline std::vector<Case> inkling_suite() {
  std::vector<Case> cases;
  struct Shape {
    char const* tag;
    int hidden;
    int inter;  // unsharded shared_d_mlp
  };
  Shape const shapes[] = {
      {"ckpt_h768_i384", 768, 384},
      {"cfg_h1536_i384", 1536, 384},
      {"cfg_h1536_i3072", 1536, 3072},
      {"prod_h6144_i3072", 6144, 3072},
  };
  int const tps[] = {1, 2, 4, 8};
  int const tokens[] = {1, 9, 144, 4096};
  for (Shape const& shape : shapes) {
    for (int tp : tps) {
      int inter = shape.inter / tp;
      if (inter <= 0) {
        continue;
      }
      for (int t : tokens) {
        // Keep the CPU reference bounded: verify small token counts, and T=144
        // only for the cheap shapes.
        double ref_ops = 6.0 * 2 * t * shape.hidden * inter;
        bool allow_verify = ref_ops <= kMaxReferenceOps;
        std::ostringstream name;
        name << shape.tag << "_tp" << tp << "_t" << t;
        cases.push_back(Case{name.str(), t, shape.hidden, inter, 2, tp, 0.0, 0.0, allow_verify});
      }
    }
  }
  // max_prefill_tokens at the production and config shapes (TP1 and TP8 only,
  // to keep the suite's wall time reasonable).
  cases.push_back(Case{"prod_h6144_i3072_tp1_t16384", 16384, 6144, 3072, 2, 1, 0.0, 0.0, false});
  cases.push_back(Case{"prod_h6144_i3072_tp8_t16384", 16384, 6144, 384, 2, 8, 0.0, 0.0, false});
  cases.push_back(Case{"cfg_h1536_i384_tp1_t16384", 16384, 1536, 384, 2, 1, 0.0, 0.0, false});
  return cases;
}

// Perf bands: decode (t=1), MTP verify (t=9), and prefill (t=4096, 16384) at
// both shipped hidden sizes across the TP shard ladder. Gates are report-only
// (0.0); see the README for the measured B60 table.
inline std::vector<Case> perf_suite() {
  std::vector<Case> cases;
  struct Shape {
    char const* tag;
    int hidden;
    int inter;
  };
  Shape const shapes[] = {
      {"prod_h6144_i3072", 6144, 3072},
      {"cfg_h1536_i384", 1536, 384},
  };
  int const tps[] = {1, 2, 4, 8};
  int const tokens[] = {1, 9, 4096, 16384};
  for (Shape const& shape : shapes) {
    for (int tp : tps) {
      int inter = shape.inter / tp;
      if (inter <= 0) {
        continue;
      }
      for (int t : tokens) {
        std::ostringstream name;
        name << "perf_" << shape.tag << "_tp" << tp << "_t" << t;
        cases.push_back(Case{name.str(), t, shape.hidden, inter, 2, tp, 0.0, 0.0, false});
      }
    }
  }
  return cases;
}

inline std::vector<Case> make_suite(std::string const& suite) {
  if (suite == "quick") {
    return quick_suite();
  }
  if (suite == "inkling") {
    return inkling_suite();
  }
  if (suite == "perf") {
    return perf_suite();
  }
  return {};
}

// Returns false (rather than throwing) on any malformed key or value: this runs
// before main's try block, so an escaping std::invalid_argument from stoi/stod
// would abort with a core dump instead of printing the usage message.
inline bool parse_shape(std::string const& text, Case& cfg) try {
  if (text.empty()) {
    return true;
  }
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto eq = item.find('=');
    if (eq == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, eq);
    std::string value = item.substr(eq + 1);
    if (key == "name") {
      cfg.name = value;
    } else if (key == "tokens" || key == "t") {
      cfg.tokens = std::stoi(value);
    } else if (key == "hidden" || key == "d" || key == "d_model") {
      cfg.hidden = std::stoi(value);
    } else if (key == "inter" || key == "f" || key == "shared_d_mlp") {
      cfg.inter = std::stoi(value);
    } else if (key == "experts" || key == "s" || key == "n_shared_experts") {
      cfg.experts = std::stoi(value);
    } else if (key == "tp") {
      cfg.tp = std::stoi(value);
    } else if (key == "target_tops" || key == "target-tops") {
      cfg.target_tops = std::stod(value);
    } else if (key == "target_gbps" || key == "target-gbps") {
      cfg.target_gbps = std::stod(value);
    } else {
      return false;
    }
  }
  return true;
} catch (std::invalid_argument const&) {
  return false;
} catch (std::out_of_range const&) {
  return false;
}

inline void validate_case(Case& cfg) {
  if (cfg.tokens <= 0 || cfg.hidden <= 0 || cfg.inter <= 0 || cfg.experts <= 0) {
    throw std::invalid_argument("case has non-positive shape");
  }
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

inline bool check_target(
    char const* label,
    char const* metric,
    double value,
    double target,
    double threshold_scale) {
  if (target <= 0.0 || threshold_scale <= 0.0) {
    return true;
  }
  double threshold = target * threshold_scale;
  if (value >= threshold) {
    return true;
  }
  std::cerr << "    perf=FAIL " << label << " " << metric << "=" << value
            << " target=" << threshold << " scale=" << threshold_scale << "\n";
  return false;
}

inline bool run_case(sycl::queue& queue, Case cfg, Options const& options) {
  validate_case(cfg);

  std::size_t t = static_cast<std::size_t>(cfg.tokens);
  std::size_t s = static_cast<std::size_t>(cfg.experts);
  std::size_t dm = static_cast<std::size_t>(cfg.hidden);
  std::size_t f = static_cast<std::size_t>(cfg.inter);

  bool run_bmm = options.path != Path::kLinearized;
  bool run_lin = options.path != Path::kBmm;

  std::cout << "  " << cfg.name << " t=" << cfg.tokens << " d=" << cfg.hidden
            << " f=" << cfg.inter << " s=" << cfg.experts << " tp=" << cfg.tp << "\n";

  float x_amp = fan_in_scale(cfg.hidden, 1.0f);
  // act has std ~0.3 after silu*up*gamma; size w2 so z is O(1) as well.
  float w2_amp = static_cast<float>(
      std::sqrt(3.0) / (0.3 * std::sqrt(static_cast<double>(cfg.inter))));

  std::vector<Bf16> host_x = make_random(t * dm, x_amp, 11);
  std::vector<Bf16> host_w13 = make_random(s * 2 * f * dm, x_amp, 23);
  std::vector<Bf16> host_w2 = make_random(s * dm * f, w2_amp, 37);
  std::vector<Bf16> host_gammas = make_gammas(t * s, 53);
  std::vector<Bf16> host_w2_lin = make_w2_lin(cfg, host_w2);

  DeviceBuffer<Bf16> d_x(queue, t * dm);
  DeviceBuffer<Bf16> d_w13(queue, s * 2 * f * dm);
  DeviceBuffer<Bf16> d_w2(queue, s * dm * f);
  DeviceBuffer<Bf16> d_w2_lin(queue, run_lin ? s * f * dm : 0);
  DeviceBuffer<Bf16> d_gammas(queue, t * s);
  DeviceBuffer<Bf16> d_y(queue, s * t * 2 * f);
  DeviceBuffer<Bf16> d_act(queue, s * t * f);
  DeviceBuffer<Bf16> d_z(queue, run_bmm ? s * t * dm : 0);
  DeviceBuffer<Bf16> d_xs(queue, (run_bmm && options.replicate_x) ? s * t * dm : 0);
  DeviceBuffer<Bf16> d_out(queue, t * dm);

  d_x.copy_from(host_x);
  d_w13.copy_from(host_w13);
  d_w2.copy_from(host_w2);
  d_gammas.copy_from(host_gammas);
  if (run_lin) {
    d_w2_lin.copy_from(host_w2_lin);
  }

  DeviceTensors d;
  d.x = d_x.get();
  d.xs = d_xs.get();
  d.w13 = d_w13.get();
  d.w2 = d_w2.get();
  d.w2_lin = d_w2_lin.get();
  d.gammas = d_gammas.get();
  d.y = d_y.get();
  d.act = d_act.get();
  d.z = d_z.get();
  d.out = d_out.get();

  bool passed = true;
  bool verified = options.verify && cfg.allow_verify;

  std::vector<Bf16> got_bmm;
  std::vector<Bf16> got_lin;
  if (verified) {
    ReferenceOutputs expected =
        reference_shared_expert_mlp(cfg, host_x, host_w13, host_w2, host_gammas);
    if (run_bmm) {
      got_bmm.assign(t * dm, to_bf16(0.0f));
      enqueue_bmm_path(queue, cfg, d, options.replicate_x);
      queue.wait_and_throw();
      d_out.copy_to(got_bmm);
      VerifyResult result = compare(got_bmm, expected.bmm);
      if (!result.passed) {
        std::cerr << "    verify=FAIL path=bmm index=" << result.index << " got_bits=0x"
                  << std::hex << result.got_bits << " expected_bits=0x" << result.expected_bits
                  << std::dec << " max_abs=" << result.max_abs << " max_rel=" << result.max_rel
                  << " max_ulps=" << result.max_ulps << "\n";
        passed = false;
      } else {
        std::cout << "    verify=PASS path=bmm max_abs=" << result.max_abs
                  << " max_rel=" << result.max_rel << " max_ulps=" << result.max_ulps << "\n";
      }
    }
    if (run_lin) {
      got_lin.assign(t * dm, to_bf16(0.0f));
      enqueue_linearized_path(queue, cfg, d);
      queue.wait_and_throw();
      d_out.copy_to(got_lin);
      VerifyResult result = compare(got_lin, expected.linearized);
      if (!result.passed) {
        std::cerr << "    verify=FAIL path=linearized index=" << result.index << " got_bits=0x"
                  << std::hex << result.got_bits << " expected_bits=0x" << result.expected_bits
                  << std::dec << " max_abs=" << result.max_abs << " max_rel=" << result.max_rel
                  << " max_ulps=" << result.max_ulps << "\n";
        passed = false;
      } else {
        std::cout << "    verify=PASS path=linearized max_abs=" << result.max_abs
                  << " max_rel=" << result.max_rel << " max_ulps=" << result.max_ulps << "\n";
      }
    }
    if (run_bmm && run_lin) {
      // Report-only: the two paths differ by the bf16 rounding of z that the
      // bmm path applies before its fp32 expert sum.
      std::cout << "    path_delta max_abs=" << max_abs_diff(got_bmm, got_lin) << "\n";
    }
  } else if (options.verify) {
    std::cout << "    verify=SKIP CPU reference too large for this shape\n";
  }

  if (!options.benchmark) {
    return passed;
  }

  double flops = case_flops(cfg);
  int timing_iterations = std::max(options.iterations, 1);

  struct PathRun {
    bool enabled;
    Path path;
    char const* label;
  };
  PathRun const runs[] = {
      {run_bmm, Path::kBmm, "bmm"},
      {run_lin, Path::kLinearized, "linearized"},
  };

  for (PathRun const& run : runs) {
    if (!run.enabled) {
      continue;
    }
    auto enqueue = [&]() {
      if (run.path == Path::kBmm) {
        enqueue_bmm_path(queue, cfg, d, options.replicate_x);
      } else {
        enqueue_linearized_path(queue, cfg, d);
      }
    };

    for (int i = 0; i < options.warmup; ++i) {
      enqueue();
    }
    queue.wait_and_throw();

    GPU_Clock clock;
    clock.start();
    for (int i = 0; i < timing_iterations; ++i) {
      enqueue();
    }
    queue.wait_and_throw();
    double total_ms = clock.milliseconds();
    double avg_ms = total_ms / static_cast<double>(timing_iterations);

    double bytes = case_bytes(cfg, run.path, options.replicate_x);
    double tops = flops / kOpsPerTOP / (avg_ms * 1.0e-3);
    double gbps = bytes / kBytesPerGB / (avg_ms * 1.0e-3);
    std::cout << std::fixed << std::setprecision(4) << "    path=" << run.label
              << " avg_ms=" << avg_ms << std::setprecision(2) << " TFLOPs=" << tops
              << " est_GBps=" << gbps;
    if (cfg.target_tops > 0.0) {
      std::cout << " target_TFLOPs=" << cfg.target_tops * options.perf_threshold_scale;
    }
    if (cfg.target_gbps > 0.0) {
      std::cout << " target_GBps=" << cfg.target_gbps * options.perf_threshold_scale;
    }
    std::cout << std::defaultfloat << "\n";
    passed &= check_target(run.label, "TFLOPs", tops, cfg.target_tops, options.perf_threshold_scale);
    passed &= check_target(run.label, "GBps", gbps, cfg.target_gbps, options.perf_threshold_scale);

    if (options.breakdown) {
      StageTimes times = run.path == Path::kBmm
                             ? time_bmm_stages(queue, cfg, d, options.replicate_x)
                             : time_linearized_stages(queue, cfg, d);
      std::cout << std::fixed << std::setprecision(4) << "      stages replicate_ms="
                << times.replicate_ms << " gemm1_ms=" << times.gemm1_ms
                << " swiglu_ms=" << times.swiglu_ms << " gemm2_ms=" << times.gemm2_ms
                << " reduce_ms=" << times.reduce_ms << std::defaultfloat << "\n";
    }
  }

  return passed;
}

inline void print_usage(char const* name) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=quick|inkling|perf     Built-in shape suite (default quick)\n"
      << "  --shape=t=<int>,d=<int>,f=<int>,s=<int>,tp=<int>[,target_tops=..,target_gbps=..]\n"
      << "                                 Run one custom shape instead of a suite\n"
      << "  --dtype=bf16                   Element dtype; the layer is bf16 with fp32 accumulate\n"
      << "  --path=both|bmm|linearized     Which dispatch path to run (default both)\n"
      << "  --replicate-x=0|1              Materialize x.expand(s,-1,-1).contiguous() (default 1;\n"
      << "                                 0 uses a stride-0 batched GEMM instead)\n"
      << "  --iterations=<int>             Timed iterations (default 20)\n"
      << "  --warmup=<int>                 Warmup iterations (default 5)\n"
      << "  --verify=0|1                   CPU reference comparison where the case permits\n"
      << "  --benchmark=0|1                Run timing (default 1)\n"
      << "  --breakdown=0|1                Also report per-stage timings (default 0)\n"
      << "  --perf-threshold-scale=<float>  Scale perf gates; 0 disables them\n";
}

}  // namespace cutlass::examples::bmg_shared_expert_mlp

int main(int argc, char const** argv) {
  using namespace cutlass::examples::bmg_shared_expert_mlp;

  cutlass::CommandLine cmd(argc, argv);
  if (cmd.check_cmd_line_flag("help")) {
    print_usage(argv[0]);
    return 0;
  }

  Options options;
  cmd.get_cmd_line_argument("suite", options.suite, options.suite);
  cmd.get_cmd_line_argument("shape", options.shape, options.shape);
  cmd.get_cmd_line_argument("dtype", options.dtype, options.dtype);
  cmd.get_cmd_line_argument("iterations", options.iterations, options.iterations);
  cmd.get_cmd_line_argument("warmup", options.warmup, options.warmup);
  cmd.get_cmd_line_argument(
      "perf-threshold-scale", options.perf_threshold_scale, options.perf_threshold_scale);

  int verify = options.verify ? 1 : 0;
  cmd.get_cmd_line_argument("verify", verify, verify);
  options.verify = verify != 0;
  int benchmark = options.benchmark ? 1 : 0;
  cmd.get_cmd_line_argument("benchmark", benchmark, benchmark);
  options.benchmark = benchmark != 0;
  int replicate_x = options.replicate_x ? 1 : 0;
  cmd.get_cmd_line_argument("replicate-x", replicate_x, replicate_x);
  options.replicate_x = replicate_x != 0;
  int breakdown = options.breakdown ? 1 : 0;
  cmd.get_cmd_line_argument("breakdown", breakdown, breakdown);
  options.breakdown = breakdown != 0;

  std::string path_arg = path_text(options.path);
  cmd.get_cmd_line_argument("path", path_arg, path_arg);
  if (!parse_path(path_arg, options.path)) {
    std::cerr << "Unknown path: " << path_arg << "\n";
    print_usage(argv[0]);
    return -1;
  }
  // InklingBatchDenseMLP is bf16 (params_dtype) with fp32 accumulation; the FP4
  // serving strategy is a different code path and is out of scope here.
  if (options.dtype != "bf16") {
    std::cerr << "Unsupported dtype: " << options.dtype
              << " (InklingBatchDenseMLP's bf16 path is the one modeled here)\n";
    return -1;
  }
  if (options.iterations < 0 || options.warmup < 0) {
    std::cerr << "iterations and warmup must be non-negative\n";
    return -1;
  }

  std::vector<Case> cases;
  if (!options.shape.empty()) {
    Case cfg;
    cfg.name = "custom";
    if (!parse_shape(options.shape, cfg)) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    // Same reference budget the suites apply: without this, a production-sized
    // custom shape with --verify=1 starts a reference that takes ~15 minutes.
    if (cfg.tokens > 0 && cfg.hidden > 0 && cfg.inter > 0 && cfg.experts > 0 &&
        6.0 * cfg.experts * cfg.tokens * cfg.hidden * cfg.inter > kMaxReferenceOps) {
      if (options.verify) {
        std::cerr << "note: --shape is too large for the CPU reference ("
                  << 6.0 * cfg.experts * cfg.tokens * cfg.hidden * cfg.inter << " ops > "
                  << kMaxReferenceOps << "); skipping verification for it\n";
      }
      cfg.allow_verify = false;
    }
    cases.push_back(cfg);
  } else {
    cases = make_suite(options.suite);
    if (cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = compat::get_default_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "26_bmg_shared_expert_mlp: InklingBatchDenseMLP shared-expert / dense MLP\n"
              << "  bmm path        : bmm(x[s,t,d], w13.mT) -> silu_and_mul(., gammas) -> "
                 "bmm(., w2.mT) -> fp32 sum over s\n"
              << "  linearized path : mm(x, w13_lin.T) -> swiglu -> mm(act, w2_lin)\n";
    // A custom --shape overrides the suite entirely, so don't print a suite name
    // that was not run.
    std::cout << (options.shape.empty() ? "Suite=" + options.suite : "Shape=" + options.shape)
              << " path=" << path_text(options.path)
              << " dtype=" << options.dtype << " iterations=" << options.iterations
              << " warmup=" << options.warmup << " verify=" << (options.verify ? 1 : 0)
              << " benchmark=" << (options.benchmark ? 1 : 0)
              << " replicate_x=" << (options.replicate_x ? 1 : 0)
              << " perf_threshold_scale=" << options.perf_threshold_scale << "\n";

    bool all_passed = true;
    for (Case const& cfg : cases) {
      all_passed &= run_case(queue, cfg, options);
    }
    if (!all_passed) {
      std::cerr << "FAILED\n";
      return -1;
    }
    std::cout << "PASSED\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
