/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling relative-attention helper examples for CUTLASS SYCL on BMG.
 *
 * Roofline summary:
 *   row_scale_bf16 / row_compact_bf16 are memory-bound. row_scale performs one
 *   FP32 multiply for each bf16/fp16 element while streaming input and output,
 *   so arithmetic intensity is roughly 0.25 FLOP/B before tau amortization.
 *   row_compact is a strided-row to contiguous-row copy with no math. Sustained
 *   effective bandwidth is the useful metric; perf cases use production-sized
 *   rows so cache-only behavior is not mistaken for bandwidth.
 *
 *   rel_proj_small_t computes T * H independent [D] x [D, E] projections. For
 *   Inkling's production D=16, E=1024 small-token path, each output element
 *   performs 32 FLOPs but streams projection data unless it is hot in cache.
 *   This is still bandwidth/latency sensitive at small T, so the production
 *   path keeps one launch, no local-memory staging, and reuses each register
 *   tile of projection weights across several output rows.
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cutlass::examples::relative_helpers {

constexpr int kDefaultBlock = 256;
constexpr int kRowPackBytes = 16;
constexpr int kRowPackWords = kRowPackBytes / static_cast<int>(sizeof(uint32_t));
constexpr int kRowPackElems = kRowPackBytes / static_cast<int>(sizeof(cutlass::bfloat16_t));
constexpr int kRowSmallLaneElems = kRowPackElems;
constexpr int kRowLargePacksPerLane = 4;
constexpr int kRowLargeLaneElems = kRowPackElems * kRowLargePacksPerLane;
constexpr int kRowEsimdCopyWords = 64;
constexpr int kRowEsimdMinRows = 512;
constexpr int kRowSmallWorkItemsThreshold = 8192;
constexpr int kRelProjVec = 8;
constexpr int kRelProjProductionD = 16;
constexpr double kMinSustainedTargetBytes = 32.0 * 1024.0 * 1024.0;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

// rel_proj is one expression, with tau either folded in or absent:
//
//   out[t, h, e] = sum_d bf16(r[t*r_stride_t + h*d + d] * tau[t]) * proj[d, e]
//
// tau is a scalar per token, so it commutes with the projection exactly; the only
// reason folding it in is observable at all is the bf16 round-trip above, which
// is where the caller's fused-log-tau path rounds. The kernel therefore
// reproduces that rounding point rather than scaling the fp32 accumulator, and
// there is no second placement or second tau shape to select between.
// fuse_tau == false is the SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU=0 path: tau is
// applied by the caller and this kernel never reads it.

struct Options {
  std::string suite = "quick";
  std::string shape;
  DType dtype = DType::kAll;
  int iterations = 20;
  int warmup = 5;
  bool verify = true;
  bool benchmark = true;
  bool target_gbps_set = false;
  double target_gbps = 0.0;
  bool help = false;
};

struct RowCase {
  std::string name;
  int rows = 1;
  int inner = 1;
  int stride = 1;
  double target_gbps = 0.0;
};

struct RelProjCase {
  std::string name;
  int t = 1;
  int h = 1;
  int d = 16;
  int e = 16;
  int r_stride_t = 16;
  bool proj_per_head = false;
  bool fuse_tau = true;
  double target_gbps = 0.0;
};

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
    ptr = sycl::malloc_device<T>(std::max<std::size_t>(count, 1), q);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept {
    queue = other.queue;
    ptr = other.ptr;
    count = other.count;
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
    if (!host.empty()) {
      queue->memcpy(ptr, host.data(), sizeof(T) * host.size()).wait();
    }
  }

  void copy_to(std::vector<T>& host) const {
    if (!host.empty()) {
      queue->memcpy(host.data(), ptr, sizeof(T) * host.size()).wait();
    }
  }
};

template <typename Element>
struct RowParams {
  Element const* __restrict__ x = nullptr;
  float const* __restrict__ tau = nullptr;
  Element* __restrict__ out = nullptr;
  int rows = 0;
  int inner = 0;
  int stride = 0;
  int lanes_per_row = 0;
  int vec_count = 0;
};

template <typename Element>
struct RelProjParams {
  Element const* __restrict__ r = nullptr;
  Element const* __restrict__ proj = nullptr;
  float const* __restrict__ tau = nullptr;
  Element* __restrict__ out = nullptr;
  int t = 0;
  int h = 0;
  int d = 0;
  int e = 0;
  int r_stride_t = 0;
  int proj_stride_h = 0;
};

struct VerifyResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::size_t index = 0;
};

template <typename Element>
CUTLASS_HOST_DEVICE
float to_float(Element x) {
#if defined(__SYCL_DEVICE_ONLY__)
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    uint32_t bits = static_cast<uint32_t>(x.raw()) << 16;
    return sycl::bit_cast<float>(bits);
  } else {
    return static_cast<float>(x);
  }
#else
  return static_cast<float>(x);
#endif
}

template <typename Element>
CUTLASS_HOST_DEVICE
Element from_float(float x) {
  return static_cast<Element>(x);
}

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline int round_up(int x, int multiple) {
  return ceil_div(x, multiple) * multiple;
}

inline bool parse_bool(std::string const& value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

inline std::string bool_text(bool value) {
  return value ? "true" : "false";
}

inline std::string dtype_text(DType dtype) {
  switch (dtype) {
    case DType::kAll: return "all";
    case DType::kBf16: return "bf16";
    case DType::kFp16: return "fp16";
  }
  return "unknown";
}

inline bool parse_dtype(std::string const& text, DType& dtype) {
  if (text == "all") {
    dtype = DType::kAll;
    return true;
  }
  if (text == "bf16") {
    dtype = DType::kBf16;
    return true;
  }
  if (text == "fp16") {
    dtype = DType::kFp16;
    return true;
  }
  return false;
}

inline bool parse_fuse_tau(std::string const& text, bool& fuse_tau) {
  if (text == "fused" || text == "1") {
    fuse_tau = true;
    return true;
  }
  if (text == "none" || text == "0") {
    fuse_tau = false;
    return true;
  }
  return false;
}

inline Options parse_options(int argc, char const** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto eq = arg.find('=');
    std::string key = eq == std::string::npos ? arg : arg.substr(0, eq);
    std::string value = eq == std::string::npos ? "" : arg.substr(eq + 1);
    if (key == "--help" || key == "-h") {
      options.help = true;
    } else if (key == "--suite") {
      options.suite = value;
    } else if (key == "--shape") {
      options.shape = value;
    } else if (key == "--dtype") {
      if (!parse_dtype(value, options.dtype)) {
        throw std::invalid_argument("unknown dtype: " + value);
      }
    } else if (key == "--iterations") {
      options.iterations = std::stoi(value);
    } else if (key == "--warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "--verify") {
      options.verify = parse_bool(value);
    } else if (key == "--benchmark") {
      options.benchmark = parse_bool(value);
    } else if (key == "--target-gbps") {
      options.target_gbps = std::stod(value);
      options.target_gbps_set = true;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return options;
}

inline void print_common_usage(
    char const* name,
    char const* suite_text,
    char const* shape_text) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=" << suite_text << "       Built-in suite (default quick)\n"
      << "  --shape=" << shape_text << "\n"
      << "  --dtype=all|bf16|fp16    Element dtype (default all)\n"
      << "  --iterations=<int>       Timed kernel iterations (default 20)\n"
      << "  --warmup=<int>           Warmup launches before timing (default 5)\n"
      << "  --verify=0|1             Run CPU reference comparison (default 1)\n"
      << "  --benchmark=0|1          Run profiling-event timing (default 1)\n"
      << "  --target-gbps=<float>    Optional sustained effective GB/s gate; 0 disables\n";
}

inline bool parse_row_shape(std::string const& text, RowCase& cfg) {
  if (text.empty()) {
    return true;
  }
  int pad = -1;
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
    } else if (key == "rows") {
      cfg.rows = std::stoi(value);
    } else if (key == "inner") {
      cfg.inner = std::stoi(value);
    } else if (key == "stride") {
      cfg.stride = std::stoi(value);
    } else if (key == "pad") {
      pad = std::stoi(value);
    } else {
      return false;
    }
  }
  if (pad >= 0) {
    cfg.stride = cfg.inner + pad;
  }
  return true;
}

inline bool parse_rel_shape(std::string const& text, RelProjCase& cfg) {
  if (text.empty()) {
    return true;
  }
  int rpad = -1;
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
    } else if (key == "t") {
      cfg.t = std::stoi(value);
    } else if (key == "h") {
      cfg.h = std::stoi(value);
    } else if (key == "d") {
      cfg.d = std::stoi(value);
    } else if (key == "e") {
      cfg.e = std::stoi(value);
    } else if (key == "r_stride_t") {
      cfg.r_stride_t = std::stoi(value);
    } else if (key == "rpad") {
      rpad = std::stoi(value);
    } else if (key == "proj") {
      if (value == "head") {
        cfg.proj_per_head = true;
      } else if (value == "shared") {
        cfg.proj_per_head = false;
      } else {
        return false;
      }
    } else if (key == "tau") {
      if (!parse_fuse_tau(value, cfg.fuse_tau)) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (rpad >= 0) {
    cfg.r_stride_t = cfg.h * cfg.d + rpad;
  }
  return true;
}

inline void validate_row_case(RowCase& cfg) {
  if (cfg.rows <= 0 || cfg.inner <= 0) {
    throw std::invalid_argument("row case has non-positive shape");
  }
  if (cfg.stride < cfg.inner) {
    cfg.stride = cfg.inner;
  }
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
}

inline void validate_rel_case(RelProjCase& cfg) {
  if (cfg.t <= 0 || cfg.h <= 0 || cfg.d <= 0 || cfg.e <= 0) {
    throw std::invalid_argument("rel-proj case has non-positive shape");
  }
  if (cfg.r_stride_t < cfg.h * cfg.d) {
    cfg.r_stride_t = cfg.h * cfg.d;
  }
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
}

template <typename Element>
std::string element_dtype_text() {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  }
  return "fp16";
}

inline double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

template <typename Element>
VerifyResult compare_output(
    std::vector<Element> const& got,
    std::vector<Element> const& expected,
    double atol,
    double rtol) {
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double g = static_cast<double>(to_float(got[i]));
    double e = static_cast<double>(to_float(expected[i]));
    double abs_err = std::abs(g - e);
    double rel_err = abs_err / std::max(1.0, std::abs(e));
    if (abs_err > result.max_abs) {
      result.max_abs = abs_err;
      result.max_rel = rel_err;
      result.index = i;
    }
    if (abs_err > atol + rtol * std::abs(e)) {
      result.passed = false;
    }
  }
  return result;
}

inline void print_verify_result(char const* name, VerifyResult const& result) {
  std::cerr << "    " << name << " mismatch index=" << result.index
            << " max_abs=" << result.max_abs
            << " max_rel=" << result.max_rel << "\n";
}

template <typename Element, bool HasTau, int Vec>
class RowScaleKernel;

template <typename Element>
class RowCompactEsimdKernel {
 public:
  RowParams<Element> params;
  int chunks_per_row;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int total = params.rows * chunks_per_row;
    if (linear >= total) {
      return;
    }
    int row = linear / chunks_per_row;
    int chunk = linear - row * chunks_per_row;
    Element const* src_row = params.x + static_cast<int64_t>(row) * params.stride;
    Element* dst_row = params.out + static_cast<int64_t>(row) * params.inner;
    auto value = sycl::ext::intel::esimd::block_load<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<uint32_t const*>(src_row) + chunk * kRowEsimdCopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<uint32_t*>(dst_row) + chunk * kRowEsimdCopyWords, value);
  }
};

class RowScaleBf16EsimdKernel {
 public:
  RowParams<cutlass::bfloat16_t> params;
  int chunks_per_row;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int total = params.rows * chunks_per_row;
    if (linear >= total) {
      return;
    }
    int row = linear / chunks_per_row;
    int chunk = linear - row * chunks_per_row;
    cutlass::bfloat16_t const* src_row = params.x + static_cast<int64_t>(row) * params.stride;
    cutlass::bfloat16_t* dst_row = params.out + static_cast<int64_t>(row) * params.inner;
    float scale = params.tau[row];

    auto raw = sycl::ext::intel::esimd::block_load<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<uint32_t const*>(src_row) + chunk * kRowEsimdCopyWords);
    auto lo_bits = (raw & 0x0000ffffu) << 16;
    auto hi_bits = raw & 0xffff0000u;
    auto lo = lo_bits.template bit_cast_view<float>();
    auto hi = hi_bits.template bit_cast_view<float>();
    lo = lo * scale;
    hi = hi * scale;

    auto lo_fbits = lo.template bit_cast_view<uint32_t>();
    auto hi_fbits = hi.template bit_cast_view<uint32_t>();
    auto lo_round = ((lo_fbits >> 16) & 1u) + 0x7fffu;
    auto hi_round = ((hi_fbits >> 16) & 1u) + 0x7fffu;
    auto out = ((lo_fbits + lo_round) >> 16) |
        (((hi_fbits + hi_round) >> 16) << 16);
    sycl::ext::intel::esimd::block_store<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<uint32_t*>(dst_row) + chunk * kRowEsimdCopyWords, out);
  }
};

template <typename Element>
sycl::event launch_row_compact_esimd(
    sycl::queue& queue,
    RowParams<Element> const& params,
    int chunks_per_row) {
  int total = params.rows * chunks_per_row;
  RowCompactEsimdKernel<Element> kernel{params, chunks_per_row};
  return queue.parallel_for<RowCompactEsimdKernel<Element>>(
      sycl::range<1>(static_cast<std::size_t>(total)), kernel);
}

inline sycl::event launch_row_scale_bf16_esimd(
    sycl::queue& queue,
    RowParams<cutlass::bfloat16_t> const& params,
    int chunks_per_row) {
  int total = params.rows * chunks_per_row;
  RowScaleBf16EsimdKernel kernel{params, chunks_per_row};
  return queue.parallel_for<RowScaleBf16EsimdKernel>(
      sycl::range<1>(static_cast<std::size_t>(total)), kernel);
}

template <typename Element>
CUTLASS_DEVICE
Element element_from_raw16(uint16_t raw) {
  return Element::bitcast(raw);
}

template <typename Element>
CUTLASS_DEVICE
uint16_t element_to_raw16(Element value) {
  return value.raw();
}

template <typename Element>
CUTLASS_DEVICE
float raw16_to_float(uint16_t raw) {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(raw) << 16);
  } else {
    return to_float(element_from_raw16<Element>(raw));
  }
}

template <typename Element>
CUTLASS_DEVICE
uint16_t float_to_raw16(float value) {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    uint32_t bits = sycl::bit_cast<uint32_t>(value);
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
  } else {
    return element_to_raw16(from_float<Element>(value));
  }
}

template <typename Element>
CUTLASS_DEVICE
uint64_t scale_pack4(uint64_t raw, float scale) {
  uint64_t out = 0;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    uint16_t in_bits = static_cast<uint16_t>(raw >> (16 * lane));
    uint16_t out_bits = float_to_raw16<Element>(raw16_to_float<Element>(in_bits) * scale);
    out |= static_cast<uint64_t>(out_bits) << (16 * lane);
  }
  return out;
}

// bf16 round-trip of an fp32 intermediate. The software path below is four
// integer ops per value; on device the same RNE rounding is a single hardware
// convert pair, which is worth 8% of a decode launch because the fused-tau path
// round-trips all 16 D values per work item.
CUTLASS_DEVICE
inline float bf16_round_trip(float value) {
#if defined(__SYCL_DEVICE_ONLY__)
  return static_cast<float>(static_cast<sycl::ext::oneapi::bfloat16>(value));
#else
  return raw16_to_float<cutlass::bfloat16_t>(float_to_raw16<cutlass::bfloat16_t>(value));
#endif
}

template <typename Element, bool HasTau, int Vec>
sycl::event launch_row_kernel_static(sycl::queue& queue, RowParams<Element> const& params) {
  static_assert(Vec % kRowPackElems == 0, "row lane must contain whole 16B packs");
  constexpr int kPacksPerLane = Vec / kRowPackElems;
  RowParams<Element> launch_params = params;
  bool aligned = (params.inner % kRowPackElems == 0) &&
      (params.stride % kRowPackElems == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.x) % kRowPackBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.out) % kRowPackBytes == 0);
  int row_bytes = params.inner * static_cast<int>(sizeof(Element));
  int row_words = row_bytes / static_cast<int>(sizeof(uint32_t));
  if constexpr (!HasTau) {
    if (params.rows >= kRowEsimdMinRows &&
        aligned &&
        row_bytes % static_cast<int>(sizeof(uint32_t)) == 0 &&
        row_words % kRowEsimdCopyWords == 0) {
      return launch_row_compact_esimd<Element>(queue, params, row_words / kRowEsimdCopyWords);
    }
  }
  if constexpr (HasTau && std::is_same_v<Element, cutlass::bfloat16_t>) {
    if (params.rows >= kRowEsimdMinRows &&
        aligned &&
        row_bytes % static_cast<int>(sizeof(uint32_t)) == 0 &&
        row_words % kRowEsimdCopyWords == 0) {
      return launch_row_scale_bf16_esimd(queue, params, row_words / kRowEsimdCopyWords);
    }
  }
  launch_params.vec_count = aligned ? params.inner / Vec : 0;
  int scalar_tail = params.inner - launch_params.vec_count * Vec;
  launch_params.lanes_per_row = launch_params.vec_count + scalar_tail;
  int total = params.rows * launch_params.lanes_per_row;
  int local = kDefaultBlock;
  int global = round_up(total, local);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<RowScaleKernel<Element, HasTau, Vec>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local))),
        [=](sycl::nd_item<1> item) {
          int idx = static_cast<int>(item.get_global_id(0));
          if (idx >= total) {
            return;
          }
          int row = idx / launch_params.lanes_per_row;
          int lane = idx - row * launch_params.lanes_per_row;
          Element const* src_row = launch_params.x + static_cast<int64_t>(row) * launch_params.stride;
          Element* dst_row = launch_params.out + static_cast<int64_t>(row) * launch_params.inner;
          float scale = 1.0f;
          if constexpr (HasTau) {
            scale = launch_params.tau[row];
          }
          if (lane < launch_params.vec_count) {
            int col0 = lane * Vec;
            if constexpr (HasTau) {
#pragma unroll
              for (int pack = 0; pack < kPacksPerLane; ++pack) {
                Element const* src_pack = src_row + col0 + pack * kRowPackElems;
                Element* dst_pack = dst_row + col0 + pack * kRowPackElems;
                uint64_t raw0 = *reinterpret_cast<uint64_t const*>(src_pack);
                uint64_t raw1 = *reinterpret_cast<uint64_t const*>(src_pack + 4);
                *reinterpret_cast<uint64_t*>(dst_pack) = scale_pack4<Element>(raw0, scale);
                *reinterpret_cast<uint64_t*>(dst_pack + 4) = scale_pack4<Element>(raw1, scale);
              }
            } else {
              using pack_t = sycl::vec<uint32_t, kRowPackWords>;
#pragma unroll
              for (int pack = 0; pack < kPacksPerLane; ++pack) {
                int pack_idx = lane * kPacksPerLane + pack;
                pack_t value;
                value.load(pack_idx, reinterpret_cast<uint32_t const*>(src_row));
                value.store(pack_idx, reinterpret_cast<uint32_t*>(dst_row));
              }
            }
            return;
          }
          int col = launch_params.vec_count * Vec + (lane - launch_params.vec_count);
          Element value = src_row[col];
          if constexpr (HasTau) {
            value = from_float<Element>(to_float(value) * scale);
          }
          dst_row[col] = value;
        });
  });
}

template <typename Element, bool HasTau>
sycl::event launch_row_kernel(sycl::queue& queue, RowParams<Element> const& params) {
  int logical_work_items = params.rows * ceil_div(params.inner, kRowPackElems);
  if (logical_work_items <= kRowSmallWorkItemsThreshold) {
    return launch_row_kernel_static<Element, HasTau, kRowSmallLaneElems>(queue, params);
  }
  return launch_row_kernel_static<Element, HasTau, kRowLargeLaneElems>(queue, params);
}

template <typename Element, bool ProjPerHead, bool FuseTau, int Vec>
class RelProjKernel;

// A runtime integer division costs tens of instructions on Xe and sits ahead of
// every address computation, so at T=1 it is a large share of a launch that is
// already within 15% of an empty kernel: replacing the two divisions below with
// a multiply-shift took the 12-row decode case from 1.376 us to 1.192 us.
// Both divisors are known on the host, and every quotient this kernel needs has
// an exact 32-bit multiply-shift form: magic 1 for a power of two, magic 0 for a
// divisor larger than any value it will ever divide (the T=1 rows, where t is
// always 0), and a Granlund-Montgomery magic otherwise. Only the fallback needs
// a branch, so the hot path is a single multiply and shift.
//
// The two knobs below exist because a wrong magic is silent in exactly the way
// that costs the most time to find: the multiply is only exact while the
// host-side bound on the dividend holds, and an overflowing magic once broke
// every T > 1 case while every T = 1 case still passed (t is 0 there, so the
// quotient is 0 either way).
//   -DCUTLASS_RELPROJ_FAST_DIV=0        build the kernel with plain divisions,
//                                       the reference for both results and cost
//                                       (it reproduces the delta above: the
//                                       12-row decode cases read 1.379 us /
//                                       338.6 GB/s against 1.207 / 386.8, and
//                                       every case is 5-14% slower)
//   -DCUTLASS_RELPROJ_FAST_DIV_VERIFY=1 assert every magic against real division
//                                       over its whole declared dividend range
// Verification is host-side and O(max_value) per launch, so it is off by default
// and belongs in a debug or single-shot (--benchmark=0) run.
#if !defined(CUTLASS_RELPROJ_FAST_DIV)
#define CUTLASS_RELPROJ_FAST_DIV 1
#endif
#if !defined(CUTLASS_RELPROJ_FAST_DIV_VERIFY)
#define CUTLASS_RELPROJ_FAST_DIV_VERIFY 0
#endif

struct RelProjFastDiv {
  uint32_t magic = 0;
  int shift = -1;  // negative falls back to a real division
};

inline RelProjFastDiv make_rel_proj_fast_div_magic(int divisor, int max_value) {
  RelProjFastDiv fd;
  if (divisor <= 0 || max_value < 0) {
    return fd;
  }
  if (max_value < divisor) {
    fd.magic = 0;
    fd.shift = 0;
    return fd;
  }
  int l = 0;
  while ((1 << l) < divisor) {
    ++l;
  }
  if ((1 << l) == divisor) {
    fd.magic = 1;
    fd.shift = l;
    return fd;
  }
  // Smallest shift with max_value * divisor < 2^shift keeps the multiply exact
  // and the magic as small as possible; anything that still does not fit in 32
  // bits degrades to a correct division rather than a wrong quotient.
  uint64_t need = static_cast<uint64_t>(max_value) * static_cast<uint64_t>(divisor) + 1ull;
  int shift = 0;
  while ((1ull << shift) < need) {
    ++shift;
  }
  uint64_t magic = (1ull << shift) / static_cast<uint64_t>(divisor) + 1ull;
  if (magic > 0xffffffffull ||
      magic * static_cast<uint64_t>(max_value) > 0xffffffffull) {
    return fd;
  }
  fd.magic = static_cast<uint32_t>(magic);
  fd.shift = shift;
  return fd;
}

// All three encodings share the device expression below, so one loop checks any
// of them; the magic is built once per launch, so the check is affordable enough
// to leave enabled through a whole suite when a t/h index is under suspicion.
inline RelProjFastDiv make_rel_proj_fast_div(int divisor, int max_value) {
#if CUTLASS_RELPROJ_FAST_DIV
  RelProjFastDiv fd = make_rel_proj_fast_div_magic(divisor, max_value);
#if CUTLASS_RELPROJ_FAST_DIV_VERIFY
  if (fd.shift >= 0) {
    for (int value = 0; value <= max_value; ++value) {
      int got = static_cast<int>((static_cast<uint32_t>(value) * fd.magic) >> fd.shift);
      if (got != value / divisor) {
        std::cerr << "rel_proj fast div is wrong: " << value << " / " << divisor
                  << " gave " << got << ", expected " << (value / divisor)
                  << " (magic " << fd.magic << ", shift " << fd.shift
                  << ", max_value " << max_value << ")\n";
        std::abort();
      }
    }
  }
#endif
  return fd;
#else
  (void)divisor;
  (void)max_value;
  return RelProjFastDiv{};  // shift < 0: the device path divides for real
#endif
}

CUTLASS_DEVICE
inline int rel_proj_fast_div(int value, RelProjFastDiv fd, int divisor) {
#if CUTLASS_RELPROJ_FAST_DIV
  if (fd.shift < 0) {
    return value / divisor;
  }
  return static_cast<int>((static_cast<uint32_t>(value) * fd.magic) >> fd.shift);
#else
  (void)fd;  // no branch at all, so the division's cost is measured on its own
  return value / divisor;
#endif
}

// out[t, h, :] = bf16(tau[t] * r[t, h, :]) @ proj
//
// Production uses a 16 x 1024 bf16 projection (32 KiB), while the output has
// 6 to 768 rows. Each work-item therefore owns a vector of output columns,
// loads that projection slice once, and applies it to MTile rows. The launcher
// composes 8-, 4-, and 2-column kernels for an arbitrary even E, preserving
// this reuse without making E=1024 an interface constraint.
template <int MTile, int Vec, bool FuseTau>
class RelProjBf16D16SimtKernel {
 public:
  static_assert(Vec % 2 == 0, "Vec must be even so proj/out move as 32-bit pairs");

  RelProjParams<cutlass::bfloat16_t> params;
  int total;
  int e_offset;
  int col_slices;
  RelProjFastDiv col_div;
  RelProjFastDiv h_div;

  void operator()(sycl::nd_item<1> item) const {
    int idx = static_cast<int>(item.get_global_id(0));
    if (idx >= total) {
      return;
    }

    int m_tile = rel_proj_fast_div(idx, col_div, col_slices);
    int col_slice = idx - m_tile * col_slices;
    int e0 = e_offset + col_slice * Vec;

    float proj_tile[kRelProjProductionD][Vec];
#pragma unroll
    for (int d = 0; d < kRelProjProductionD; ++d) {
      uint32_t const* proj_row = reinterpret_cast<uint32_t const*>(
          params.proj + static_cast<int64_t>(d) * params.e);
#pragma unroll
      for (int i = 0; i < Vec / 2; ++i) {
        uint32_t pair = proj_row[e0 / 2 + i];
        proj_tile[d][2 * i] = raw16_to_float<cutlass::bfloat16_t>(
            static_cast<uint16_t>(pair & 0xffffu));
        proj_tile[d][2 * i + 1] = raw16_to_float<cutlass::bfloat16_t>(
            static_cast<uint16_t>(pair >> 16));
      }
    }

    int m = m_tile * MTile;
    int rows = params.t * params.h;
    int ti = rel_proj_fast_div(m, h_div, params.h);
    int hi = m - ti * params.h;

#pragma unroll
    for (int mm = 0; mm < MTile; ++mm) {
      if (m >= rows) {
        return;
      }

      float scale = 1.0f;
      if constexpr (FuseTau) {
        scale = params.tau[ti];
      }
      cutlass::bfloat16_t const* r_row =
          params.r + static_cast<int64_t>(ti) * params.r_stride_t + hi * kRelProjProductionD;
      float acc[Vec];
#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        acc[i] = 0.0f;
      }

      // Two D values per dword halves r's load messages; keeping only the two
      // live values (instead of an r[16] array) avoids the register pressure
      // that a full-row prepass costs.
      uint32_t const* r_pairs = reinterpret_cast<uint32_t const*>(r_row);
#pragma unroll
      for (int d = 0; d < kRelProjProductionD; d += 2) {
        uint32_t pair = r_pairs[d / 2];
        float r_lo = raw16_to_float<cutlass::bfloat16_t>(static_cast<uint16_t>(pair & 0xffffu));
        float r_hi = raw16_to_float<cutlass::bfloat16_t>(static_cast<uint16_t>(pair >> 16));
        if constexpr (FuseTau) {
          r_lo = bf16_round_trip(r_lo * scale);
          r_hi = bf16_round_trip(r_hi * scale);
        }
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          acc[i] += r_lo * proj_tile[d][i];
          acc[i] += r_hi * proj_tile[d + 1][i];
        }
      }

      uint32_t* out_row = reinterpret_cast<uint32_t*>(
          params.out + static_cast<int64_t>(m) * params.e);
#pragma unroll
      for (int i = 0; i < Vec / 2; ++i) {
        out_row[e0 / 2 + i] =
            static_cast<uint32_t>(float_to_raw16<cutlass::bfloat16_t>(acc[2 * i])) |
            (static_cast<uint32_t>(float_to_raw16<cutlass::bfloat16_t>(acc[2 * i + 1])) << 16);
      }

      ++m;
      if (++hi == params.h) {
        hi = 0;
        ++ti;
      }
    }
  }
};

// Launch shape for the register-tiled kernel, measured on B60 across the
// Inkling per-rank shapes (T in {1, 9, 32} x H in {3..48}, E in {512, 1024}).
// Two effects trade off against each other:
//   * A small Vec and a small work-group keep the launch short, which is all
//     that matters for the T=1 decode rows -- there the whole launch is within
//     15% of an empty kernel, so any extra code or work-group setup shows up
//     one-for-one (12 rows read 393 GB/s at Vec=2/local=16 against 332 at
//     Vec=8/local=256).
//   * Once there are enough rows to fill the machine, wider columns per
//     work-item amortize the 16 projection loads and win instead (768 rows read
//     3608 GB/s at Vec=8 against 1332 at Vec=2).
// MTile row reuse only pays in the middle band; past ~256 rows there are enough
// row tiles already and tiling them costs more than the reuse returns.
struct RelProjLaunchPlan {
  int vec = kRelProjVec;
  int local = kDefaultBlock;
  int mtile = 1;
};

inline RelProjLaunchPlan rel_proj_launch_plan(int rows) {
  if (rows <= 12) {
    return {2, 16, 1};
  }
  if (rows <= 40) {
    return {2, 64, 1};
  }
  if (rows <= 64) {
    return {8, 16, 1};
  }
  if (rows <= 256) {
    constexpr int kMinRowTiles = 40;
    int mtile = 1;
    while (mtile < 16 && rows >= kMinRowTiles * mtile * 2) {
      mtile *= 2;
    }
    return {4, 64, mtile};
  }
  return {8, 16, 1};
}

template <int MTile, bool FuseTau, int Vec>
sycl::event launch_rel_proj_bf16_d16_simt_static(
    sycl::queue& queue,
    RelProjParams<cutlass::bfloat16_t> const& params,
    int e_offset,
    int col_slices,
    int local_size,
    std::vector<sycl::event> const& dependencies = {}) {
  int total = ceil_div(params.t * params.h, MTile) * col_slices;
  int local = std::max(1, std::min(local_size, total));
  int global = round_up(total, local);
  RelProjBf16D16SimtKernel<MTile, Vec, FuseTau> kernel{
      params, total, e_offset, col_slices,
      make_rel_proj_fast_div(col_slices, total),
      make_rel_proj_fast_div(params.h, params.t * params.h)};
  return queue.submit([&](sycl::handler& cgh) {
    if (!dependencies.empty()) {
      cgh.depends_on(dependencies);
    }
    cgh.parallel_for<RelProjBf16D16SimtKernel<MTile, Vec, FuseTau>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local))),
        kernel);
  });
}

template <int MTile, bool FuseTau, int VecMain>
sycl::event launch_rel_proj_bf16_d16_simt_segments(
    sycl::queue& queue,
    RelProjParams<cutlass::bfloat16_t> const& params,
    int local_size) {
  std::vector<sycl::event> dependencies;
  sycl::event last;
  int e_offset = 0;
  int remaining = params.e;

  int col_slices = remaining / VecMain;
  if (col_slices > 0) {
    last = launch_rel_proj_bf16_d16_simt_static<MTile, FuseTau, VecMain>(
        queue, params, e_offset, col_slices, local_size, dependencies);
    dependencies = {last};
    e_offset += col_slices * VecMain;
    remaining -= col_slices * VecMain;
  }
  if constexpr (VecMain > 4) {
    if (remaining >= 4) {
      last = launch_rel_proj_bf16_d16_simt_static<MTile, FuseTau, 4>(
          queue, params, e_offset, 1, local_size, dependencies);
      dependencies = {last};
      e_offset += 4;
      remaining -= 4;
    }
  }
  if constexpr (VecMain > 2) {
    if (remaining >= 2) {
      last = launch_rel_proj_bf16_d16_simt_static<MTile, FuseTau, 2>(
          queue, params, e_offset, 1, local_size, dependencies);
    }
  }
  return last;
}

// Only the Vec=4 band ever asks for MTile > 1, so the other widths are
// instantiated once and the kernel count stays bounded.
template <bool FuseTau>
sycl::event launch_rel_proj_bf16_d16_simt_mtile(
    sycl::queue& queue,
    RelProjParams<cutlass::bfloat16_t> const& params) {
  RelProjLaunchPlan plan = rel_proj_launch_plan(params.t * params.h);
  if (plan.vec == 2) {
    return launch_rel_proj_bf16_d16_simt_segments<1, FuseTau, 2>(queue, params, plan.local);
  }
  if (plan.vec == 4) {
    switch (plan.mtile) {
      case 1: return launch_rel_proj_bf16_d16_simt_segments<1, FuseTau, 4>(queue, params, plan.local);
      case 2: return launch_rel_proj_bf16_d16_simt_segments<2, FuseTau, 4>(queue, params, plan.local);
      case 4: return launch_rel_proj_bf16_d16_simt_segments<4, FuseTau, 4>(queue, params, plan.local);
      case 8: return launch_rel_proj_bf16_d16_simt_segments<8, FuseTau, 4>(queue, params, plan.local);
      default: return launch_rel_proj_bf16_d16_simt_segments<16, FuseTau, 4>(queue, params, plan.local);
    }
  }
  return launch_rel_proj_bf16_d16_simt_segments<1, FuseTau, 8>(queue, params, plan.local);
}

// Both tau settings reuse the same register-tiled kernel. Leaving the tau=none
// case on the generic fallback cost 5x (12.5 us against 2.5 us for the identical
// shape with tau fused) because the fallback re-reads proj per output element
// instead of holding a register tile across MTile rows.
inline sycl::event launch_rel_proj_bf16_d16_simt(
    sycl::queue& queue,
    RelProjParams<cutlass::bfloat16_t> const& params,
    bool fuse_tau) {
  return fuse_tau ? launch_rel_proj_bf16_d16_simt_mtile<true>(queue, params)
                  : launch_rel_proj_bf16_d16_simt_mtile<false>(queue, params);
}

template <typename Element, bool ProjPerHead, bool FuseTau, int Vec = kRelProjVec>
sycl::event launch_rel_proj_static(sycl::queue& queue, RelProjParams<Element> const& params) {
  int e_vecs = ceil_div(params.e, Vec);
  int total = params.t * params.h * e_vecs;
  int local = kDefaultBlock;
  int global = round_up(total, local);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<RelProjKernel<Element, ProjPerHead, FuseTau, Vec>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local))),
        [=](sycl::nd_item<1> item) {
          int idx = static_cast<int>(item.get_global_id(0));
          if (idx >= total) {
            return;
          }
          int ev = idx % e_vecs;
          int th = idx / e_vecs;
          int ti = th / params.h;
          int hi = th - ti * params.h;
          int e0 = ev * Vec;

          float scale = 1.0f;
          if constexpr (FuseTau) {
            scale = params.tau[ti];
          }

          float acc[Vec];
#pragma unroll
          for (int i = 0; i < Vec; ++i) {
            acc[i] = 0.0f;
          }

          Element const* r_row = params.r + static_cast<int64_t>(ti) * params.r_stride_t +
              static_cast<int64_t>(hi) * params.d;
          Element const* proj_base = params.proj;
          if constexpr (ProjPerHead) {
            proj_base += static_cast<int64_t>(hi) * params.proj_stride_h;
          }

          for (int d = 0; d < params.d; ++d) {
            float rv = to_float(r_row[d]);
            if constexpr (FuseTau) {
              rv = to_float(from_float<Element>(rv * scale));
            }
            Element const* proj_row = proj_base + static_cast<int64_t>(d) * params.e;
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
              int e_col = e0 + i;
              if (e_col < params.e) {
                acc[i] += rv * to_float(proj_row[e_col]);
              }
            }
          }

          Element* out_row = params.out +
              (static_cast<int64_t>(ti) * params.h + hi) * params.e;
#pragma unroll
          for (int i = 0; i < Vec; ++i) {
            int e_col = e0 + i;
            if (e_col < params.e) {
              out_row[e_col] = from_float<Element>(acc[i]);
            }
          }
        });
  });
}

template <typename Element>
sycl::event launch_rel_proj(sycl::queue& queue, RelProjParams<Element> const& params, RelProjCase const& cfg) {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    if (!cfg.proj_per_head &&
        params.d == kRelProjProductionD &&
        params.e % 2 == 0 &&
        params.r_stride_t % 2 == 0) {
      return launch_rel_proj_bf16_d16_simt(queue, params, cfg.fuse_tau);
    }
  }
  if (cfg.proj_per_head) {
    return cfg.fuse_tau ? launch_rel_proj_static<Element, true, true>(queue, params)
                        : launch_rel_proj_static<Element, true, false>(queue, params);
  }
  return cfg.fuse_tau ? launch_rel_proj_static<Element, false, true>(queue, params)
                      : launch_rel_proj_static<Element, false, false>(queue, params);
}

template <typename Element>
struct RowHostTensors {
  std::vector<Element> x;
  std::vector<float> tau;
  std::vector<Element> out;
  std::vector<Element> ref;
};

template <typename Element>
RowHostTensors<Element> initialize_row_case(
    RowCase const& cfg,
    bool has_tau,
    bool build_reference,
    uint32_t seed) {
  RowHostTensors<Element> h;
  h.x.resize(static_cast<std::size_t>(cfg.rows) * cfg.stride);
  h.tau.resize(static_cast<std::size_t>(cfg.rows));

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> value_dist(-0.75f, 0.75f);
  std::uniform_real_distribution<float> tau_dist(0.70f, 1.35f);
  for (Element& x : h.x) {
    x = from_float<Element>(value_dist(gen));
  }
  for (float& t : h.tau) {
    t = tau_dist(gen);
  }

  if (build_reference) {
    h.out.resize(static_cast<std::size_t>(cfg.rows) * cfg.inner);
    h.ref.resize(h.out.size());
    std::fill(h.out.begin(), h.out.end(), from_float<Element>(0.0f));

    for (int row = 0; row < cfg.rows; ++row) {
      float scale = h.tau[row];
      for (int col = 0; col < cfg.inner; ++col) {
        Element value = h.x[static_cast<std::size_t>(row) * cfg.stride + col];
        if (has_tau) {
          value = from_float<Element>(to_float(value) * scale);
        }
        h.ref[static_cast<std::size_t>(row) * cfg.inner + col] = value;
      }
    }
  }
  return h;
}

template <typename Element>
struct RelHostTensors {
  std::vector<Element> r;
  std::vector<Element> proj;
  std::vector<float> tau;
  std::vector<Element> out;
  std::vector<Element> ref;
};

template <typename Element>
RelHostTensors<Element> initialize_rel_case(
    RelProjCase const& cfg,
    bool build_reference,
    uint32_t seed) {
  RelHostTensors<Element> h;
  std::size_t proj_count = static_cast<std::size_t>(cfg.proj_per_head ? cfg.h : 1) * cfg.d * cfg.e;
  std::size_t tau_count = cfg.fuse_tau ? static_cast<std::size_t>(cfg.t) : 0;

  h.r.resize(static_cast<std::size_t>(cfg.t) * cfg.r_stride_t);
  h.proj.resize(proj_count);
  h.tau.resize(std::max<std::size_t>(tau_count, 1), 1.0f);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> value_dist(-0.45f, 0.45f);
  std::uniform_real_distribution<float> tau_dist(0.75f, 1.25f);
  for (Element& x : h.r) {
    x = from_float<Element>(value_dist(gen));
  }
  for (Element& x : h.proj) {
    x = from_float<Element>(value_dist(gen));
  }
  for (std::size_t i = 0; i < tau_count; ++i) {
    h.tau[i] = tau_dist(gen);
  }

  if (build_reference) {
    h.out.resize(static_cast<std::size_t>(cfg.t) * cfg.h * cfg.e);
    h.ref.resize(h.out.size());
    std::fill(h.out.begin(), h.out.end(), from_float<Element>(0.0f));

    for (int ti = 0; ti < cfg.t; ++ti) {
      for (int hi = 0; hi < cfg.h; ++hi) {
        float scale = cfg.fuse_tau ? h.tau[ti] : 1.0f;
        for (int e_col = 0; e_col < cfg.e; ++e_col) {
          float acc = 0.0f;
          for (int d = 0; d < cfg.d; ++d) {
            float rv = to_float(h.r[static_cast<std::size_t>(ti) * cfg.r_stride_t + hi * cfg.d + d]);
            if (cfg.fuse_tau) {
              rv = to_float(from_float<Element>(rv * scale));
            }
            std::size_t proj_offset = static_cast<std::size_t>(d) * cfg.e + e_col;
            if (cfg.proj_per_head) {
              proj_offset += static_cast<std::size_t>(hi) * cfg.d * cfg.e;
            }
            acc += rv * to_float(h.proj[proj_offset]);
          }
          h.ref[(static_cast<std::size_t>(ti) * cfg.h + hi) * cfg.e + e_col] = from_float<Element>(acc);
        }
      }
    }
  }
  return h;
}

template <typename Element, bool HasTau>
bool run_row_case(
    sycl::queue& queue,
    RowCase cfg,
    Options const& options,
    char const* op_name) {
  validate_row_case(cfg);
  RowHostTensors<Element> h = initialize_row_case<Element>(
      cfg,
      HasTau,
      options.verify,
      1337u + static_cast<uint32_t>(cfg.rows * 13 + cfg.inner * 17 + cfg.stride));
  std::size_t out_count = static_cast<std::size_t>(cfg.rows) * cfg.inner;

  DeviceBuffer<Element> d_x(queue, h.x.size());
  DeviceBuffer<float> d_tau(queue, h.tau.size());
  DeviceBuffer<Element> d_out(queue, out_count);
  d_x.copy_from(h.x);
  d_tau.copy_from(h.tau);
  if (options.verify) {
    d_out.copy_from(h.out);
  }

  RowParams<Element> params{};
  params.x = d_x.get();
  params.tau = d_tau.get();
  params.out = d_out.get();
  params.rows = cfg.rows;
  params.inner = cfg.inner;
  params.stride = cfg.stride;

  auto launch = [&]() {
    return launch_row_kernel<Element, HasTau>(queue, params);
  };

  bool passed = true;
  if (options.verify) {
    launch().wait_and_throw();
    d_out.copy_to(h.out);
    double atol = std::is_same_v<Element, cutlass::bfloat16_t> ? 1.0e-2 : 1.0e-3;
    double rtol = std::is_same_v<Element, cutlass::bfloat16_t> ? 1.0e-2 : 1.0e-3;
    VerifyResult result = compare_output(h.out, h.ref, atol, rtol);
    passed = result.passed;
    if (!passed) {
      print_verify_result("out", result);
    }
  }

  double avg_ms = 0.0;
  double gbps = 0.0;
  double tops = 0.0;
  bool perf_passed = true;
  double target_gbps = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
  if constexpr (HasTau && std::is_same_v<Element, cutlass::half_t>) {
    if (!options.target_gbps_set) {
      target_gbps = 0.0;
    }
  }
  double bytes = static_cast<double>(cfg.rows) * cfg.inner * sizeof(Element) * 2.0;
  if constexpr (HasTau) {
    bytes += static_cast<double>(cfg.rows) * sizeof(float);
  }
  double flops = HasTau ? static_cast<double>(cfg.rows) * cfg.inner : 0.0;

  if (options.benchmark) {
    int warmup = std::max(0, options.warmup);
    for (int i = 0; i < warmup; ++i) {
      launch().wait_and_throw();
    }
    int timing_iterations = std::max(1, options.iterations);
    std::vector<sycl::event> events;
    events.reserve(static_cast<std::size_t>(timing_iterations));
    for (int i = 0; i < timing_iterations; ++i) {
      events.push_back(launch());
    }
    queue.wait_and_throw();
    double total_ms = 0.0;
    for (sycl::event const& event : events) {
      total_ms += event_ms(event);
    }
    avg_ms = total_ms / static_cast<double>(timing_iterations);
    gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
    tops = flops / (avg_ms * 1.0e-3) / 1.0e12;
    perf_passed = target_gbps <= 0.0 || bytes < kMinSustainedTargetBytes || gbps >= target_gbps;
    passed = passed && perf_passed;
  }

  std::cout << "  [" << element_dtype_text<Element>() << "] "
            << std::left << std::setw(28) << cfg.name << std::right
            << " rows=" << cfg.rows
            << " inner=" << cfg.inner
            << " stride=" << cfg.stride
            << " op=" << op_name;
  if (options.benchmark) {
    std::cout << "  " << std::fixed << std::setprecision(3) << (avg_ms * 1000.0) << " us"
              << "  " << std::setprecision(2) << gbps << " GB/s"
              << "  " << std::setprecision(3) << tops << " TOPS";
    if (target_gbps > 0.0) {
      std::cout << " target=" << std::setprecision(2) << target_gbps << " GB/s";
    }
  }
  std::cout << "  " << (options.verify ? (passed ? "passed" : "FAILED") :
                             (perf_passed ? "verification skipped" : "FAILED"))
            << "\n";
  if (!perf_passed) {
    std::cerr << "    performance target failed: " << gbps
              << " GB/s < " << target_gbps << " GB/s\n";
  }
  return passed;
}

template <typename Element>
bool run_rel_case(sycl::queue& queue, RelProjCase cfg, Options const& options) {
  validate_rel_case(cfg);
  RelHostTensors<Element> h = initialize_rel_case<Element>(
      cfg,
      options.verify,
      2027u + static_cast<uint32_t>(cfg.t * 19 + cfg.h * 23 + cfg.d * 29 + cfg.e));
  std::size_t out_count = static_cast<std::size_t>(cfg.t) * cfg.h * cfg.e;

  DeviceBuffer<Element> d_r(queue, h.r.size());
  DeviceBuffer<Element> d_proj(queue, h.proj.size());
  DeviceBuffer<float> d_tau(queue, h.tau.size());
  DeviceBuffer<Element> d_out(queue, out_count);
  d_r.copy_from(h.r);
  d_proj.copy_from(h.proj);
  d_tau.copy_from(h.tau);
  if (options.verify) {
    d_out.copy_from(h.out);
  }

  RelProjParams<Element> params{};
  params.r = d_r.get();
  params.proj = d_proj.get();
  params.tau = d_tau.get();
  params.out = d_out.get();
  params.t = cfg.t;
  params.h = cfg.h;
  params.d = cfg.d;
  params.e = cfg.e;
  params.r_stride_t = cfg.r_stride_t;
  params.proj_stride_h = cfg.d * cfg.e;

  auto launch = [&]() {
    return launch_rel_proj<Element>(queue, params, cfg);
  };

  bool passed = true;
  if (options.verify) {
    launch().wait_and_throw();
    d_out.copy_to(h.out);
    double atol = std::is_same_v<Element, cutlass::bfloat16_t> ? 1.8e-2 : 2.5e-3;
    double rtol = std::is_same_v<Element, cutlass::bfloat16_t> ? 1.8e-2 : 2.5e-3;
    VerifyResult result = compare_output(h.out, h.ref, atol, rtol);
    passed = result.passed;
    if (!passed) {
      print_verify_result("out", result);
    }
  }

  double avg_ms = 0.0;
  double gbps = 0.0;
  double tops = 0.0;
  bool perf_passed = true;
  double target_gbps = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
  int e_vecs = ceil_div(cfg.e, kRelProjVec);
  double r_bytes = static_cast<double>(cfg.t) * cfg.h * e_vecs * cfg.d * sizeof(Element);
  double proj_bytes = static_cast<double>(cfg.t) * cfg.h * cfg.d * cfg.e * sizeof(Element);
  double out_bytes = static_cast<double>(cfg.t) * cfg.h * cfg.e * sizeof(Element);
  double tau_bytes = cfg.fuse_tau ? static_cast<double>(cfg.t) * sizeof(float) : 0.0;
  double bytes = r_bytes + proj_bytes + out_bytes + tau_bytes;
  double flops = static_cast<double>(cfg.t) * cfg.h * cfg.d * cfg.e * 2.0;
  if (cfg.fuse_tau) {
    flops += static_cast<double>(cfg.t) * cfg.h * cfg.d;
  }

  if (options.benchmark) {
    int warmup = std::max(0, options.warmup);
    for (int i = 0; i < warmup; ++i) {
      launch().wait_and_throw();
    }
    int timing_iterations = std::max(1, options.iterations);
    std::vector<sycl::event> events;
    events.reserve(static_cast<std::size_t>(timing_iterations));
    for (int i = 0; i < timing_iterations; ++i) {
      events.push_back(launch());
    }
    queue.wait_and_throw();
    double total_ms = 0.0;
    for (sycl::event const& event : events) {
      total_ms += event_ms(event);
    }
    avg_ms = total_ms / static_cast<double>(timing_iterations);
    gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
    tops = flops / (avg_ms * 1.0e-3) / 1.0e12;
    perf_passed = target_gbps <= 0.0 || bytes < kMinSustainedTargetBytes || gbps >= target_gbps;
    passed = passed && perf_passed;
  }

  std::cout << "  [" << element_dtype_text<Element>() << "] "
            << std::left << std::setw(28) << cfg.name << std::right
            << " T=" << cfg.t
            << " H=" << cfg.h
            << " D=" << cfg.d
            << " E=" << cfg.e
            << " r_stride_t=" << cfg.r_stride_t
            << " proj=" << (cfg.proj_per_head ? "head" : "shared")
            << " tau=" << (cfg.fuse_tau ? "fused" : "none");
  if (options.benchmark) {
    std::cout << "  " << std::fixed << std::setprecision(3) << (avg_ms * 1000.0) << " us"
              << "  " << std::setprecision(2) << gbps << " GB/s"
              << "  " << std::setprecision(3) << tops << " TOPS";
    if (target_gbps > 0.0) {
      std::cout << " target=" << std::setprecision(2) << target_gbps << " GB/s";
    }
  }
  std::cout << "  " << (options.verify ? (passed ? "passed" : "FAILED") :
                             (perf_passed ? "verification skipped" : "FAILED"))
            << "\n";
  if (!perf_passed) {
    std::cerr << "    performance target failed: " << gbps
              << " GB/s < " << target_gbps << " GB/s\n";
  }
  return passed;
}

template <typename Element, bool HasTau>
bool run_row_cases_for_dtype(
    sycl::queue& queue,
    std::vector<RowCase> const& cases,
    Options const& options,
    char const* op_name) {
  bool all_passed = true;
  for (RowCase cfg : cases) {
    all_passed &= run_row_case<Element, HasTau>(queue, cfg, options, op_name);
  }
  return all_passed;
}

template <typename Element>
bool run_rel_cases_for_dtype(
    sycl::queue& queue,
    std::vector<RelProjCase> const& cases,
    Options const& options) {
  bool all_passed = true;
  for (RelProjCase cfg : cases) {
    all_passed &= run_rel_case<Element>(queue, cfg, options);
  }
  return all_passed;
}

inline sycl::queue make_queue() {
  return sycl::queue(
      sycl::gpu_selector_v,
      sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
}

}  // namespace cutlass::examples::relative_helpers
