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

enum TauMode : int {
  kTauNone = 0,
  kTauPreToken = 1,
  kTauPreRow = 2,
  kTauPostToken = 3,
  kTauPostRow = 4
};

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
  TauMode tau_mode = kTauPreToken;
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

inline std::string tau_mode_text(TauMode mode) {
  switch (mode) {
    case kTauNone: return "none";
    case kTauPreToken: return "pre_token";
    case kTauPreRow: return "pre_row";
    case kTauPostToken: return "post_token";
    case kTauPostRow: return "post_row";
  }
  return "unknown";
}

inline bool parse_tau_mode(std::string const& text, TauMode& mode) {
  if (text == "none") {
    mode = kTauNone;
    return true;
  }
  if (text == "pre_token") {
    mode = kTauPreToken;
    return true;
  }
  if (text == "pre_row") {
    mode = kTauPreRow;
    return true;
  }
  if (text == "post_token") {
    mode = kTauPostToken;
    return true;
  }
  if (text == "post_row") {
    mode = kTauPostRow;
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
      if (!parse_tau_mode(value, cfg.tau_mode)) {
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

template <int Tau>
static constexpr bool tau_is_pre_v = Tau == kTauPreToken || Tau == kTauPreRow;

template <int Tau>
static constexpr bool tau_is_post_v = Tau == kTauPostToken || Tau == kTauPostRow;

template <int Tau>
static constexpr bool tau_is_row_v = Tau == kTauPreRow || Tau == kTauPostRow;

template <typename Element, bool ProjPerHead, int Tau, int Vec>
class RelProjKernel;

// out[t, h, :] = bf16(tau[t] * r[t, h, :]) @ proj
//
// Production uses a 16 x 1024 bf16 projection (32 KiB), while the output has
// 6 to 768 rows. Each work-item therefore owns a vector of output columns,
// loads that projection slice once, and applies it to MTile rows. The launcher
// composes 8-, 4-, and 2-column kernels for an arbitrary even E, preserving
// this reuse without making E=1024 an interface constraint.
template <int MTile, int Vec>
class RelProjBf16D16SimtKernel {
 public:
  static_assert(Vec % 2 == 0, "Vec must be even so proj/out move as 32-bit pairs");

  RelProjParams<cutlass::bfloat16_t> params;
  int total;
  int e_offset;
  int col_slices;

  void operator()(sycl::nd_item<1> item) const {
    int idx = static_cast<int>(item.get_global_id(0));
    if (idx >= total) {
      return;
    }

    int col_slice = idx % col_slices;
    int m_tile = idx / col_slices;
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
    int ti = m / params.h;
    int hi = m - ti * params.h;

#pragma unroll
    for (int mm = 0; mm < MTile; ++mm) {
      if (m >= rows) {
        return;
      }

      float scale = params.tau[ti];
      cutlass::bfloat16_t const* r_row =
          params.r + static_cast<int64_t>(ti) * params.r_stride_t + hi * kRelProjProductionD;
      float acc[Vec];
#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        acc[i] = 0.0f;
      }

#pragma unroll
      for (int d = 0; d < kRelProjProductionD; ++d) {
        uint16_t r_bits = element_to_raw16(r_row[d]);
        float r_value = raw16_to_float<cutlass::bfloat16_t>(
            float_to_raw16<cutlass::bfloat16_t>(
                raw16_to_float<cutlass::bfloat16_t>(r_bits) * scale));
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          acc[i] += r_value * proj_tile[d][i];
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

template <int MTile, int Vec = kRelProjVec>
sycl::event launch_rel_proj_bf16_d16_simt_static(
    sycl::queue& queue,
    RelProjParams<cutlass::bfloat16_t> const& params,
    int e_offset,
    int col_slices,
    std::vector<sycl::event> const& dependencies = {}) {
  int total = ceil_div(params.t * params.h, MTile) * col_slices;
  int local = std::min(kDefaultBlock, col_slices);
  int global = round_up(total, local);
  RelProjBf16D16SimtKernel<MTile, Vec> kernel{params, total, e_offset, col_slices};
  return queue.submit([&](sycl::handler& cgh) {
    if (!dependencies.empty()) {
      cgh.depends_on(dependencies);
    }
    cgh.parallel_for<RelProjBf16D16SimtKernel<MTile, Vec>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local))),
        kernel);
  });
}

template <int MTile>
sycl::event launch_rel_proj_bf16_d16_simt_segments(
    sycl::queue& queue,
    RelProjParams<cutlass::bfloat16_t> const& params) {
  std::vector<sycl::event> dependencies;
  sycl::event last;
  int e_offset = 0;
  int remaining = params.e;

  int col_slices = remaining / kRelProjVec;
  if (col_slices > 0) {
    last = launch_rel_proj_bf16_d16_simt_static<MTile, kRelProjVec>(
        queue, params, e_offset, col_slices, dependencies);
    dependencies = {last};
    e_offset += col_slices * kRelProjVec;
    remaining -= col_slices * kRelProjVec;
  }
  if (remaining >= 4) {
    last = launch_rel_proj_bf16_d16_simt_static<MTile, 4>(
        queue, params, e_offset, 1, dependencies);
    dependencies = {last};
    e_offset += 4;
    remaining -= 4;
  }
  if (remaining >= 2) {
    last = launch_rel_proj_bf16_d16_simt_static<MTile, 2>(
        queue, params, e_offset, 1, dependencies);
  }
  return last;
}

inline sycl::event launch_rel_proj_bf16_d16_simt(
    sycl::queue& queue,
    RelProjParams<cutlass::bfloat16_t> const& params) {
  constexpr int kMinRowTiles = 40;
  int rows = params.t * params.h;
  if (rows >= kMinRowTiles * 16) {
    return launch_rel_proj_bf16_d16_simt_segments<16>(queue, params);
  }
  if (rows >= kMinRowTiles * 8) {
    return launch_rel_proj_bf16_d16_simt_segments<8>(queue, params);
  }
  if (rows >= kMinRowTiles * 4) {
    return launch_rel_proj_bf16_d16_simt_segments<4>(queue, params);
  }
  if (rows >= kMinRowTiles * 2) {
    return launch_rel_proj_bf16_d16_simt_segments<2>(queue, params);
  }
  return launch_rel_proj_bf16_d16_simt_segments<1>(queue, params);
}

template <typename Element, bool ProjPerHead, int Tau, int Vec = kRelProjVec>
sycl::event launch_rel_proj_static(sycl::queue& queue, RelProjParams<Element> const& params) {
  int e_vecs = ceil_div(params.e, Vec);
  int total = params.t * params.h * e_vecs;
  int local = kDefaultBlock;
  int global = round_up(total, local);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<RelProjKernel<Element, ProjPerHead, Tau, Vec>>(
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
          if constexpr (Tau != kTauNone) {
            if constexpr (tau_is_row_v<Tau>) {
              scale = params.tau[ti * params.h + hi];
            } else {
              scale = params.tau[ti];
            }
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
            if constexpr (tau_is_pre_v<Tau>) {
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
              float value = acc[i];
              if constexpr (tau_is_post_v<Tau>) {
                value *= scale;
              }
              out_row[e_col] = from_float<Element>(value);
            }
          }
        });
  });
}

template <typename Element>
sycl::event launch_rel_proj(sycl::queue& queue, RelProjParams<Element> const& params, RelProjCase const& cfg) {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    if (!cfg.proj_per_head &&
        cfg.tau_mode == kTauPreToken &&
        params.d == kRelProjProductionD &&
        params.e % 2 == 0) {
      return launch_rel_proj_bf16_d16_simt(queue, params);
    }
  }
  if (cfg.proj_per_head) {
    switch (cfg.tau_mode) {
      case kTauNone: return launch_rel_proj_static<Element, true, kTauNone>(queue, params);
      case kTauPreToken: return launch_rel_proj_static<Element, true, kTauPreToken>(queue, params);
      case kTauPreRow: return launch_rel_proj_static<Element, true, kTauPreRow>(queue, params);
      case kTauPostToken: return launch_rel_proj_static<Element, true, kTauPostToken>(queue, params);
      case kTauPostRow: return launch_rel_proj_static<Element, true, kTauPostRow>(queue, params);
    }
  } else {
    switch (cfg.tau_mode) {
      case kTauNone: return launch_rel_proj_static<Element, false, kTauNone>(queue, params);
      case kTauPreToken: return launch_rel_proj_static<Element, false, kTauPreToken>(queue, params);
      case kTauPreRow: return launch_rel_proj_static<Element, false, kTauPreRow>(queue, params);
      case kTauPostToken: return launch_rel_proj_static<Element, false, kTauPostToken>(queue, params);
      case kTauPostRow: return launch_rel_proj_static<Element, false, kTauPostRow>(queue, params);
    }
  }
  throw std::invalid_argument("unknown tau mode");
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
  std::size_t tau_count = 0;
  if (cfg.tau_mode == kTauPreToken || cfg.tau_mode == kTauPostToken) {
    tau_count = static_cast<std::size_t>(cfg.t);
  } else if (cfg.tau_mode == kTauPreRow || cfg.tau_mode == kTauPostRow) {
    tau_count = static_cast<std::size_t>(cfg.t) * cfg.h;
  }

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
        float scale = 1.0f;
        if (cfg.tau_mode == kTauPreToken || cfg.tau_mode == kTauPostToken) {
          scale = h.tau[ti];
        } else if (cfg.tau_mode == kTauPreRow || cfg.tau_mode == kTauPostRow) {
          scale = h.tau[static_cast<std::size_t>(ti) * cfg.h + hi];
        }
        for (int e_col = 0; e_col < cfg.e; ++e_col) {
          float acc = 0.0f;
          for (int d = 0; d < cfg.d; ++d) {
            float rv = to_float(h.r[static_cast<std::size_t>(ti) * cfg.r_stride_t + hi * cfg.d + d]);
            if (cfg.tau_mode == kTauPreToken || cfg.tau_mode == kTauPreRow) {
              rv = to_float(from_float<Element>(rv * scale));
            }
            std::size_t proj_offset = static_cast<std::size_t>(d) * cfg.e + e_col;
            if (cfg.proj_per_head) {
              proj_offset += static_cast<std::size_t>(hi) * cfg.d * cfg.e;
            }
            acc += rv * to_float(h.proj[proj_offset]);
          }
          if (cfg.tau_mode == kTauPostToken || cfg.tau_mode == kTauPostRow) {
            acc *= scale;
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
  double tau_bytes = cfg.tau_mode == kTauNone ? 0.0 :
      static_cast<double>(cfg.t) * (cfg.tau_mode == kTauPreRow || cfg.tau_mode == kTauPostRow ? cfg.h : 1) * sizeof(float);
  double bytes = r_bytes + proj_bytes + out_bytes + tau_bytes;
  double flops = static_cast<double>(cfg.t) * cfg.h * cfg.d * cfg.e * 2.0;
  if (cfg.tau_mode != kTauNone) {
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
            << " tau=" << tau_mode_text(cfg.tau_mode);
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
