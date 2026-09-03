/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling quantization examples for CUTLASS SYCL on BMG.
 *
 * Roofline summary:
 *   NVFP4 layout quantization is memory-bound for inference weight preparation.
 *   Each 16-value group streams one input element once, writes one packed FP4
 *   byte per two values, and writes one FP8 scale byte. For bf16/fp16 inputs
 *   this is roughly 2.56 bytes/value before metadata padding, while the math is
 *   a max reduction plus scalar FP4/FP8 encoding. Sustained bandwidth is the
 *   relevant target, so perf cases use large matrices and the kernel keeps
 *   group dimensions runtime-variable.
 *
 *   MXFP4 per-token group quantization is also memory-bound. Each 32-value
 *   group streams one input element, writes 16 packed bytes, and writes one
 *   UE8M0 scale byte. The optimized bf16/fp16 path keeps the group in raw
 *   16-bit words for the max pass, then converts once for FP4 packing. Perf
 *   cases use production-like token/hidden sizes to avoid measuring cache-only
 *   behavior.
 *
 *   MXFP8 KV-cache quantization (the activation quantizer the Inkling model
 *   actually runs) is memory-bound as well: per 32-channel group it streams one
 *   input element, writes 32 E4M3 payload bytes, and writes one E8M0 scale byte
 *   into the interleaved FA4 scale-factor buffer. Static per-tensor E4M3 scaling
 *   is the cheapest of the three: no reduction at all, one multiply by a
 *   preloaded reciprocal scale plus a byte store.
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cutlass::examples::bmg_quantization {

constexpr int kDefaultBlock = 256;
constexpr int kNvfp4GroupSize = 16;
constexpr int kMxfp4GroupSize = 32;
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3FnMax = 448.0f;
constexpr double kMinSustainedTargetBytes = 32.0 * 1024.0 * 1024.0;

// MXFP8 (float8_e4m3fn payload + float8_e8m0fnu scales) constants. The Inkling
// KV cache quantizes 32 contiguous channels per scale byte and the FA4
// scale-factor buffer is indexed per (page, kv head, 32-channel block).
constexpr int kMxfp8GroupSize = 32;
constexpr int kMxfp8HeadDim = 128;
constexpr int kMxfp8ScalesPerHead = kMxfp8HeadDim / kMxfp8GroupSize;
// sglang's MXFP8 quantizer floors amax so log2 never sees zero
// (see kernels/ops/quantization/mxfp8_quant.py).
constexpr float kMxfp8AmaxFloor = 1.0e-30f;
// input_scale = input_amax / (448 * 6): FP8-e4m3 max times FP4-e2m1 max. This is
// the ModelOpt NVFP4 activation *global scale* convention, which Inkling's
// checkpoint loader reproduces for the experts' w13_input_scale / w2_input_scale
// (models/inkling.py::_ckpt_scale_to_modelopt). It is consumed by fp4_quantize,
// not by an E4M3 store -- see per_tensor_input_scale_from_amax() below.
constexpr float kNvfp4InputScaleDivisor = kE4M3FnMax * kE2M1Max;

enum class DType {
  kAll,
  kFloat,
  kBf16,
  kFp16
};

// Which quantizer family a run exercises. kMxfp4 is the legacy group-32 FP4
// mapping (NOT used by the Inkling model, kept as a reference kernel). kMxfp8Kv
// is the KV-cache quantizer Inkling runs. kFp8PerTensor is static per-tensor
// E4M3 scaling; its NVFP4 divisor is Inkling's own scale derivation, but the
// E4M3 payload is static_quant_fp8's (see 21_bmg_mxfp4_mapping.cpp's header).
enum class QuantMode {
  kAll,
  kMxfp4,
  kMxfp8Kv,
  kFp8PerTensor
};

struct Options {
  std::string suite = "quick";
  std::string shape;
  DType dtype = DType::kAll;
  QuantMode mode = QuantMode::kAll;
  int iterations = 20;
  int warmup = 5;
  bool verify = true;
  bool benchmark = true;
  bool target_gbps_set = false;
  double target_gbps = 0.0;
  bool help = false;
};

struct ByteCompareResult {
  bool passed = true;
  std::size_t mismatches = 0;
  std::size_t first_index = 0;
  uint8_t got = 0;
  uint8_t expected = 0;
};

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline int round_up(int x, int multiple) {
  return ceil_div(x, multiple) * multiple;
}

inline int choose_group_block(int groups) {
  if (groups >= 256 && groups % 256 == 0) {
    return 256;
  }
  if (groups >= 128 && groups % 128 == 0) {
    return 128;
  }
  if (groups >= 64 && groups % 64 == 0) {
    return 64;
  }
  if (groups >= 32 && groups % 32 == 0) {
    return 32;
  }
  if (groups >= 16 && groups % 16 == 0) {
    return 16;
  }
  if (groups >= 8 && groups % 8 == 0) {
    return 8;
  }
  return std::min(kDefaultBlock, groups);
}

inline int choose_row_block_for_group_tile(int groups) {
  int rows = std::max(1, kDefaultBlock / std::max(1, groups));
  while (rows > 1 && (rows * groups) % 16 != 0) {
    --rows;
  }
  return rows;
}

CUTLASS_HOST_DEVICE
float abs_f(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::fabs(x);
#else
  return std::fabs(x);
#endif
}

CUTLASS_HOST_DEVICE
float floor_f(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::floor(x);
#else
  return std::floor(x);
#endif
}

CUTLASS_HOST_DEVICE
float log2_f(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::log2(x);
#else
  return std::log2(x);
#endif
}

CUTLASS_HOST_DEVICE
float exp2_f(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::exp2(x);
#else
  return std::exp2(x);
#endif
}

CUTLASS_HOST_DEVICE
float pow2_int(int exponent) {
  if (exponent < -126) {
    return 0.0f;
  }
  if (exponent > 127) {
    exponent = 127;
  }
  uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
  return sycl::bit_cast<float>(bits);
}

// Exact scaling by a power of two. Used instead of `x / descale` in the MXFP8
// payload path: icpx lowers device fp32 division to an approximation that can
// land ~1 ulp away from the host's std::div, which flips a payload code whenever
// the quotient sits on a rounding boundary. ldexp/scalb is exact on both sides
// (a pure exponent adjust) for every power-of-two scale, including the ones whose
// reciprocal would be subnormal and could not be formed as a float at all.
CUTLASS_HOST_DEVICE
float scalb_f(float x, int exponent) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::ldexp(x, exponent);
#else
  return std::ldexp(x, exponent);
#endif
}

CUTLASS_HOST_DEVICE
int floor_log2_positive(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  int exponent = static_cast<int>((bits >> 23) & 0xffu);
  if (exponent == 0) {
    return -126;
  }
  return exponent - 127;
}

CUTLASS_HOST_DEVICE
float clamp_f(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

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
CUTLASS_DEVICE
Element element_from_raw16(uint16_t raw) {
  return Element::bitcast(raw);
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
CUTLASS_HOST_DEVICE
Element from_float(float x) {
  return static_cast<Element>(x);
}

CUTLASS_HOST_DEVICE
uint16_t raw16_abs_bits(uint16_t raw) {
  return static_cast<uint16_t>(raw & 0x7fffu);
}

template <typename Element>
CUTLASS_HOST_DEVICE
int raw16_floor_log2_positive(uint16_t raw_abs) {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    int exponent = static_cast<int>((raw_abs >> 7) & 0xffu);
    return exponent == 0 ? -126 : exponent - 127;
  } else {
    int exponent = static_cast<int>((raw_abs >> 10) & 0x1fu);
    if (exponent != 0) {
      return exponent - 15;
    }
    int mantissa = static_cast<int>(raw_abs & 0x03ffu);
    if (mantissa == 0) {
      return -126;
    }
    int leading_bit = 0;
    if (mantissa >= 512) {
      leading_bit = 9;
    } else if (mantissa >= 256) {
      leading_bit = 8;
    } else if (mantissa >= 128) {
      leading_bit = 7;
    } else if (mantissa >= 64) {
      leading_bit = 6;
    } else if (mantissa >= 32) {
      leading_bit = 5;
    } else if (mantissa >= 16) {
      leading_bit = 4;
    } else if (mantissa >= 8) {
      leading_bit = 3;
    } else if (mantissa >= 4) {
      leading_bit = 2;
    } else if (mantissa >= 2) {
      leading_bit = 1;
    }
    return leading_bit - 24;
  }
}

template <typename Element>
std::string element_dtype_text() {
  if constexpr (std::is_same_v<Element, float>) {
    return "float";
  } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  } else if constexpr (std::is_same_v<Element, cutlass::half_t>) {
    return "fp16";
  } else {
    return "unknown";
  }
}

inline char const* dtype_text(DType dtype) {
  switch (dtype) {
    case DType::kAll:
      return "all";
    case DType::kFloat:
      return "float";
    case DType::kBf16:
      return "bf16";
    case DType::kFp16:
      return "fp16";
  }
  return "unknown";
}

inline bool parse_dtype(std::string const& text, DType& dtype) {
  if (text == "all") {
    dtype = DType::kAll;
    return true;
  }
  if (text == "float" || text == "fp32") {
    dtype = DType::kFloat;
    return true;
  }
  if (text == "bf16") {
    dtype = DType::kBf16;
    return true;
  }
  if (text == "fp16" || text == "half") {
    dtype = DType::kFp16;
    return true;
  }
  return false;
}

inline char const* mode_text(QuantMode mode) {
  switch (mode) {
    case QuantMode::kAll:
      return "all";
    case QuantMode::kMxfp4:
      return "mxfp4";
    case QuantMode::kMxfp8Kv:
      return "mxfp8_kv";
    case QuantMode::kFp8PerTensor:
      return "fp8_pertensor";
  }
  return "unknown";
}

inline bool parse_mode(std::string const& text, QuantMode& mode) {
  if (text == "all") {
    mode = QuantMode::kAll;
    return true;
  }
  if (text == "mxfp4") {
    mode = QuantMode::kMxfp4;
    return true;
  }
  if (text == "mxfp8_kv" || text == "mxfp8-kv" || text == "mxfp8") {
    mode = QuantMode::kMxfp8Kv;
    return true;
  }
  if (text == "fp8_pertensor" || text == "fp8-pertensor" || text == "fp8") {
    mode = QuantMode::kFp8PerTensor;
    return true;
  }
  return false;
}

inline bool mode_selected(QuantMode selected, QuantMode family) {
  return selected == QuantMode::kAll || selected == family;
}

inline bool parse_bool(std::string const& value) {
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

inline std::string bool_text(bool value) {
  return value ? "true" : "false";
}

inline bool starts_with(std::string const& text, char const* prefix) {
  std::string p(prefix);
  return text.size() >= p.size() && text.compare(0, p.size(), p) == 0;
}

inline std::vector<std::string> split(std::string const& text, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::stringstream ss(text);
  while (std::getline(ss, cur, sep)) {
    if (!cur.empty()) {
      out.push_back(cur);
    }
  }
  return out;
}

inline Options parse_common_options(int argc, char const** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      options.help = true;
      continue;
    }
    auto eq = arg.find('=');
    if (eq == std::string::npos || !starts_with(arg, "--")) {
      throw std::invalid_argument("expected --key=value or --help, got: " + arg);
    }
    std::string key = arg.substr(2, eq - 2);
    std::string value = arg.substr(eq + 1);
    if (key == "suite") {
      options.suite = value;
    } else if (key == "shape") {
      options.shape = value;
    } else if (key == "dtype") {
      if (!parse_dtype(value, options.dtype)) {
        throw std::invalid_argument("unknown dtype: " + value);
      }
    } else if (key == "mode") {
      if (!parse_mode(value, options.mode)) {
        throw std::invalid_argument("unknown mode: " + value);
      }
    } else if (key == "iterations") {
      options.iterations = std::stoi(value);
    } else if (key == "warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "verify") {
      options.verify = parse_bool(value);
    } else if (key == "benchmark") {
      options.benchmark = parse_bool(value);
    } else if (key == "target-gbps") {
      options.target_gbps = std::stod(value);
      options.target_gbps_set = true;
    } else {
      throw std::invalid_argument("unknown option: --" + key);
    }
  }
  if (options.iterations < 0 || options.warmup < 0) {
    throw std::invalid_argument("iterations and warmup must be non-negative");
  }
  return options;
}

// `show_mode` is opt-in because --mode only does something in a binary that
// consults Options::mode; advertising it unconditionally would promise a flag
// that other consumers of this header silently ignore.
inline void print_common_usage(
    char const* name, char const* suites, char const* shape_text, bool show_mode = false) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=" << suites << "       Built-in suite (default quick)\n"
      << "  --shape=" << shape_text << "\n"
      << "  --dtype=all|float|bf16|fp16   Element dtype (default all)\n";
  if (show_mode) {
    std::cout
        << "  --mode=all|mxfp4|mxfp8_kv|fp8_pertensor\n"
        << "                                Quantizer family to run (default all)\n";
  }
  std::cout
      << "  --iterations=<int>            Timed kernel iterations (default 20)\n"
      << "  --warmup=<int>                Warmup launches before timing (default 5)\n"
      << "  --verify=0|1                  Run CPU reference comparison (default 1)\n"
      << "  --benchmark=0|1               Run profiling-event timing (default 1)\n"
      << "  --target-gbps=<float>         Optional sustained effective GB/s gate\n";
}

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

  void zero() {
    if (count > 0) {
      queue->memset(ptr, 0, sizeof(T) * count).wait();
    }
  }
};

CUTLASS_HOST_DEVICE
int round_nearest_even_int(float x) {
  float base_f = floor_f(x);
  int base = static_cast<int>(base_f);
  float frac = x - base_f;
  if (frac > 0.5f || (frac == 0.5f && (base & 1))) {
    ++base;
  }
  return base;
}

CUTLASS_HOST_DEVICE
uint8_t quantize_e2m1_code(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint8_t sign = static_cast<uint8_t>((bits >> 28) & 0x8u);
  float x = sycl::bit_cast<float>(bits & 0x7fffffffu);
  constexpr float kTieTol = 1.0e-6f;
  uint8_t mag = 0u;
  mag += static_cast<uint8_t>(x > 0.25f + kTieTol);
  mag += static_cast<uint8_t>(x >= 0.75f - kTieTol);
  mag += static_cast<uint8_t>(x > 1.25f + kTieTol);
  mag += static_cast<uint8_t>(x >= 1.75f - kTieTol);
  mag += static_cast<uint8_t>(x > 2.5f + kTieTol);
  mag += static_cast<uint8_t>(x >= 3.5f - kTieTol);
  mag += static_cast<uint8_t>(x > 5.0f + kTieTol);
  return mag == 0u ? 0u : static_cast<uint8_t>(sign | mag);
}

CUTLASS_HOST_DEVICE
uint8_t quantize_e2m1_pair(float first, float second) {
  uint32_t bits0 = sycl::bit_cast<uint32_t>(first);
  uint32_t bits1 = sycl::bit_cast<uint32_t>(second);
  uint8_t sign0 = static_cast<uint8_t>((bits0 >> 28) & 0x8u);
  uint8_t sign1 = static_cast<uint8_t>((bits1 >> 28) & 0x8u);
  float x0 = sycl::bit_cast<float>(bits0 & 0x7fffffffu);
  float x1 = sycl::bit_cast<float>(bits1 & 0x7fffffffu);
  constexpr float kTieTol = 1.0e-6f;
  uint8_t mag0 = 0u;
  uint8_t mag1 = 0u;
  mag0 += static_cast<uint8_t>(x0 > 0.25f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 0.25f + kTieTol);
  mag0 += static_cast<uint8_t>(x0 >= 0.75f - kTieTol);
  mag1 += static_cast<uint8_t>(x1 >= 0.75f - kTieTol);
  mag0 += static_cast<uint8_t>(x0 > 1.25f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 1.25f + kTieTol);
  mag0 += static_cast<uint8_t>(x0 >= 1.75f - kTieTol);
  mag1 += static_cast<uint8_t>(x1 >= 1.75f - kTieTol);
  mag0 += static_cast<uint8_t>(x0 > 2.5f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 2.5f + kTieTol);
  mag0 += static_cast<uint8_t>(x0 >= 3.5f - kTieTol);
  mag1 += static_cast<uint8_t>(x1 >= 3.5f - kTieTol);
  mag0 += static_cast<uint8_t>(x0 > 5.0f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 5.0f + kTieTol);
  uint8_t code0 = mag0 == 0u ? 0u : static_cast<uint8_t>(sign0 | mag0);
  uint8_t code1 = mag1 == 0u ? 0u : static_cast<uint8_t>(sign1 | mag1);
  return static_cast<uint8_t>(code0 | (code1 << 4));
}

CUTLASS_HOST_DEVICE
uint8_t quantize_e2m1_pair_ordered(float first, float second) {
  uint32_t bits0 = sycl::bit_cast<uint32_t>(first);
  uint32_t bits1 = sycl::bit_cast<uint32_t>(second);
  uint8_t sign0 = static_cast<uint8_t>((bits0 >> 28) & 0x8u);
  uint8_t sign1 = static_cast<uint8_t>((bits1 >> 28) & 0x8u);
  constexpr float kTieTol = 1.0e-6f;
  constexpr uint32_t kThreshold0 = sycl::bit_cast<uint32_t>(0.25f + kTieTol);
  constexpr uint32_t kThreshold1 = sycl::bit_cast<uint32_t>(0.75f - kTieTol);
  constexpr uint32_t kThreshold2 = sycl::bit_cast<uint32_t>(1.25f + kTieTol);
  constexpr uint32_t kThreshold3 = sycl::bit_cast<uint32_t>(1.75f - kTieTol);
  constexpr uint32_t kThreshold4 = sycl::bit_cast<uint32_t>(2.5f + kTieTol);
  constexpr uint32_t kThreshold5 = sycl::bit_cast<uint32_t>(3.5f - kTieTol);
  constexpr uint32_t kThreshold6 = sycl::bit_cast<uint32_t>(5.0f + kTieTol);
  uint32_t x0 = bits0 & 0x7fffffffu;
  uint32_t x1 = bits1 & 0x7fffffffu;
  x0 = x0 > 0x7f800000u ? 0u : x0;
  x1 = x1 > 0x7f800000u ? 0u : x1;
  uint8_t mag0 = 0u;
  uint8_t mag1 = 0u;
  mag0 += static_cast<uint8_t>(x0 > kThreshold0);
  mag1 += static_cast<uint8_t>(x1 > kThreshold0);
  mag0 += static_cast<uint8_t>(x0 >= kThreshold1);
  mag1 += static_cast<uint8_t>(x1 >= kThreshold1);
  mag0 += static_cast<uint8_t>(x0 > kThreshold2);
  mag1 += static_cast<uint8_t>(x1 > kThreshold2);
  mag0 += static_cast<uint8_t>(x0 >= kThreshold3);
  mag1 += static_cast<uint8_t>(x1 >= kThreshold3);
  mag0 += static_cast<uint8_t>(x0 > kThreshold4);
  mag1 += static_cast<uint8_t>(x1 > kThreshold4);
  mag0 += static_cast<uint8_t>(x0 >= kThreshold5);
  mag1 += static_cast<uint8_t>(x1 >= kThreshold5);
  mag0 += static_cast<uint8_t>(x0 > kThreshold6);
  mag1 += static_cast<uint8_t>(x1 > kThreshold6);
  uint8_t code0 = mag0 == 0u ? 0u : static_cast<uint8_t>(sign0 | mag0);
  uint8_t code1 = mag1 == 0u ? 0u : static_cast<uint8_t>(sign1 | mag1);
  return static_cast<uint8_t>(code0 | (code1 << 4));
}

template <typename Element, bool Ordered = false>
CUTLASS_DEVICE
uint32_t quantize_e2m1_raw_word_pairs(uint64_t raw, float scale) {
  uint16_t bits0 = static_cast<uint16_t>(raw);
  uint16_t bits1 = static_cast<uint16_t>(raw >> 16);
  float scaled0 = raw16_to_float<Element>(bits0) * scale;
  float scaled1 = raw16_to_float<Element>(bits1) * scale;
  uint32_t packed01 = 0;
  if constexpr (Ordered) {
    packed01 = quantize_e2m1_pair_ordered(scaled0, scaled1);
  } else {
    packed01 = quantize_e2m1_pair(scaled0, scaled1);
  }
  uint16_t bits2 = static_cast<uint16_t>(raw >> 32);
  uint16_t bits3 = static_cast<uint16_t>(raw >> 48);
  float scaled2 = raw16_to_float<Element>(bits2) * scale;
  float scaled3 = raw16_to_float<Element>(bits3) * scale;
  uint32_t packed23 = 0;
  if constexpr (Ordered) {
    packed23 = quantize_e2m1_pair_ordered(scaled2, scaled3);
  } else {
    packed23 = quantize_e2m1_pair(scaled2, scaled3);
  }
  return packed01 | (packed23 << 8);
}

CUTLASS_HOST_DEVICE
float dequantize_e2m1_code(uint8_t code) {
  float value = 0.0f;
  switch (code & 0x7u) {
    case 0:
      value = 0.0f;
      break;
    case 1:
      value = 0.5f;
      break;
    case 2:
      value = 1.0f;
      break;
    case 3:
      value = 1.5f;
      break;
    case 4:
      value = 2.0f;
      break;
    case 5:
      value = 3.0f;
      break;
    case 6:
      value = 4.0f;
      break;
    default:
      value = 6.0f;
      break;
  }
  return (code & 0x8u) ? -value : value;
}

CUTLASS_HOST_DEVICE
uint8_t pack_e2m1_pair(uint8_t first, uint8_t second) {
  return static_cast<uint8_t>((first & 0x0fu) | ((second & 0x0fu) << 4));
}

CUTLASS_HOST_DEVICE
uint8_t e4m3fn_encode(float value) {
  uint8_t sign = value < 0.0f ? 0x80u : 0u;
  float x = abs_f(value);
  if (!(x > 0.0f)) {
    return sign;
  }
  if (x >= kE4M3FnMax) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  constexpr float kMinNormal = 0.015625f;      // 2^-6
  constexpr float kSubnormalStep = 0.001953125f;  // 2^-9

  if (x < kMinNormal) {
    int mantissa = round_nearest_even_int(x / kSubnormalStep);
    if (mantissa <= 0) {
      return sign;
    }
    if (mantissa >= 8) {
      return static_cast<uint8_t>(sign | (1u << 3));
    }
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(mantissa));
  }

  int exponent = floor_log2_positive(x);
  if (exponent > 8) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }
  if (exponent < -6) {
    exponent = -6;
  }
  float scale = pow2_int(exponent);
  int mantissa = round_nearest_even_int((x / scale - 1.0f) * 8.0f);
  int exponent_field = exponent + 7;
  if (mantissa >= 8) {
    mantissa = 0;
    ++exponent_field;
  }
  if (exponent_field >= 15 && mantissa > 6) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }
  if (exponent_field > 15) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }
  return static_cast<uint8_t>(sign | static_cast<uint8_t>((exponent_field << 3) | mantissa));
}

CUTLASS_HOST_DEVICE
uint8_t e4m3fn_encode_positive(float x) {
  if (!(x > 0.0f)) {
    return 0u;
  }
  if (x >= kE4M3FnMax) {
    return 0x7eu;
  }

  constexpr float kMinNormal = 0.015625f;  // 2^-6

  if (x < kMinNormal) {
    // x / 2^-9 as an exact exponent adjust; a device fp32 divide here could land
    // a ulp off the host and flip the subnormal payload code.
    int mantissa = round_nearest_even_int(scalb_f(x, 9));
    if (mantissa <= 0) {
      return 0u;
    }
    if (mantissa >= 8) {
      return 0x08u;
    }
    return static_cast<uint8_t>(mantissa);
  }

  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  uint32_t mantissa_bits = bits & 0x007fffffu;
  int mantissa = static_cast<int>(mantissa_bits >> 20);
  uint32_t round_bits = mantissa_bits & 0x000fffffu;
  if (round_bits > 0x00080000u || (round_bits == 0x00080000u && (mantissa & 1))) {
    ++mantissa;
  }
  int exponent_field = exponent + 7;
  if (mantissa >= 8) {
    mantissa = 0;
    ++exponent_field;
  }
  if (exponent_field >= 15 && mantissa > 6) {
    return 0x7eu;
  }
  if (exponent_field > 15) {
    return 0x7eu;
  }
  return static_cast<uint8_t>((exponent_field << 3) | mantissa);
}

// Sign-magnitude E4M3 encode built on the pure-bit-op positive encoder. Unlike
// e4m3fn_encode() the normal path needs no float division, reciprocal or floor,
// which matters because the MXFP8 and per-tensor FP8 kernels below are
// encode-bound rather than bandwidth-bound. Checked against e4m3fn_encode()
// over every third finite float bit pattern (1.43e9 values): identical for all
// of them. The only input where the two differ is -0.0f, which this form maps
// to 0x80 (signed zero, matching torch's .to(float8_e4m3fn)) instead of 0x00.
CUTLASS_HOST_DEVICE
uint8_t e4m3fn_encode_signed(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  float magnitude = sycl::bit_cast<float>(bits & 0x7fffffffu);
  return static_cast<uint8_t>(sign | e4m3fn_encode_positive(magnitude));
}

struct E4M3FnEncodeInvResult {
  uint8_t code = 0;
  float inv_decoded = 0.0f;
};

CUTLASS_HOST_DEVICE
float e4m3fn_mantissa_inv(int mantissa) {
  switch (mantissa) {
    case 0:
      return 1.0f;
    case 1:
      return 0.8888888888888888f;
    case 2:
      return 0.8f;
    case 3:
      return 0.7272727272727273f;
    case 4:
      return 0.6666666666666666f;
    case 5:
      return 0.6153846153846154f;
    case 6:
      return 0.5714285714285714f;
    default:
      return 0.5333333333333333f;
  }
}

CUTLASS_HOST_DEVICE
float e4m3fn_subnormal_inv(int mantissa) {
  switch (mantissa) {
    case 1:
      return 512.0f;
    case 2:
      return 256.0f;
    case 3:
      return 170.66666666666666f;
    case 4:
      return 128.0f;
    case 5:
      return 102.4f;
    case 6:
      return 85.33333333333333f;
    default:
      return 73.14285714285714f;
  }
}

CUTLASS_HOST_DEVICE
E4M3FnEncodeInvResult e4m3fn_encode_positive_with_inv_decode(float x) {
  if (!(x > 0.0f)) {
    return {};
  }
  if (x >= kE4M3FnMax) {
    return {0x7eu, 0.002232142857142857f};
  }

  constexpr float kMinNormal = 0.015625f;      // 2^-6
  constexpr float kSubnormalStep = 0.001953125f;  // 2^-9

  if (x < kMinNormal) {
    int mantissa = round_nearest_even_int(x / kSubnormalStep);
    if (mantissa <= 0) {
      return {};
    }
    if (mantissa >= 8) {
      return {0x08u, 64.0f};
    }
    return {static_cast<uint8_t>(mantissa), e4m3fn_subnormal_inv(mantissa)};
  }

  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  int decoded_exponent = exponent;
  uint32_t mantissa_bits = bits & 0x007fffffu;
  int mantissa = static_cast<int>(mantissa_bits >> 20);
  uint32_t round_bits = mantissa_bits & 0x000fffffu;
  if (round_bits > 0x00080000u || (round_bits == 0x00080000u && (mantissa & 1))) {
    ++mantissa;
  }
  int exponent_field = exponent + 7;
  if (mantissa >= 8) {
    mantissa = 0;
    ++exponent_field;
    ++decoded_exponent;
  }
  if (exponent_field >= 15 && mantissa > 6) {
    return {0x7eu, 0.002232142857142857f};
  }
  if (exponent_field > 15) {
    return {0x7eu, 0.002232142857142857f};
  }
  uint8_t code = static_cast<uint8_t>((exponent_field << 3) | mantissa);
  float inv_decoded = e4m3fn_mantissa_inv(mantissa) * pow2_int(-decoded_exponent);
  return {code, inv_decoded};
}

CUTLASS_HOST_DEVICE
float e4m3fn_decode(uint8_t code) {
  int sign = (code & 0x80u) ? -1 : 1;
  int exponent_field = (code >> 3) & 0x0f;
  int mantissa = code & 0x07;
  if ((code & 0x7fu) == 0u) {
    return 0.0f;
  }
  if ((code & 0x7fu) == 0x7fu) {
    return 0.0f;
  }
  float value;
  if (exponent_field == 0) {
    value = static_cast<float>(mantissa) * pow2_int(-9);
  } else {
    value = (1.0f + static_cast<float>(mantissa) * 0.125f) *
        pow2_int(exponent_field - 7);
  }
  return sign < 0 ? -value : value;
}

CUTLASS_HOST_DEVICE
int nvfp4_swizzled_scale_index(int row, int group, int rounded_groups) {
  int row_block = row / 128;
  int row_rem = row - row_block * 128;
  int e = row_rem / 32;
  int d = row_rem - e * 32;
  int c = group / 4;
  int f = group - c * 4;
  int groups4 = rounded_groups / 4;
  return (((row_block * groups4 + c) * 32 + d) * 4 + e) * 4 + f;
}

CUTLASS_HOST_DEVICE
int clamp_exponent_to_ue8m0(int exponent) {
  return exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
}

CUTLASS_HOST_DEVICE
uint8_t encode_ue8m0_exponent(int exponent) {
  return static_cast<uint8_t>(clamp_exponent_to_ue8m0(exponent) + 127);
}

CUTLASS_HOST_DEVICE
int decode_ue8m0_exponent(uint8_t scale) {
  return static_cast<int>(scale) - 127;
}

// ---------------------------------------------------------------------------
// MXFP8 (E4M3 payload + E8M0 scales) helpers.
//
// Scale rule, transcribed from sglang
// python/sglang/kernels/ops/quantization/mxfp8_quant.py::_mxfp8_quant_kernel:
//
//   amax          = max(|x| over the 32-element group, 1e-30)
//   scale_biased  = clamp(ceil(log2(amax / 448.0)) + 127.0, 0.0, 254.0)
//   descale       = exp2(scale_biased - 127.0)
//   payload       = e4m3(clamp(x / descale, -448.0, 448.0))
//                   (applied as scalb_f(x, -(scale_biased - 127)); descale is a
//                    power of two, so this is the same value computed exactly)
//   scale byte    = uint8(scale_biased)          // float8_e8m0fnu, bias 127
//
// The scale byte is the raw biased exponent, so 254 is the largest value ever
// written and 0xFF (E8M0 NaN) never appears.
//
// Example 15 (15_bmg_attn_prologue_mxfp8_store_tau.cpp) quantizes the same KV
// cache and carries its own private copies of this rule and of the interleaved
// offset (its mxfp8_scale_byte / kv_scale_offset). The two agree numerically on
// every value the suites exercise, but nothing enforces that: if you change
// either side, change both. Its copy still uses the float log2 discussed below.
// ---------------------------------------------------------------------------
// ceil(log2(amax / 448)) evaluated exactly with integer arithmetic instead of a
// libm log2. This matters because the scale byte and its 32-byte payload group
// are compared byte-for-byte between the host reference and the device kernel:
// std::log2 and sycl::log2 differ by ~1-4 ulp, so whenever amax/448 lands just
// above a power of two the two sides ceil() to different integers, the byte
// differs by one and the whole group mismatches. With e = floor_log2(amax) and
// the mantissa m = amax / 2^e in [1, 2), amax/448 = (m / 1.75) * 2^(e - 8), and
// log2(m / 1.75) is <= 0 exactly when m <= 1.75, hence
//
//   ceil(log2(amax / 448)) == e - 8 + (m > 1.75 ? 1 : 0)
//
// and m > 1.75 is the mantissa-field compare (bits & 0x7fffff) > 0x600000.
// The MXFP4 helpers in this header avoid libm the same way.
//
// Checked against the ceil(std::log2(...)) form over every third non-negative
// finite float (7.13e8 values): they agree everywhere except 1848 inputs whose
// mantissa sits within a few ulp above 0x600000, i.e. amax just above 1.75*2^e.
// There the float form returns the ceil of a log2 that rounded down onto the
// integer, so it is the float form that is off by one; this one is exact.
CUTLASS_HOST_DEVICE
int mxfp8_ceil_log2_ratio(float amax) {
  uint32_t bits = sycl::bit_cast<uint32_t>(amax);
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  bool mantissa_above_1p75 = (bits & 0x007fffffu) > 0x00600000u;
  return exponent - 8 + (mantissa_above_1p75 ? 1 : 0);
}

// Returns the E8M0 scale byte and, in `descale_exponent`, the unbiased exponent
// it encodes: descale == 2^descale_exponent. Divide the payload with
// scalb_f(x, -descale_exponent) rather than `x / pow2_int(descale_exponent)`, so
// the host reference and the device kernel agree bit for bit.
CUTLASS_HOST_DEVICE
uint8_t mxfp8_scale_byte(float amax, int& descale_exponent) {
  // The 1e-30 floor is normal in fp32, so mxfp8_ceil_log2_ratio() never sees a
  // subnormal (and the resulting byte is >= 19, never 0).
  float safe_amax = amax > kMxfp8AmaxFloor ? amax : kMxfp8AmaxFloor;
  int biased = mxfp8_ceil_log2_ratio(safe_amax) + 127;
  biased = biased < 0 ? 0 : (biased > 254 ? 254 : biased);
  descale_exponent = biased - 127;
  return static_cast<uint8_t>(biased);
}

// Byte offset of the E8M0 scale for KV slot `slot`, channel `channel` in the
// interleaved FA4 BlockScaledBasicChunk buffer
//
//   (slots / page_size, dkv / 128, 32, page_size / 32, 128 / 32)
//
// which is how sglang's MXFP8 KV pool allocates k_scale_buffer / v_scale_buffer
// at page_size == 128 (srt/mem_cache/memory_pool.py) and how
// mxfp8_interleave_sf.py / _mxfp8_quant_store_qkv_kernel index it. Matches
// example 15's kv_scale_offset().
CUTLASS_HOST_DEVICE
int64_t mxfp8_interleaved_sf_offset(int64_t slot, int channel, int dkv, int page_size) {
  int heads = dkv / kMxfp8HeadDim;
  int page_chunks = page_size / kMxfp8GroupSize;
  int64_t page = slot / page_size;
  int64_t page_offset = slot % page_size;
  int head = channel / kMxfp8HeadDim;
  int block = (channel % kMxfp8HeadDim) / kMxfp8GroupSize;
  return ((page * heads + head) * (kMxfp8GroupSize * page_chunks * kMxfp8ScalesPerHead)) +
      ((page_offset % kMxfp8GroupSize) * (page_chunks * kMxfp8ScalesPerHead)) +
      ((page_offset / kMxfp8GroupSize) * kMxfp8ScalesPerHead) +
      block;
}

// Static per-tensor activation scale from a checkpoint amax. Two divisors ship:
//
//   kE4M3FnMax (448)                the plain per-tensor FP8-E4M3 convention,
//                                   scale = amax / e4m3_max, whose runtime is
//                                   static_quant_fp8 (kernels/ops/quantization/
//                                   fp8_kernel.py) -- it multiplies by
//                                   1 / scale and clamps to +-448.
//   kNvfp4InputScaleDivisor (448*6) input_scale for Inkling's NVFP4 experts
//                                   (models/inkling.py::_ckpt_scale_to_modelopt,
//                                   gated on ".experts." / ".shared_experts.").
//                                   This is the only per-tensor activation scale
//                                   Inkling derives; Inkling instantiates no FP8
//                                   linear method. Its consumer is fp4_quantize
//                                   (modelopt_quant.py), i.e. an E2M1 payload
//                                   plus per-16 E4M3 block scales, so do not
//                                   read this divisor as an FP8 store rule: the
//                                   trailing 6 is the E2M1 max, and feeding it
//                                   to an E4M3 store saturates everything above
//                                   amax / 6.
inline float per_tensor_input_scale_from_amax(float amax, float amax_divisor) {
  return amax / amax_divisor;
}

inline ByteCompareResult compare_bytes(std::vector<uint8_t> const& got, std::vector<uint8_t> const& expected) {
  if (got.size() != expected.size()) {
    throw std::invalid_argument("compare_bytes size mismatch");
  }
  ByteCompareResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      if (result.mismatches == 0) {
        result.first_index = i;
        result.got = got[i];
        result.expected = expected[i];
      }
      ++result.mismatches;
    }
  }
  result.passed = result.mismatches == 0;
  return result;
}

inline void print_byte_compare(char const* name, ByteCompareResult const& result) {
  if (result.passed) {
    return;
  }
  std::cerr << "    " << name << " mismatch count=" << result.mismatches
            << " first_index=" << result.first_index
            << " got=0x" << std::hex << static_cast<int>(result.got)
            << " expected=0x" << static_cast<int>(result.expected)
            << std::dec << "\n";
}

inline double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

struct EventBundle {
  sycl::event first;
  sycl::event second;
  bool has_second = false;

  explicit EventBundle(sycl::event event) : first(event) {}

  EventBundle(sycl::event first_event, sycl::event second_event)
      : first(first_event), second(second_event), has_second(true) {}

  void wait() {
    if (has_second) {
      second.wait();
    } else {
      first.wait();
    }
  }
};

inline double event_ms(EventBundle const& events) {
  if (events.has_second) {
    auto start = events.first.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = events.second.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(end - start) * 1.0e-6;
  }
  return event_ms(events.first);
}

template <typename Launcher>
double benchmark_ms(Launcher&& launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch().wait();
  }
  if (iterations == 0) {
    launch().wait();
    return 0.0;
  }
  using Event = std::decay_t<decltype(launch())>;
  std::vector<Event> events;
  events.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    events.push_back(launch());
  }
  for (Event& event : events) {
    event.wait();
  }
  double total = 0.0;
  for (Event const& event : events) {
    total += event_ms(event);
  }
  return total / static_cast<double>(iterations);
}

inline double effective_gbps(double bytes, double mean_ms) {
  if (mean_ms <= 0.0) {
    return 0.0;
  }
  return bytes / (mean_ms * 1.0e-3) / 1.0e9;
}

inline sycl::queue make_queue() {
  return sycl::queue(
      sycl::gpu_selector_v,
      sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
}

template <typename Element>
std::vector<Element> make_input(std::size_t count, uint32_t seed, float lo = -3.0f, float hi = 3.0f) {
  std::vector<Element> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (Element& x : data) {
    x = from_float<Element>(dist(gen));
  }
  return data;
}

template <typename Element>
float max_abs_host(std::vector<Element> const& data) {
  float max_abs = 0.0f;
  for (Element x : data) {
    max_abs = std::max(max_abs, std::fabs(to_float(x)));
  }
  return max_abs;
}

}  // namespace cutlass::examples::bmg_quantization
