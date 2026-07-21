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

enum class DType {
  kAll,
  kFloat,
  kBf16,
  kFp16
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

inline void print_common_usage(char const* name, char const* suites, char const* shape_text) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=" << suites << "       Built-in suite (default quick)\n"
      << "  --shape=" << shape_text << "\n"
      << "  --dtype=all|float|bf16|fp16   Element dtype (default all)\n"
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

  constexpr float kMinNormal = 0.015625f;      // 2^-6
  constexpr float kSubnormalStep = 0.001953125f;  // 2^-9

  if (x < kMinNormal) {
    int mantissa = round_nearest_even_int(x / kSubnormalStep);
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
