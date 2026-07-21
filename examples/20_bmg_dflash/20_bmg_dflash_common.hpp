/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling DFLASH helper examples for CUTLASS SYCL on BMG.
 *
 * Roofline summary:
 *   07.1 device guard is a host/runtime gating check. The example uses a tiny
 *   byte-prefix kernel only to keep the validation one-to-one with the DFLASH
 *   work item; there is no meaningful arithmetic roofline.
 *
 *   07.2 masked req_to_token gather is memory-bound: each accepted token
 *   streams metadata and one int64 table entry, then writes one int64 result,
 *   so arithmetic intensity is effectively 0 FLOP/B. Greedy sampling computes
 *   hidden @ weight.T and then argmax. Its GEMM intensity is approximately
 *   2*N*V*H / (4*(N*H + V*H + N*V)) FLOP/B for FP32 logits; production-like
 *   N=512,V=8192,H=1536 is above 200 FLOP/B, so the dot product is compute-bound
 *   and uses oneMKL SGEMM while the custom kernel only does the argmax.
 *
 *   07.3 Mamba/conv verify commit is memory-bound. It copies selected
 *   intermediate rows back to persistent cache slots, with a step >= 0 mask
 *   and an optional second tracking pass. Sustained effective read/write
 *   bandwidth is the relevant metric, and perf cases use large working sets.
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include <algorithm>
#include <chrono>
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

namespace cutlass::examples::bmg_dflash {

constexpr int kThreads = 256;
constexpr int kSubGroup = 16;
constexpr int kGreedyRowsPerGroup = 16;
constexpr int kCopyPackBytes = 16;
constexpr int kSkipReturnCode = 77;
constexpr double kMinSustainedTargetBytes = 32.0 * 1024.0 * 1024.0;

class NoGpuDevice : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

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
  int iterations = 5;
  int warmup = 2;
  bool verify = true;
  bool benchmark = true;
  bool help = false;
  bool target_gbps_set = false;
  double target_gbps = 0.0;
  bool target_tops_set = false;
  double target_tops = 0.0;
};

inline bool starts_with(std::string const& text, char const* prefix) {
  std::string p(prefix);
  return text.size() >= p.size() && text.compare(0, p.size(), p) == 0;
}

inline bool parse_bool_value(std::string const& text) {
  if (text == "1" || text == "true" || text == "on" || text == "yes") {
    return true;
  }
  if (text == "0" || text == "false" || text == "off" || text == "no") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + text);
}

inline DType parse_dtype(std::string const& text) {
  if (text == "all") {
    return DType::kAll;
  }
  if (text == "float" || text == "fp32") {
    return DType::kFloat;
  }
  if (text == "bf16") {
    return DType::kBf16;
  }
  if (text == "fp16" || text == "half") {
    return DType::kFp16;
  }
  throw std::invalid_argument("unknown dtype: " + text);
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

template <typename Element>
char const* element_dtype_text() {
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
      options.dtype = parse_dtype(value);
    } else if (key == "iterations") {
      options.iterations = std::stoi(value);
    } else if (key == "warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "verify") {
      options.verify = parse_bool_value(value);
    } else if (key == "benchmark") {
      options.benchmark = parse_bool_value(value);
    } else if (key == "target-gbps") {
      options.target_gbps = std::stod(value);
      options.target_gbps_set = true;
    } else if (key == "target-tops") {
      options.target_tops = std::stod(value);
      options.target_tops_set = true;
    } else {
      throw std::invalid_argument("unknown option: --" + key);
    }
  }
  if (options.iterations < 0 || options.warmup < 0) {
    throw std::invalid_argument("iterations and warmup must be non-negative");
  }
  return options;
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

inline bool parse_shape_ints(std::string const& shape, std::vector<std::pair<std::string, int*>> fields) {
  if (shape.empty()) {
    return true;
  }
  for (std::string const& item : split(shape, ',')) {
    auto eq = item.find('=');
    if (eq == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, eq);
    int value = std::stoi(item.substr(eq + 1));
    bool matched = false;
    for (auto const& field : fields) {
      if (field.first == key) {
        *field.second = value;
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

inline sycl::queue make_queue() {
  auto async_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (std::exception const& ex) {
        std::cerr << "Asynchronous SYCL exception: " << ex.what() << "\n";
      }
    }
  };
  std::vector<sycl::device> devices;
  try {
    devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  } catch (sycl::exception const& e) {
    throw NoGpuDevice(std::string("No SYCL GPU device available: ") + e.what());
  }
  if (devices.empty()) {
    throw NoGpuDevice("No SYCL GPU device available");
  }
  return sycl::queue(
      devices.front(),
      async_handler,
      sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
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
    if (!host.empty()) {
      if (host.size() > count) {
        throw std::runtime_error("copy_from exceeds device buffer");
      }
      queue->memcpy(ptr, host.data(), sizeof(T) * host.size()).wait();
    }
  }

  void copy_to(std::vector<T>& host) const {
    if (!host.empty()) {
      if (host.size() > count) {
        throw std::runtime_error("copy_to exceeds device buffer");
      }
      queue->memcpy(host.data(), ptr, sizeof(T) * host.size()).wait();
    }
  }
};

template <typename Element>
CUTLASS_HOST_DEVICE
float elem_to_float(Element x) {
  if constexpr (std::is_same_v<Element, float>) {
    return x;
  } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
#if defined(__SYCL_DEVICE_ONLY__)
    uint32_t bits = static_cast<uint32_t>(x.raw()) << 16;
    return sycl::bit_cast<float>(bits);
#else
    return static_cast<float>(x);
#endif
  } else {
    return static_cast<float>(x);
  }
}

template <typename Element>
Element elem_from_float(float x) {
  return static_cast<Element>(x);
}

template <>
inline float elem_from_float<float>(float x) {
  return x;
}

inline int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline double elapsed_ms(std::chrono::steady_clock::time_point begin,
                         std::chrono::steady_clock::time_point end,
                         int iterations) {
  if (iterations <= 0) {
    return 0.0;
  }
  double us = std::chrono::duration<double, std::micro>(end - begin).count();
  return us / 1000.0 / static_cast<double>(iterations);
}

inline std::string bool_text(bool value) {
  return value ? "true" : "false";
}

struct CompareResult {
  bool passed = true;
  double max_abs = 0.0;
  std::size_t index = 0;
};

template <typename Element>
CompareResult compare_vectors(std::vector<Element> const& got,
                              std::vector<Element> const& expected,
                              double atol = 0.0) {
  if (got.size() != expected.size()) {
    return {false, std::numeric_limits<double>::infinity(), 0};
  }
  CompareResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double diff = std::abs(static_cast<double>(elem_to_float(got[i])) -
                           static_cast<double>(elem_to_float(expected[i])));
    if (diff > result.max_abs) {
      result.max_abs = diff;
      result.index = i;
    }
    if (diff > atol) {
      result.passed = false;
    }
  }
  return result;
}

inline double patterned_value(std::size_t i, int salt = 0) {
  double x = static_cast<double>((i * 1315423911ull + static_cast<std::size_t>(salt) * 2654435761ull) & 0xffffu);
  return (x / 32768.0 - 1.0) * 0.75;
}

}  // namespace cutlass::examples::bmg_dflash
