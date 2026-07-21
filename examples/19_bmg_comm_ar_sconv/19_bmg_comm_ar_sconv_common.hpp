/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

#include <algorithm>
#include <chrono>
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

namespace cutlass::examples::comm_ar_sconv {

constexpr int kPadSlot = -1;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

inline std::string bool_text(bool value) {
  return value ? "true" : "false";
}

inline std::string dtype_text(DType dtype) {
  switch (dtype) {
    case DType::kAll:
      return "all";
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

template <typename Element>
inline char const* element_dtype_text() {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  } else {
    return "fp16";
  }
}

template <typename Element>
CUTLASS_HOST_DEVICE
float element_to_float(Element x) {
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
Element round_to_element(float x) {
  return Element(x);
}

CUTLASS_HOST_DEVICE
float silu(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return x * sycl::native::recip(1.0f + sycl::native::exp(-x));
#else
  return x / (1.0f + std::exp(-x));
#endif
}

inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

inline std::size_t ceil_div_size(std::size_t a, std::size_t b) {
  return (a + b - 1) / b;
}

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
    if (count == 0) {
      return;
    }
    ptr = sycl::malloc_device<T>(count, q);
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
void fill_random(std::vector<Element>& values, uint32_t seed, float lo = -0.5f, float hi = 0.5f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& v : values) {
    v = Element(dist(gen));
  }
}

template <typename Element>
bool compare_vectors(
    std::string const& label,
    std::vector<Element> const& got,
    std::vector<Element> const& ref,
    float atol,
    float rtol,
    int max_report = 6) {
  if (got.size() != ref.size()) {
    std::cerr << label << " size mismatch got=" << got.size() << " ref=" << ref.size() << "\n";
    return false;
  }
  bool passed = true;
  int reports = 0;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  std::size_t max_idx = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    float g = element_to_float(got[i]);
    float r = element_to_float(ref[i]);
    float abs_err = std::abs(g - r);
    float rel_err = abs_err / std::max(std::abs(r), 1.0f);
    if (abs_err > max_abs) {
      max_abs = abs_err;
      max_rel = rel_err;
      max_idx = i;
    }
    if (abs_err > atol + rtol * std::abs(r)) {
      passed = false;
      if (reports < max_report) {
        std::cerr << label << " mismatch[" << i << "] got=" << g << " ref=" << r
                  << " abs=" << abs_err << " rel=" << rel_err << "\n";
      }
      ++reports;
    }
  }
  if (!passed) {
    std::cerr << label << " failed mismatches=" << reports << " max_abs=" << max_abs
              << " max_rel=" << max_rel << " max_idx=" << max_idx << "\n";
  }
  return passed;
}

template <typename Element>
float default_atol() {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return 4.0e-2f;
  } else {
    return 6.0e-3f;
  }
}

template <typename Element>
float default_rtol() {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return 4.0e-2f;
  } else {
    return 6.0e-3f;
  }
}

template <typename LaunchFn>
double time_ms(sycl::queue& q, int iterations, LaunchFn&& launch) {
  q.wait();
  for (int i = 0; i < 2; ++i) {
    launch().wait();
  }
  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    launch().wait();
  }
  q.wait();
  auto stop = std::chrono::steady_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  return total_ms / std::max(iterations, 1);
}

inline void print_device(sycl::queue const& q) {
  auto dev = q.get_device();
  std::cout << "device: " << dev.get_info<sycl::info::device::name>() << "\n";
}

}  // namespace cutlass::examples::comm_ar_sconv
