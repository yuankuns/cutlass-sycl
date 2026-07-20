/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

/*! \file
    \brief Shared relative-attention backend example utilities for CUTLASS SYCL on BMG.

    The device kernel intentionally keeps one unified control flow for the two
    Inkling backend requirements:

      * relative_bias_score_mod: add Aux0[q_idx, head, q_pos - kv_pos] before softmax
      * XPU flash-attention integration: honor causal and local-window masks

    Template parameters remove unused score-mod/window branches in each example.
    The public launcher returns a sycl::event and never allocates inside the
    launch path, which keeps it friendly to event chaining and graph capture.

    Roofline: for production D=128,Dv=128 local attention, the row/head path
    streams Q/K/V for every valid query-key pair and keeps probabilities in
    local memory for the value update. Arithmetic intensity is roughly
    0.5-1 FLOP/B depending on the active window, so sustained bandwidth is the
    primary metric. The perf suites use multi-MB working sets and report
    effective GB/s; small correctness cases are not used as performance evidence.
*/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

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

namespace cutlass::examples::relative_attention {

constexpr int kValueTile = 16;
constexpr int kDefaultLocalSize = 128;
constexpr double kMinSustainedTargetBytes = 32.0 * 1024.0 * 1024.0;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

struct Options {
  std::string suite = "quick";
  std::string shape;
  DType dtype = DType::kAll;
  int iterations = 20;
  int warmup = 5;
  int local_size = kDefaultLocalSize;
  bool verify = true;
  bool benchmark = true;
  bool target_gbps_set = false;
  double target_gbps = 0.0;
  bool help = false;
};

struct AttentionCase {
  std::string name;
  int batch = 1;
  int max_seq_len = 1;
  int heads = 1;
  int kv_heads = 1;
  int d = 64;
  int dv = 64;
  int rel_len = 0;
  int q_padding = 0;
  int k_padding = 0;
  int v_padding = 0;
  int o_padding = 0;
  int window_left = -1;
  int window_right = -1;
  float softcap = 0.0f;
  bool use_relative_bias = false;
  bool use_window = false;
  bool causal = true;
  bool varied_lengths = false;
  bool decode_tail = false;
  bool include_zero_length = false;
  double target_gbps = 0.0;
};

template <typename Element_>
struct AttentionParams {
  using Element = Element_;

  Element const* __restrict__ q;
  Element const* __restrict__ k;
  Element const* __restrict__ v;
  float const* __restrict__ rel_bias;
  int32_t const* __restrict__ q_to_seq;
  int32_t const* __restrict__ q_pos;
  int32_t const* __restrict__ cu_k;
  Element* __restrict__ out;
  float* __restrict__ lse;
  float scale;
  float softcap;
  int total_q;
  int total_k;
  int batch;
  int heads;
  int kv_heads;
  int d;
  int dv;
  int rel_len;
  int q_stride_t;
  int q_stride_h;
  int k_stride_t;
  int k_stride_h;
  int v_stride_t;
  int v_stride_h;
  int o_stride_t;
  int o_stride_h;
  int bias_stride_t;
  int bias_stride_h;
  int window_left;
  int window_right;
};

template <typename Element_>
struct HostTensors {
  using Element = Element_;

  std::vector<Element> q;
  std::vector<Element> k;
  std::vector<Element> v;
  std::vector<float> rel_bias;
  std::vector<int32_t> q_to_seq;
  std::vector<int32_t> q_pos;
  std::vector<int32_t> cu_k;
  std::vector<Element> out;
  std::vector<Element> ref;
  std::vector<float> lse;
  std::vector<float> ref_lse;
  int total_q = 0;
  int total_k = 0;
  int q_stride_t = 0;
  int q_stride_h = 0;
  int k_stride_t = 0;
  int k_stride_h = 0;
  int v_stride_t = 0;
  int v_stride_h = 0;
  int o_stride_t = 0;
  int o_stride_h = 0;
  int bias_stride_t = 0;
  int bias_stride_h = 0;
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

inline bool is_power_of_two(int x) {
  return x > 0 && (x & (x - 1)) == 0;
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
    } else if (key == "--local-size") {
      options.local_size = std::stoi(value);
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
  if (!is_power_of_two(options.local_size)) {
    throw std::invalid_argument("--local-size must be a power of two");
  }
  return options;
}

inline void print_common_usage(char const* name, char const* suite_text) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=" << suite_text << "       Built-in suite (default quick)\n"
      << "  --shape=k=v,...          Custom case; keys include batch,seq,heads,kv_heads,d,dv,rel,\n"
      << "                            bias,causal,window,window_right,decode,varied,zero,softcap\n"
      << "  --dtype=all|bf16|fp16    Element dtype (default all)\n"
      << "  --iterations=<int>       Timed kernel iterations (default 20)\n"
      << "  --warmup=<int>           Warmup launches before timing (default 5)\n"
      << "  --local-size=<int>       Work-items per row/value tile, power of two (default 128)\n"
      << "  --verify=0|1             Run CPU reference comparison (default 1)\n"
      << "  --benchmark=0|1          Run profiling-event timing (default 1)\n"
      << "  --target-gbps=<float>    Optional sustained effective GB/s gate; 0 disables\n";
}

inline bool parse_shape(std::string const& text, AttentionCase& cfg) {
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
    } else if (key == "batch") {
      cfg.batch = std::stoi(value);
    } else if (key == "seq") {
      cfg.max_seq_len = std::stoi(value);
    } else if (key == "heads") {
      cfg.heads = std::stoi(value);
    } else if (key == "kv_heads") {
      cfg.kv_heads = std::stoi(value);
    } else if (key == "d") {
      cfg.d = std::stoi(value);
    } else if (key == "dv") {
      cfg.dv = std::stoi(value);
    } else if (key == "rel") {
      cfg.rel_len = std::stoi(value);
    } else if (key == "qpad") {
      cfg.q_padding = std::stoi(value);
    } else if (key == "kpad") {
      cfg.k_padding = std::stoi(value);
    } else if (key == "vpad") {
      cfg.v_padding = std::stoi(value);
    } else if (key == "opad") {
      cfg.o_padding = std::stoi(value);
    } else if (key == "bias") {
      cfg.use_relative_bias = parse_bool(value);
    } else if (key == "causal") {
      cfg.causal = parse_bool(value);
    } else if (key == "window") {
      cfg.use_window = true;
      cfg.window_left = std::stoi(value);
      if (cfg.window_right < 0) {
        cfg.window_right = 0;
      }
    } else if (key == "window_right") {
      cfg.use_window = true;
      cfg.window_right = std::stoi(value);
    } else if (key == "decode") {
      cfg.decode_tail = parse_bool(value);
    } else if (key == "varied") {
      cfg.varied_lengths = parse_bool(value);
    } else if (key == "zero") {
      cfg.include_zero_length = parse_bool(value);
      if (cfg.include_zero_length) {
        cfg.varied_lengths = true;
      }
    } else if (key == "softcap") {
      cfg.softcap = std::stof(value);
    } else {
      return false;
    }
  }
  return true;
}

inline void validate_case(AttentionCase const& cfg) {
  if (cfg.batch <= 0 || cfg.max_seq_len <= 0 || cfg.heads <= 0 || cfg.kv_heads <= 0 ||
      cfg.d <= 0 || cfg.dv <= 0) {
    throw std::invalid_argument("case has non-positive shape");
  }
  if (cfg.heads % cfg.kv_heads != 0) {
    throw std::invalid_argument("heads must be divisible by kv_heads");
  }
  if (cfg.use_relative_bias && cfg.rel_len <= 0) {
    throw std::invalid_argument("relative-bias case requires rel_len > 0");
  }
}

inline std::vector<int> make_sequence_lengths(AttentionCase const& cfg) {
  std::vector<int> lengths(static_cast<std::size_t>(cfg.batch), cfg.max_seq_len);
  if (!cfg.varied_lengths) {
    return lengths;
  }
  int span = std::max(1, cfg.max_seq_len / 2);
  for (int b = 0; b < cfg.batch; ++b) {
    lengths[b] = std::max(1, cfg.max_seq_len - ((b * 7 + 3) % span));
  }
  if (cfg.batch > 1) {
    lengths[1] = std::min(lengths[1], std::max(1, cfg.max_seq_len / 4));
  }
  if (cfg.include_zero_length) {
    lengths.back() = 0;
  }
  return lengths;
}

template <typename Element, bool UseRelativeBias, bool UseWindow, bool UseCausal>
CUTLASS_DEVICE
bool key_is_valid(AttentionParams<Element> const& params, int q_pos, int k_pos) {
  bool valid = true;
  if constexpr (UseCausal) {
    valid = valid && (k_pos <= q_pos);
  }
  if constexpr (UseWindow) {
    if (params.window_left >= 0) {
      valid = valid && (k_pos >= q_pos - params.window_left);
    }
    if (params.window_right >= 0) {
      valid = valid && (k_pos <= q_pos + params.window_right);
    }
  }
  return valid;
}

template <typename Element, bool UseRelativeBias, bool UseWindow, bool UseCausal>
CUTLASS_DEVICE
float compute_score(
    AttentionParams<Element> const& params,
    int q_row,
    int head,
    int kv_head,
    int q_abs_pos,
    int k_global,
    int k_pos) {
  if (!key_is_valid<Element, UseRelativeBias, UseWindow, UseCausal>(params, q_abs_pos, k_pos)) {
    return -3.402823466e+38F;
  }

  int64_t q_base = static_cast<int64_t>(q_row) * params.q_stride_t +
      static_cast<int64_t>(head) * params.q_stride_h;
  int64_t k_base = static_cast<int64_t>(k_global) * params.k_stride_t +
      static_cast<int64_t>(kv_head) * params.k_stride_h;
  float score = 0.0f;
  for (int d = 0; d < params.d; ++d) {
    score += to_float(params.q[q_base + d]) * to_float(params.k[k_base + d]);
  }
  score *= params.scale;

  if constexpr (UseRelativeBias) {
    int rel = q_abs_pos - k_pos;
    if (rel >= 0 && rel < params.rel_len) {
      int64_t bias_offset = static_cast<int64_t>(q_row) * params.bias_stride_t +
          static_cast<int64_t>(head) * params.bias_stride_h + rel;
      score += params.rel_bias[bias_offset];
    }
  }

  if (params.softcap > 0.0f) {
    score = params.softcap * sycl::tanh(score / params.softcap);
  }
  return score;
}

template <typename Element, bool UseRelativeBias, bool UseWindow, bool UseCausal, int ValueTile>
class RelativeAttentionKernel;

template <typename Element, bool UseRelativeBias, bool UseWindow, bool UseCausal>
class RelativeAttentionRowKernel;

template <typename Element, bool UseRelativeBias, bool UseWindow, bool UseCausal>
sycl::event launch_attention_row_static(
    sycl::queue& queue,
    AttentionParams<Element> const& params,
    int local_size) {
  if (params.total_q == 0) {
    return sycl::event{};
  }

  int64_t groups = static_cast<int64_t>(params.total_q) * params.heads;
  int64_t global = groups * local_size;

  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> p_scratch(sycl::range<1>(static_cast<std::size_t>(local_size)), cgh);
    cgh.parallel_for<RelativeAttentionRowKernel<Element, UseRelativeBias, UseWindow, UseCausal>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local_size))),
        [=](sycl::nd_item<1> item) {
          sycl::sub_group sg = item.get_sub_group();
          int local_id = static_cast<int>(item.get_local_id(0));
          int sg_lane = static_cast<int>(sg.get_local_id());
          int sg_id = static_cast<int>(sg.get_group_id());
          int sg_size = static_cast<int>(sg.get_local_range()[0]);
          int sg_count = (local_size + sg_size - 1) / sg_size;
          int group_id = static_cast<int>(item.get_group(0));
          int head = group_id % params.heads;
          int q_row = group_id / params.heads;
          int kv_group = params.heads / params.kv_heads;
          int kv_head = head / kv_group;
          int seq = params.q_to_seq[q_row];
          int kv_begin = params.cu_k[seq];
          int kv_end = params.cu_k[seq + 1];
          int kv_len = kv_end - kv_begin;
          int q_abs_pos = params.q_pos[q_row];
          int valid_begin = 0;
          int valid_end = kv_len;
          if constexpr (UseCausal) {
            valid_end = sycl::min(valid_end, q_abs_pos + 1);
          }
          if constexpr (UseWindow) {
            if (params.window_left >= 0) {
              valid_begin = sycl::max(valid_begin, q_abs_pos - params.window_left);
            }
            if (params.window_right >= 0) {
              valid_end = sycl::min(valid_end, q_abs_pos + params.window_right + 1);
            }
          }
          valid_begin = sycl::max(0, sycl::min(valid_begin, kv_len));
          valid_end = sycl::max(valid_begin, sycl::min(valid_end, kv_len));
          int valid_len = valid_end - valid_begin;

          float e_max = -3.402823466e+38F;
          float denom = 0.0f;
          float acc = 0.0f;
          bool owns_value = local_id < params.dv;

          for (int tile_begin = 0; tile_begin < valid_len; tile_begin += local_size) {
            int tile_count = sycl::min(local_size, valid_len - tile_begin);
            int k_local = valid_begin + tile_begin + local_id;
            float score = -3.402823466e+38F;
            if (local_id < tile_count) {
              score = compute_score<Element, UseRelativeBias, UseWindow, UseCausal>(
                  params, q_row, head, kv_head, q_abs_pos, kv_begin + k_local, k_local);
            }

            float sg_max = sycl::reduce_over_group(sg, score, sycl::maximum<float>());
            if (sg_lane == 0) {
              p_scratch[sg_id] = sg_max;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (local_id == 0) {
              float reduced = -3.402823466e+38F;
              for (int i = 0; i < sg_count; ++i) {
                float candidate = p_scratch[i];
                reduced = candidate > reduced ? candidate : reduced;
              }
              p_scratch[0] = reduced;
            }
            item.barrier(sycl::access::fence_space::local_space);
            float tile_max = p_scratch[0];
            item.barrier(sycl::access::fence_space::local_space);
            float n_e_max = tile_max > e_max ? tile_max : e_max;
            float re_scale = sycl::exp(e_max - n_e_max);
            float p = local_id < tile_count ? sycl::exp(score - n_e_max) : 0.0f;
            float sg_sum = sycl::reduce_over_group(sg, p, sycl::plus<float>());
            if (sg_lane == 0) {
              p_scratch[sg_id] = sg_sum;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (local_id == 0) {
              float reduced = 0.0f;
              for (int i = 0; i < sg_count; ++i) {
                reduced += p_scratch[i];
              }
              p_scratch[0] = reduced;
            }
            item.barrier(sycl::access::fence_space::local_space);
            float tile_sum = p_scratch[0];
            item.barrier(sycl::access::fence_space::local_space);

            p_scratch[local_id] = p;
            item.barrier(sycl::access::fence_space::local_space);

            if (owns_value) {
              acc *= re_scale;
              for (int n = 0; n < tile_count; ++n) {
                int v_k_local = valid_begin + tile_begin + n;
                int64_t v_base = static_cast<int64_t>(kv_begin + v_k_local) * params.v_stride_t +
                    static_cast<int64_t>(kv_head) * params.v_stride_h;
                acc += p_scratch[n] * to_float(params.v[v_base + local_id]);
              }
            }

            denom = denom * re_scale + tile_sum;
            e_max = n_e_max;
            item.barrier(sycl::access::fence_space::local_space);
          }

          int64_t o_base = static_cast<int64_t>(q_row) * params.o_stride_t +
              static_cast<int64_t>(head) * params.o_stride_h;
          if (owns_value) {
            float value = denom > 0.0f ? acc / denom : 0.0f;
            params.out[o_base + local_id] = from_float<Element>(value);
          }
          if (local_id == 0) {
            int64_t lse_offset = static_cast<int64_t>(q_row) * params.heads + head;
            params.lse[lse_offset] = denom > 0.0f ? sycl::log(denom) + e_max :
                -std::numeric_limits<float>::infinity();
          }
        });
  });
}

template <typename Element, bool UseRelativeBias, bool UseWindow, bool UseCausal, int ValueTile>
sycl::event launch_attention_static(
    sycl::queue& queue,
    AttentionParams<Element> const& params,
    int local_size) {
  if (params.total_q == 0) {
    return sycl::event{};
  }

  int value_tiles = ceil_div(params.dv, ValueTile);
  int64_t groups = static_cast<int64_t>(params.total_q) * params.heads * value_tiles;
  int64_t global = groups * local_size;
  std::size_t scratch_floats = static_cast<std::size_t>(local_size) * (1 + ValueTile);

  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(scratch_floats), cgh);
    cgh.parallel_for<RelativeAttentionKernel<Element, UseRelativeBias, UseWindow, UseCausal, ValueTile>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local_size))),
        [=](sycl::nd_item<1> item) {
          int local_id = static_cast<int>(item.get_local_id(0));
          int group_id = static_cast<int>(item.get_group(0));
          int value_tiles_local = (params.dv + ValueTile - 1) / ValueTile;
          int value_tile = group_id % value_tiles_local;
          int head = (group_id / value_tiles_local) % params.heads;
          int q_row = group_id / (value_tiles_local * params.heads);
          int kv_group = params.heads / params.kv_heads;
          int kv_head = head / kv_group;
          int seq = params.q_to_seq[q_row];
          int kv_begin = params.cu_k[seq];
          int kv_end = params.cu_k[seq + 1];
          int kv_len = kv_end - kv_begin;
          int q_abs_pos = params.q_pos[q_row];
          int valid_begin = 0;
          int valid_end = kv_len;
          if constexpr (UseCausal) {
            valid_end = sycl::min(valid_end, q_abs_pos + 1);
          }
          if constexpr (UseWindow) {
            if (params.window_left >= 0) {
              valid_begin = sycl::max(valid_begin, q_abs_pos - params.window_left);
            }
            if (params.window_right >= 0) {
              valid_end = sycl::min(valid_end, q_abs_pos + params.window_right + 1);
            }
          }
          valid_begin = sycl::max(0, sycl::min(valid_begin, kv_len));
          valid_end = sycl::max(valid_begin, sycl::min(valid_end, kv_len));
          int valid_len = valid_end - valid_begin;

          float local_max = -3.402823466e+38F;
          for (int offset = local_id; offset < valid_len; offset += local_size) {
            int k_local = valid_begin + offset;
            float score = compute_score<Element, UseRelativeBias, UseWindow, UseCausal>(
                params, q_row, head, kv_head, q_abs_pos, kv_begin + k_local, k_local);
            local_max = score > local_max ? score : local_max;
          }

          scratch[local_id] = local_max;
          item.barrier(sycl::access::fence_space::local_space);
          for (int offset = local_size >> 1; offset > 0; offset >>= 1) {
            if (local_id < offset) {
              float rhs = scratch[local_id + offset];
              scratch[local_id] = rhs > scratch[local_id] ? rhs : scratch[local_id];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float row_max = scratch[0];
          item.barrier(sycl::access::fence_space::local_space);

          int dv_base = value_tile * ValueTile;
          float acc[ValueTile];
          for (int i = 0; i < ValueTile; ++i) {
            acc[i] = 0.0f;
          }
          float denom = 0.0f;

          if (row_max > -3.0e38F) {
            for (int offset = local_id; offset < valid_len; offset += local_size) {
              int k_local = valid_begin + offset;
              float score = compute_score<Element, UseRelativeBias, UseWindow, UseCausal>(
                  params, q_row, head, kv_head, q_abs_pos, kv_begin + k_local, k_local);
              float p = sycl::exp(score - row_max);
              denom += p;
              int64_t v_base = static_cast<int64_t>(kv_begin + k_local) * params.v_stride_t +
                  static_cast<int64_t>(kv_head) * params.v_stride_h;
              for (int i = 0; i < ValueTile; ++i) {
                int dv = dv_base + i;
                if (dv < params.dv) {
                  acc[i] += p * to_float(params.v[v_base + dv]);
                }
              }
            }
          }

          scratch[local_id] = denom;
          item.barrier(sycl::access::fence_space::local_space);
          for (int offset = local_size >> 1; offset > 0; offset >>= 1) {
            if (local_id < offset) {
              scratch[local_id] += scratch[local_id + offset];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float row_denom = scratch[0];
          item.barrier(sycl::access::fence_space::local_space);

          for (int i = 0; i < ValueTile; ++i) {
            scratch[(i + 1) * local_size + local_id] = acc[i];
          }
          item.barrier(sycl::access::fence_space::local_space);
          for (int offset = local_size >> 1; offset > 0; offset >>= 1) {
            if (local_id < offset) {
              for (int i = 0; i < ValueTile; ++i) {
                scratch[(i + 1) * local_size + local_id] +=
                    scratch[(i + 1) * local_size + local_id + offset];
              }
            }
            item.barrier(sycl::access::fence_space::local_space);
          }

          if (local_id == 0) {
            int64_t o_base = static_cast<int64_t>(q_row) * params.o_stride_t +
                static_cast<int64_t>(head) * params.o_stride_h;
            for (int i = 0; i < ValueTile; ++i) {
              int dv = dv_base + i;
              if (dv < params.dv) {
                float value = row_denom > 0.0f ? scratch[(i + 1) * local_size] / row_denom : 0.0f;
                params.out[o_base + dv] = from_float<Element>(value);
              }
            }
            if (value_tile == 0) {
              int64_t lse_offset = static_cast<int64_t>(q_row) * params.heads + head;
              params.lse[lse_offset] = row_denom > 0.0f ? sycl::log(row_denom) + row_max :
                  -std::numeric_limits<float>::infinity();
            }
          }
        });
  });
}

template <typename Element>
sycl::event launch_attention(
    sycl::queue& queue,
    AttentionParams<Element> const& params,
    AttentionCase const& cfg,
    int local_size) {
  if (params.dv <= local_size) {
    if (cfg.use_relative_bias) {
      if (cfg.use_window) {
        if (cfg.causal) {
          return launch_attention_row_static<Element, true, true, true>(queue, params, local_size);
        }
        return launch_attention_row_static<Element, true, true, false>(queue, params, local_size);
      }
      if (cfg.causal) {
        return launch_attention_row_static<Element, true, false, true>(queue, params, local_size);
      }
      return launch_attention_row_static<Element, true, false, false>(queue, params, local_size);
    }

    if (cfg.use_window) {
      if (cfg.causal) {
        return launch_attention_row_static<Element, false, true, true>(queue, params, local_size);
      }
      return launch_attention_row_static<Element, false, true, false>(queue, params, local_size);
    }
    if (cfg.causal) {
      return launch_attention_row_static<Element, false, false, true>(queue, params, local_size);
    }
    return launch_attention_row_static<Element, false, false, false>(queue, params, local_size);
  }

  if (cfg.use_relative_bias) {
    if (cfg.use_window) {
      if (cfg.causal) {
        return launch_attention_static<Element, true, true, true, kValueTile>(queue, params, local_size);
      }
      return launch_attention_static<Element, true, true, false, kValueTile>(queue, params, local_size);
    }
    if (cfg.causal) {
      return launch_attention_static<Element, true, false, true, kValueTile>(queue, params, local_size);
    }
    return launch_attention_static<Element, true, false, false, kValueTile>(queue, params, local_size);
  }

  if (cfg.use_window) {
    if (cfg.causal) {
      return launch_attention_static<Element, false, true, true, kValueTile>(queue, params, local_size);
    }
    return launch_attention_static<Element, false, true, false, kValueTile>(queue, params, local_size);
  }
  if (cfg.causal) {
    return launch_attention_static<Element, false, false, true, kValueTile>(queue, params, local_size);
  }
  return launch_attention_static<Element, false, false, false, kValueTile>(queue, params, local_size);
}

template <typename Element>
void compute_reference(AttentionCase const& cfg, HostTensors<Element>& h) {
  std::fill(h.ref.begin(), h.ref.end(), from_float<Element>(0.0f));
  std::fill(h.ref_lse.begin(), h.ref_lse.end(), -std::numeric_limits<float>::infinity());
  int kv_group = cfg.heads / cfg.kv_heads;
  for (int q_row = 0; q_row < h.total_q; ++q_row) {
    int seq = h.q_to_seq[q_row];
    int kv_begin = h.cu_k[seq];
    int kv_end = h.cu_k[seq + 1];
    int kv_len = kv_end - kv_begin;
    int q_abs_pos = h.q_pos[q_row];
    for (int head = 0; head < cfg.heads; ++head) {
      int kv_head = head / kv_group;
      std::vector<float> scores(static_cast<std::size_t>(kv_len), -std::numeric_limits<float>::infinity());
      float row_max = -std::numeric_limits<float>::infinity();
      for (int k_local = 0; k_local < kv_len; ++k_local) {
        bool valid = true;
        if (cfg.causal) {
          valid = valid && (k_local <= q_abs_pos);
        }
        if (cfg.use_window) {
          if (cfg.window_left >= 0) {
            valid = valid && (k_local >= q_abs_pos - cfg.window_left);
          }
          if (cfg.window_right >= 0) {
            valid = valid && (k_local <= q_abs_pos + cfg.window_right);
          }
        }
        if (!valid) {
          continue;
        }

        int64_t q_base = static_cast<int64_t>(q_row) * h.q_stride_t +
            static_cast<int64_t>(head) * h.q_stride_h;
        int64_t k_base = static_cast<int64_t>(kv_begin + k_local) * h.k_stride_t +
            static_cast<int64_t>(kv_head) * h.k_stride_h;
        float score = 0.0f;
        for (int d = 0; d < cfg.d; ++d) {
          score += to_float(h.q[q_base + d]) * to_float(h.k[k_base + d]);
        }
        score *= 1.0f / std::sqrt(static_cast<float>(cfg.d));

        if (cfg.use_relative_bias) {
          int rel = q_abs_pos - k_local;
          if (rel >= 0 && rel < cfg.rel_len) {
            int64_t bias_offset = static_cast<int64_t>(q_row) * h.bias_stride_t +
                static_cast<int64_t>(head) * h.bias_stride_h + rel;
            score += h.rel_bias[bias_offset];
          }
        }
        if (cfg.softcap > 0.0f) {
          score = cfg.softcap * std::tanh(score / cfg.softcap);
        }
        scores[k_local] = score;
        row_max = std::max(row_max, score);
      }

      int64_t o_base = static_cast<int64_t>(q_row) * h.o_stride_t +
          static_cast<int64_t>(head) * h.o_stride_h;
      int64_t lse_offset = static_cast<int64_t>(q_row) * cfg.heads + head;
      if (!std::isfinite(row_max)) {
        for (int dv = 0; dv < cfg.dv; ++dv) {
          h.ref[o_base + dv] = from_float<Element>(0.0f);
        }
        h.ref_lse[lse_offset] = -std::numeric_limits<float>::infinity();
        continue;
      }

      float denom = 0.0f;
      for (float score : scores) {
        if (std::isfinite(score)) {
          denom += std::exp(score - row_max);
        }
      }
      h.ref_lse[lse_offset] = std::log(denom) + row_max;

      for (int dv = 0; dv < cfg.dv; ++dv) {
        float acc = 0.0f;
        for (int k_local = 0; k_local < kv_len; ++k_local) {
          if (!std::isfinite(scores[k_local])) {
            continue;
          }
          float p = std::exp(scores[k_local] - row_max) / denom;
          int64_t v_base = static_cast<int64_t>(kv_begin + k_local) * h.v_stride_t +
              static_cast<int64_t>(kv_head) * h.v_stride_h;
          acc += p * to_float(h.v[v_base + dv]);
        }
        h.ref[o_base + dv] = from_float<Element>(acc);
      }
    }
  }
}

template <typename Element>
HostTensors<Element> initialize_case(AttentionCase const& cfg, uint32_t seed) {
  validate_case(cfg);
  HostTensors<Element> h;

  std::vector<int> lengths = make_sequence_lengths(cfg);
  h.cu_k.resize(static_cast<std::size_t>(cfg.batch + 1), 0);
  for (int b = 0; b < cfg.batch; ++b) {
    h.cu_k[b + 1] = h.cu_k[b] + lengths[b];
  }
  h.total_k = h.cu_k.back();

  for (int b = 0; b < cfg.batch; ++b) {
    int q_len = cfg.decode_tail ? (lengths[b] > 0 ? 1 : 0) : lengths[b];
    int q_start_pos = lengths[b] - q_len;
    for (int q = 0; q < q_len; ++q) {
      h.q_to_seq.push_back(b);
      h.q_pos.push_back(q_start_pos + q);
    }
  }
  h.total_q = static_cast<int>(h.q_to_seq.size());
  if (h.total_q == 0 || h.total_k == 0) {
    throw std::invalid_argument("case produced no query or key tokens");
  }

  h.q_stride_h = round_up(cfg.d + cfg.q_padding, 1);
  h.k_stride_h = round_up(cfg.d + cfg.k_padding, 1);
  h.v_stride_h = round_up(cfg.dv + cfg.v_padding, 1);
  h.o_stride_h = round_up(cfg.dv + cfg.o_padding, 1);
  h.q_stride_t = cfg.heads * h.q_stride_h;
  h.k_stride_t = cfg.kv_heads * h.k_stride_h;
  h.v_stride_t = cfg.kv_heads * h.v_stride_h;
  h.o_stride_t = cfg.heads * h.o_stride_h;
  h.bias_stride_h = std::max(cfg.rel_len, 1);
  h.bias_stride_t = cfg.heads * h.bias_stride_h;

  h.q.resize(static_cast<std::size_t>(h.total_q) * h.q_stride_t);
  h.k.resize(static_cast<std::size_t>(h.total_k) * h.k_stride_t);
  h.v.resize(static_cast<std::size_t>(h.total_k) * h.v_stride_t);
  h.rel_bias.resize(static_cast<std::size_t>(std::max(1, h.total_q * h.bias_stride_t)));
  h.out.resize(static_cast<std::size_t>(h.total_q) * h.o_stride_t);
  h.ref.resize(h.out.size());
  h.lse.resize(static_cast<std::size_t>(h.total_q) * cfg.heads);
  h.ref_lse.resize(h.lse.size());

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> value_dist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> bias_dist(-0.15f, 0.15f);

  for (Element& x : h.q) {
    x = from_float<Element>(value_dist(gen));
  }
  for (Element& x : h.k) {
    x = from_float<Element>(value_dist(gen));
  }
  for (Element& x : h.v) {
    x = from_float<Element>(value_dist(gen));
  }
  for (float& x : h.rel_bias) {
    x = cfg.use_relative_bias ? bias_dist(gen) : 0.0f;
  }
  std::fill(h.out.begin(), h.out.end(), from_float<Element>(0.0f));
  std::fill(h.lse.begin(), h.lse.end(), 0.0f);

  compute_reference(cfg, h);
  return h;
}

struct VerifyResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::size_t index = 0;
};

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

inline VerifyResult compare_lse(
    std::vector<float> const& got,
    std::vector<float> const& expected,
    double atol,
    double rtol) {
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double g = static_cast<double>(got[i]);
    double e = static_cast<double>(expected[i]);
    if (!std::isfinite(g) || !std::isfinite(e)) {
      if (std::isfinite(g) != std::isfinite(e)) {
        result.passed = false;
        result.index = i;
      }
      continue;
    }
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

inline double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

template <typename Element>
int64_t count_valid_pairs(AttentionCase const& cfg, HostTensors<Element> const& h) {
  int64_t pairs = 0;
  for (int q_row = 0; q_row < h.total_q; ++q_row) {
    int seq = h.q_to_seq[q_row];
    int kv_len = h.cu_k[seq + 1] - h.cu_k[seq];
    int q_abs_pos = h.q_pos[q_row];
    for (int k = 0; k < kv_len; ++k) {
      bool valid = true;
      if (cfg.causal) {
        valid = valid && (k <= q_abs_pos);
      }
      if (cfg.use_window) {
        if (cfg.window_left >= 0) {
          valid = valid && (k >= q_abs_pos - cfg.window_left);
        }
        if (cfg.window_right >= 0) {
          valid = valid && (k <= q_abs_pos + cfg.window_right);
        }
      }
      pairs += valid ? 1 : 0;
    }
  }
  return pairs;
}

inline double estimate_flops(AttentionCase const& cfg, int64_t valid_pairs, int local_size) {
  if (cfg.dv <= local_size) {
    double pairs = static_cast<double>(valid_pairs) * cfg.heads;
    return pairs * (2.0 * cfg.d + 3.0 * cfg.dv + 8.0);
  }
  int value_tiles = ceil_div(cfg.dv, kValueTile);
  double pairs = static_cast<double>(valid_pairs) * cfg.heads;
  double per_pair_per_tile = 2.0 * cfg.d + 3.0 * kValueTile + 8.0;
  return pairs * value_tiles * per_pair_per_tile;
}

inline double estimate_bytes(
    AttentionCase const& cfg,
    int total_q,
    int64_t valid_pairs,
    std::size_t element_bytes,
    int local_size) {
  if (cfg.dv <= local_size) {
    double pairs = static_cast<double>(valid_pairs) * cfg.heads;
    double qkv = static_cast<double>(2 * cfg.d + cfg.dv) * element_bytes;
    double bias = cfg.use_relative_bias ? sizeof(float) : 0.0;
    double streamed = pairs * (qkv + bias);
    double output = static_cast<double>(total_q) * cfg.heads * cfg.dv * element_bytes;
    double lse = static_cast<double>(total_q) * cfg.heads * sizeof(float);
    return streamed + output + lse;
  }
  int value_tiles = ceil_div(cfg.dv, kValueTile);
  double pairs = static_cast<double>(valid_pairs) * cfg.heads;
  double qk_per_tile = static_cast<double>(2 * cfg.d) * element_bytes;
  double v_per_tile = static_cast<double>(kValueTile) * element_bytes;
  double bias_per_tile = cfg.use_relative_bias ? sizeof(float) : 0.0;
  double streamed = pairs * value_tiles * (qk_per_tile + v_per_tile + bias_per_tile);
  double output = static_cast<double>(total_q) * cfg.heads * cfg.dv * element_bytes;
  double lse = static_cast<double>(total_q) * cfg.heads * sizeof(float);
  return streamed + output + lse;
}

template <typename Element>
std::string element_dtype_text() {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  }
  return "fp16";
}

template <typename Element>
bool run_case(
    sycl::queue& queue,
    AttentionCase const& cfg,
    Options const& options,
    double target_gbps) {
  HostTensors<Element> h = initialize_case<Element>(
      cfg,
      2027u + static_cast<uint32_t>(cfg.max_seq_len * 17 + cfg.heads * 31 + cfg.d));

  DeviceBuffer<Element> d_q(queue, h.q.size());
  DeviceBuffer<Element> d_k(queue, h.k.size());
  DeviceBuffer<Element> d_v(queue, h.v.size());
  DeviceBuffer<float> d_bias(queue, h.rel_bias.size());
  DeviceBuffer<int32_t> d_q_to_seq(queue, h.q_to_seq.size());
  DeviceBuffer<int32_t> d_q_pos(queue, h.q_pos.size());
  DeviceBuffer<int32_t> d_cu_k(queue, h.cu_k.size());
  DeviceBuffer<Element> d_out(queue, h.out.size());
  DeviceBuffer<float> d_lse(queue, h.lse.size());

  d_q.copy_from(h.q);
  d_k.copy_from(h.k);
  d_v.copy_from(h.v);
  d_bias.copy_from(h.rel_bias);
  d_q_to_seq.copy_from(h.q_to_seq);
  d_q_pos.copy_from(h.q_pos);
  d_cu_k.copy_from(h.cu_k);
  d_out.copy_from(h.out);
  d_lse.copy_from(h.lse);

  AttentionParams<Element> params{};
  params.q = d_q.get();
  params.k = d_k.get();
  params.v = d_v.get();
  params.rel_bias = d_bias.get();
  params.q_to_seq = d_q_to_seq.get();
  params.q_pos = d_q_pos.get();
  params.cu_k = d_cu_k.get();
  params.out = d_out.get();
  params.lse = d_lse.get();
  params.scale = 1.0f / std::sqrt(static_cast<float>(cfg.d));
  params.softcap = cfg.softcap;
  params.total_q = h.total_q;
  params.total_k = h.total_k;
  params.batch = cfg.batch;
  params.heads = cfg.heads;
  params.kv_heads = cfg.kv_heads;
  params.d = cfg.d;
  params.dv = cfg.dv;
  params.rel_len = cfg.rel_len;
  params.q_stride_t = h.q_stride_t;
  params.q_stride_h = h.q_stride_h;
  params.k_stride_t = h.k_stride_t;
  params.k_stride_h = h.k_stride_h;
  params.v_stride_t = h.v_stride_t;
  params.v_stride_h = h.v_stride_h;
  params.o_stride_t = h.o_stride_t;
  params.o_stride_h = h.o_stride_h;
  params.bias_stride_t = h.bias_stride_t;
  params.bias_stride_h = h.bias_stride_h;
  params.window_left = cfg.window_left;
  params.window_right = cfg.window_right;

  auto launch = [&]() {
    return launch_attention<Element>(queue, params, cfg, options.local_size);
  };

  bool passed = true;
  if (options.verify) {
    launch().wait_and_throw();
    d_out.copy_to(h.out);
    d_lse.copy_to(h.lse);
    double out_atol = std::is_same_v<Element, cutlass::bfloat16_t> ? 5.5e-2 : 8.0e-3;
    double out_rtol = std::is_same_v<Element, cutlass::bfloat16_t> ? 5.5e-2 : 8.0e-3;
    double lse_atol = std::is_same_v<Element, cutlass::bfloat16_t> ? 4.0e-2 : 5.0e-3;
    double lse_rtol = std::is_same_v<Element, cutlass::bfloat16_t> ? 4.0e-2 : 5.0e-3;
    VerifyResult out_result = compare_output(h.out, h.ref, out_atol, out_rtol);
    VerifyResult lse_result = compare_lse(h.lse, h.ref_lse, lse_atol, lse_rtol);
    passed = out_result.passed && lse_result.passed;
    if (!passed) {
      print_verify_result("out", out_result);
      print_verify_result("lse", lse_result);
    }
  }

  double avg_ms = 0.0;
  double gbps = 0.0;
  double tops = 0.0;
  bool perf_passed = true;
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
    int64_t valid_pairs = count_valid_pairs(cfg, h);
    double bytes = estimate_bytes(cfg, h.total_q, valid_pairs, sizeof(Element), options.local_size);
    double flops = estimate_flops(cfg, valid_pairs, options.local_size);
    gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
    tops = flops / (avg_ms * 1.0e-3) / 1.0e12;
    perf_passed = target_gbps <= 0.0 || bytes < kMinSustainedTargetBytes || gbps >= target_gbps;
    passed = passed && perf_passed;
  }

  std::ostringstream suffix;
  if (target_gbps > 0.0) {
    suffix << " target=" << std::fixed << std::setprecision(2) << target_gbps << " GB/s";
  }
  std::cout << "  [" << element_dtype_text<Element>() << "] "
            << std::left << std::setw(28) << cfg.name << std::right
            << " B=" << cfg.batch
            << " Tq=" << h.total_q
            << " Tk=" << h.total_k
            << " H=" << cfg.heads
            << "/" << cfg.kv_heads
            << " D=" << cfg.d
            << " Dv=" << cfg.dv
            << " rel=" << (cfg.use_relative_bias ? cfg.rel_len : 0)
            << " window=" << (cfg.use_window ? cfg.window_left : -1)
            << " causal=" << bool_text(cfg.causal);
  if (options.benchmark) {
    std::cout << "  " << std::fixed << std::setprecision(3) << (avg_ms * 1000.0) << " us"
              << "  " << std::setprecision(2) << gbps << " GB/s"
              << suffix.str()
              << "  " << std::setprecision(3) << tops << " TOPS";
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

inline int run_suite(
    int argc,
    char const** argv,
    char const* suite_text,
    AttentionCase custom_default,
    std::vector<AttentionCase> (*make_suite)(std::string const&),
    char const* example_description) {
  Options options;
  try {
    options = parse_options(argc, argv);
    if (options.help) {
      std::cout << example_description << "\n\n";
      print_common_usage(argv[0], suite_text);
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<AttentionCase> cases;
  if (!options.shape.empty()) {
    if (!parse_shape(options.shape, custom_default)) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    custom_default.name = custom_default.name.empty() ? "custom" : custom_default.name;
    cases.push_back(custom_default);
  } else {
    cases = make_suite(options.suite);
    if (cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue(
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << example_description << "\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " local_size=" << options.local_size
              << " verify=" << bool_text(options.verify)
              << " benchmark=" << bool_text(options.benchmark) << "\n";

    bool all_passed = true;
    for (AttentionCase const& cfg : cases) {
      double target_gbps = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
      if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
        all_passed &= run_case<cutlass::bfloat16_t>(queue, cfg, options, target_gbps);
      }
      if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
        all_passed &= run_case<cutlass::half_t>(queue, cfg, options, target_gbps);
      }
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}

}  // namespace cutlass::examples::relative_attention
