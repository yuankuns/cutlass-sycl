/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling audio mel-bin embedding lookup + sum(dim=1) for CUTLASS SYCL on BMG.
 *
 * Semantics:
 *   features[token, mel] is an int32 value in [0, mel_vocab_size).
 *   weight[(mel * mel_vocab_size + features[token, mel]), hidden] is looked up
 *   for each mel bin and accumulated in increasing mel order for every output
 *   channel. The launcher preserves the Inkling chunking policy by launching
 *   token chunks while avoiding the [tokens, n_mel_bins, hidden] temporary.
 *
 * Roofline summary:
 *   For production bf16/fp16 (n_mel_bins=80, hidden=6144), each output element
 *   streams 80 two-byte embedding values and writes one two-byte result while
 *   doing about 80 FP32 additions. Feature indices are staged once per token per
 *   channel tile in local memory, so their traffic is amortized. Arithmetic
 *   intensity is roughly 80 / (80*2 + 2) = 0.49 FLOP/B, making the kernel
 *   memory-bound. The benchmark reports estimated effective bandwidth and the
 *   optimization target is sustained read/write bandwidth rather than TOPS.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

namespace cutlass::examples::bmg_mel_embedding_sum {

constexpr int kChannelBlock = 256;
constexpr int kDefaultChunkSize = 512;
constexpr double kBytesPerGB = 1.0e9;
constexpr double kMemoryBoundTargetGBps = 350.0;
constexpr double kLargeVocabTargetGBps = 200.0;

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
  int channels_per_item = 0;
  bool verify = true;
  bool benchmark = true;
  bool target_gbps_set = false;
  double target_gbps = 0.0;
  bool help = false;
};

struct CaseConfig {
  std::string name;
  int tokens = 1;
  int n_mel_bins = 80;
  int mel_vocab_size = 16;
  int hidden = 6144;
  int chunk_size = kDefaultChunkSize;
  double target_gbps = 0.0;
  bool allow_verify = true;
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

template <typename Element>
struct KernelParams {
  int32_t const* __restrict__ features = nullptr;
  Element const* __restrict__ weight = nullptr;
  Element* __restrict__ out = nullptr;
  int tokens = 0;
  int n_mel_bins = 0;
  int mel_vocab_size = 0;
  int hidden = 0;
  int token_offset = 0;
  int chunk_tokens = 0;
};

struct VerifyResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  int max_ulps = 0;
  std::size_t index = 0;
  uint32_t got_bits = 0;
  uint32_t expected_bits = 0;
};

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline bool starts_with(std::string const& text, char const* prefix) {
  std::string p(prefix);
  return text.size() >= p.size() && text.compare(0, p.size(), p) == 0;
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
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    uint32_t bits = sycl::bit_cast<uint32_t>(x);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
      if (bits & 0x007fffffu) {
        return cutlass::bfloat16_t::bitcast(0x7fffu);
      }
      return cutlass::bfloat16_t::bitcast(static_cast<uint16_t>(bits >> 16));
    }
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return cutlass::bfloat16_t::bitcast(static_cast<uint16_t>((bits + rounding_bias) >> 16));
  } else {
    return static_cast<Element>(x);
  }
}

template <typename Element>
CUTLASS_DEVICE
Element element_from_raw(uint64_t raw, int lane) {
  return Element::bitcast(static_cast<uint16_t>(raw >> (16 * lane)));
}

template <typename Element>
CUTLASS_DEVICE
void load_add_vec8(Element const* ptr, float* accum) {
  uint64_t raw0 = *reinterpret_cast<uint64_t const*>(ptr);
  uint64_t raw1 = *reinterpret_cast<uint64_t const*>(ptr + 4);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    accum[i] += to_float(element_from_raw<Element>(raw0, i));
    accum[i + 4] += to_float(element_from_raw<Element>(raw1, i));
  }
}

template <typename Element>
CUTLASS_DEVICE
void store_vec8(Element* ptr, float const* accum) {
  uint64_t raw0 = 0;
  uint64_t raw1 = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    raw0 |= static_cast<uint64_t>(from_float<Element>(accum[i]).raw()) << (16 * i);
    raw1 |= static_cast<uint64_t>(from_float<Element>(accum[i + 4]).raw()) << (16 * i);
  }
  *reinterpret_cast<uint64_t*>(ptr) = raw0;
  *reinterpret_cast<uint64_t*>(ptr + 4) = raw1;
}

template <typename Element>
uint32_t host_bits(Element value) {
  if constexpr (std::is_same_v<Element, float>) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  } else {
    return static_cast<uint32_t>(value.raw());
  }
}

inline int ordered_raw16(uint32_t raw) {
  int value = static_cast<int>(raw & 0xffffu);
  return (value & 0x8000) ? (0x8000 - value) : (value + 0x8000);
}

inline double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

inline sycl::queue make_queue() {
  return sycl::queue(
      sycl::gpu_selector_v,
      sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
}

inline Options parse_options(int argc, char const** argv) {
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
    } else if (key == "channels-per-item") {
      options.channels_per_item = std::stoi(value);
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
  if (!(options.channels_per_item == 0 || options.channels_per_item == 1 ||
        options.channels_per_item == 2 || options.channels_per_item == 4 ||
        options.channels_per_item == 8)) {
    throw std::invalid_argument("--channels-per-item must be 0, 1, 2, 4, or 8");
  }
  return options;
}

inline void print_usage(char const* name) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=quick|inkling|perf      Built-in suite (default quick)\n"
      << "  --shape=tokens=<int>,bins=<int>,vocab=<int>,hidden=<int>,chunk=<int>\n"
      << "  --dtype=all|float|bf16|fp16     Element dtype (default all; all runs bf16/fp16)\n"
      << "  --channels-per-item=0|1|2|4|8   Channels computed per work-item, 0 auto (default 0)\n"
      << "  --iterations=<int>              Timed full-operation iterations (default 20)\n"
      << "  --warmup=<int>                  Warmup full-operation launches (default 5)\n"
      << "  --verify=0|1                    Run CPU reference comparison when case permits (default 1)\n"
      << "  --benchmark=0|1                 Run profiling-event timing (default 1)\n"
      << "  --target-gbps=<float>           Optional effective GB/s gate; 0 disables\n";
}

inline bool parse_shape(std::string const& text, CaseConfig& cfg) {
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
    } else if (key == "tokens") {
      cfg.tokens = std::stoi(value);
    } else if (key == "bins" || key == "n_mel_bins") {
      cfg.n_mel_bins = std::stoi(value);
    } else if (key == "vocab" || key == "mel_vocab_size") {
      cfg.mel_vocab_size = std::stoi(value);
    } else if (key == "hidden" || key == "decoder_dmodel") {
      cfg.hidden = std::stoi(value);
    } else if (key == "chunk" || key == "chunk_size") {
      cfg.chunk_size = std::stoi(value);
    } else if (key == "target" || key == "target_gbps" || key == "target-gbps") {
      cfg.target_gbps = std::stod(value);
    } else {
      return false;
    }
  }
  return true;
}

inline void validate_case(CaseConfig& cfg) {
  if (cfg.tokens <= 0 || cfg.n_mel_bins <= 0 || cfg.mel_vocab_size <= 0 ||
      cfg.hidden <= 0 || cfg.chunk_size <= 0) {
    throw std::invalid_argument("case has non-positive shape");
  }
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
}

inline std::vector<CaseConfig> quick_suite() {
  return {
      {"oracle_tail_chunk", 9, 5, 7, 8, 4, 0.0, true},
      {"oracle_single_chunk", 4, 3, 5, 6, 16, 0.0, true},
      {"hidden_tail_bins80", 17, 80, 16, 259, 8, 0.0, true},
      {"chunk_boundary_513", 513, 80, 16, 384, 512, 0.0, true},
  };
}

inline std::vector<CaseConfig> inkling_suite() {
  // The Inkling audio tower (InklingAudio.encoder) is a plain nn.Embedding
  // and does not participate in tensor-parallel sharding -- every rank runs
  // the full [n_mel_bins*mel_vocab_size, decoder_dmodel] lookup + reduction.
  // The two shipped decoder_dmodel values (cfg=1536 tied to text hidden_size,
  // prod=6144) are therefore covered per-rank regardless of TP=1/2/4/8, so
  // this suite mirrors the same T bands across both configs.
  //
  // Chunk-boundary T bands mirror the upstream PR-31557 CPU CI test
  // (num_tokens in {1, 511, 512, 513, 1025}), which was added specifically
  // to catch off-by-one bugs at the chunk boundary of the chunked audio
  // encoder:
  //   T=1                     -- single partial chunk, single token.
  //   T=chunk_size-1 (511)    -- single partial chunk just below full.
  //   T=chunk_size   (512)    -- exactly one full chunk, no tail.
  //   T=chunk_size+1 (513)    -- one full chunk + one-token tail.
  //   T=2*chunk_size+1 (1025) -- multiple full chunks + one-token tail.
  // T=chunk_size and T=2*chunk_size exercise the "no tail" launch shape
  // (every chunk hits chunk_tokens==chunk_size), which the other bands
  // never reach.
  return {
      // Production decoder_dmodel=6144 (matches text hidden_size=6144).
      {"prod_h6144_decode_t1", 1, 80, 16, 6144, 512, 0.0, true},
      {"prod_h6144_target_verify_t9", 9, 80, 16, 6144, 512, 0.0, true},
      {"prod_h6144_below_chunk_t511", 511, 80, 16, 6144, 512, 0.0, true},
      {"prod_h6144_full_chunk_t512", 512, 80, 16, 6144, 512, 0.0, true},
      {"prod_h6144_chunk_tail_t513", 513, 80, 16, 6144, 512, 0.0, true},
      {"prod_h6144_two_chunks_t1025", 1025, 80, 16, 6144, 512, 0.0, true},
      {"prod_h6144_irregular", 65, 80, 16, 6145, 32, 0.0, true},
      // Config-defaults decoder_dmodel=1536 (matches text hidden_size=1536).
      // The tower is replicated per-rank so these shapes are identical across
      // TP=1/2/4/8; adding them completes the shipped-config coverage.
      {"cfg_h1536_decode_t1", 1, 80, 16, 1536, 512, 0.0, true},
      {"cfg_h1536_target_verify_t9", 9, 80, 16, 1536, 512, 0.0, true},
      {"cfg_h1536_below_chunk_t511", 511, 80, 16, 1536, 512, 0.0, true},
      {"cfg_h1536_full_chunk_t512", 512, 80, 16, 1536, 512, 0.0, true},
      {"cfg_h1536_chunk_tail_t513", 513, 80, 16, 1536, 512, 0.0, true},
      {"cfg_h1536_two_chunks_t1025", 1025, 80, 16, 1536, 512, 0.0, true},
      {"cfg_h1536_irregular", 65, 80, 16, 1537, 32, 0.0, true},
  };
}

inline std::vector<CaseConfig> perf_suite() {
  return {
      // Production decoder_dmodel=6144 bands.
      {"perf_prod_h6144_t2048", 2048, 80, 16, 6144, 512, kMemoryBoundTargetGBps, false},
      {"perf_prod_h6144_t18432", 18432, 80, 16, 6144, 512, kMemoryBoundTargetGBps, false},
      {"perf_prod_h6144_t36864", 36864, 80, 16, 6144, 512, kMemoryBoundTargetGBps, false},
      // Config-defaults decoder_dmodel=1536 matching decode/prefill bands, so
      // both shipped hidden sizes have perf coverage. TP is not sharded on the
      // audio encoder, so these apply identically to TP=1/2/4/8.
      {"perf_cfg_h1536_t2048", 2048, 80, 16, 1536, 512, kMemoryBoundTargetGBps, false},
      {"perf_cfg_h1536_t18432", 18432, 80, 16, 1536, 512, kMemoryBoundTargetGBps, false},
      {"perf_cfg_h1536_t36864", 36864, 80, 16, 1536, 512, kMemoryBoundTargetGBps, false},
      // Larger mel_vocab_size bands (kept at prod hidden).
      // These intentionally defeat cache-hot table reuse. They are kept as a
      // random-row DRAM stress floor, while production/cache-reuse shapes gate
      // against the AGENTS.md 350 GB/s memory-bound target above.
      {"perf_large_vocab_t4096", 4096, 80, 256, 6144, 512, kLargeVocabTargetGBps, false},
      {"perf_large_vocab_t8192", 8192, 80, 256, 6144, 512, kLargeVocabTargetGBps, false},
  };
}

inline std::vector<CaseConfig> make_suite(std::string const& suite) {
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

inline int choose_channels_per_item(CaseConfig const& cfg, Options const& options) {
  if (options.channels_per_item != 0) {
    return options.channels_per_item;
  }
  if (cfg.hidden >= 1536 && cfg.n_mel_bins >= 64) {
    return 8;
  }
  if (cfg.hidden >= 4096 && (cfg.tokens >= 8192 || cfg.mel_vocab_size >= 128)) {
    return 4;
  }
  return cfg.hidden >= 2048 ? 2 : 1;
}

template <typename Element, int ChannelsPerItem>
class MelEmbeddingSumKernel;

template <typename Element, int ChannelsPerItem>
sycl::event launch_chunk_kernel(sycl::queue& queue, KernelParams<Element> params) {
  constexpr int kChannelsPerGroup = kChannelBlock * ChannelsPerItem;
  int channel_tiles = ceil_div(params.hidden, kChannelsPerGroup);
  int global_channels = channel_tiles * kChannelBlock;
  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_features(
        sycl::range<1>(static_cast<std::size_t>(params.n_mel_bins)), cgh);
    cgh.parallel_for<MelEmbeddingSumKernel<Element, ChannelsPerItem>>(
        sycl::nd_range<2>(
            sycl::range<2>(
                static_cast<std::size_t>(params.chunk_tokens),
                static_cast<std::size_t>(global_channels)),
            sycl::range<2>(1, kChannelBlock)),
        [=](sycl::nd_item<2> item) {
          int token_in_chunk = static_cast<int>(item.get_group(0));
          int channel_tile = static_cast<int>(item.get_group(1));
          int lane = static_cast<int>(item.get_local_id(1));
          int token = params.token_offset + token_in_chunk;

          for (int mel = lane; mel < params.n_mel_bins; mel += kChannelBlock) {
            local_features[mel] = params.features[token * params.n_mel_bins + mel];
          }
          item.barrier(sycl::access::fence_space::local_space);

          int channel_base = channel_tile * kChannelsPerGroup + lane * ChannelsPerItem;
          float accum[ChannelsPerItem];
#pragma unroll
          for (int i = 0; i < ChannelsPerItem; ++i) {
            accum[i] = 0.0f;
          }

          for (int mel = 0; mel < params.n_mel_bins; ++mel) {
            int feature = local_features[mel];
            int64_t row = static_cast<int64_t>(mel) * params.mel_vocab_size + feature;
            Element const* weight_row = params.weight + row * params.hidden;
            if constexpr (ChannelsPerItem == 8 && !std::is_same_v<Element, float>) {
              if ((params.hidden % 8) == 0 && channel_base + ChannelsPerItem - 1 < params.hidden) {
                load_add_vec8(weight_row + channel_base, accum);
                continue;
              }
            }
#pragma unroll
            for (int i = 0; i < ChannelsPerItem; ++i) {
              int channel = channel_base + i;
              if (channel < params.hidden) {
                accum[i] += to_float(weight_row[channel]);
              }
            }
          }

          Element* out_row = params.out + static_cast<int64_t>(token) * params.hidden;
          if constexpr (ChannelsPerItem == 8 && !std::is_same_v<Element, float>) {
            if ((params.hidden % 8) == 0 && channel_base + ChannelsPerItem - 1 < params.hidden) {
              store_vec8(out_row + channel_base, accum);
              return;
            }
          }
#pragma unroll
          for (int i = 0; i < ChannelsPerItem; ++i) {
            int channel = channel_base + i;
            if (channel < params.hidden) {
              out_row[channel] = from_float<Element>(accum[i]);
            }
          }
        });
  });
}

template <typename Element>
std::vector<sycl::event> launch_mel_embedding_sum(
    sycl::queue& queue,
    KernelParams<Element> base_params,
    int chunk_size,
    int channels_per_item) {
  std::vector<sycl::event> events;
  events.reserve(static_cast<std::size_t>(ceil_div(base_params.tokens, chunk_size)));
  for (int start = 0; start < base_params.tokens; start += chunk_size) {
    KernelParams<Element> params = base_params;
    params.token_offset = start;
    params.chunk_tokens = std::min(chunk_size, base_params.tokens - start);
    if (channels_per_item == 8) {
      events.push_back(launch_chunk_kernel<Element, 8>(queue, params));
    } else if (channels_per_item == 4) {
      events.push_back(launch_chunk_kernel<Element, 4>(queue, params));
    } else if (channels_per_item == 2) {
      events.push_back(launch_chunk_kernel<Element, 2>(queue, params));
    } else {
      events.push_back(launch_chunk_kernel<Element, 1>(queue, params));
    }
  }
  return events;
}

inline std::vector<int32_t> make_features(CaseConfig const& cfg, uint32_t seed) {
  std::vector<int32_t> features(static_cast<std::size_t>(cfg.tokens) * cfg.n_mel_bins);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int32_t> dist(0, cfg.mel_vocab_size - 1);
  for (int token = 0; token < cfg.tokens; ++token) {
    for (int mel = 0; mel < cfg.n_mel_bins; ++mel) {
      int32_t value = dist(gen);
      if (token == 0) {
        value = static_cast<int32_t>(mel % cfg.mel_vocab_size);
      }
      if (token == cfg.tokens - 1) {
        value = static_cast<int32_t>(cfg.mel_vocab_size - 1 - (mel % cfg.mel_vocab_size));
      }
      features[static_cast<std::size_t>(token) * cfg.n_mel_bins + mel] = value;
    }
  }
  return features;
}

template <typename Element>
std::vector<Element> make_weight(CaseConfig const& cfg, uint32_t seed) {
  std::size_t rows = static_cast<std::size_t>(cfg.n_mel_bins) * cfg.mel_vocab_size;
  std::vector<Element> weight(rows * static_cast<std::size_t>(cfg.hidden));
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (int channel = 0; channel < cfg.hidden; ++channel) {
      float pattern = (static_cast<int>((row * 131 + channel * 17) % 29) - 14) * 0.001f;
      float value = dist(gen) + pattern;
      weight[row * static_cast<std::size_t>(cfg.hidden) + channel] = from_float<Element>(value);
    }
  }
  return weight;
}

template <typename Element>
std::vector<Element> reference_mel_embedding_sum(
    CaseConfig const& cfg,
    std::vector<int32_t> const& features,
    std::vector<Element> const& weight) {
  std::vector<Element> out(static_cast<std::size_t>(cfg.tokens) * cfg.hidden);
  for (int start = 0; start < cfg.tokens; start += cfg.chunk_size) {
    int end = std::min(start + cfg.chunk_size, cfg.tokens);
    for (int token = start; token < end; ++token) {
      for (int channel = 0; channel < cfg.hidden; ++channel) {
        float accum = 0.0f;
        for (int mel = 0; mel < cfg.n_mel_bins; ++mel) {
          int32_t feature = features[static_cast<std::size_t>(token) * cfg.n_mel_bins + mel];
          std::size_t row = static_cast<std::size_t>(mel) * cfg.mel_vocab_size + feature;
          accum += to_float(weight[row * static_cast<std::size_t>(cfg.hidden) + channel]);
        }
        out[static_cast<std::size_t>(token) * cfg.hidden + channel] = from_float<Element>(accum);
      }
    }
  }
  return out;
}

template <typename Element>
VerifyResult compare_outputs(
    std::vector<Element> const& got,
    std::vector<Element> const& expected) {
  if (got.size() != expected.size()) {
    throw std::invalid_argument("compare_outputs size mismatch");
  }
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
      result.got_bits = host_bits(got[i]);
      result.expected_bits = host_bits(expected[i]);
    }

    bool ok;
    if constexpr (std::is_same_v<Element, float>) {
      ok = abs_err <= 1.0e-5 + 1.0e-5 * std::abs(e);
    } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
      int ulps = std::abs(ordered_raw16(host_bits(got[i])) - ordered_raw16(host_bits(expected[i])));
      result.max_ulps = std::max(result.max_ulps, ulps);
      ok = ulps <= 1;
    } else {
      ok = host_bits(got[i]) == host_bits(expected[i]);
    }
    if (!ok && result.passed) {
      result.passed = false;
      result.index = i;
      result.got_bits = host_bits(got[i]);
      result.expected_bits = host_bits(expected[i]);
    }
  }
  return result;
}

inline std::size_t estimated_traffic_bytes(
    CaseConfig const& cfg,
    int channels_per_item,
    std::size_t element_bytes) {
  int channel_tiles = ceil_div(cfg.hidden, kChannelBlock * channels_per_item);
  std::size_t tokens = static_cast<std::size_t>(cfg.tokens);
  std::size_t hidden = static_cast<std::size_t>(cfg.hidden);
  std::size_t bins = static_cast<std::size_t>(cfg.n_mel_bins);
  std::size_t weight_bytes = tokens * hidden * bins * element_bytes;
  std::size_t output_bytes = tokens * hidden * element_bytes;
  std::size_t feature_bytes = tokens * bins * sizeof(int32_t) * static_cast<std::size_t>(channel_tiles);
  return weight_bytes + output_bytes + feature_bytes;
}

template <typename Element>
bool run_case_for_dtype(sycl::queue& queue, CaseConfig cfg, Options const& options) {
  validate_case(cfg);
  int channels_per_item = choose_channels_per_item(cfg, options);
  std::size_t feature_count = static_cast<std::size_t>(cfg.tokens) * cfg.n_mel_bins;
  std::size_t weight_count =
      static_cast<std::size_t>(cfg.n_mel_bins) * cfg.mel_vocab_size * cfg.hidden;
  std::size_t output_count = static_cast<std::size_t>(cfg.tokens) * cfg.hidden;

  std::vector<int32_t> host_features = make_features(cfg, 17);
  std::vector<Element> host_weight = make_weight<Element>(cfg, 29);
  std::vector<Element> host_out(output_count);

  DeviceBuffer<int32_t> d_features(queue, feature_count);
  DeviceBuffer<Element> d_weight(queue, weight_count);
  DeviceBuffer<Element> d_out(queue, output_count);
  d_features.copy_from(host_features);
  d_weight.copy_from(host_weight);

  KernelParams<Element> params;
  params.features = d_features.get();
  params.weight = d_weight.get();
  params.out = d_out.get();
  params.tokens = cfg.tokens;
  params.n_mel_bins = cfg.n_mel_bins;
  params.mel_vocab_size = cfg.mel_vocab_size;
  params.hidden = cfg.hidden;

  bool passed = true;
  std::cout << "  [" << element_dtype_text<Element>() << "] " << cfg.name
            << " tokens=" << cfg.tokens
            << " bins=" << cfg.n_mel_bins
            << " vocab=" << cfg.mel_vocab_size
            << " hidden=" << cfg.hidden
            << " chunk=" << cfg.chunk_size
            << " cpi=" << channels_per_item << "\n";

  if (options.verify && cfg.allow_verify) {
    auto events = launch_mel_embedding_sum(queue, params, cfg.chunk_size, channels_per_item);
    events.back().wait();
    d_out.copy_to(host_out);
    std::vector<Element> expected = reference_mel_embedding_sum(cfg, host_features, host_weight);
    VerifyResult result = compare_outputs(host_out, expected);
    if (!result.passed) {
      std::cerr << "    verify=FAIL index=" << result.index
                << " got_bits=0x" << std::hex << result.got_bits
                << " expected_bits=0x" << result.expected_bits << std::dec
                << " max_abs=" << result.max_abs
                << " max_rel=" << result.max_rel
                << " max_ulps=" << result.max_ulps << "\n";
      passed = false;
    } else {
      std::cout << "    verify=PASS max_abs=" << result.max_abs
                << " max_rel=" << result.max_rel
                << " max_ulps=" << result.max_ulps << "\n";
    }
  } else if (options.verify) {
    std::cout << "    verify=SKIP large perf case; use quick/inkling for full CPU reference\n";
  }

  if (options.benchmark) {
    for (int i = 0; i < options.warmup; ++i) {
      auto events = launch_mel_embedding_sum(queue, params, cfg.chunk_size, channels_per_item);
      events.back().wait();
    }

    double total_ms = 0.0;
    int timing_iterations = std::max(options.iterations, 1);
    for (int i = 0; i < timing_iterations; ++i) {
      auto events = launch_mel_embedding_sum(queue, params, cfg.chunk_size, channels_per_item);
      events.back().wait();
      for (sycl::event const& event : events) {
        total_ms += event_ms(event);
      }
    }
    double avg_ms = total_ms / static_cast<double>(timing_iterations);
    std::size_t bytes = estimated_traffic_bytes(cfg, channels_per_item, sizeof(Element));
    double gbps = (static_cast<double>(bytes) / kBytesPerGB) / (avg_ms * 1.0e-3);
    double target = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
    std::cout << std::fixed << std::setprecision(3)
              << "    avg_ms=" << avg_ms
              << " estimated_GB=" << (static_cast<double>(bytes) / kBytesPerGB)
              << " effective_GBps=" << gbps;
    if (target > 0.0) {
      std::cout << " target_GBps=" << target;
    }
    std::cout << std::defaultfloat << "\n";
    if (target > 0.0 && gbps < target) {
      std::cerr << "    perf=FAIL target_GBps=" << target << "\n";
      passed = false;
    }
  }

  return passed;
}

template <typename Element>
bool run_cases_for_dtype(
    sycl::queue& queue,
    std::vector<CaseConfig> const& cases,
    Options const& options) {
  bool all_passed = true;
  for (CaseConfig cfg : cases) {
    all_passed &= run_case_for_dtype<Element>(queue, std::move(cfg), options);
  }
  return all_passed;
}

}  // namespace cutlass::examples::bmg_mel_embedding_sum

int main(int argc, char const** argv) {
  namespace mel = cutlass::examples::bmg_mel_embedding_sum;

  mel::Options options;
  try {
    options = mel::parse_options(argc, argv);
    if (options.help) {
      mel::print_usage(argv[0]);
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<mel::CaseConfig> cases;
  if (!options.shape.empty()) {
    mel::CaseConfig cfg;
    cfg.name = "custom";
    if (!mel::parse_shape(options.shape, cfg)) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    cases.push_back(cfg);
  } else {
    cases = mel::make_suite(options.suite);
    if (cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = mel::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "23_bmg_mel_embedding_sum: out[token, hidden] = sum_m weight[mel*vocab + feature[token,mel], hidden]\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << mel::dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << mel::bool_text(options.verify)
              << " benchmark=" << mel::bool_text(options.benchmark)
              << " channels_per_item=" << options.channels_per_item << "\n";

    bool all_passed = true;
    if (options.dtype == mel::DType::kAll || options.dtype == mel::DType::kBf16) {
      all_passed &= mel::run_cases_for_dtype<cutlass::bfloat16_t>(queue, cases, options);
    }
    if (options.dtype == mel::DType::kAll || options.dtype == mel::DType::kFp16) {
      all_passed &= mel::run_cases_for_dtype<cutlass::half_t>(queue, cases, options);
    }
    if (options.dtype == mel::DType::kFloat) {
      all_passed &= mel::run_cases_for_dtype<float>(queue, cases, options);
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
