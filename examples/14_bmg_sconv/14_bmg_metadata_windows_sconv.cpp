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

/*! \file
    \brief Inkling SConv metadata, prefix-cache track indices, and intermediate windows.

    Semantics match modeltune/inkling/01_sconv/01_05_metadata_and_windows and the
    upstream SGLang kernels:

      decode metadata:
        query_start_loc = cu = arange(B + 1)
        has_initial_state = true
        cache_mask[b] = cache_indices[b] != -1
        safe_idx[b] = max(cache_indices[b], 0)
        si[t] = t

      extend metadata:
        query_start_loc = cu = exclusive prefix sum of extend_seq_lens, or
        b * draft_token_num in target-verify mode. has_initial_state is selected
        by HIS_{ZEROS,PREFIX,SEQ_MINUS_EXT,ONES}. si follows searchsorted(cu, t,
        right) - 1 and clamps tail tokens to B - 1 when cu[B] < T.

      prefix-cache track indices:
        indices[b, w] = clamp(query_start_loc[b] +
                              floor((mamba_track_seqlens[b] - extend_prefix_lens[b])
                                    / chunk_size) * chunk_size -
                              (W - 1) + w,
                              0, query_start_loc[B] - 1)

      intermediate windows:
        virtual = concat(cache[cache_indices[b]], hidden[b, :])
        out[b, t, w, :] = virtual[t + 1 + w, :]

    Roofline: the metadata and track kernels move only a few bytes per request
    and have low arithmetic intensity, so the optimization target is launch
    reduction and one-pass generation. save_intermediate_conv_windows is a
    memory-bound copy/select kernel; small and irregular rows use 16-byte
    vector copies with scalar lanes for non-divisible channel tails, while
    sustained W - 1 = 3 aligned rows use 256-byte subgroup block copies
    (sycl::ext::oneapi::experimental::group_load / group_store, which lower to
    the SPIR-V SubgroupBlockRead/Write intrinsics on Xe).
*/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cutlass::examples::sconv {

constexpr int kPadSlot = -1;
constexpr int kThreads = 512;
constexpr int kMetaThreads = 64;
constexpr int kCopyBytes = 16;
constexpr int kCopyWords = kCopyBytes / static_cast<int>(sizeof(uint32_t));
constexpr int kPacksPerLane = 1;
// Wide row path: one subgroup block copy moves kWideSubGroupSize *
// kWideWordsPerLane dwords, i.e. 256 B, which is the widest LSC block message
// the part offers (d32x64t). The plain-SYCL group_load/group_store pair lowers
// to exactly that message, so this matches the byte-per-message granularity of
// a hand-written block copy without leaving standard SYCL.
constexpr int kWideSubGroupSize = 16;
constexpr int kWideWordsPerLane = 4;
constexpr int kWideCopyWords = kWideSubGroupSize * kWideWordsPerLane;
constexpr int kWideMinBatch = 512;
constexpr int kWideMaxGroupItems = 512;
// The wide path passes alignment<16> to group_store, which makes the runtime
// alignment check compile away, and it relies on the per-lane vector path's
// kCopyBytes gate to have already proven that much alignment. Keep the two
// tied together: a narrower kCopyBytes would certify less than the block store
// requires and corrupt exactly the large-batch shapes.
static_assert(kCopyBytes >= static_cast<int>(sizeof(uint32_t)) * kWideWordsPerLane,
              "wide row path needs the alignment guaranteed by kCopyBytes");
constexpr int kMaxWindow = 16;
constexpr double kMinSustainedWindowBytes = 32.0 * 1024.0 * 1024.0;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

enum class Op {
  kAll,
  kDecode,
  kExtend,
  kTrack,
  kWindows
};

enum HisMode {
  kHisZeros = 0,
  kHisPrefix = 1,
  kHisSeqMinusExt = 2,
  kHisOnes = 3
};

std::string bool_text(bool value) {
  return value ? "true" : "false";
}

std::string dtype_text(DType dtype) {
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

bool parse_dtype(std::string const& text, DType& dtype) {
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

std::string op_text(Op op) {
  switch (op) {
    case Op::kAll:
      return "all";
    case Op::kDecode:
      return "decode";
    case Op::kExtend:
      return "extend";
    case Op::kTrack:
      return "track";
    case Op::kWindows:
      return "windows";
  }
  return "unknown";
}

bool parse_op(std::string const& text, Op& op) {
  if (text == "all") {
    op = Op::kAll;
    return true;
  }
  if (text == "decode") {
    op = Op::kDecode;
    return true;
  }
  if (text == "extend") {
    op = Op::kExtend;
    return true;
  }
  if (text == "track") {
    op = Op::kTrack;
    return true;
  }
  if (text == "windows") {
    op = Op::kWindows;
    return true;
  }
  return false;
}

char const* his_mode_text(int mode) {
  switch (mode) {
    case kHisZeros:
      return "zeros";
    case kHisPrefix:
      return "prefix";
    case kHisSeqMinusExt:
      return "seq-ext";
    case kHisOnes:
      return "ones";
  }
  return "unknown";
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

template <typename LaunchFn>
double time_kernel_seconds(LaunchFn&& launch, int iterations) {
  int timing_iterations = std::max(1, iterations);
  std::vector<sycl::event> events;
  events.reserve(timing_iterations);
  for (int i = 0; i < timing_iterations; ++i) {
    events.push_back(launch());
  }
  double total_ns = 0.0;
  for (auto& event : events) {
    event.wait_and_throw();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_ns += static_cast<double>(end - start);
  }
  return total_ns * 1.0e-9 / static_cast<double>(events.size());
}

struct DecodeMetadataParams {
  int32_t const* __restrict__ cache_indices;
  int32_t* __restrict__ query_start_loc;
  uint8_t* __restrict__ has_initial_state;
  uint8_t* __restrict__ cache_mask;
  int64_t* __restrict__ safe_idx;
  int64_t* __restrict__ cu;
  int32_t* __restrict__ si;
  int batch;
  int pad_slot_id;
};

class DecodeMetadataKernel {
 public:
  DecodeMetadataParams params;

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int off = static_cast<int>(item.get_global_linear_id());
    if (off > params.batch) {
      return;
    }
    params.query_start_loc[off] = off;
    params.cu[off] = static_cast<int64_t>(off);
    if (off < params.batch) {
      int32_t ci = params.cache_indices[off];
      params.has_initial_state[off] = 1;
      params.cache_mask[off] = static_cast<uint8_t>(ci != params.pad_slot_id);
      params.safe_idx[off] = static_cast<int64_t>(ci < 0 ? 0 : ci);
      params.si[off] = off;
    }
  }
};

sycl::event launch_decode_metadata(sycl::queue& q, DecodeMetadataParams const& params) {
  int total = params.batch + 1;
  int global = ((total + kMetaThreads - 1) / kMetaThreads) * kMetaThreads;
  DecodeMetadataKernel kernel{params};
  return q.parallel_for<DecodeMetadataKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kMetaThreads)), kernel);
}

struct ExtendMetadataParams {
  int32_t const* __restrict__ cache_indices;
  int32_t const* __restrict__ extend_seq_lens;
  int32_t const* __restrict__ his_src;
  int32_t* __restrict__ query_start_loc;
  uint8_t* __restrict__ has_initial_state;
  uint8_t* __restrict__ cache_mask;
  int64_t* __restrict__ safe_idx;
  int64_t* __restrict__ cu;
  int32_t* __restrict__ si;
  int batch;
  int tokens;
  int draft_token_num;
  int pad_slot_id;
};

class EmptyExtendMetadataKernel {
 public:
  ExtendMetadataParams params;

  CUTLASS_DEVICE
  void operator()(sycl::item<1> item) const {
    if (item.get_id(0) == 0) {
      params.query_start_loc[0] = 0;
      params.cu[0] = 0;
    }
  }
};

template <int Mode>
class ExtendMetadataKernel {
 public:
  ExtendMetadataParams params;

  CUTLASS_DEVICE
  int64_t sequence_start(int group) const {
    if constexpr (Mode == kHisOnes) {
      return static_cast<int64_t>(group) * params.draft_token_num;
    } else {
      int64_t start = 0;
      for (int i = 0; i < group; ++i) {
        start += static_cast<int64_t>(params.extend_seq_lens[i]);
      }
      return start;
    }
  }

  CUTLASS_DEVICE
  int sequence_length(int group) const {
    if (group >= params.batch) {
      return 0;
    }
    if constexpr (Mode == kHisOnes) {
      return params.draft_token_num;
    } else {
      return params.extend_seq_lens[group];
    }
  }

  CUTLASS_DEVICE
  uint8_t has_initial_state(int group, int len) const {
    if constexpr (Mode == kHisZeros) {
      return 0;
    } else if constexpr (Mode == kHisPrefix) {
      return static_cast<uint8_t>(params.his_src[group] > 0);
    } else if constexpr (Mode == kHisSeqMinusExt) {
      return static_cast<uint8_t>((params.his_src[group] - len) > 0);
    } else {
      return 1;
    }
  }

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int group = static_cast<int>(item.get_group(0));
    int lane = static_cast<int>(item.get_local_id(0));

    int64_t start = sequence_start(std::min(group, params.batch));
    int len = sequence_length(group);
    int64_t end = group < params.batch ? start + static_cast<int64_t>(len)
                                       : static_cast<int64_t>(params.tokens);

    if (group < params.batch && lane == 0) {
      uint8_t his = has_initial_state(group, len);
      int32_t ci = params.cache_indices[group];
      params.query_start_loc[group] = static_cast<int32_t>(start);
      params.query_start_loc[group + 1] = static_cast<int32_t>(start + len);
      params.cu[group] = start;
      params.cu[group + 1] = start + len;
      params.has_initial_state[group] = his;
      params.cache_mask[group] = static_cast<uint8_t>(his != 0 && ci != params.pad_slot_id);
      params.safe_idx[group] = static_cast<int64_t>(ci < 0 ? 0 : ci);
    }

    if (params.batch <= 0) {
      return;
    }
    int seq = group < params.batch ? group : params.batch - 1;
    int64_t fill_begin = start;
    int64_t fill_end = end;
    if (fill_begin < 0) {
      fill_begin = 0;
    }
    if (fill_end > params.tokens) {
      fill_end = params.tokens;
    }
    for (int64_t t = fill_begin + lane; t < fill_end; t += kMetaThreads) {
      params.si[t] = seq;
    }
  }
};

template <int Mode>
sycl::event launch_extend_metadata_static(sycl::queue& q, ExtendMetadataParams const& params) {
  if (params.batch == 0) {
    EmptyExtendMetadataKernel kernel{params};
    return q.parallel_for<EmptyExtendMetadataKernel>(sycl::range<1>(1), kernel);
  }
  int groups = params.batch + 1;
  ExtendMetadataKernel<Mode> kernel{params};
  return q.parallel_for<ExtendMetadataKernel<Mode>>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(groups * kMetaThreads)),
          sycl::range<1>(kMetaThreads)),
      kernel);
}

sycl::event launch_extend_metadata(sycl::queue& q, ExtendMetadataParams const& params, int his_mode) {
  switch (his_mode) {
    case kHisZeros:
      return launch_extend_metadata_static<kHisZeros>(q, params);
    case kHisPrefix:
      return launch_extend_metadata_static<kHisPrefix>(q, params);
    case kHisSeqMinusExt:
      return launch_extend_metadata_static<kHisSeqMinusExt>(q, params);
    case kHisOnes:
      return launch_extend_metadata_static<kHisOnes>(q, params);
    default:
      throw std::invalid_argument("unsupported his_mode");
  }
}

struct TrackIndicesParams {
  int32_t const* __restrict__ query_start_loc;
  int32_t const* __restrict__ mamba_track_seqlens;
  int32_t const* __restrict__ extend_prefix_lens;
  int32_t* __restrict__ track_indices;
  int batch;
  int width_minus_one;
  int chunk_size;
  int total_tokens;
};

class TrackIndicesKernel {
 public:
  TrackIndicesParams params;

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int linear = static_cast<int>(item.get_global_linear_id());
    int total = params.batch * params.width_minus_one;
    if (linear >= total) {
      return;
    }
    int b = linear / params.width_minus_one;
    int w = linear - b * params.width_minus_one;
    int lens_to_track = params.mamba_track_seqlens[b] - params.extend_prefix_lens[b];
    if (lens_to_track < 0) {
      lens_to_track = 0;
    }
    int aligned = (lens_to_track / params.chunk_size) * params.chunk_size;
    int idx = params.query_start_loc[b] + aligned - params.width_minus_one + w;
    int max_idx = std::max(0, params.total_tokens - 1);
    idx = std::max(0, std::min(idx, max_idx));
    params.track_indices[linear] = idx;
  }
};

sycl::event launch_track_indices(sycl::queue& q, TrackIndicesParams const& params) {
  int total = params.batch * params.width_minus_one;
  if (total == 0) {
    return sycl::event{};
  }
  int global = ((total + kThreads - 1) / kThreads) * kThreads;
  TrackIndicesKernel kernel{params};
  return q.parallel_for<TrackIndicesKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename Element_>
struct WindowParams {
  using Element = Element_;

  Element const* __restrict__ cache;
  Element const* __restrict__ hidden;
  int32_t const* __restrict__ cache_indices;
  Element* __restrict__ out;
  int cache_stride_slot;
  int cache_stride_w;
  int hidden_stride_b;
  int hidden_stride_t;
  int out_stride_b;
  int out_stride_t;
  int out_stride_w;
  int batch;
  int draft_tokens;
  int width_minus_one;
  int channels;
  int pad_slot_id;
};

template <typename Element, int StaticWidthMinusOne>
class SaveWindowsKernel {
 public:
  WindowParams<Element> params;
  int lanes_per_row;
  int vec_count;
  int pack_elems;

  using pack_t = sycl::vec<uint32_t, kCopyWords>;

  CUTLASS_DEVICE
  void copy_pack(Element const* src_row_base, Element* dst_row_base, int pack_idx) const {
#pragma unroll
    for (int i = 0; i < kPacksPerLane; ++i) {
      pack_t value;
      value.load(pack_idx + i, reinterpret_cast<uint32_t const*>(src_row_base));
      value.store(pack_idx + i, reinterpret_cast<uint32_t*>(dst_row_base));
    }
  }

  CUTLASS_DEVICE
  void copy_scalar(Element const* src_row_base, Element* dst_row_base, int channel) const {
    dst_row_base[channel] = src_row_base[channel];
  }

  CUTLASS_DEVICE
  Element const* source_row_base(int b, int cache_slot, int position, int width) const {
    if (position < width) {
      return params.cache
          + static_cast<int64_t>(cache_slot) * params.cache_stride_slot
          + static_cast<int64_t>(position) * params.cache_stride_w;
    }
    int hidden_t = position - width;
    return params.hidden
        + static_cast<int64_t>(b) * params.hidden_stride_b
        + static_cast<int64_t>(hidden_t) * params.hidden_stride_t;
  }

  CUTLASS_DEVICE
  Element* destination_row_base(int b, int t, int w) const {
    return params.out
        + static_cast<int64_t>(b) * params.out_stride_b
        + static_cast<int64_t>(t) * params.out_stride_t
        + static_cast<int64_t>(w) * params.out_stride_w;
  }

  CUTLASS_DEVICE
  int lane_channel(int lane, bool is_vec_lane) const {
    int lane_elems = pack_elems * kPacksPerLane;
    return is_vec_lane ? lane * lane_elems : vec_count * lane_elems + (lane - vec_count);
  }

  CUTLASS_DEVICE
  void copy_row_lane(int b, int t, int w, int lane, int width) const {
    int cache_slot = params.cache_indices[b];
    if (cache_slot == params.pad_slot_id) {
      return;
    }

    bool is_vec_lane = lane < vec_count;
    int channel = lane_channel(lane, is_vec_lane);
    int position = t + 1 + w;
    Element const* src_row = source_row_base(b, cache_slot, position, width);
    Element* dst_row = destination_row_base(b, t, w);
    if (is_vec_lane) {
      copy_pack(src_row, dst_row, lane * kPacksPerLane);
    } else {
      copy_scalar(src_row, dst_row, channel);
    }
  }

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int width = StaticWidthMinusOne > 0 ? StaticWidthMinusOne : params.width_minus_one;
    int linear = static_cast<int>(item.get_global_linear_id());
    int rows_per_batch = params.draft_tokens * width;
    int total_lanes = params.batch * rows_per_batch * lanes_per_row;
    if (linear >= total_lanes) {
      return;
    }
    int row_lane = linear / lanes_per_row;
    int lane = linear - row_lane * lanes_per_row;
    int b = row_lane / rows_per_batch;
    int local_row = row_lane - b * rows_per_batch;
    int t = local_row / width;
    int w = local_row - t * width;
    copy_row_lane(b, t, w, lane, width);
  }
};

// Plain-SYCL wide row copy. The work decomposition is 3D on purpose:
// dim0 = request, dim1 = (t, w) row inside the request, dim2 = lane inside the
// row. Every index therefore arrives from the launch geometry, so the only
// divisions left are by compile-time constants (StaticWidthMinusOne and the
// subgroup size); the flat 1D mapping instead divided by a runtime
// lanes-per-row. dim2 is the fastest-varying dimension, so each subgroup
// covers one contiguous kWideCopyWords dword chunk of the row and the block
// load/store addresses are subgroup-uniform as group_load requires.
template <typename Element, int StaticWidthMinusOne>
class SaveWindowsWideRowKernel {
 public:
  WindowParams<Element> params;

  using lane_vec_t = sycl::vec<uint32_t, kWideWordsPerLane>;

  [[sycl::reqd_sub_group_size(kWideSubGroupSize)]]
  void operator()(sycl::nd_item<3> item) const {
    int width = StaticWidthMinusOne > 0 ? StaticWidthMinusOne : params.width_minus_one;
    int b = static_cast<int>(item.get_global_id(0));
    int local_row = static_cast<int>(item.get_global_id(1));
    int chunk = static_cast<int>(item.get_global_id(2)) / kWideSubGroupSize;
    int t = local_row / width;
    int w = local_row - t * width;

    int cache_slot = params.cache_indices[b];
    if (cache_slot == params.pad_slot_id) {
      return;
    }

    int position = t + 1 + w;
    Element const* src_row = nullptr;
    if (position < width) {
      src_row = params.cache
          + static_cast<int64_t>(cache_slot) * params.cache_stride_slot
          + static_cast<int64_t>(position) * params.cache_stride_w;
    } else {
      int hidden_t = position - width;
      src_row = params.hidden
          + static_cast<int64_t>(b) * params.hidden_stride_b
          + static_cast<int64_t>(hidden_t) * params.hidden_stride_t;
    }
    Element* dst_row = params.out
        + static_cast<int64_t>(b) * params.out_stride_b
        + static_cast<int64_t>(t) * params.out_stride_t
        + static_cast<int64_t>(w) * params.out_stride_w;

    // full_group + contiguous_memory + striped placement is the exact property
    // set that lowers group_load/group_store to SubgroupBlockRead/Write. The
    // striped placement matters: with the default blocked placement the
    // implementation's has_builtin test (sizeof(word) * words_per_lane <= 8)
    // fails for 4 dwords per lane and the copy silently degrades to a per-lane
    // d32x4 gather plus four d32 scatters, which measured 309 GB/s on
    // b1024/T9/W-1=3/D=512 versus the 570 GB/s the block message reaches.
    // Placement is irrelevant to a pure copy as long as load and store agree.
    auto props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::contiguous_memory,
        sycl::ext::oneapi::experimental::full_group,
        sycl::ext::oneapi::experimental::data_placement_striped,
        sycl::ext::oneapi::experimental::alignment<sizeof(lane_vec_t)>};
    // Decorated global pointers: from a plain raw pointer the deduced address
    // space is generic, so the implementation emits a runtime
    // dynamic_address_cast and keeps the whole per-lane gather/scatter fallback
    // alongside the block message (~600 vs 404 lines of generated ISA). Naming
    // the address space leaves only the two block messages. Throughput is
    // unchanged within noise; this is a code-size/clarity win.
    auto sg = item.get_sub_group();
    auto src_words = sycl::address_space_cast<
        sycl::access::address_space::global_space, sycl::access::decorated::yes>(
        reinterpret_cast<uint32_t const*>(src_row) + chunk * kWideCopyWords)
        .get_decorated();
    auto dst_words = sycl::address_space_cast<
        sycl::access::address_space::global_space, sycl::access::decorated::yes>(
        reinterpret_cast<uint32_t*>(dst_row) + chunk * kWideCopyWords)
        .get_decorated();
    lane_vec_t value;
    sycl::ext::oneapi::experimental::group_load(sg, src_words, value, props);
    sycl::ext::oneapi::experimental::group_store(sg, value, dst_words, props);
  }
};

// Largest divisor of extent that keeps stride * divisor within budget. The
// kernel carries no bounds check (a partially populated subgroup would violate
// group_load's full_group contract), so every work-group extent has to divide
// its global extent exactly.
inline int wide_largest_divisor(int extent, int stride, int budget) {
  int best = 1;
  for (int candidate = 1; candidate <= extent; ++candidate) {
    if (extent % candidate == 0 && candidate * stride <= budget) {
      best = candidate;
    }
  }
  return best;
}

template <typename Element, int StaticWidthMinusOne>
sycl::event launch_save_windows_wide_row(
    sycl::queue& q,
    WindowParams<Element> const& params,
    int chunks_per_row) {
  int width = StaticWidthMinusOne > 0 ? StaticWidthMinusOne : params.width_minus_one;
  int rows_per_batch = params.draft_tokens * width;
  int lanes_per_row = chunks_per_row * kWideSubGroupSize;
  // Fill the work-group along the lanes of one row first; when the chunk count
  // is odd (D = 384 gives 3, D = 640 gives 5) that leaves a single subgroup per
  // group, so stack whole rows on top of it instead of launching thousands of
  // 16-item groups.
  int group_lanes =
      wide_largest_divisor(chunks_per_row, kWideSubGroupSize, kWideMaxGroupItems) *
      kWideSubGroupSize;
  int group_rows =
      wide_largest_divisor(rows_per_batch, group_lanes, kWideMaxGroupItems);
  sycl::range<3> global(
      static_cast<std::size_t>(params.batch),
      static_cast<std::size_t>(rows_per_batch),
      static_cast<std::size_t>(lanes_per_row));
  sycl::range<3> local(
      1,
      static_cast<std::size_t>(group_rows),
      static_cast<std::size_t>(group_lanes));
  SaveWindowsWideRowKernel<Element, StaticWidthMinusOne> kernel{params};
  return q.parallel_for<SaveWindowsWideRowKernel<Element, StaticWidthMinusOne>>(
      sycl::nd_range<3>(global, local), kernel);
}

template <typename Element, int StaticWidthMinusOne>
sycl::event launch_save_windows_static(sycl::queue& q, WindowParams<Element> const& params) {
  if (params.batch == 0 || params.draft_tokens == 0 || params.width_minus_one == 0 || params.channels == 0) {
    return sycl::event{};
  }
  if (params.width_minus_one > kMaxWindow) {
    throw std::invalid_argument("window width_minus_one exceeds kMaxWindow");
  }

  int pack_elems = kCopyBytes / static_cast<int>(sizeof(Element));
  int lane_elems = pack_elems * kPacksPerLane;
  auto aligned_elems = [pack_elems](int stride) {
    return stride % pack_elems == 0;
  };
  bool aligned = aligned_elems(params.cache_stride_slot) &&
      aligned_elems(params.cache_stride_w) &&
      aligned_elems(params.hidden_stride_b) &&
      aligned_elems(params.hidden_stride_t) &&
      aligned_elems(params.out_stride_b) &&
      aligned_elems(params.out_stride_t) &&
      aligned_elems(params.out_stride_w) &&
      (reinterpret_cast<std::uintptr_t>(params.cache) % kCopyBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.hidden) % kCopyBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.out) % kCopyBytes == 0);

  int vec_count = aligned ? params.channels / lane_elems : 0;
  int scalar_tail = params.channels - vec_count * lane_elems;
  int lanes_per_row = vec_count + scalar_tail;
  int row_bytes = params.channels * static_cast<int>(sizeof(Element));
  int row_words = row_bytes / static_cast<int>(sizeof(uint32_t));
  int wide_chunks_per_row = row_words / kWideCopyWords;
  // 256-byte subgroup block copies amortize per-item index setup on sustained
  // large-B rows; small launch-limited shapes stay on the lighter row-driven
  // per-lane vector path, which is cache-resident there and already faster.
  // if constexpr keeps the wide kernel out of the binary for every window
  // width the dispatch can never select.
  if constexpr (StaticWidthMinusOne == 3) {
    if (params.batch >= kWideMinBatch &&
        params.width_minus_one == 3 &&
        aligned &&
        scalar_tail == 0 &&
        row_bytes % static_cast<int>(sizeof(uint32_t)) == 0 &&
        row_words % kWideCopyWords == 0) {
      return launch_save_windows_wide_row<Element, StaticWidthMinusOne>(q, params, wide_chunks_per_row);
    }
  }
  int rows_per_batch = params.draft_tokens * params.width_minus_one;
  int total_lanes = params.batch * rows_per_batch * lanes_per_row;
  if (total_lanes == 0) {
    return sycl::event{};
  }

  int global = ((total_lanes + kThreads - 1) / kThreads) * kThreads;
  SaveWindowsKernel<Element, StaticWidthMinusOne> kernel{
      params, lanes_per_row, vec_count, pack_elems};
  return q.parallel_for<SaveWindowsKernel<Element, StaticWidthMinusOne>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename Element>
sycl::event launch_save_windows(sycl::queue& q, WindowParams<Element> const& params) {
  switch (params.width_minus_one) {
    case 1:
      return launch_save_windows_static<Element, 1>(q, params);
    case 2:
      return launch_save_windows_static<Element, 2>(q, params);
    case 3:
      return launch_save_windows_static<Element, 3>(q, params);
    case 5:
      return launch_save_windows_static<Element, 5>(q, params);
    case 7:
      return launch_save_windows_static<Element, 7>(q, params);
    case 8:
      return launch_save_windows_static<Element, 8>(q, params);
    default:
      return launch_save_windows_static<Element, 0>(q, params);
  }
}

template <typename T>
bool verify_exact(std::vector<T> const& got, std::vector<T> const& ref, int& bad_index) {
  if (got.size() != ref.size()) {
    bad_index = -1;
    return false;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != ref[i]) {
      bad_index = static_cast<int>(i);
      return false;
    }
  }
  return true;
}

template <typename Element>
float to_float(Element value) {
  return static_cast<float>(value);
}

template <typename Element>
bool verify_element_exact(std::vector<Element> const& got, std::vector<Element> const& ref, int& bad_index) {
  if (got.size() != ref.size()) {
    bad_index = -1;
    return false;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (to_float(got[i]) != to_float(ref[i])) {
      bad_index = static_cast<int>(i);
      return false;
    }
  }
  return true;
}

struct DecodeCase {
  std::string name;
  int batch = 1;
  bool include_pad = false;
};

struct ExtendCase {
  std::string name;
  int batch = 1;
  int qlen = 1;
  int tokens_extra = 0;
  int his_mode = kHisPrefix;
  int draft_token_num = 1;
  bool mixed_lengths = false;
  bool include_pad = false;
  bool include_zero = false;
  unsigned seed = 0;
};

struct TrackCase {
  std::string name;
  int batch = 1;
  int width_minus_one = 3;
  int qlen = 8;
  int chunk_size = 4;
  bool mixed_lengths = false;
};

struct WindowCase {
  std::string name;
  int batch = 1;
  int draft_tokens = 1;
  int width_minus_one = 3;
  int channels = 128;
  bool include_pad = false;
  int hidden_padding = 0;
  int cache_padding = 0;
  int out_padding = 0;
};

struct DecodeHost {
  std::vector<int32_t> cache_indices;
  std::vector<int32_t> query_start_loc;
  std::vector<int32_t> ref_query_start_loc;
  std::vector<uint8_t> has_initial_state;
  std::vector<uint8_t> ref_has_initial_state;
  std::vector<uint8_t> cache_mask;
  std::vector<uint8_t> ref_cache_mask;
  std::vector<int64_t> safe_idx;
  std::vector<int64_t> ref_safe_idx;
  std::vector<int64_t> cu;
  std::vector<int64_t> ref_cu;
  std::vector<int32_t> si;
  std::vector<int32_t> ref_si;
};

DecodeHost make_decode_host(DecodeCase const& cfg) {
  DecodeHost h;
  h.cache_indices.resize(cfg.batch);
  h.query_start_loc.assign(cfg.batch + 1, -77);
  h.ref_query_start_loc.resize(cfg.batch + 1);
  h.has_initial_state.assign(cfg.batch, 0);
  h.ref_has_initial_state.resize(cfg.batch);
  h.cache_mask.assign(cfg.batch, 0);
  h.ref_cache_mask.resize(cfg.batch);
  h.safe_idx.assign(cfg.batch, -77);
  h.ref_safe_idx.resize(cfg.batch);
  h.cu.assign(cfg.batch + 1, -77);
  h.ref_cu.resize(cfg.batch + 1);
  h.si.assign(cfg.batch, -77);
  h.ref_si.resize(cfg.batch);
  for (int b = 0; b < cfg.batch; ++b) {
    h.cache_indices[b] = cfg.include_pad && (b % 5 == 2) ? kPadSlot : b + 3;
  }
  for (int i = 0; i <= cfg.batch; ++i) {
    h.ref_query_start_loc[i] = i;
    h.ref_cu[i] = i;
  }
  for (int b = 0; b < cfg.batch; ++b) {
    int32_t ci = h.cache_indices[b];
    h.ref_has_initial_state[b] = 1;
    h.ref_cache_mask[b] = static_cast<uint8_t>(ci != kPadSlot);
    h.ref_safe_idx[b] = ci < 0 ? 0 : ci;
    h.ref_si[b] = b;
  }
  return h;
}

std::vector<int32_t> make_lengths(ExtendCase const& cfg) {
  std::vector<int32_t> lens(cfg.batch, cfg.qlen);
  if (cfg.his_mode == kHisOnes) {
    std::fill(lens.begin(), lens.end(), cfg.draft_token_num);
    return lens;
  }
  if (cfg.mixed_lengths) {
    for (int b = 0; b < cfg.batch; ++b) {
      int pattern = b % 7;
      if (pattern == 0 && cfg.include_zero) {
        lens[b] = 0;
      } else if (pattern == 1) {
        lens[b] = 1;
      } else if (pattern == 2) {
        lens[b] = 2;
      } else if (pattern == 3) {
        lens[b] = std::max(1, cfg.qlen / 2);
      } else if (pattern == 4) {
        lens[b] = cfg.qlen + 3;
      } else {
        lens[b] = cfg.qlen;
      }
    }
  }
  if (cfg.seed != 0) {
    std::mt19937 gen(cfg.seed);
    std::uniform_int_distribution<int> dist(0, std::max(1, cfg.qlen + 5));
    for (int b = 0; b < cfg.batch; ++b) {
      lens[b] = dist(gen);
    }
  }
  return lens;
}

struct ExtendHost {
  std::vector<int32_t> cache_indices;
  std::vector<int32_t> extend_seq_lens;
  std::vector<int32_t> his_src;
  std::vector<int32_t> query_start_loc;
  std::vector<int32_t> ref_query_start_loc;
  std::vector<uint8_t> has_initial_state;
  std::vector<uint8_t> ref_has_initial_state;
  std::vector<uint8_t> cache_mask;
  std::vector<uint8_t> ref_cache_mask;
  std::vector<int64_t> safe_idx;
  std::vector<int64_t> ref_safe_idx;
  std::vector<int64_t> cu;
  std::vector<int64_t> ref_cu;
  std::vector<int32_t> si;
  std::vector<int32_t> ref_si;
  int tokens = 0;
};

ExtendHost make_extend_host(ExtendCase const& cfg) {
  ExtendHost h;
  h.cache_indices.resize(cfg.batch);
  h.extend_seq_lens = make_lengths(cfg);
  h.his_src.assign(cfg.batch, 0);
  h.ref_query_start_loc.resize(cfg.batch + 1);
  h.ref_cu.resize(cfg.batch + 1);
  h.ref_query_start_loc[0] = 0;
  h.ref_cu[0] = 0;
  for (int b = 0; b < cfg.batch; ++b) {
    int len = cfg.his_mode == kHisOnes ? cfg.draft_token_num : h.extend_seq_lens[b];
    int next = cfg.his_mode == kHisOnes ? (b + 1) * cfg.draft_token_num
                                        : h.ref_query_start_loc[b] + len;
    h.ref_query_start_loc[b + 1] = next;
    h.ref_cu[b + 1] = next;
  }
  h.tokens = h.ref_query_start_loc.back() + cfg.tokens_extra;
  h.query_start_loc.assign(cfg.batch + 1, -77);
  h.has_initial_state.assign(cfg.batch, 0);
  h.cache_mask.assign(cfg.batch, 0);
  h.safe_idx.assign(cfg.batch, -77);
  h.cu.assign(cfg.batch + 1, -77);
  h.si.assign(std::max(0, h.tokens), -77);
  h.ref_has_initial_state.resize(cfg.batch);
  h.ref_cache_mask.resize(cfg.batch);
  h.ref_safe_idx.resize(cfg.batch);
  h.ref_si.resize(std::max(0, h.tokens));
  for (int b = 0; b < cfg.batch; ++b) {
    h.cache_indices[b] = cfg.include_pad && (b % 6 == 3) ? kPadSlot : b + 11;
    int len = h.extend_seq_lens[b];
    if (cfg.his_mode == kHisPrefix) {
      h.his_src[b] = (b % 3 == 0) ? 0 : (b + 1);
    } else if (cfg.his_mode == kHisSeqMinusExt) {
      h.his_src[b] = len + ((b % 4 == 1) ? 0 : 5);
    } else {
      h.his_src[b] = 0;
    }
  }
  for (int b = 0; b < cfg.batch; ++b) {
    int len = cfg.his_mode == kHisOnes ? cfg.draft_token_num : h.extend_seq_lens[b];
    uint8_t his = 0;
    if (cfg.his_mode == kHisZeros) {
      his = 0;
    } else if (cfg.his_mode == kHisPrefix) {
      his = static_cast<uint8_t>(h.his_src[b] > 0);
    } else if (cfg.his_mode == kHisSeqMinusExt) {
      his = static_cast<uint8_t>((h.his_src[b] - len) > 0);
    } else {
      his = 1;
    }
    int32_t ci = h.cache_indices[b];
    h.ref_has_initial_state[b] = his;
    h.ref_cache_mask[b] = static_cast<uint8_t>(his != 0 && ci != kPadSlot);
    h.ref_safe_idx[b] = ci < 0 ? 0 : ci;
  }
  for (int t = 0; t < h.tokens; ++t) {
    int count = 0;
    for (int b = 1; b <= cfg.batch; ++b) {
      if (h.ref_cu[b] <= t) {
        ++count;
      }
    }
    h.ref_si[t] = std::min(count, cfg.batch - 1);
  }
  return h;
}

bool run_decode_case(sycl::queue& q, DecodeCase const& cfg, int iterations, bool verify) {
  DecodeHost h = make_decode_host(cfg);
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<int32_t> d_query_start_loc(q, h.query_start_loc.size());
  DeviceBuffer<uint8_t> d_has_initial_state(q, h.has_initial_state.size());
  DeviceBuffer<uint8_t> d_cache_mask(q, h.cache_mask.size());
  DeviceBuffer<int64_t> d_safe_idx(q, h.safe_idx.size());
  DeviceBuffer<int64_t> d_cu(q, h.cu.size());
  DeviceBuffer<int32_t> d_si(q, h.si.size());
  d_cache_indices.copy_from(h.cache_indices);
  d_query_start_loc.copy_from(h.query_start_loc);
  d_has_initial_state.copy_from(h.has_initial_state);
  d_cache_mask.copy_from(h.cache_mask);
  d_safe_idx.copy_from(h.safe_idx);
  d_cu.copy_from(h.cu);
  d_si.copy_from(h.si);

  DecodeMetadataParams params{
      d_cache_indices.get(),
      d_query_start_loc.get(),
      d_has_initial_state.get(),
      d_cache_mask.get(),
      d_safe_idx.get(),
      d_cu.get(),
      d_si.get(),
      cfg.batch,
      kPadSlot};

  auto launch = [&]() {
    return launch_decode_metadata(q, params);
  };
  launch().wait_and_throw();

  bool passed = true;
  int bad = -1;
  if (verify) {
    d_query_start_loc.copy_to(h.query_start_loc);
    d_has_initial_state.copy_to(h.has_initial_state);
    d_cache_mask.copy_to(h.cache_mask);
    d_safe_idx.copy_to(h.safe_idx);
    d_cu.copy_to(h.cu);
    d_si.copy_to(h.si);
    passed = verify_exact(h.query_start_loc, h.ref_query_start_loc, bad) &&
        verify_exact(h.has_initial_state, h.ref_has_initial_state, bad) &&
        verify_exact(h.cache_mask, h.ref_cache_mask, bad) &&
        verify_exact(h.safe_idx, h.ref_safe_idx, bad) &&
        verify_exact(h.cu, h.ref_cu, bad) &&
        verify_exact(h.si, h.ref_si, bad);
  }

  for (int i = 0; i < 3; ++i) {
    launch();
  }
  q.wait_and_throw();
  double avg_s = time_kernel_seconds(launch, iterations);
  std::cout << std::left << std::setw(34) << cfg.name
            << " op=decode"
            << " B=" << std::setw(5) << cfg.batch
            << " pad=" << bool_text(cfg.include_pad)
            << "  " << std::fixed << std::setprecision(4)
            << (avg_s * 1000.0) << " ms"
            << "  " << (verify ? (passed ? "passed" : "failed") : "verification skipped");
  if (!passed) {
    std::cout << " bad_index=" << bad;
  }
  std::cout << "\n";
  return passed;
}

bool run_extend_case(sycl::queue& q, ExtendCase const& cfg, int iterations, bool verify) {
  ExtendHost h = make_extend_host(cfg);
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<int32_t> d_extend_seq_lens(q, h.extend_seq_lens.size());
  DeviceBuffer<int32_t> d_his_src(q, h.his_src.size());
  DeviceBuffer<int32_t> d_query_start_loc(q, h.query_start_loc.size());
  DeviceBuffer<uint8_t> d_has_initial_state(q, h.has_initial_state.size());
  DeviceBuffer<uint8_t> d_cache_mask(q, h.cache_mask.size());
  DeviceBuffer<int64_t> d_safe_idx(q, h.safe_idx.size());
  DeviceBuffer<int64_t> d_cu(q, h.cu.size());
  DeviceBuffer<int32_t> d_si(q, h.si.size());
  d_cache_indices.copy_from(h.cache_indices);
  d_extend_seq_lens.copy_from(h.extend_seq_lens);
  d_his_src.copy_from(h.his_src);
  d_query_start_loc.copy_from(h.query_start_loc);
  d_has_initial_state.copy_from(h.has_initial_state);
  d_cache_mask.copy_from(h.cache_mask);
  d_safe_idx.copy_from(h.safe_idx);
  d_cu.copy_from(h.cu);
  d_si.copy_from(h.si);

  ExtendMetadataParams params{
      d_cache_indices.get(),
      d_extend_seq_lens.get(),
      d_his_src.get(),
      d_query_start_loc.get(),
      d_has_initial_state.get(),
      d_cache_mask.get(),
      d_safe_idx.get(),
      d_cu.get(),
      d_si.get(),
      cfg.batch,
      h.tokens,
      cfg.draft_token_num,
      kPadSlot};

  auto launch = [&]() {
    return launch_extend_metadata(q, params, cfg.his_mode);
  };
  launch().wait_and_throw();

  bool passed = true;
  int bad = -1;
  if (verify) {
    d_query_start_loc.copy_to(h.query_start_loc);
    d_has_initial_state.copy_to(h.has_initial_state);
    d_cache_mask.copy_to(h.cache_mask);
    d_safe_idx.copy_to(h.safe_idx);
    d_cu.copy_to(h.cu);
    d_si.copy_to(h.si);
    passed = verify_exact(h.query_start_loc, h.ref_query_start_loc, bad) &&
        verify_exact(h.has_initial_state, h.ref_has_initial_state, bad) &&
        verify_exact(h.cache_mask, h.ref_cache_mask, bad) &&
        verify_exact(h.safe_idx, h.ref_safe_idx, bad) &&
        verify_exact(h.cu, h.ref_cu, bad) &&
        verify_exact(h.si, h.ref_si, bad);
  }

  for (int i = 0; i < 3; ++i) {
    launch();
  }
  q.wait_and_throw();
  double avg_s = time_kernel_seconds(launch, iterations);
  std::cout << std::left << std::setw(34) << cfg.name
            << " op=extend"
            << " B=" << std::setw(5) << cfg.batch
            << " T=" << std::setw(7) << h.tokens
            << " mode=" << std::setw(8) << his_mode_text(cfg.his_mode)
            << "  " << std::fixed << std::setprecision(4)
            << (avg_s * 1000.0) << " ms"
            << "  " << (verify ? (passed ? "passed" : "failed") : "verification skipped");
  if (!passed) {
    std::cout << " bad_index=" << bad;
  }
  std::cout << "\n";
  return passed;
}

struct TrackHost {
  std::vector<int32_t> query_start_loc;
  std::vector<int32_t> mamba_track_seqlens;
  std::vector<int32_t> extend_prefix_lens;
  std::vector<int32_t> track_indices;
  std::vector<int32_t> ref_track_indices;
  int total_tokens = 0;
};

TrackHost make_track_host(TrackCase const& cfg) {
  TrackHost h;
  h.query_start_loc.resize(cfg.batch + 1);
  h.mamba_track_seqlens.resize(cfg.batch);
  h.extend_prefix_lens.resize(cfg.batch);
  h.track_indices.assign(cfg.batch * cfg.width_minus_one, -77);
  h.ref_track_indices.resize(cfg.batch * cfg.width_minus_one);
  h.query_start_loc[0] = 0;
  for (int b = 0; b < cfg.batch; ++b) {
    int len = cfg.mixed_lengths ? (cfg.qlen + (b % 5) - 2) : cfg.qlen;
    len = std::max(0, len);
    h.query_start_loc[b + 1] = h.query_start_loc[b] + len;
    h.extend_prefix_lens[b] = b % 4;
    h.mamba_track_seqlens[b] = h.extend_prefix_lens[b] + len + (b % 3) * cfg.chunk_size;
  }
  h.total_tokens = h.query_start_loc.back();
  for (int b = 0; b < cfg.batch; ++b) {
    int lens_to_track = std::max(0, h.mamba_track_seqlens[b] - h.extend_prefix_lens[b]);
    int aligned = (lens_to_track / cfg.chunk_size) * cfg.chunk_size;
    int start = h.query_start_loc[b] + aligned - cfg.width_minus_one;
    int max_idx = std::max(0, h.total_tokens - 1);
    for (int w = 0; w < cfg.width_minus_one; ++w) {
      int idx = std::max(0, std::min(start + w, max_idx));
      h.ref_track_indices[b * cfg.width_minus_one + w] = idx;
    }
  }
  return h;
}

bool run_track_case(sycl::queue& q, TrackCase const& cfg, int iterations, bool verify) {
  TrackHost h = make_track_host(cfg);
  DeviceBuffer<int32_t> d_query_start_loc(q, h.query_start_loc.size());
  DeviceBuffer<int32_t> d_mamba_track_seqlens(q, h.mamba_track_seqlens.size());
  DeviceBuffer<int32_t> d_extend_prefix_lens(q, h.extend_prefix_lens.size());
  DeviceBuffer<int32_t> d_track_indices(q, h.track_indices.size());
  d_query_start_loc.copy_from(h.query_start_loc);
  d_mamba_track_seqlens.copy_from(h.mamba_track_seqlens);
  d_extend_prefix_lens.copy_from(h.extend_prefix_lens);
  d_track_indices.copy_from(h.track_indices);

  TrackIndicesParams params{
      d_query_start_loc.get(),
      d_mamba_track_seqlens.get(),
      d_extend_prefix_lens.get(),
      d_track_indices.get(),
      cfg.batch,
      cfg.width_minus_one,
      cfg.chunk_size,
      h.total_tokens};
  auto launch = [&]() {
    return launch_track_indices(q, params);
  };
  launch().wait_and_throw();

  bool passed = true;
  int bad = -1;
  if (verify) {
    d_track_indices.copy_to(h.track_indices);
    passed = verify_exact(h.track_indices, h.ref_track_indices, bad);
  }

  for (int i = 0; i < 3; ++i) {
    launch();
  }
  q.wait_and_throw();
  double avg_s = time_kernel_seconds(launch, iterations);
  std::cout << std::left << std::setw(34) << cfg.name
            << " op=track "
            << " B=" << std::setw(5) << cfg.batch
            << " W-1=" << std::setw(3) << cfg.width_minus_one
            << " chunk=" << std::setw(5) << cfg.chunk_size
            << "  " << std::fixed << std::setprecision(4)
            << (avg_s * 1000.0) << " ms"
            << "  " << (verify ? (passed ? "passed" : "failed") : "verification skipped");
  if (!passed) {
    std::cout << " bad_index=" << bad;
  }
  std::cout << "\n";
  return passed;
}

template <typename Element_>
struct WindowHost {
  using Element = Element_;

  std::vector<Element> cache;
  std::vector<Element> hidden;
  std::vector<int32_t> cache_indices;
  std::vector<Element> out;
  std::vector<Element> ref_out;
  int slots = 0;
  int cache_stride_slot = 0;
  int cache_stride_w = 0;
  int hidden_stride_b = 0;
  int hidden_stride_t = 0;
  int out_stride_b = 0;
  int out_stride_t = 0;
  int out_stride_w = 0;
};

template <typename Element>
WindowHost<Element> make_window_host(WindowCase const& cfg) {
  WindowHost<Element> h;
  h.slots = std::max(8, cfg.batch + 4);
  h.cache_stride_w = cfg.channels + cfg.cache_padding;
  h.cache_stride_slot = cfg.width_minus_one * h.cache_stride_w;
  h.hidden_stride_t = cfg.channels + cfg.hidden_padding;
  h.hidden_stride_b = cfg.draft_tokens * h.hidden_stride_t;
  h.out_stride_w = cfg.channels + cfg.out_padding;
  h.out_stride_t = cfg.width_minus_one * h.out_stride_w;
  h.out_stride_b = cfg.draft_tokens * h.out_stride_t;
  h.cache.resize(static_cast<std::size_t>(h.slots) * h.cache_stride_slot);
  h.hidden.resize(static_cast<std::size_t>(cfg.batch) * h.hidden_stride_b);
  h.cache_indices.resize(cfg.batch);
  h.out.resize(static_cast<std::size_t>(cfg.batch) * h.out_stride_b);
  h.ref_out.resize(h.out.size());

  unsigned seed = 20260718u + static_cast<unsigned>(cfg.batch * 17 + cfg.channels + cfg.width_minus_one * 13);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h.cache) {
    v = Element(dist(gen) * 0.25f);
  }
  for (auto& v : h.hidden) {
    v = Element(dist(gen));
  }
  for (std::size_t i = 0; i < h.out.size(); ++i) {
    h.out[i] = Element(-7.0f);
    h.ref_out[i] = h.out[i];
  }
  for (int b = 0; b < cfg.batch; ++b) {
    h.cache_indices[b] = cfg.include_pad && (b % 7 == 4) ? kPadSlot : b;
  }

  for (int b = 0; b < cfg.batch; ++b) {
    int slot = h.cache_indices[b];
    if (slot == kPadSlot) {
      continue;
    }
    for (int t = 0; t < cfg.draft_tokens; ++t) {
      for (int w = 0; w < cfg.width_minus_one; ++w) {
        int position = t + 1 + w;
        for (int d = 0; d < cfg.channels; ++d) {
          Element value;
          if (position < cfg.width_minus_one) {
            value = h.cache[static_cast<std::size_t>(slot) * h.cache_stride_slot
                + static_cast<std::size_t>(position) * h.cache_stride_w + d];
          } else {
            int hidden_t = position - cfg.width_minus_one;
            value = h.hidden[static_cast<std::size_t>(b) * h.hidden_stride_b
                + static_cast<std::size_t>(hidden_t) * h.hidden_stride_t + d];
          }
          h.ref_out[static_cast<std::size_t>(b) * h.out_stride_b
              + static_cast<std::size_t>(t) * h.out_stride_t
              + static_cast<std::size_t>(w) * h.out_stride_w + d] = value;
        }
      }
    }
  }
  return h;
}

template <typename Element>
double window_effective_bytes(WindowCase const& cfg) {
  int active_batches = cfg.batch;
  if (cfg.include_pad) {
    active_batches = 0;
    for (int b = 0; b < cfg.batch; ++b) {
      if (b % 7 != 4) {
        ++active_batches;
      }
    }
  }
  return static_cast<double>(active_batches) * cfg.draft_tokens * cfg.width_minus_one *
      cfg.channels * sizeof(Element) * 2.0;
}

template <typename Element>
bool run_window_case(
    sycl::queue& q,
    WindowCase const& cfg,
    int iterations,
    bool verify,
    double target_gbps) {
  WindowHost<Element> h = make_window_host<Element>(cfg);
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<Element> d_hidden(q, h.hidden.size());
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<Element> d_out(q, h.out.size());
  d_cache.copy_from(h.cache);
  d_hidden.copy_from(h.hidden);
  d_cache_indices.copy_from(h.cache_indices);
  d_out.copy_from(h.out);

  WindowParams<Element> params{
      d_cache.get(),
      d_hidden.get(),
      d_cache_indices.get(),
      d_out.get(),
      h.cache_stride_slot,
      h.cache_stride_w,
      h.hidden_stride_b,
      h.hidden_stride_t,
      h.out_stride_b,
      h.out_stride_t,
      h.out_stride_w,
      cfg.batch,
      cfg.draft_tokens,
      cfg.width_minus_one,
      cfg.channels,
      kPadSlot};
  auto launch = [&]() {
    return launch_save_windows<Element>(q, params);
  };
  launch().wait_and_throw();

  bool passed = true;
  int bad = -1;
  if (verify) {
    d_out.copy_to(h.out);
    passed = verify_element_exact(h.out, h.ref_out, bad);
  }

  for (int i = 0; i < 5; ++i) {
    launch();
  }
  q.wait_and_throw();
  double avg_s = time_kernel_seconds(launch, iterations);
  double bytes = window_effective_bytes<Element>(cfg);
  double gbps = (bytes / 1.0e9) / avg_s;
  bool applies_target = target_gbps > 0.0 && bytes >= kMinSustainedWindowBytes;
  if (applies_target && gbps < target_gbps) {
    passed = false;
  }

  std::cout << std::left << std::setw(34) << cfg.name
            << " op=windows"
            << " B=" << std::setw(5) << cfg.batch
            << " T=" << std::setw(4) << cfg.draft_tokens
            << " W-1=" << std::setw(3) << cfg.width_minus_one
            << " D=" << std::setw(6) << cfg.channels
            << "  " << std::fixed << std::setprecision(4)
            << (avg_s * 1000.0) << " ms"
            << "  " << std::setprecision(3) << gbps << " GB/s";
  if (applies_target) {
    std::cout << " target=" << target_gbps << " GB/s";
  } else if (target_gbps > 0.0) {
    std::cout << " target=skipped-cache-smoke";
  }
  std::cout << "  " << (verify ? (passed ? "passed" : "failed") : "verification skipped");
  if (!passed) {
    std::cout << " bad_index=" << bad;
  }
  std::cout << "\n";
  return passed;
}

std::vector<DecodeCase> decode_quick_suite() {
  return {
      {"decode_b5_pad", 5, true},
      {"decode_b128_prod", 128, true},
  };
}

std::vector<ExtendCase> extend_quick_suite() {
  return {
      {"extend_prefix_zero_tail", 8, 7, 5, kHisPrefix, 1, true, true, true, 0},
      {"extend_seq_minus_ext", 17, 9, 3, kHisSeqMinusExt, 1, true, true, true, 0},
      {"extend_zeros_boundary", 11, 5, 0, kHisZeros, 1, true, true, true, 0},
      {"target_verify_draft9", 16, 0, 0, kHisOnes, 9, false, true, false, 0},
  };
}

std::vector<TrackCase> track_quick_suite() {
  return {
      {"track_small_w3", 6, 3, 11, 4, true},
      {"track_w7_irregular", 13, 7, 23, 8, true},
  };
}

std::vector<WindowCase> window_quick_suite() {
  return {
      {"windows_small_tail_pad", 5, 4, 2, 7, true, 0, 0, 0},
      {"windows_nondiv_w7", 7, 9, 7, 257, true, 3, 5, 7},
      {"windows_inkling_d1536", 16, 9, 3, 1536, false, 0, 0, 0},
      {"windows_inkling_d512", 16, 9, 3, 512, false, 0, 0, 0},
      // Smallest batch that selects the 256-byte subgroup block path, with pad
      // slots on, so --verify=1 covers the wide kernel and not just the
      // per-lane vector kernel.
      {"windows_wide_b512_d512_pad", kWideMinBatch, 9, 3, 512, true, 0, 0, 0},
  };
}

std::vector<DecodeCase> decode_perf_suite() {
  return {
      {"decode_b1024_pad", 1024, true},
      {"decode_b4096", 4096, false},
  };
}

std::vector<ExtendCase> extend_perf_suite() {
  return {
      {"extend_b64_t65536_prefix", 64, 1024, 0, kHisPrefix, 1, false, true, false, 0},
      {"extend_b256_t65536_seq", 256, 256, 19, kHisSeqMinusExt, 1, false, true, false, 0},
      {"verify_b256_draft9", 256, 0, 0, kHisOnes, 9, false, true, false, 0},
  };
}

std::vector<TrackCase> track_perf_suite() {
  return {
      {"track_b256_w3", 256, 3, 1024, 256, false},
      {"track_b1024_w7", 1024, 7, 256, 128, true},
  };
}

std::vector<WindowCase> window_perf_suite() {
  return {
      {"windows_b128_t9_w3_d1536_launch", 128, 9, 3, 1536, false, 0, 0, 0},
      {"windows_b128_t9_w3_d512_launch", 128, 9, 3, 512, false, 0, 0, 0},
      {"windows_b512_t9_w3_d1536", 512, 9, 3, 1536, false, 0, 0, 0},
      {"windows_b1024_t9_w3_d768", 1024, 9, 3, 768, false, 0, 0, 0},
      {"windows_b1024_t9_w3_d512", 1024, 9, 3, 512, false, 0, 0, 0},
      {"windows_b1024_t9_w3_d384", 1024, 9, 3, 384, false, 0, 0, 0},
  };
}

std::vector<DecodeCase> decode_suite(std::string const& suite) {
  if (suite == "quick" || suite == "stress") {
    return decode_quick_suite();
  }
  return decode_perf_suite();
}

std::vector<ExtendCase> extend_suite(std::string const& suite) {
  if (suite == "quick") {
    return extend_quick_suite();
  }
  if (suite == "stress") {
    auto cases = extend_quick_suite();
    cases.push_back({"extend_random_b33", 33, 12, 7, kHisPrefix, 1, false, true, true, 1234u});
    cases.push_back({"extend_random_seq_b65", 65, 16, 11, kHisSeqMinusExt, 1, false, true, true, 5678u});
    return cases;
  }
  return extend_perf_suite();
}

std::vector<TrackCase> track_suite(std::string const& suite) {
  if (suite == "quick") {
    return track_quick_suite();
  }
  if (suite == "stress") {
    auto cases = track_quick_suite();
    cases.push_back({"track_b257_w8", 257, 8, 17, 16, true});
    return cases;
  }
  return track_perf_suite();
}

std::vector<WindowCase> window_suite(std::string const& suite) {
  if (suite == "quick") {
    return window_quick_suite();
  }
  if (suite == "stress") {
    auto cases = window_quick_suite();
    cases.push_back({"windows_stress_b9_w8_d31", 9, 11, 8, 31, true, 1, 2, 3});
    cases.push_back({"windows_stress_b15_w5_d1025", 15, 13, 5, 1025, true, 7, 11, 13});
    return cases;
  }
  return window_perf_suite();
}

template <typename Element>
bool run_windows_for_dtype(
    sycl::queue& q,
    std::vector<WindowCase> const& cases,
    int iterations,
    bool verify,
    double target_gbps) {
  bool passed = true;
  for (auto const& cfg : cases) {
    passed &= run_window_case<Element>(q, cfg, iterations, verify, target_gbps);
  }
  return passed;
}

struct Options {
  bool help = false;
  bool valid = true;
  bool verify = true;
  int iterations = 20;
  std::string suite = "quick";
  std::string dtype_name = "all";
  DType dtype = DType::kAll;
  std::string op_name = "all";
  Op op = Op::kAll;
  double target_gbps = 0.0;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    int verify_int = 1;
    cmd.get_cmd_line_argument("verify", verify_int, 1);
    verify = verify_int != 0;
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("suite", suite, std::string("quick"));
    cmd.get_cmd_line_argument("dtype", dtype_name, std::string("all"));
    cmd.get_cmd_line_argument("op", op_name, std::string("all"));
    cmd.get_cmd_line_argument("target-gbps", target_gbps, 0.0);
    if (!parse_dtype(dtype_name, dtype) || !parse_op(op_name, op)) {
      valid = false;
    }
    if (suite != "quick" && suite != "stress" && suite != "perf") {
      valid = false;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Inkling BMG SConv Metadata and Windows Example\n\n"
        << "Options:\n"
        << "  --help                         Print this message\n"
        << "  --op=<all|decode|extend|track|windows>\n"
        << "  --suite=<quick|stress|perf>     Built-in shape suite (default: quick)\n"
        << "  --dtype=<all|bf16|fp16>         Window dtype; metadata is integer-only\n"
        << "  --iterations=<int>              Timed kernel iterations\n"
        << "  --verify=<0|1>                  Run CPU reference comparison\n"
        << "  --target-gbps=<float>           Fail window cases below this effective GB/s\n\n"
        << "Examples:\n"
        << "  ./examples/14_bmg_sconv/14_bmg_metadata_windows_sconv --suite=quick\n"
        << "  ./examples/14_bmg_sconv/14_bmg_metadata_windows_sconv --op=windows --dtype=fp16\n"
        << "  ./examples/14_bmg_sconv/14_bmg_metadata_windows_sconv --suite=perf --verify=0 --iterations=100\n";
    return out;
  }
};

}  // namespace cutlass::examples::sconv

int main(int argc, char const** argv) {
  using namespace cutlass::examples::sconv;

  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }
  if (!options.valid) {
    std::cerr << "Invalid options: op=" << options.op_name
              << " dtype=" << options.dtype_name
              << " suite=" << options.suite << "\n";
    options.print_usage(std::cerr);
    return 1;
  }

  try {
    sycl::queue base_queue = compat::get_default_queue();
    sycl::queue q(
        base_queue.get_context(),
        base_queue.get_device(),
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Suite: " << options.suite
              << " op=" << op_text(options.op)
              << " dtype=" << dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " verify=" << bool_text(options.verify) << "\n";

    bool passed = true;
    if (options.op == Op::kAll || options.op == Op::kDecode) {
      for (auto const& cfg : decode_suite(options.suite)) {
        passed &= run_decode_case(q, cfg, options.iterations, options.verify);
      }
    }
    if (options.op == Op::kAll || options.op == Op::kExtend) {
      for (auto const& cfg : extend_suite(options.suite)) {
        passed &= run_extend_case(q, cfg, options.iterations, options.verify);
      }
    }
    if (options.op == Op::kAll || options.op == Op::kTrack) {
      for (auto const& cfg : track_suite(options.suite)) {
        passed &= run_track_case(q, cfg, options.iterations, options.verify);
      }
    }
    if (options.op == Op::kAll || options.op == Op::kWindows) {
      auto cases = window_suite(options.suite);
      if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
        passed &= run_windows_for_dtype<cutlass::bfloat16_t>(
            q, cases, options.iterations, options.verify, options.target_gbps);
      }
      if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
        passed &= run_windows_for_dtype<cutlass::half_t>(
            q, cases, options.iterations, options.verify, options.target_gbps);
      }
    }
    return passed ? 0 : 1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
