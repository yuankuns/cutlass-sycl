/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling HMLP helper kernels for CUTLASS SYCL on BMG.
 *
 * Roofline summary:
 *   fold_timespace_to_depth is a reshape/permute/reshape materialization with
 *   no arithmetic. Each element is read once and written once, so arithmetic
 *   intensity is 0 FLOP/B and the useful metric is sustained effective memory
 *   bandwidth over input + output bytes. The optimized path uses ESIMD 256-512B
 *   block copies for sustained large-row shuffles, with 16-128B vector-lane
 *   fallbacks, and keeps t_fold/hw_fold runtime parameters because Inkling
 *   derives them from prime factors of the model patch sizes.
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cutlass::examples::bmg_hmlp {

constexpr int kBlockSize = 256;
constexpr int kPackBytes = 16;
constexpr int kMediumLaneBytes = 64;
constexpr int kLargeLaneBytes = 128;
constexpr int kEsimdCopyWords = 64;
constexpr int kLargeEsimdCopyWords = 128;
constexpr int64_t kLargeLaneElementsThreshold = 1 << 20;

inline int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int64_t round_up(int64_t x, int64_t multiple) {
  return ceil_div(x, multiple) * multiple;
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
struct FoldParams {
  Element const* x = nullptr;
  Element* out = nullptr;
  int B = 0;
  int T = 0;
  int H = 0;
  int W = 0;
  int C = 0;
  int t_fold = 1;
  int hw_fold = 1;
  int t_new = 0;
  int h_new = 0;
  int w_new = 0;
  int fold_count = 1;
  int64_t total_elements = 0;
  int64_t lanes_per_segment = 0;
  int64_t vec_count = 0;
};

template <typename Element>
void validate_fold_params(FoldParams<Element> const& params) {
  if (params.x == nullptr || params.out == nullptr) {
    throw std::invalid_argument("fold_timespace_to_depth got a null pointer");
  }
  if (params.B <= 0 || params.T <= 0 || params.H <= 0 || params.W <= 0 || params.C <= 0) {
    throw std::invalid_argument("fold_timespace_to_depth shape dimensions must be positive");
  }
  if (params.t_fold <= 0 || params.hw_fold <= 0) {
    throw std::invalid_argument("fold factors must be positive");
  }
  if (params.T % params.t_fold != 0 || params.H % params.hw_fold != 0 || params.W % params.hw_fold != 0) {
    throw std::invalid_argument("fold factors must divide T, H, and W");
  }
}

template <typename Element>
FoldParams<Element> make_fold_params(
    Element const* x,
    Element* out,
    int B,
    int T,
    int H,
    int W,
    int C,
    int t_fold,
    int hw_fold) {
  FoldParams<Element> params;
  params.x = x;
  params.out = out;
  params.B = B;
  params.T = T;
  params.H = H;
  params.W = W;
  params.C = C;
  params.t_fold = t_fold;
  params.hw_fold = hw_fold;
  validate_fold_params(params);
  params.t_new = T / t_fold;
  params.h_new = H / hw_fold;
  params.w_new = W / hw_fold;
  params.fold_count = t_fold * hw_fold * hw_fold;
  params.total_elements = static_cast<int64_t>(B) * T * H * W * C;
  return params;
}

template <typename Element>
bool is_contiguous_reinterpret(FoldParams<Element> const& params) {
  if (params.t_fold == 1 && params.hw_fold == 1) {
    return true;
  }
  if (params.h_new == 1 && params.w_new == 1) {
    return true;
  }
  if (params.t_fold == 1 && params.w_new == 1) {
    return true;
  }
  return false;
}

template <typename Element>
int64_t segment_count(FoldParams<Element> const& params) {
  return static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new * params.fold_count;
}

template <typename Element, int LaneElems>
class FoldTimespaceToDepthKernel;

template <typename Element, int LaneElems>
class FoldTimespaceToDepthRowSliceKernel;

template <typename Element, int HwFold>
class FoldTimespaceToDepthSpatialC3T1Kernel;

template <typename Element, int HwFold>
class FoldTimespaceToDepthSpatialC3T1PairKernel;

template <typename Element, int HwFold>
class FoldTimespaceToDepthSpatialC3T1RowKernel;

template <typename Element, int CopyWords>
class FoldTimespaceToDepthSegmentEsimdKernel {
 public:
  FoldParams<Element> params;
  int chunks_per_segment = 0;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int segment = linear / chunks_per_segment;
    int chunk = linear - segment * chunks_per_segment;

    int fold = segment % params.fold_count;
    int outer = segment / params.fold_count;
    int w_out = outer % params.w_new;
    outer /= params.w_new;
    int h_out = outer % params.h_new;
    outer /= params.h_new;
    int t_out = outer % params.t_new;
    int b = outer / params.t_new;

    int wf = fold % params.hw_fold;
    int fold_tmp = fold / params.hw_fold;
    int hf = fold_tmp % params.hw_fold;
    int tf = fold_tmp / params.hw_fold;

    int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) *
                        params.H +
                    h_out * params.hw_fold + hf) *
                       params.W +
                   w_out * params.hw_fold + wf) *
        params.C;
    int64_t dst = static_cast<int64_t>(segment) * params.C;

    auto value = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src) + chunk * CopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst) + chunk * CopyWords, value);
  }
};

template <typename Element, int CopyWords>
class FoldTimespaceToDepthRowSliceEsimdKernel {
 public:
  FoldParams<Element> params;
  int chunks_per_slice = 0;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int slice = linear / chunks_per_slice;
    int chunk = linear - slice * chunks_per_slice;

    int hf = slice % params.hw_fold;
    slice /= params.hw_fold;
    int tf = slice % params.t_fold;
    int outer = slice / params.t_fold;
    int w_out = outer % params.w_new;
    outer /= params.w_new;
    int h_out = outer % params.h_new;
    outer /= params.h_new;
    int t_out = outer % params.t_new;
    int b = outer / params.t_new;

    int64_t outer_cell = (((static_cast<int64_t>(b) * params.t_new + t_out) * params.h_new + h_out) *
                              params.w_new +
                          w_out);
    int64_t dst = (outer_cell * params.fold_count +
                   (static_cast<int64_t>(tf) * params.hw_fold + hf) * params.hw_fold) *
        params.C;
    int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) *
                        params.H +
                    h_out * params.hw_fold + hf) *
                       params.W +
                   w_out * params.hw_fold) *
        params.C;

    auto value = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src) + chunk * CopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst) + chunk * CopyWords, value);
  }
};

template <typename Element, int CopyWords>
class FoldTimespaceToDepthHwf2PairRowsEsimdKernel {
 public:
  FoldParams<Element> params;
  int chunks_per_slice = 0;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int linear = static_cast<int>(item.get_linear_id());
    int tf_cell = linear / chunks_per_slice;
    int chunk = linear - tf_cell * chunks_per_slice;

    int tf = tf_cell % params.t_fold;
    int outer_cell_i = tf_cell / params.t_fold;
    int outer = outer_cell_i;
    int w_out = outer % params.w_new;
    outer /= params.w_new;
    int h_out = outer % params.h_new;
    outer /= params.h_new;
    int t_out = outer % params.t_new;
    int b = outer / params.t_new;

    int64_t src0 = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) * params.H +
                     h_out * 2) *
                        params.W +
                    w_out * 2) *
        params.C;
    int64_t src1 = src0 + static_cast<int64_t>(params.W) * params.C;
    int64_t outer_cell = (((static_cast<int64_t>(b) * params.t_new + t_out) * params.h_new + h_out) *
                              params.w_new +
                          w_out);
    int64_t dst0 = (outer_cell * params.fold_count + static_cast<int64_t>(tf) * 4) * params.C;
    int64_t dst1 = dst0 + 2 * params.C;

    auto row0 = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src0) + chunk * CopyWords);
    auto row1 = sycl::ext::intel::esimd::block_load<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t const*>(params.x + src1) + chunk * CopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst0) + chunk * CopyWords, row0);
    sycl::ext::intel::esimd::block_store<uint32_t, CopyWords>(
        reinterpret_cast<uint32_t*>(params.out + dst1) + chunk * CopyWords, row1);
  }
};

template <typename Element>
bool pointer_aligned(Element const* ptr, int bytes) {
  return reinterpret_cast<std::uintptr_t>(ptr) % static_cast<std::uintptr_t>(bytes) == 0;
}

template <typename Element>
bool pointer_aligned(Element* ptr, int bytes) {
  return reinterpret_cast<std::uintptr_t>(ptr) % static_cast<std::uintptr_t>(bytes) == 0;
}

template <int CopyWords, typename Element>
bool can_use_esimd_segment_copy(FoldParams<Element> const& params, int& chunks_per_segment) {
  constexpr int kCopyBytes = CopyWords * static_cast<int>(sizeof(uint32_t));
  int segment_bytes = params.C * static_cast<int>(sizeof(Element));
  if (segment_bytes % kCopyBytes != 0 ||
      !pointer_aligned(params.x, kPackBytes) ||
      !pointer_aligned(params.out, kPackBytes)) {
    return false;
  }
  chunks_per_segment = segment_bytes / kCopyBytes;
  return chunks_per_segment > 0;
}

template <int CopyWords, typename Element>
bool can_use_esimd_row_slice_copy(FoldParams<Element> const& params, int& chunks_per_slice) {
  constexpr int kCopyBytes = CopyWords * static_cast<int>(sizeof(uint32_t));
  int slice_bytes = params.hw_fold * params.C * static_cast<int>(sizeof(Element));
  if (slice_bytes % kCopyBytes != 0 ||
      !pointer_aligned(params.x, kPackBytes) ||
      !pointer_aligned(params.out, kPackBytes)) {
    return false;
  }
  chunks_per_slice = slice_bytes / kCopyBytes;
  return chunks_per_slice > 0;
}

template <typename Element, int HwFold>
inline int spatial_c3_t1_src_index(int idx, FoldParams<Element> const& params) {
  constexpr int kChannels = 3;
  constexpr int kElemsPerOutputCell = HwFold * HwFold * kChannels;

  int elem = idx % kElemsPerOutputCell;
  int outer = idx / kElemsPerOutputCell;
  int c = elem % kChannels;
  int fold = elem / kChannels;
  int wf = fold % HwFold;
  int hf = fold / HwFold;

  int w_out = outer % params.w_new;
  outer /= params.w_new;
  int h_out = outer % params.h_new;
  int b = outer / params.h_new;

  return (((b * params.H + h_out * HwFold + hf) * params.W + w_out * HwFold + wf) * kChannels) + c;
}

template <int CopyWords, typename Element>
sycl::event launch_fold_segment_esimd(sycl::queue& queue, FoldParams<Element> params, int chunks_per_segment) {
  int64_t total_chunks = segment_count(params) * static_cast<int64_t>(chunks_per_segment);
  if (total_chunks > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth ESIMD segment launch exceeds 32-bit indexing");
  }
  FoldTimespaceToDepthSegmentEsimdKernel<Element, CopyWords> kernel{params, chunks_per_segment};
  return queue.parallel_for<FoldTimespaceToDepthSegmentEsimdKernel<Element, CopyWords>>(
      sycl::range<1>(static_cast<std::size_t>(total_chunks)), kernel);
}

template <int CopyWords, typename Element>
sycl::event launch_fold_row_slice_esimd(sycl::queue& queue, FoldParams<Element> params, int chunks_per_slice) {
  int64_t slices = static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new *
      params.t_fold * params.hw_fold;
  int64_t total_chunks = slices * static_cast<int64_t>(chunks_per_slice);
  if (total_chunks > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth ESIMD row-slice launch exceeds 32-bit indexing");
  }
  FoldTimespaceToDepthRowSliceEsimdKernel<Element, CopyWords> kernel{params, chunks_per_slice};
  return queue.parallel_for<FoldTimespaceToDepthRowSliceEsimdKernel<Element, CopyWords>>(
      sycl::range<1>(static_cast<std::size_t>(total_chunks)), kernel);
}

template <int CopyWords, typename Element>
sycl::event launch_fold_hwf2_pair_rows_esimd(
    sycl::queue& queue,
    FoldParams<Element> params,
    int chunks_per_slice) {
  int64_t tf_cells = static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new * params.t_fold;
  int64_t total_chunks = tf_cells * static_cast<int64_t>(chunks_per_slice);
  if (total_chunks > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth ESIMD hwf2 launch exceeds 32-bit indexing");
  }
  FoldTimespaceToDepthHwf2PairRowsEsimdKernel<Element, CopyWords> kernel{params, chunks_per_slice};
  return queue.parallel_for<FoldTimespaceToDepthHwf2PairRowsEsimdKernel<Element, CopyWords>>(
      sycl::range<1>(static_cast<std::size_t>(total_chunks)), kernel);
}

template <typename Element, int LaneElems>
sycl::event launch_fold_kernel_static(sycl::queue& queue, FoldParams<Element> params) {
  constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(Element));
  constexpr int kPackWords = kPackBytes / static_cast<int>(sizeof(uint32_t));
  constexpr int kPacksPerLane = LaneElems / kPackElems;
  static_assert(LaneElems % kPackElems == 0, "lane must contain whole 16B packs");

  bool aligned = (params.C % kPackElems == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.x) % kPackBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.out) % kPackBytes == 0);

  params.vec_count = aligned ? params.C / LaneElems : 0;
  int64_t scalar_tail = params.C - params.vec_count * LaneElems;
  params.lanes_per_segment = params.vec_count + scalar_tail;

  int64_t segments = segment_count(params);
  int64_t total_lanes = segments * params.lanes_per_segment;
  if (total_lanes > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth launch grid exceeds 32-bit work-item indexing");
  }
  int64_t global = round_up(total_lanes, kBlockSize);
  int total_lanes_i = static_cast<int>(total_lanes);
  int lanes_per_segment_i = static_cast<int>(params.lanes_per_segment);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<FoldTimespaceToDepthKernel<Element, LaneElems>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
        [=](sycl::nd_item<1> item) {
          int idx = static_cast<int>(item.get_global_id(0));
          if (idx >= total_lanes_i) {
            return;
          }

          int segment = idx / lanes_per_segment_i;
          int lane = idx - segment * lanes_per_segment_i;

          int fold = segment % params.fold_count;
          int outer = segment / params.fold_count;
          int w_out = outer % params.w_new;
          outer /= params.w_new;
          int h_out = outer % params.h_new;
          outer /= params.h_new;
          int t_out = outer % params.t_new;
          int b = outer / params.t_new;

          int wf = fold % params.hw_fold;
          int fold_tmp = fold / params.hw_fold;
          int hf = fold_tmp % params.hw_fold;
          int tf = fold_tmp / params.hw_fold;

          int c = 0;
          if (lane < params.vec_count) {
            c = lane * LaneElems;
          } else {
            c = static_cast<int>(params.vec_count) * LaneElems + (lane - static_cast<int>(params.vec_count));
          }

          int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) *
                              params.H +
                          h_out * params.hw_fold + hf) *
                             params.W +
                         w_out * params.hw_fold + wf) *
                            params.C +
              c;
          int64_t dst = static_cast<int64_t>(segment) * params.C + c;

          if (lane < params.vec_count) {
            using pack_t = sycl::vec<uint32_t, kPackWords>;
#pragma unroll
            for (int pack = 0; pack < kPacksPerLane; ++pack) {
              pack_t value;
              value.load(0, reinterpret_cast<uint32_t const*>(params.x + src + pack * kPackElems));
              value.store(0, reinterpret_cast<uint32_t*>(params.out + dst + pack * kPackElems));
            }
          } else {
            params.out[dst] = params.x[src];
          }
        });
  });
}

template <typename Element, int HwFold>
sycl::event launch_fold_spatial_c3_t1_kernel_static(sycl::queue& queue, FoldParams<Element> params) {
  static_assert(HwFold > 1, "spatial C3 fast path is only useful for spatial folds");
  constexpr int kChannels = 3;
  constexpr int kElemsPerOutputCell = HwFold * HwFold * kChannels;
  constexpr int kElemsPerInputRow = HwFold * kChannels;

  if (params.total_elements > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth spatial C3 launch grid exceeds 32-bit indexing");
  }
  int total_elements_i = static_cast<int>(params.total_elements);

  if constexpr (sizeof(Element) == 2) {
    int total_pairs_i = static_cast<int>(ceil_div(params.total_elements, 2));
    int64_t global = round_up(total_pairs_i, kBlockSize);

    return queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<FoldTimespaceToDepthSpatialC3T1PairKernel<Element, HwFold>>(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<std::size_t>(global)),
              sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
          [=](sycl::nd_item<1> item) {
            int pair_idx = static_cast<int>(item.get_global_id(0));
            if (pair_idx >= total_pairs_i) {
              return;
            }

            int idx0 = pair_idx * 2;
            int src0 = spatial_c3_t1_src_index<Element, HwFold>(idx0, params);
            if constexpr (HwFold == 2) {
              uint32_t value = *reinterpret_cast<uint32_t const*>(params.x + src0);
              *reinterpret_cast<uint32_t*>(params.out + idx0) = value;
              return;
            }
            params.out[idx0] = params.x[src0];

            int idx1 = idx0 + 1;
            if (idx1 < total_elements_i) {
              int elem0 = idx0 % kElemsPerOutputCell;
              int row_elem0 = elem0 % kElemsPerInputRow;
              int src1 = (row_elem0 + 1 < kElemsPerInputRow)
                  ? src0 + 1
                  : spatial_c3_t1_src_index<Element, HwFold>(idx1, params);
              params.out[idx1] = params.x[src1];
            }
          });
    });
  }

  int64_t global = round_up(params.total_elements, kBlockSize);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<FoldTimespaceToDepthSpatialC3T1Kernel<Element, HwFold>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
        [=](sycl::nd_item<1> item) {
          int idx = static_cast<int>(item.get_global_id(0));
          if (idx >= total_elements_i) {
            return;
          }

          int src = spatial_c3_t1_src_index<Element, HwFold>(idx, params);
          params.out[idx] = params.x[src];
        });
  });
}

template <typename Element, int HwFold>
sycl::event launch_fold_spatial_c3_t1_row_kernel_static(sycl::queue& queue, FoldParams<Element> params) {
  static_assert(HwFold > 1, "spatial C3 row fast path is only useful for spatial folds");
  constexpr int kChannels = 3;
  constexpr int kElemsPerOutputCell = HwFold * HwFold * kChannels;
  constexpr int kElemsPerInputRow = HwFold * kChannels;

  int64_t rows = static_cast<int64_t>(params.B) * params.h_new * params.w_new * HwFold;
  if (rows > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth spatial C3 row launch exceeds 32-bit indexing");
  }
  int total_rows_i = static_cast<int>(rows);
  int64_t global = round_up(total_rows_i, kBlockSize);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<FoldTimespaceToDepthSpatialC3T1RowKernel<Element, HwFold>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
        [=](sycl::nd_item<1> item) {
          int row = static_cast<int>(item.get_global_id(0));
          if (row >= total_rows_i) {
            return;
          }

          int hf = row % HwFold;
          int cell = row / HwFold;
          int outer = cell;
          int w_out = outer % params.w_new;
          outer /= params.w_new;
          int h_out = outer % params.h_new;
          int b = outer / params.h_new;

          int64_t dst = static_cast<int64_t>(cell) * kElemsPerOutputCell + hf * kElemsPerInputRow;
          int64_t src = ((static_cast<int64_t>(b) * params.H + h_out * HwFold + hf) *
                             params.W +
                         w_out * HwFold) *
              kChannels;

          if constexpr (sizeof(Element) == 2 && HwFold == 7) {
            if ((src & 1) == (dst & 1)) {
              int start = static_cast<int>(src & 1);
              if (start != 0) {
                params.out[dst] = params.x[src];
              }
              uint32_t const* src_words = reinterpret_cast<uint32_t const*>(params.x + src + start);
              uint32_t* dst_words = reinterpret_cast<uint32_t*>(params.out + dst + start);
#pragma unroll
              for (int word = 0; word < (kElemsPerInputRow - 1) / 2; ++word) {
                dst_words[word] = src_words[word];
              }
              if (start == 0) {
                params.out[dst + kElemsPerInputRow - 1] = params.x[src + kElemsPerInputRow - 1];
              }
              return;
            }
          }
#pragma unroll
          for (int row_elem = 0; row_elem < kElemsPerInputRow; ++row_elem) {
            params.out[dst + row_elem] = params.x[src + row_elem];
          }
        });
  });
}

template <typename Element, int LaneElems>
sycl::event launch_fold_row_slice_kernel_static(sycl::queue& queue, FoldParams<Element> params) {
  constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(Element));
  constexpr int kPackWords = kPackBytes / static_cast<int>(sizeof(uint32_t));
  constexpr int kPacksPerLane = LaneElems / kPackElems;
  static_assert(LaneElems % kPackElems == 0, "lane must contain whole 16B packs");

  int slice_elems = params.hw_fold * params.C;
  bool aligned = (params.C % kPackElems == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.x) % kPackBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.out) % kPackBytes == 0);

  params.vec_count = aligned ? slice_elems / LaneElems : 0;
  int64_t scalar_tail = slice_elems - params.vec_count * LaneElems;
  params.lanes_per_segment = params.vec_count + scalar_tail;

  int64_t slices = static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new *
      params.t_fold * params.hw_fold;
  int64_t total_lanes = slices * params.lanes_per_segment;
  if (total_lanes > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth row-slice launch grid exceeds 32-bit work-item indexing");
  }
  int64_t global = round_up(total_lanes, kBlockSize);
  int total_lanes_i = static_cast<int>(total_lanes);
  int lanes_per_segment_i = static_cast<int>(params.lanes_per_segment);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<FoldTimespaceToDepthRowSliceKernel<Element, LaneElems>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
        [=](sycl::nd_item<1> item) {
          int idx = static_cast<int>(item.get_global_id(0));
          if (idx >= total_lanes_i) {
            return;
          }

          int slice = idx / lanes_per_segment_i;
          int lane = idx - slice * lanes_per_segment_i;

          int hf = slice % params.hw_fold;
          slice /= params.hw_fold;
          int tf = slice % params.t_fold;
          int outer = slice / params.t_fold;
          int w_out = outer % params.w_new;
          outer /= params.w_new;
          int h_out = outer % params.h_new;
          outer /= params.h_new;
          int t_out = outer % params.t_new;
          int b = outer / params.t_new;

          int c = 0;
          if (lane < params.vec_count) {
            c = lane * LaneElems;
          } else {
            c = static_cast<int>(params.vec_count) * LaneElems + (lane - static_cast<int>(params.vec_count));
          }

          int64_t outer_cell = (((static_cast<int64_t>(b) * params.t_new + t_out) * params.h_new + h_out) *
                                    params.w_new +
                                w_out);
          int64_t dst = (outer_cell * params.fold_count +
                         (static_cast<int64_t>(tf) * params.hw_fold + hf) * params.hw_fold) *
                  params.C +
              c;
          int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) *
                              params.H +
                          h_out * params.hw_fold + hf) *
                             params.W +
                         w_out * params.hw_fold) *
                            params.C +
              c;

          if (lane < params.vec_count) {
            using pack_t = sycl::vec<uint32_t, kPackWords>;
#pragma unroll
            for (int pack = 0; pack < kPacksPerLane; ++pack) {
              pack_t value;
              value.load(0, reinterpret_cast<uint32_t const*>(params.x + src + pack * kPackElems));
              value.store(0, reinterpret_cast<uint32_t*>(params.out + dst + pack * kPackElems));
            }
          } else {
            params.out[dst] = params.x[src];
          }
        });
  });
}

template <typename Element>
sycl::event launch_fold_timespace_to_depth(sycl::queue& queue, FoldParams<Element> params) {
  validate_fold_params(params);
  params.t_new = params.T / params.t_fold;
  params.h_new = params.H / params.hw_fold;
  params.w_new = params.W / params.hw_fold;
  params.fold_count = params.t_fold * params.hw_fold * params.hw_fold;
  params.total_elements = static_cast<int64_t>(params.B) * params.T * params.H * params.W * params.C;
  if (params.total_elements == 0) {
    return {};
  }

  if (is_contiguous_reinterpret(params)) {
    return queue.memcpy(params.out, params.x, static_cast<std::size_t>(params.total_elements * sizeof(Element)));
  }

  constexpr int kSmallLaneElems = kPackBytes / static_cast<int>(sizeof(Element));
  constexpr int kMediumLaneElems = kMediumLaneBytes / static_cast<int>(sizeof(Element));
  constexpr int kLargeLaneElems = kLargeLaneBytes / static_cast<int>(sizeof(Element));
  if (params.T == 1 && params.t_fold == 1 && params.C == 3) {
    if (params.hw_fold == 2) {
      return launch_fold_spatial_c3_t1_kernel_static<Element, 2>(queue, params);
    }
    if (params.hw_fold == 7) {
      if (params.total_elements >= kLargeLaneElementsThreshold) {
        return launch_fold_spatial_c3_t1_kernel_static<Element, 7>(queue, params);
      }
      return launch_fold_spatial_c3_t1_row_kernel_static<Element, 7>(queue, params);
    }
  }
  if (params.hw_fold > 1) {
    int slice_elems = params.hw_fold * params.C;
    int chunks_per_slice = 0;
    if constexpr (sizeof(Element) == 2) {
      if (params.hw_fold == 2 &&
          can_use_esimd_row_slice_copy<kLargeEsimdCopyWords>(params, chunks_per_slice)) {
        return launch_fold_hwf2_pair_rows_esimd<kLargeEsimdCopyWords>(queue, params, chunks_per_slice);
      }
      if (params.hw_fold == 2 &&
          can_use_esimd_row_slice_copy<kEsimdCopyWords>(params, chunks_per_slice)) {
        return launch_fold_hwf2_pair_rows_esimd<kEsimdCopyWords>(queue, params, chunks_per_slice);
      }
    }
    if (can_use_esimd_row_slice_copy<kLargeEsimdCopyWords>(params, chunks_per_slice)) {
      return launch_fold_row_slice_esimd<kLargeEsimdCopyWords>(queue, params, chunks_per_slice);
    }
    if (can_use_esimd_row_slice_copy<kEsimdCopyWords>(params, chunks_per_slice)) {
      return launch_fold_row_slice_esimd<kEsimdCopyWords>(queue, params, chunks_per_slice);
    }
    if (params.total_elements >= kLargeLaneElementsThreshold &&
        slice_elems >= kLargeLaneElems &&
        slice_elems % kLargeLaneElems == 0 &&
        params.C % (kPackBytes / static_cast<int>(sizeof(Element))) == 0) {
      return launch_fold_row_slice_kernel_static<Element, kLargeLaneElems>(queue, params);
    }
    if (params.total_elements >= kLargeLaneElementsThreshold && slice_elems >= kMediumLaneElems) {
      return launch_fold_row_slice_kernel_static<Element, kMediumLaneElems>(queue, params);
    }
    return launch_fold_row_slice_kernel_static<Element, kSmallLaneElems>(queue, params);
  }
  int chunks_per_segment = 0;
  if (can_use_esimd_segment_copy<kLargeEsimdCopyWords>(params, chunks_per_segment)) {
    return launch_fold_segment_esimd<kLargeEsimdCopyWords>(queue, params, chunks_per_segment);
  }
  if (can_use_esimd_segment_copy<kEsimdCopyWords>(params, chunks_per_segment)) {
    return launch_fold_segment_esimd<kEsimdCopyWords>(queue, params, chunks_per_segment);
  }
  if (params.total_elements >= kLargeLaneElementsThreshold &&
      params.C >= kLargeLaneElems &&
      params.C % kLargeLaneElems == 0) {
    return launch_fold_kernel_static<Element, kLargeLaneElems>(queue, params);
  }
  if (params.total_elements >= kLargeLaneElementsThreshold && params.C >= kMediumLaneElems) {
    return launch_fold_kernel_static<Element, kMediumLaneElems>(queue, params);
  }
  return launch_fold_kernel_static<Element, kSmallLaneElems>(queue, params);
}

template <typename Element>
sycl::event launch_fold_timespace_to_depth(
    sycl::queue& queue,
    Element const* x,
    Element* out,
    int B,
    int T,
    int H,
    int W,
    int C,
    int t_fold,
    int hw_fold) {
  return launch_fold_timespace_to_depth(queue, make_fold_params(x, out, B, T, H, W, C, t_fold, hw_fold));
}

}  // namespace cutlass::examples::bmg_hmlp
