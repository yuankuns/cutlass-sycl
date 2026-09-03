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
 *   bandwidth over input + output bytes. The copy engine is plain SYCL only (no
 *   ESIMD): each work item moves a whole number of 16B packs, and neighbouring
 *   work items cover neighbouring packs so that one subgroup instruction lands
 *   as a single fully-coalesced 16-lane x 16B = 256B LSC message -- exactly the
 *   message shape the ESIMD block_load<uint32_t, 64> this file used to carry
 *   produced from one thread. Measured on B60 the rewrite lands 1-3% under the
 *   ESIMD it replaces on the shuffle rows (f32: 390.8 vs 399.9, 393.4 vs 397.7,
 *   391.5 vs 402.4 GB/s; bf16/fp16 neutral), which is where the part runs out of
 *   DRAM: the same buffers moved by plain queue.memcpy only reach 379-392 GB/s.
 *   There is no headroom left for a wider message, so no ESIMD, inline vISA or
 *   group_load/striped formulation can buy anything back here. t_fold/hw_fold stay
 *   runtime parameters because Inkling derives them from the prime factors of
 *   the model patch sizes, so non-power-of-two folds (hw_fold=5 at patch_size
 *   40, hw_fold=7 at patch_size 14) must stay correct on the general path.
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>

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

template <typename Element, bool Packed>
class FoldTimespaceToDepthSegmentKernel;

template <typename Element, bool Packed>
class FoldTimespaceToDepthRowSliceKernel;

template <typename Element, int HwFold>
class FoldTimespaceToDepthSpatialC3T1Kernel;

template <typename Element, int HwFold>
class FoldTimespaceToDepthSpatialC3T1PairKernel;

template <typename Element>
bool pointer_aligned(Element const* ptr, int bytes) {
  return reinterpret_cast<std::uintptr_t>(ptr) % static_cast<std::uintptr_t>(bytes) == 0;
}

template <typename Element>
bool pointer_aligned(Element* ptr, int bytes) {
  return reinterpret_cast<std::uintptr_t>(ptr) % static_cast<std::uintptr_t>(bytes) == 0;
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

template <typename Element, bool Packed>
sycl::event launch_fold_segment_kernel(sycl::queue& queue, FoldParams<Element> params) {
  // One work item moves one 16B pack (Packed) or one element (fallback). The
  // lane index is the fastest-varying part of the launch index, so the 16 lanes
  // of a subgroup cover 16 consecutive packs and the load/store issues as a
  // single 256B LSC message. Wider per-lane copies were measured and rejected:
  // a 64B/128B lane makes neighbouring lanes 64B/128B apart, which splits every
  // message into 16 scattered chunks and costs 35-38% of peak on B60 -- e.g.
  // perf_spatial_hw2_c256 f32 measures 248 GB/s with a 128B lane against 393
  // GB/s with the one-pack lane here. (That 248 is an intermediate variant of
  // this rewrite, not the ESIMD baseline it replaces, which was 402 GB/s.)
  constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(Element));
  constexpr int kPackWords = kPackBytes / static_cast<int>(sizeof(uint32_t));
  constexpr int kLaneElems = Packed ? kPackElems : 1;

  int lanes_per_segment_i = params.C / kLaneElems;

  int64_t segments = segment_count(params);
  int64_t total_lanes = segments * lanes_per_segment_i;
  if (total_lanes > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth launch grid exceeds 32-bit work-item indexing");
  }
  int64_t global = round_up(total_lanes, kBlockSize);
  int total_lanes_i = static_cast<int>(total_lanes);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<FoldTimespaceToDepthSegmentKernel<Element, Packed>>(
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

          int c = lane * kLaneElems;
          int64_t src = (((static_cast<int64_t>(b) * params.T + t_out * params.t_fold + tf) *
                              params.H +
                          h_out * params.hw_fold + hf) *
                             params.W +
                         w_out * params.hw_fold + wf) *
                            params.C +
              c;
          int64_t dst = static_cast<int64_t>(segment) * params.C + c;

          if constexpr (Packed) {
            sycl::vec<uint32_t, kPackWords> value;
            value.load(0, reinterpret_cast<uint32_t const*>(params.x + src));
            value.store(0, reinterpret_cast<uint32_t*>(params.out + dst));
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

template <typename Element, bool Packed>
sycl::event launch_fold_row_slice_kernel(sycl::queue& queue, FoldParams<Element> params) {
  // Row-slice form: the hw_fold consecutive w positions of one input row map to
  // hw_fold consecutive output segments, so hw_fold * C elements are contiguous
  // on both sides and one launch index walks that whole slice. Same one-pack-per
  // -lane rule as launch_fold_segment_kernel, for the same measured reason.
  constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(Element));
  constexpr int kPackWords = kPackBytes / static_cast<int>(sizeof(uint32_t));
  constexpr int kLaneElems = Packed ? kPackElems : 1;

  int slice_elems = params.hw_fold * params.C;
  int lanes_per_slice_i = slice_elems / kLaneElems;

  int64_t slices = static_cast<int64_t>(params.B) * params.t_new * params.h_new * params.w_new *
      params.t_fold * params.hw_fold;
  int64_t total_lanes = slices * lanes_per_slice_i;
  if (total_lanes > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("fold_timespace_to_depth row-slice launch grid exceeds 32-bit work-item indexing");
  }
  int64_t global = round_up(total_lanes, kBlockSize);
  int total_lanes_i = static_cast<int>(total_lanes);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<FoldTimespaceToDepthRowSliceKernel<Element, Packed>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kBlockSize))),
        [=](sycl::nd_item<1> item) {
          int idx = static_cast<int>(item.get_global_id(0));
          if (idx >= total_lanes_i) {
            return;
          }

          int slice = idx / lanes_per_slice_i;
          int lane = idx - slice * lanes_per_slice_i;

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

          int c = lane * kLaneElems;
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

          if constexpr (Packed) {
            sycl::vec<uint32_t, kPackWords> value;
            value.load(0, reinterpret_cast<uint32_t const*>(params.x + src));
            value.store(0, reinterpret_cast<uint32_t*>(params.out + dst));
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

  constexpr int kPackElems = kPackBytes / static_cast<int>(sizeof(Element));

  // A purely spatial fold (t_fold == 1) never mixes two different t indices, and
  // the output keeps t as the second-slowest axis, so (b, t) can be flattened
  // into a single batch index. That is exact -- not an approximation -- and it
  // removes one runtime division per work item from every kernel below while
  // also letting the T > 1 shapes reach the C == 3 fast paths. The shipped
  // patch_size=40 layer 0 fold (T=2, H=W=40, C=3, hw_fold=5) needs both.
  if (params.t_fold == 1 && params.T > 1) {
    if (static_cast<int64_t>(params.B) * params.T > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("fold_timespace_to_depth B*T exceeds 32-bit indexing");
    }
    params.B *= params.T;
    params.T = 1;
    params.t_new = 1;
  }

  if (params.T == 1 && params.t_fold == 1 && params.C == 3) {
    // C == 3 leaves nothing to vectorize inside a segment (6B/12B), so this
    // path indexes the output element-wise instead: stores stay perfectly
    // coalesced and only the loads gather. A one-work-item-per-input-row
    // variant (3x fewer items, contiguous 2-byte word copies inside each row)
    // was built and measured and loses at every size on B60 -- at hw_fold=5,
    // (t=2,h=w=40,c=3) it runs 31.6/145.0/115.4 GB/s f32 at b=2/16/100 against
    // 137.5/208.9/242.9 GB/s here, and the same ordering holds for hw_fold=7 --
    // so there is no size threshold worth switching on.
    if (params.hw_fold == 2) {
      return launch_fold_spatial_c3_t1_kernel_static<Element, 2>(queue, params);
    }
    if (params.hw_fold == 5) {
      // patch_size 40 -> prime factors {2,2,2,5}; the first HMLP layer folds 5.
      return launch_fold_spatial_c3_t1_kernel_static<Element, 5>(queue, params);
    }
    if (params.hw_fold == 7) {
      // patch_size 14 -> prime factors {2,7}; the first layer folds 7.
      return launch_fold_spatial_c3_t1_kernel_static<Element, 7>(queue, params);
    }
  }

  // 16B packs need C to be a whole number of packs, because a segment starts at
  // a multiple of C elements and an unaligned sycl::vec load is not allowed.
  bool packed = (params.C % kPackElems == 0) &&
      pointer_aligned(params.x, kPackBytes) &&
      pointer_aligned(params.out, kPackBytes);

  if (params.hw_fold > 1) {
    if (packed) {
      return launch_fold_row_slice_kernel<Element, true>(queue, params);
    }
    return launch_fold_row_slice_kernel<Element, false>(queue, params);
  }

  // Pure temporal fold: only a single C-wide segment is contiguous.
  if (packed) {
    return launch_fold_segment_kernel<Element, true>(queue, params);
  }
  return launch_fold_segment_kernel<Element, false>(queue, params);
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
