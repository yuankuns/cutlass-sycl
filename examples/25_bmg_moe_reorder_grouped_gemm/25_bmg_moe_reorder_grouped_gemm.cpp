/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling MoE dispatch pipeline (reorder + grouped GEMM) for CUTLASS SYCL on BMG.
 *
 * Mirrors sglang's Inkling routed-expert path one-for-one:
 *   python/sglang/srt/models/inkling_common/moe.py  :: run_moe_preprocess / moe_tp_forward
 *   python/sglang/kernels/ops/moe/inkling_moe.py    :: every kernel below
 *
 * Stages (each individually timed, each with a CPU reference):
 *
 *   1. fused_moe_preprocess          One work-group does the whole preprocess: stable sort of
 *                                    the packed (expert_id << 12 | position) keys, src2dst,
 *                                    expert offsets, counts, block offsets and the block
 *                                    schedule. Taken when n = T * top_k <= 1024
 *                                    (FUSED_PREPROCESS_WIN_TOKENS); stage 3 is then skipped,
 *                                    exactly as run_moe_preprocess does.
 *   2. stable sort + get_src2dst     The n > 1024 path. torch.sort(int16, stable=True) is
 *                                    reproduced by a stable block counting sort (per-chunk
 *                                    histogram -> column-major scan -> in-order scatter),
 *                                    then src2dst[reorder_ids[dst]] = dst.
 *   3. compute_grouped_gemm_metadata expert_token_offs (binary search over the sorted ids),
 *                                    num_tokens_per_expert, expert_block_offs (cumsum of
 *                                    cdiv(count, BLOCK_M)) and the packed per-block schedule
 *                                    (block_id << 16 | expert_id, -1 for padding slots).
 *   4. pre_reorder                   Gather hidden_states rows into expert-contiguous order,
 *                                    replicating each token top_k times.
 *   5a. grouped GEMM 1               A[n, H] x w13[E, 2*I_p, H]^T -> [n, 2*I_p]
 *   5b. silu_and_mul                 Interleaved [g0,u0,g1,u1,...] -> silu(g)*u, fp32 math.
 *   5c. grouped GEMM 2               A[n, I_p] x w2[E, H, I_p]^T  -> [n, H]
 *   6. post_reorder                  Scatter back and reduce the top_k contributions weighted
 *                                    by topk_weights.
 *
 * Findings, all read out of the sglang source rather than guessed:
 *
 *   - pre_reorder folds NOTHING. It is a pure gather/replicate; the triton kernel casts to
 *     fp32 and stores back into the input dtype, a bitwise copy for bf16. The per-token top-k
 *     weight is applied at the very end, in post_reorder. (silu_and_mul CAN fold topk_weights,
 *     but Inkling's moe_tp_forward calls activation() without them.)
 *   - The sort key is the expert id cast to int16, sorted stably; the values are the flat slot
 *     positions (t * top_k + k). src2dst is the INVERSE permutation, indexed by source slot.
 *   - Index dtypes: topk_ids int32 -> int16 for the sort; reorder_ids (the argsort output)
 *     int64; src2dst / num_tokens_per_expert / expert_token_offs / expert_block_offs /
 *     expert_block_schedule all int32. The fused single-CTA path keeps reorder_topk_ids in
 *     int32 while the sort path leaves it int16; both dtypes are covered here.
 *   - Weights are b[E, N, K] and the GEMM is A x B^T per expert:
 *     c[m, n] = sum_k a[m, k] * b[expert, n, k].
 *   - Block-config selection rule (select_grouped_gemm_block_m, verified in source):
 *       n = T * top_k <= GROUPED_GEMM_SMALL_M_MAX (6144) -> BLOCK_M 16, BLOCK_N 128, BLOCK_K 128
 *       n >  6144                                       -> BLOCK_M 128, BLOCK_N 256, BLOCK_K 64
 *     BLOCK_M is load-bearing: expert_block_schedule is built for it, and the GEMM decodes it
 *     out of the schedule. BLOCK_N / BLOCK_K are pure perf knobs.
 *   - Inkling passes w13_bias / w2_bias as None, so apply_grouped_bias never fires and is not
 *     modelled here.
 *
 * Performance status: stages 1-4 and 6 (the Inkling-specific reorder/metadata machinery) are
 * the tuned part -- one-launch preprocess, atomic-free counting sort, vectorized copies. The
 * grouped GEMM (5a/5c) is a correctness-first SLM-staged SYCL kernel: it reproduces the triton
 * kernel's schedule, masking and fp32 accumulation order exactly, but it is NOT DPAS-tuned, so
 * its TOPS number is a floor rather than a target. See README.md.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cutlass::examples::bmg_moe_reorder_grouped_gemm {

using Element = cutlass::bfloat16_t;

// ---------------------------------------------------------------------------
// Constants mirroring sglang/kernels/ops/moe/inkling_moe.py
// ---------------------------------------------------------------------------
constexpr int kSmallMBlockSizeM = 16;            // SMALL_M_BLOCK_SIZE_M
constexpr int kBlockSizeM = 128;                 // BLOCK_SIZE_M
constexpr int kGroupedGemmSmallMMax = 6144;      // GROUPED_GEMM_SMALL_M_MAX
constexpr int kFusedPreprocessWinTokens = 1024;  // FUSED_PREPROCESS_WIN_TOKENS
constexpr int kMemsetBlockSize = 512;            // BLOCK_SIZE_MEMSET

// Example-local constants (not part of the model contract).
constexpr int kWgSize = 256;
constexpr int kSortChunk = 1024;   // positions per counting-sort work-group
constexpr int kMaxVerifyRows = 192;  // sampling caps keep --verify=1 usable at T = 16384
constexpr int kMaxVerifyCols = 48;
constexpr double kBytesPerGB = 1.0e9;
constexpr double kOpsPerTOP = 1.0e12;
constexpr double kClockRampMs = 2500.0;  // BMG needs ~2 s to reach 2400 MHz
constexpr int kMaxWarmupIterations = 400;
// Caps the fill kernel's grid so its global size always fits in an int.
constexpr std::size_t kMaxFillGroups = 65536;

// select_grouped_gemm_block_m(num_routed_tokens)
inline int select_grouped_gemm_block_m(int n) {
  return n <= kGroupedGemmSmallMMax ? kSmallMBlockSizeM : kBlockSizeM;
}

CUTLASS_HOST_DEVICE
int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

CUTLASS_HOST_DEVICE
int int_min(int a, int b) {
  return a < b ? a : b;
}

// _get_max_num_blocks(num_routed_tokens, [block_size_m], num_experts)
inline int get_max_num_blocks(int n, int block_m, int num_experts) {
  return ceil_div(n, block_m) + num_experts - 1;
}

enum class PreprocessPath { kAuto, kFused, kSort };

inline char const* path_text(PreprocessPath p) {
  switch (p) {
    case PreprocessPath::kAuto: return "auto";
    case PreprocessPath::kFused: return "fused";
    case PreprocessPath::kSort: return "sort";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// bf16 <-> float helpers (round-to-nearest-even, identical on host and device)
// ---------------------------------------------------------------------------
CUTLASS_HOST_DEVICE
float bf16_to_float(Element x) {
#if defined(__SYCL_DEVICE_ONLY__)
  uint32_t bits = static_cast<uint32_t>(x.raw()) << 16;
  return sycl::bit_cast<float>(bits);
#else
  return static_cast<float>(x);
#endif
}

CUTLASS_HOST_DEVICE
Element float_to_bf16(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
#else
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
#endif
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    if (bits & 0x007fffffu) {
      return Element::bitcast(uint16_t(0x7fffu));
    }
    return Element::bitcast(static_cast<uint16_t>(bits >> 16));
  }
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  return Element::bitcast(static_cast<uint16_t>((bits + rounding_bias) >> 16));
}

CUTLASS_HOST_DEVICE
Element bf16_zero() {
  return Element::bitcast(uint16_t(0));
}

// Counter-based RNG so the multi-GiB weight tensors never need a host staging
// copy: the fill kernel and the CPU reference evaluate the *same* function of
// the linear index, so verification needs no host weight buffer at all. Values
// are genuinely random per element -- Xe memory compression would otherwise
// inflate every bandwidth number measured against constant data.
CUTLASS_HOST_DEVICE
uint32_t mix32(uint32_t x) {
  x += 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x21f0aaadu;
  x ^= x >> 15;
  x *= 0x735a2d97u;
  x ^= x >> 15;
  return x;
}

CUTLASS_HOST_DEVICE
float hash_signed_unit(uint64_t index, uint32_t seed) {
  uint32_t lo = static_cast<uint32_t>(index);
  uint32_t hi = static_cast<uint32_t>(index >> 32);
  uint32_t h = mix32(lo ^ mix32(hi ^ seed));
  // 24 random mantissa bits -> [-1, 1)
  return static_cast<float>(h >> 8) * (1.0f / 8388608.0f) - 1.0f;
}

CUTLASS_HOST_DEVICE
Element hash_bf16(uint64_t index, uint32_t seed, float scale) {
  return float_to_bf16(scale * hash_signed_unit(index, seed));
}

// ---------------------------------------------------------------------------
// Small device-buffer RAII wrapper (same shape as example 23's)
// ---------------------------------------------------------------------------
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

  // Read a sampled subset of equally sized rows. Verification at T = 16384
  // would otherwise pull >1 GiB per buffer back to the host.
  std::vector<T> read_rows(std::vector<int> const& rows, std::size_t row_len) const {
    std::vector<T> out(rows.size() * row_len);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      std::size_t offset = static_cast<std::size_t>(rows[i]) * row_len;
      if (offset + row_len > count) {
        throw std::runtime_error("read_rows out of range");
      }
      queue->memcpy(out.data() + i * row_len, ptr + offset, sizeof(T) * row_len);
    }
    queue->wait();
    return out;
  }
};

inline double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

inline sycl::queue make_queue() {
  return sycl::queue(
      sycl::gpu_selector_v,
      sycl::property_list{sycl::property::queue::in_order{},
                          sycl::property::queue::enable_profiling{}});
}

// Strided sample of [0, count) that always includes the first and last index.
// The stride is forced odd: every real N here is a power of two, so an even
// stride would alias with the GEMM's own kRN / c_slot tile geometry and the
// column sample would only ever land on j == 0, hiding a bug in 15 of the 16
// accumulator slots.
inline std::vector<int> sample_indices(int count, int max_samples) {
  std::vector<int> out;
  if (count <= 0) {
    return out;
  }
  if (count <= max_samples) {
    out.resize(static_cast<std::size_t>(count));
    std::iota(out.begin(), out.end(), 0);
    return out;
  }
  int stride = ceil_div(count, max_samples);
  if (stride % 2 == 0) {
    ++stride;
  }
  for (int i = 0; i < count; i += stride) {
    out.push_back(i);
  }
  if (out.back() != count - 1) {
    out.push_back(count - 1);
  }
  return out;
}

// ===========================================================================
// Stage 0: weight / activation fill (random data, generated on device)
// ===========================================================================
class FillHashKernel;

inline sycl::event fill_hash(
    sycl::queue& queue, Element* dst, std::size_t count, uint32_t seed, float scale) {
  // Grid-stride: the largest weight tensor here is E * 2*I_p * H = 2.4e9
  // elements, and a one-work-item-per-element launch overflows the int that
  // DPC++ uses for global ids ("range and/or offset does not fit in int").
  std::size_t groups =
      std::min<std::size_t>((count + kWgSize - 1) / kWgSize, kMaxFillGroups);
  std::size_t stride = groups * kWgSize;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<FillHashKernel>(
        sycl::nd_range<1>(sycl::range<1>(stride), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          for (std::size_t i = item.get_global_id(0); i < count; i += stride) {
            dst[i] = hash_bf16(i, seed, scale);
          }
        });
  });
}

// ===========================================================================
// Stage 1: fused_moe_preprocess (_fused_moe_preprocess_kernel)
// ===========================================================================
class FusedPreprocessKernel;

struct MetaBuffers {
  int32_t* num_tokens_per_expert = nullptr;
  int32_t* expert_token_offs = nullptr;
  int32_t* expert_block_offs = nullptr;
  int32_t* expert_block_schedule = nullptr;
};

inline int next_pow2(int v) {
  int p = 1;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

// SLM the fused single-work-group preprocess needs: the sort buffer plus three
// per-expert arrays. Checked before dispatch -- an over-budget local_accessor
// otherwise hangs the device instead of failing.
inline std::size_t fused_preprocess_slm_bytes(int n, int num_experts) {
  std::size_t keys = static_cast<std::size_t>(std::max(next_pow2(n), 16));
  std::size_t per_expert = static_cast<std::size_t>(num_experts + 1) * 2 +
                           static_cast<std::size_t>(num_experts);
  return (keys + per_expert) * sizeof(int32_t);
}

// Single work-group: packed-key bitonic sort + src2dst + offsets + counts +
// block offsets + block schedule. The (expert_id << 12 | position) packing is
// what makes the sort stable, and it caps the path at n <= 4096 (the model
// gates far below that, at 1024).
inline sycl::event launch_fused_preprocess(
    sycl::queue& queue,
    int32_t const* topk_ids,
    int32_t* src2dst,
    int32_t* reorder_topk_ids,
    MetaBuffers const& meta,
    int n,
    int num_experts,
    int block_m,
    int max_num_blocks) {
  int block_n = std::max(next_pow2(n), 16);
  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> keys(sycl::range<1>(static_cast<std::size_t>(block_n)), cgh);
    sycl::local_accessor<int32_t, 1> loc_offs(
        sycl::range<1>(static_cast<std::size_t>(num_experts + 1)), cgh);
    sycl::local_accessor<int32_t, 1> loc_nblocks(
        sycl::range<1>(static_cast<std::size_t>(num_experts)), cgh);
    sycl::local_accessor<int32_t, 1> loc_block_offs(
        sycl::range<1>(static_cast<std::size_t>(num_experts + 1)), cgh);
    cgh.parallel_for<FusedPreprocessKernel>(
        sycl::nd_range<1>(sycl::range<1>(kWgSize), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int lid = static_cast<int>(item.get_local_id(0));

          // Pack (id << 12) | position. Padding slots get id == num_experts so
          // they sort strictly after every real key (triton's `other=E`).
          for (int i = lid; i < block_n; i += kWgSize) {
            int id = (i < n) ? topk_ids[i] : num_experts;
            keys[i] = (id << 12) | i;
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Bitonic sort. Every key is unique (the position field breaks all
          // ties), so an ascending sort of the packed keys is exactly
          // torch.sort(stable=True) on the ids.
          for (int k = 2; k <= block_n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
              for (int i = lid; i < block_n; i += kWgSize) {
                int partner = i ^ j;
                if (partner > i) {
                  bool ascending = ((i & k) == 0);
                  int a = keys[i];
                  int b = keys[partner];
                  if ((a > b) == ascending) {
                    keys[i] = b;
                    keys[partner] = a;
                  }
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }
          }

          for (int i = lid; i < n; i += kWgSize) {
            int key = keys[i];
            reorder_topk_ids[i] = key >> 12;
            src2dst[key & 0xfff] = i;
          }

          // expert_token_offs[e] = first sorted index whose id is >= e.
          for (int e = lid; e <= num_experts; e += kWgSize) {
            int lo = 0;
            int hi = n;
            while (lo < hi) {
              int mid = (lo + hi) >> 1;
              if ((keys[mid] >> 12) < e) {
                lo = mid + 1;
              } else {
                hi = mid;
              }
            }
            loc_offs[e] = lo;
            meta.expert_token_offs[e] = lo;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int e = lid; e < num_experts; e += kWgSize) {
            int count = loc_offs[e + 1] - loc_offs[e];
            meta.num_tokens_per_expert[e] = count;
            loc_nblocks[e] = ceil_div(count, block_m);
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Exclusive scan of the per-expert block counts. num_experts is a few
          // hundred at most, so one work-item beats a tree here.
          if (lid == 0) {
            int running = 0;
            for (int e = 0; e < num_experts; ++e) {
              loc_block_offs[e] = running;
              running += loc_nblocks[e];
            }
            loc_block_offs[num_experts] = running;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int e = lid; e <= num_experts; e += kWgSize) {
            meta.expert_block_offs[e] = loc_block_offs[e];
          }

          // schedule[s] = ((s - block_offs[e]) << 16) | e for the last e with
          // block_offs[e] <= s; -1 for the padding slots.
          int total_blocks = loc_block_offs[num_experts];
          for (int s = lid; s < max_num_blocks; s += kWgSize) {
            if (s >= total_blocks) {
              meta.expert_block_schedule[s] = -1;
              continue;
            }
            int lo = 0;
            int hi = num_experts;
            while (lo < hi) {
              int mid = (lo + hi + 1) >> 1;
              if (loc_block_offs[mid] > s) {
                hi = mid - 1;
              } else {
                lo = mid;
              }
            }
            meta.expert_block_schedule[s] = ((s - loc_block_offs[lo]) << 16) | lo;
          }
        });
  });
}

// ===========================================================================
// Stage 2: the n > 1024 path -- stable sort of the int16 expert ids, then
// get_src2dst. torch.sort(stable=True) over a tiny key domain is reproduced by
// a stable block counting sort. No atomics anywhere: each work-item owns a set
// of experts and scans its chunk in position order, which is what makes the
// scatter (and therefore the stability) exact.
// ===========================================================================
class SortHistogramKernel;
class SortBlockScanKernel;
class SortExpertBaseKernel;
class SortScatterKernel;
class Src2DstKernel;

inline sycl::event launch_sort_histogram(
    sycl::queue& queue,
    int32_t const* topk_ids,
    int32_t* block_hist,  // [num_chunks, E]
    int n,
    int num_experts,
    int num_chunks) {
  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> ids(sycl::range<1>(kSortChunk), cgh);
    cgh.parallel_for<SortHistogramKernel>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_chunks) * kWgSize),
                          sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int chunk = static_cast<int>(item.get_group(0));
          int lid = static_cast<int>(item.get_local_id(0));
          int base = chunk * kSortChunk;
          int len = int_min(kSortChunk, n - base);
          for (int i = lid; i < len; i += kWgSize) {
            ids[i] = topk_ids[base + i];
          }
          item.barrier(sycl::access::fence_space::local_space);
          for (int e = lid; e < num_experts; e += kWgSize) {
            int count = 0;
            for (int i = 0; i < len; ++i) {
              count += (ids[i] == e) ? 1 : 0;
            }
            block_hist[static_cast<int64_t>(chunk) * num_experts + e] = count;
          }
        });
  });
}

// Per expert: exclusive running sum over chunks (-> block_rel) and the total.
inline sycl::event launch_sort_block_scan(
    sycl::queue& queue,
    int32_t const* block_hist,
    int32_t* block_rel,
    int32_t* expert_total,
    int num_experts,
    int num_chunks) {
  std::size_t groups = (static_cast<std::size_t>(num_experts) + kWgSize - 1) / kWgSize;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SortBlockScanKernel>(
        sycl::nd_range<1>(sycl::range<1>(groups * kWgSize), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int e = static_cast<int>(item.get_global_id(0));
          if (e >= num_experts) {
            return;
          }
          int running = 0;
          for (int c = 0; c < num_chunks; ++c) {
            int64_t idx = static_cast<int64_t>(c) * num_experts + e;
            block_rel[idx] = running;
            running += block_hist[idx];
          }
          expert_total[e] = running;
        });
  });
}

// Exclusive scan of the per-expert totals -> each expert's base row.
inline sycl::event launch_sort_expert_base(
    sycl::queue& queue, int32_t const* expert_total, int32_t* expert_base, int num_experts) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SortExpertBaseKernel>(
        sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1>) {
          int running = 0;
          for (int e = 0; e < num_experts; ++e) {
            expert_base[e] = running;
            running += expert_total[e];
          }
        });
  });
}

inline sycl::event launch_sort_scatter(
    sycl::queue& queue,
    int32_t const* topk_ids,
    int32_t const* block_rel,
    int32_t const* expert_base,
    int16_t* reorder_topk_ids,
    int64_t* reorder_ids,
    int n,
    int num_experts,
    int num_chunks) {
  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> ids(sycl::range<1>(kSortChunk), cgh);
    cgh.parallel_for<SortScatterKernel>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_chunks) * kWgSize),
                          sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int chunk = static_cast<int>(item.get_group(0));
          int lid = static_cast<int>(item.get_local_id(0));
          int base = chunk * kSortChunk;
          int len = int_min(kSortChunk, n - base);
          for (int i = lid; i < len; i += kWgSize) {
            ids[i] = topk_ids[base + i];
          }
          item.barrier(sycl::access::fence_space::local_space);
          for (int e = lid; e < num_experts; e += kWgSize) {
            int pos = expert_base[e] + block_rel[static_cast<int64_t>(chunk) * num_experts + e];
            for (int i = 0; i < len; ++i) {
              if (ids[i] == e) {
                reorder_topk_ids[pos] = static_cast<int16_t>(e);
                reorder_ids[pos] = static_cast<int64_t>(base) + i;
                ++pos;
              }
            }
          }
        });
  });
}

// _compute_src2dst_kernel: src2dst[reorder_ids[dst]] = dst
inline sycl::event launch_get_src2dst(
    sycl::queue& queue, int64_t const* reorder_ids, int32_t* src2dst, int n) {
  std::size_t groups = (static_cast<std::size_t>(n) + kWgSize - 1) / kWgSize;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Src2DstKernel>(
        sycl::nd_range<1>(sycl::range<1>(groups * kWgSize), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int dst = static_cast<int>(item.get_global_id(0));
          if (dst < n) {
            src2dst[reorder_ids[dst]] = dst;
          }
        });
  });
}

// ===========================================================================
// Stage 3: compute_grouped_gemm_metadata (4 launches, mirroring the model)
// ===========================================================================
template <typename IdT>
class ExpertOffsetsKernel;
class ExpertCountsKernel;
class MemsetBlockMetadataKernel;
class BlockMetadataKernel;

// _compute_expert_offsets_kernel: offs[e + 1] = #{sorted ids <= e}, offs[0] = 0.
template <typename IdT>
inline sycl::event launch_expert_offsets(
    sycl::queue& queue, IdT const* sorted_ids, int32_t* offs, int n, int num_experts) {
  std::size_t groups = (static_cast<std::size_t>(num_experts) + kWgSize - 1) / kWgSize;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<ExpertOffsetsKernel<IdT>>(
        sycl::nd_range<1>(sycl::range<1>(groups * kWgSize), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int expert = static_cast<int>(item.get_global_id(0));
          if (expert >= num_experts) {
            return;
          }
          if (expert == 0) {
            offs[0] = 0;
          }
          int low = 0;
          int high = n - 1;
          int target = -1;
          while (low <= high) {
            int mid = (low + high) / 2;
            if (static_cast<int>(sorted_ids[mid]) > expert) {
              high = mid - 1;
            } else {
              low = mid + 1;
              target = mid;
            }
          }
          offs[expert + 1] = target + 1;
        });
  });
}

inline sycl::event launch_expert_counts(
    sycl::queue& queue, int32_t const* offs, int32_t* counts, int num_experts) {
  std::size_t groups = (static_cast<std::size_t>(num_experts) + kWgSize - 1) / kWgSize;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<ExpertCountsKernel>(
        sycl::nd_range<1>(sycl::range<1>(groups * kWgSize), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int e = static_cast<int>(item.get_global_id(0));
          if (e < num_experts) {
            counts[e] = offs[e + 1] - offs[e];
          }
        });
  });
}

// _memset_block_metadata_kernel: group 0 builds the block-offset cumsum while
// the remaining groups fill the schedule with -1. Disjoint writes, one launch.
inline sycl::event launch_memset_block_metadata(
    sycl::queue& queue,
    int32_t const* counts,
    int32_t* block_offs,
    int32_t* schedule,
    int max_num_blocks,
    int num_experts,
    int block_m) {
  std::size_t groups = 1 + static_cast<std::size_t>(ceil_div(max_num_blocks, kMemsetBlockSize));
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<MemsetBlockMetadataKernel>(
        sycl::nd_range<1>(sycl::range<1>(groups * kWgSize), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int group = static_cast<int>(item.get_group(0));
          int lid = static_cast<int>(item.get_local_id(0));
          if (group == 0) {
            if (lid == 0) {
              int running = 0;
              for (int e = 0; e < num_experts; ++e) {
                block_offs[e] = running;
                running += ceil_div(counts[e], block_m);
              }
              block_offs[num_experts] = running;
            }
          } else {
            int base = (group - 1) * kMemsetBlockSize;
            for (int i = lid; i < kMemsetBlockSize; i += kWgSize) {
              if (base + i < max_num_blocks) {
                schedule[base + i] = -1;
              }
            }
          }
        });
  });
}

// _compute_block_metadata_kernel: one work-group per expert.
inline sycl::event launch_block_metadata(
    sycl::queue& queue,
    int32_t const* counts,
    int32_t const* block_offs,
    int32_t* schedule,
    int num_experts,
    int block_m) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<BlockMetadataKernel>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_experts) * kWgSize),
                          sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int expert = static_cast<int>(item.get_group(0));
          int lid = static_cast<int>(item.get_local_id(0));
          int num_blocks = ceil_div(counts[expert], block_m);
          int off = block_offs[expert];
          for (int b = lid; b < num_blocks; b += kWgSize) {
            schedule[off + b] = (b << 16) | expert;
          }
        });
  });
}

// ===========================================================================
// Stage 4: pre_reorder. Nothing is folded here -- no scale, no topk weight.
// ===========================================================================
template <bool Vec8>
class PreReorderKernel;

template <bool Vec8>
inline sycl::event launch_pre_reorder_impl(
    sycl::queue& queue,
    Element const* input,
    Element* output,
    int32_t const* src2dst,
    int tokens,
    int top_k,
    int hidden) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<PreReorderKernel<Vec8>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(tokens) * kWgSize),
                          sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int token = static_cast<int>(item.get_group(0));
          int lid = static_cast<int>(item.get_local_id(0));
          Element const* src = input + static_cast<int64_t>(token) * hidden;
          int32_t const* map = src2dst + static_cast<int64_t>(token) * top_k;
          for (int k = 0; k < top_k; ++k) {
            Element* dst = output + static_cast<int64_t>(map[k]) * hidden;
            if constexpr (Vec8) {
              int vec = hidden / 8;
              auto const* src_v = reinterpret_cast<sycl::vec<uint32_t, 4> const*>(src);
              auto* dst_v = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst);
              for (int i = lid; i < vec; i += kWgSize) {
                dst_v[i] = src_v[i];
              }
            } else {
              for (int i = lid; i < hidden; i += kWgSize) {
                dst[i] = src[i];
              }
            }
          }
        });
  });
}

inline sycl::event launch_pre_reorder(
    sycl::queue& queue,
    Element const* input,
    Element* output,
    int32_t const* src2dst,
    int tokens,
    int top_k,
    int hidden) {
  if (hidden % 8 == 0) {
    return launch_pre_reorder_impl<true>(queue, input, output, src2dst, tokens, top_k, hidden);
  }
  return launch_pre_reorder_impl<false>(queue, input, output, src2dst, tokens, top_k, hidden);
}

// ===========================================================================
// Stage 5b: silu_and_mul. Inkling's moe_tp_forward calls activation() without
// topk_weights, so the weight is NOT folded here. Interleaved is the shipped
// layout (inference_moe_w13_interleaved defaults to True).
// ===========================================================================
template <bool Interleaved>
class SiluAndMulKernel;

template <bool Interleaved>
inline sycl::event launch_silu_and_mul_impl(
    sycl::queue& queue, Element const* gateup, Element* down_input, int rows, int n) {
  int64_t total = static_cast<int64_t>(rows) * n;
  std::size_t groups = static_cast<std::size_t>((total + kWgSize - 1) / kWgSize);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SiluAndMulKernel<Interleaved>>(
        sycl::nd_range<1>(sycl::range<1>(groups * kWgSize), sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int64_t idx = static_cast<int64_t>(item.get_global_id(0));
          if (idx >= total) {
            return;
          }
          int64_t row = idx / n;
          int64_t col = idx - row * n;
          Element const* src = gateup + row * (2 * static_cast<int64_t>(n));
          float gate;
          float up;
          if constexpr (Interleaved) {
            gate = bf16_to_float(src[2 * col]);
            up = bf16_to_float(src[2 * col + 1]);
          } else {
            gate = bf16_to_float(src[col]);
            up = bf16_to_float(src[col + n]);
          }
          float value = gate * (1.0f / (1.0f + sycl::exp(-gate))) * up;
          down_input[idx] = float_to_bf16(value);
        });
  });
}

inline sycl::event launch_silu_and_mul(
    sycl::queue& queue,
    Element const* gateup,
    Element* down_input,
    int rows,
    int n,
    bool interleaved) {
  if (interleaved) {
    return launch_silu_and_mul_impl<true>(queue, gateup, down_input, rows, n);
  }
  return launch_silu_and_mul_impl<false>(queue, gateup, down_input, rows, n);
}

// ===========================================================================
// Stage 5a/5c: the grouped GEMM. c[m, n] = sum_k a[m, k] * b[expert, n, k].
//
// The schedule decode (expert_block_schedule / expert_block_offs /
// expert_token_offs / num_tokens_per_expert), the padding-block early exit and
// the C-store masking all mirror _grouped_gemm_kernel, and the fp32
// accumulation runs in the same sequential k order, so the CPU reference agrees
// to within one FMA-contraction rounding per term.
//
// The inner tiling is correctness-first, NOT DPAS-tuned: a work-item owns one
// row and kRN contiguous columns, with the A and B tiles staged in SLM. The B
// tile is staged k-major and padded by kBPad so that (a) the kRN reads in the
// inner loop are contiguous and (b) the strided SLM writes during staging do
// not all land in the same bank -- the natural n-major layout costs ~7x.
// ===========================================================================
template <int SchedBlockM, int TileN, int TileK, bool VecStage>
class GroupedGemmKernel;

// Number of bf16 elements moved by one 16-byte staging load.
constexpr int kStageVec = 8;

template <int SchedBlockM, int TileN, int TileK, bool VecStage>
inline sycl::event launch_grouped_gemm_impl(
    sycl::queue& queue,
    Element const* A,
    Element const* B,
    Element* C,
    int32_t const* counts,
    int32_t const* token_offs,
    int32_t const* block_offs,
    int32_t const* schedule,
    int num_experts,
    int N,
    int K,
    int max_num_blocks) {
  constexpr int kRowsPerPass = 16;
  constexpr int kRN = 16;
  constexpr int kColSlots = TileN / kRN;
  constexpr int kItems = kRowsPerPass * kColSlots;
  constexpr int kPasses = SchedBlockM / kRowsPerPass;
  constexpr int kBPad = 2;              // breaks the SLM bank conflict on staging writes
  constexpr int kBRow = TileN + kBPad;  // k-major row pitch of the staged B tile
  static_assert(SchedBlockM % kRowsPerPass == 0, "schedule block must be a multiple of 16");
  static_assert(TileN % kRN == 0, "TileN must be a multiple of 16");
  static_assert(TileK % kStageVec == 0, "TileK must be a multiple of the staging vector width");

  int n_tiles = ceil_div(N, TileN);
  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<Element, 1> a_tile(sycl::range<1>(kRowsPerPass * TileK), cgh);
    sycl::local_accessor<Element, 1> b_tile(sycl::range<1>(TileK * kBRow), cgh);
    cgh.parallel_for<GroupedGemmKernel<SchedBlockM, TileN, TileK, VecStage>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(max_num_blocks),
                           static_cast<std::size_t>(n_tiles) * kItems),
            sycl::range<2>(1, kItems)),
        [=](sycl::nd_item<2> item) {
          int pid_m = static_cast<int>(item.get_group(0));
          int pid_n = static_cast<int>(item.get_group(1));
          int lid = static_cast<int>(item.get_local_id(1));
          int r_slot = lid / kColSlots;
          int c_slot = lid % kColSlots;

          // Padding-only blocks exit, exactly as the triton kernel does.
          if (pid_m >= block_offs[num_experts]) {
            return;
          }
          int data = schedule[pid_m];
          int expert_id = data & 0xffff;
          int block_id = data >> 16;
          int token_start = token_offs[expert_id];
          int block_start = token_start + block_id * SchedBlockM;
          int block_end = int_min(block_start + SchedBlockM, token_start + counts[expert_id]);

          int n_base = pid_n * TileN;
          Element const* b_expert = B + static_cast<int64_t>(expert_id) * N * K;

          for (int pass = 0; pass < kPasses; ++pass) {
            int row_base = block_start + pass * kRowsPerPass;
            // Work-group uniform: block_start/block_end do not depend on lid,
            // so leaving the loop early keeps every barrier below collective.
            if (row_base >= block_end) {
              break;
            }
            float acc[kRN];
#pragma unroll
            for (int j = 0; j < kRN; ++j) {
              acc[j] = 0.0f;
            }

            for (int k0 = 0; k0 < K; k0 += TileK) {
              item.barrier(sycl::access::fence_space::local_space);
              // Staging dominates this kernel's instruction count, so when the
              // K extent allows it each item moves kStageVec elements per
              // global load instead of one (worth ~4x).
              if constexpr (VecStage) {
                constexpr int kAVecs = TileK / kStageVec;
                for (int idx = lid; idx < kRowsPerPass * kAVecs; idx += kItems) {
                  int r = idx / kAVecs;
                  int kv = idx - r * kAVecs;
                  int row = row_base + r;
                  int kg = k0 + kv * kStageVec;
                  Element* slot = &a_tile[r * TileK + kv * kStageVec];
                  // Rows past block_end contribute zeros; they never reach C.
                  // The k tail is guarded too: K need not divide TileK (I_p is
                  // 192 / 96 / 48 at the higher TP degrees). Only the *global*
                  // side is vectorized -- the SLM destination is a
                  // local_accessor<Element>, whose base is only guaranteed
                  // 2-byte aligned, so writing it as a 16-byte vector would be
                  // undefined behaviour that happens to work today.
                  alignas(16) Element buf[kStageVec];
                  if (row < block_end && kg + kStageVec <= K) {
                    *reinterpret_cast<sycl::vec<uint32_t, 4>*>(buf) =
                        *reinterpret_cast<sycl::vec<uint32_t, 4> const*>(
                            A + static_cast<int64_t>(row) * K + kg);
                  } else {
#pragma unroll
                    for (int j = 0; j < kStageVec; ++j) {
                      buf[j] = (row < block_end && kg + j < K)
                                   ? A[static_cast<int64_t>(row) * K + kg + j]
                                   : bf16_zero();
                    }
                  }
#pragma unroll
                  for (int j = 0; j < kStageVec; ++j) {
                    slot[j] = buf[j];
                  }
                }
                constexpr int kBVecs = TileK / kStageVec;
                for (int idx = lid; idx < TileN * kBVecs; idx += kItems) {
                  int nn = idx / kBVecs;
                  int kv = idx - nn * kBVecs;
                  int ng = n_base + nn;
                  int kg = k0 + kv * kStageVec;
                  alignas(16) Element buf[kStageVec];
                  if (ng < N && kg + kStageVec <= K) {
                    *reinterpret_cast<sycl::vec<uint32_t, 4>*>(buf) =
                        *reinterpret_cast<sycl::vec<uint32_t, 4> const*>(
                            b_expert + static_cast<int64_t>(ng) * K + kg);
                  } else {
#pragma unroll
                    for (int j = 0; j < kStageVec; ++j) {
                      buf[j] = (ng < N && kg + j < K)
                                   ? b_expert[static_cast<int64_t>(ng) * K + kg + j]
                                   : bf16_zero();
                    }
                  }
#pragma unroll
                  for (int j = 0; j < kStageVec; ++j) {
                    b_tile[(kv * kStageVec + j) * kBRow + nn] = buf[j];
                  }
                }
              } else {
                for (int idx = lid; idx < kRowsPerPass * TileK; idx += kItems) {
                  int r = idx / TileK;
                  int kk = idx - r * TileK;
                  int row = row_base + r;
                  int kg = k0 + kk;
                  a_tile[idx] = (row < block_end && kg < K)
                                    ? A[static_cast<int64_t>(row) * K + kg]
                                    : bf16_zero();
                }
                // Global reads stay contiguous in k (B is b[E, N, K]) while the
                // SLM tile is written k-major so the inner loop can read kRN
                // consecutive n values per k.
                for (int idx = lid; idx < TileN * TileK; idx += kItems) {
                  int nn = idx / TileK;
                  int kk = idx - nn * TileK;
                  int ng = n_base + nn;
                  int kg = k0 + kk;
                  b_tile[kk * kBRow + nn] = (ng < N && kg < K)
                                                ? b_expert[static_cast<int64_t>(ng) * K + kg]
                                                : bf16_zero();
                }
              }
              item.barrier(sycl::access::fence_space::local_space);

              int a_base = r_slot * TileK;
              int b_base = c_slot * kRN;
              for (int kk = 0; kk < TileK; ++kk) {
                float a = bf16_to_float(a_tile[a_base + kk]);
                int b_row = kk * kBRow + b_base;
#pragma unroll
                for (int j = 0; j < kRN; ++j) {
                  acc[j] += a * bf16_to_float(b_tile[b_row + j]);
                }
              }
            }

            int row = row_base + r_slot;
            if (row >= block_end) {
              continue;
            }
            Element* c_row = C + static_cast<int64_t>(row) * N;
#pragma unroll
            for (int j = 0; j < kRN; ++j) {
              int col = n_base + c_slot * kRN + j;
              if (col < N) {
                c_row[col] = float_to_bf16(acc[j]);
              }
            }
          }
        });
  });
}

// SLM required by a config, so an unsupported device fails loudly up front.
template <int TileN, int TileK>
constexpr std::size_t grouped_gemm_slm_bytes() {
  return (static_cast<std::size_t>(16) * TileK +
          static_cast<std::size_t>(TileK) * (TileN + 2)) *
         sizeof(Element);
}

inline sycl::event launch_grouped_gemm(
    sycl::queue& queue,
    Element const* A,
    Element const* B,
    Element* C,
    int32_t const* counts,
    int32_t const* token_offs,
    int32_t const* block_offs,
    int32_t const* schedule,
    int num_experts,
    int N,
    int K,
    int max_num_blocks,
    int block_m) {
  // The vectorized staging path needs every row of A and B to start 16-byte
  // aligned, i.e. K divisible by kStageVec. Real Inkling shapes always are; the
  // quick suite deliberately includes shapes that are not.
  bool vec_stage = (K % kStageVec == 0);
  if (block_m == kSmallMBlockSizeM) {
    // Decode / small-M config: BLOCK_M 16, BLOCK_N 128, BLOCK_K 128.
    if (vec_stage) {
      return launch_grouped_gemm_impl<kSmallMBlockSizeM, 128, 128, true>(
          queue, A, B, C, counts, token_offs, block_offs, schedule, num_experts, N, K,
          max_num_blocks);
    }
    return launch_grouped_gemm_impl<kSmallMBlockSizeM, 128, 128, false>(
        queue, A, B, C, counts, token_offs, block_offs, schedule, num_experts, N, K,
        max_num_blocks);
  }
  // Prefill config: BLOCK_M 128, BLOCK_N 256, BLOCK_K 64.
  if (vec_stage) {
    return launch_grouped_gemm_impl<kBlockSizeM, 256, 64, true>(
        queue, A, B, C, counts, token_offs, block_offs, schedule, num_experts, N, K,
        max_num_blocks);
  }
  return launch_grouped_gemm_impl<kBlockSizeM, 256, 64, false>(
      queue, A, B, C, counts, token_offs, block_offs, schedule, num_experts, N, K,
      max_num_blocks);
}

// ===========================================================================
// Stage 6: post_reorder -- scatter back and reduce the top_k contributions.
// ===========================================================================
class PostReorderKernel;

inline sycl::event launch_post_reorder(
    sycl::queue& queue,
    Element const* down_output,
    Element* output,
    int32_t const* src2dst,
    float const* topk_weights,
    int tokens,
    int top_k,
    int hidden) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<PostReorderKernel>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(tokens) * kWgSize),
                          sycl::range<1>(kWgSize)),
        [=](sycl::nd_item<1> item) {
          int token = static_cast<int>(item.get_group(0));
          int lid = static_cast<int>(item.get_local_id(0));
          int32_t const* map = src2dst + static_cast<int64_t>(token) * top_k;
          float const* weights = topk_weights + static_cast<int64_t>(token) * top_k;
          Element* out = output + static_cast<int64_t>(token) * hidden;
          for (int i = lid; i < hidden; i += kWgSize) {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) {
              Element const* src = down_output + static_cast<int64_t>(map[k]) * hidden;
              sum += bf16_to_float(src[i]) * weights[k];
            }
            out[i] = float_to_bf16(sum);
          }
        });
  });
}

// ===========================================================================
// Options / cases
// ===========================================================================
struct Options {
  std::string suite = "quick";
  std::string shape;
  std::string dtype = "bf16";
  PreprocessPath path = PreprocessPath::kAuto;
  int iterations = 5;
  int warmup = 2;
  bool verify = true;
  bool benchmark = true;
  bool interleaved = true;
  double perf_threshold_scale = 1.0;
  double mem_budget_gb = 12.0;
  bool help = false;
};

struct Case {
  std::string name;
  int tokens = 1;
  int top_k = 6;
  int num_experts = 256;
  int hidden = 1536;       // H
  int intermediate = 384;  // I / P (per-rank routed intermediate size)
  // Gates are report-only (0.0). A shared BMG card plus a correctness-first
  // grouped GEMM makes any number here a CI flake risk, and nothing about this
  // pipeline has a calibrated target yet -- see README.md and example 17, which
  // ships the same report-only convention.
  double target_gbps = 0.0;
  double target_tops = 0.0;
  bool verify_gemm = true;  // sampled CPU grouped-GEMM reference
};

// One timed stage. Bytes/ops are the *work* of the stage; rates are always
// derived from the accumulated ms so the printed rate and the printed avg_ms
// stay mutually consistent (averaging per-iteration rates does not).
struct StageTime {
  std::string name;
  double ms = 0.0;
  double bytes = 0.0;  // 0 => no GB/s reported
  double ops = 0.0;    // 0 => no TOPS reported
};

inline bool parse_path(std::string const& text, PreprocessPath& path) {
  if (text == "auto") {
    path = PreprocessPath::kAuto;
    return true;
  }
  if (text == "fused") {
    path = PreprocessPath::kFused;
    return true;
  }
  if (text == "sort") {
    path = PreprocessPath::kSort;
    return true;
  }
  return false;
}

inline bool parse_shape(std::string const& text, Case& cfg) {
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
    try {
      if (key == "name") {
        cfg.name = value;
      } else if (key == "tokens" || key == "T") {
        cfg.tokens = std::stoi(value);
      } else if (key == "topk" || key == "top_k") {
        cfg.top_k = std::stoi(value);
      } else if (key == "experts" || key == "E") {
        cfg.num_experts = std::stoi(value);
      } else if (key == "hidden" || key == "H") {
        cfg.hidden = std::stoi(value);
      } else if (key == "intermediate" || key == "I") {
        cfg.intermediate = std::stoi(value);
      } else if (key == "target_gbps") {
        cfg.target_gbps = std::stod(value);
      } else if (key == "target_tops") {
        cfg.target_tops = std::stod(value);
      } else if (key == "verify_gemm") {
        cfg.verify_gemm = std::stoi(value) != 0;
      } else {
        return false;
      }
    } catch (std::exception const&) {
      return false;
    }
  }
  return true;
}

inline void validate_case(Case const& cfg) {
  if (cfg.tokens <= 0 || cfg.top_k <= 0 || cfg.num_experts <= 0 || cfg.hidden <= 0 ||
      cfg.intermediate <= 0) {
    throw std::invalid_argument("case has a non-positive shape");
  }
  if (cfg.top_k > cfg.num_experts) {
    throw std::invalid_argument("top_k must be <= num_experts");
  }
  if (static_cast<int64_t>(cfg.tokens) * cfg.top_k > 0x7fffffffLL / 4) {
    throw std::invalid_argument("tokens * top_k overflows the int32 index domain");
  }
  // The sort path stores the expert id as int16 (mirroring the model's
  // topk_ids.to(torch.int16)) and the block schedule packs it into the low 16
  // bits, so the id must fit in a *signed* 16-bit value.
  if (cfg.num_experts > 0x7fff) {
    throw std::invalid_argument("num_experts must fit in a signed 16-bit value");
  }
}

// Quick suite: tiny shapes chosen to hit the branchy edges of the reorder
// machinery -- empty experts, tail blocks, the fused/sort path boundary, the
// BLOCK_M 16 -> 128 boundary, and a hidden size that is not a multiple of 8.
inline std::vector<Case> quick_suite() {
  return {
      //                        T   K   E    H    I
      {"tiny_single_token",     1,  2,  8,  64,  32},
      {"tiny_empty_experts",    3,  1, 16,  64,  32},
      {"tiny_tail_blocks",     17,  3,  8, 128,  64},
      {"odd_hidden_no_vec8",    9,  2,  8,  70,  34},
      // I_p = 48 is a real TP=8 shard and is NOT a multiple of BLOCK_K (128),
      // so GEMM2's k tail exercises the guarded staging path.
      {"ip_tail_k_lt_blockk",  40,  3,  8, 128,  48},
      // n = 1024 exactly: the largest shape that takes the fused single-CTA path.
      {"fused_boundary_n1024", 512, 2, 32, 128,  64},
      // n = 1026: the first shape on the sort path, still BLOCK_M 16.
      {"sort_boundary_n1026",  513, 2, 32, 128,  64},
      // n = 6144 / 6150 straddle the BLOCK_M 16 -> BLOCK_M 128 boundary.
      {"blockm16_max_n6144",  1024, 6, 32,  64,  32},
      {"blockm128_min_n6150", 1025, 6, 32,  64,  32},
  };
}

// Inkling suite: E = 256, top_k = 6, the three shipped hidden sizes and both
// routed intermediate sizes at every TP degree (I_p = I / P). The grouped-GEMM
// reference is sampled, so it stays affordable at every T; the reorder and
// metadata arrays are always verified exactly and in full.
inline std::vector<Case> inkling_suite() {
  std::vector<Case> cases;
  int const hiddens[] = {768, 1536, 6144};
  // 1 = decode, 3 and 9 = the two shipped draft_token_num values, 144 and 512 =
  // mid-band, 4096 = chunked prefill, 16384 = max_prefill_tokens.
  int const tokens[] = {1, 3, 9, 144, 512, 4096, 16384};
  struct IShard {
    char const* label;
    int intermediate;
  };
  // I = 384 (checkpoint) and I = 3072 (production), sharded over P = 1/2/4/8.
  IShard const shards[] = {
      {"i384_p1", 384},   {"i384_p2", 192},   {"i384_p4", 96},   {"i384_p8", 48},
      {"i3072_p1", 3072}, {"i3072_p2", 1536}, {"i3072_p4", 768}, {"i3072_p8", 384},
  };
  for (int hidden : hiddens) {
    for (IShard const& shard : shards) {
      for (int t : tokens) {
        Case cfg;
        cfg.name = "h" + std::to_string(hidden) + "_" + shard.label + "_t" + std::to_string(t);
        cfg.tokens = t;
        cfg.top_k = 6;
        cfg.num_experts = 256;
        cfg.hidden = hidden;
        cfg.intermediate = shard.intermediate;
        cases.push_back(cfg);
      }
    }
  }
  // T = 170 / 171 straddle the fused-preprocess window at production E/top_k
  // (n = 1020 / 1026), which the T grid above steps over. T = 1024 / 1025
  // straddle the BLOCK_M 16 -> 128 boundary (n = 6144 / 6150).
  for (int t : {170, 171, 1024, 1025}) {
    Case cfg;
    cfg.name = "h1536_i3072_p8_boundary_t" + std::to_string(t);
    cfg.tokens = t;
    cfg.top_k = 6;
    cfg.num_experts = 256;
    cfg.hidden = 1536;
    cfg.intermediate = 384;
    cases.push_back(cfg);
  }
  return cases;
}

// Perf suite: the decode / draft / prefill bands at both shipped hidden sizes,
// verification off. Deliberately small: the correctness-first grouped GEMM
// dominates the wall clock at prefill row counts.
inline std::vector<Case> perf_suite() {
  std::vector<Case> cases;
  int const hiddens[] = {1536, 6144};
  int const tokens[] = {1, 9, 144, 4096, 16384};
  // I_p: TP=8 of I=3072 is numerically the same shard as TP=1 of I=384.
  int const shards[] = {384, 3072};
  for (int hidden : hiddens) {
    for (int ip : shards) {
      for (int t : tokens) {
        Case cfg;
        cfg.name = "perf_h" + std::to_string(hidden) + "_ip" + std::to_string(ip) + "_t" +
                   std::to_string(t);
        cfg.tokens = t;
        cfg.top_k = 6;
        cfg.num_experts = 256;
        cfg.hidden = hidden;
        cfg.intermediate = ip;
        cfg.verify_gemm = false;
        cases.push_back(cfg);
      }
    }
  }
  return cases;
}

inline std::vector<Case> make_suite(std::string const& suite) {
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

// ===========================================================================
// Host reference for the whole preprocess (run_moe_preprocess)
// ===========================================================================
struct HostMeta {
  std::vector<int32_t> src2dst;
  std::vector<int32_t> reorder_topk_ids;
  std::vector<int64_t> reorder_ids;
  std::vector<int32_t> num_tokens_per_expert;
  std::vector<int32_t> expert_token_offs;
  std::vector<int32_t> expert_block_offs;
  std::vector<int32_t> expert_block_schedule;
  int total_blocks = 0;
};

inline HostMeta reference_preprocess(
    std::vector<int32_t> const& topk_ids, int num_experts, int block_m, int max_num_blocks) {
  int n = static_cast<int>(topk_ids.size());
  HostMeta meta;
  meta.reorder_ids.resize(n);
  std::iota(meta.reorder_ids.begin(), meta.reorder_ids.end(), int64_t{0});
  std::stable_sort(meta.reorder_ids.begin(), meta.reorder_ids.end(), [&](int64_t a, int64_t b) {
    return topk_ids[static_cast<std::size_t>(a)] < topk_ids[static_cast<std::size_t>(b)];
  });
  meta.reorder_topk_ids.resize(n);
  meta.src2dst.resize(n);
  for (int dst = 0; dst < n; ++dst) {
    int64_t src = meta.reorder_ids[dst];
    meta.reorder_topk_ids[dst] = topk_ids[static_cast<std::size_t>(src)];
    meta.src2dst[static_cast<std::size_t>(src)] = dst;
  }

  meta.expert_token_offs.assign(num_experts + 1, 0);
  meta.num_tokens_per_expert.assign(num_experts, 0);
  for (int id : meta.reorder_topk_ids) {
    ++meta.num_tokens_per_expert[id];
  }
  for (int e = 0; e < num_experts; ++e) {
    meta.expert_token_offs[e + 1] = meta.expert_token_offs[e] + meta.num_tokens_per_expert[e];
  }

  meta.expert_block_offs.assign(num_experts + 1, 0);
  int running = 0;
  for (int e = 0; e < num_experts; ++e) {
    meta.expert_block_offs[e] = running;
    running += ceil_div(meta.num_tokens_per_expert[e], block_m);
  }
  meta.expert_block_offs[num_experts] = running;
  meta.total_blocks = running;

  meta.expert_block_schedule.assign(max_num_blocks, -1);
  for (int e = 0; e < num_experts; ++e) {
    int num_blocks = ceil_div(meta.num_tokens_per_expert[e], block_m);
    for (int b = 0; b < num_blocks; ++b) {
      meta.expert_block_schedule[meta.expert_block_offs[e] + b] = (b << 16) | e;
    }
  }
  return meta;
}

template <typename T>
bool check_exact(
    char const* label, std::vector<T> const& got, std::vector<T> const& expected,
    std::string& message) {
  if (got.size() != expected.size()) {
    message = std::string(label) + " size mismatch";
    return false;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      std::stringstream ss;
      ss << label << " mismatch at " << i << " got=" << static_cast<int64_t>(got[i])
         << " expected=" << static_cast<int64_t>(expected[i]);
      message = ss.str();
      return false;
    }
  }
  return true;
}

struct FloatCheck {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::size_t index = 0;
  std::size_t compared = 0;
};

// bf16 outputs: one bf16 rounding is already 2^-9 relative, and the device may
// contract a*b+c into an FMA where the host reference does not, so the right
// gate is relative, a few bf16 ULPs wide, with an absolute floor for values
// that cancelled to near zero.
inline void accumulate_check(FloatCheck& result, double got, double expected, double rel_tol,
                             double abs_floor, std::size_t index) {
  double abs_err = std::abs(got - expected);
  double rel_err = abs_err / std::max(std::abs(expected), 1.0e-30);
  ++result.compared;
  if (abs_err > result.max_abs) {
    result.max_abs = abs_err;
    result.max_rel = rel_err;
    result.index = index;
  }
  if (abs_err > abs_floor && rel_err > rel_tol) {
    result.passed = false;
  }
}

// ===========================================================================
// Case plan
// ===========================================================================
struct Plan {
  int n = 0;  // routed rows = T * top_k
  int block_m = 0;
  int block_n = 0;
  int block_k = 0;
  int max_num_blocks = 0;
  bool fused = false;          // fused single-work-group preprocess
  bool fused_slm_over = false; // fused path would exceed the device's SLM
  int gemm1_n = 0;             // 2 * I_p
  int gemm1_k = 0;     // H
  int gemm2_n = 0;     // H
  int gemm2_k = 0;     // I_p
};

inline Plan make_plan(Case const& cfg, PreprocessPath path, std::size_t max_slm) {
  Plan plan;
  plan.n = cfg.tokens * cfg.top_k;
  plan.block_m = select_grouped_gemm_block_m(plan.n);
  if (plan.block_m == kSmallMBlockSizeM) {
    plan.block_n = 128;
    plan.block_k = 128;
  } else {
    plan.block_n = 256;
    plan.block_k = 64;
  }
  plan.max_num_blocks = get_max_num_blocks(plan.n, plan.block_m, cfg.num_experts);
  bool fused_possible = plan.n <= kFusedPreprocessWinTokens;
  // The fused kernel's per-expert SLM arrays scale with num_experts (~12 B
  // each), so a very large E does not fit even though the model's E=256 does.
  plan.fused_slm_over = fused_possible &&
                        fused_preprocess_slm_bytes(plan.n, cfg.num_experts) > max_slm;
  plan.fused =
      (path == PreprocessPath::kSort) ? false : (fused_possible && !plan.fused_slm_over);
  plan.gemm1_n = 2 * cfg.intermediate;
  plan.gemm1_k = cfg.hidden;
  plan.gemm2_n = cfg.hidden;
  plan.gemm2_k = cfg.intermediate;
  return plan;
}

inline double plan_bytes_gib(Case const& cfg, Plan const& plan) {
  double e = static_cast<double>(sizeof(Element));
  double n = static_cast<double>(plan.n);
  double t = static_cast<double>(cfg.tokens);
  double bytes = 0.0;
  bytes += t * cfg.hidden * e;                                            // hidden_states
  bytes += n * cfg.hidden * e;                                            // gateup_input
  bytes += n * plan.gemm1_n * e;                                          // gateup_output
  bytes += n * cfg.intermediate * e;                                      // down_input
  bytes += n * cfg.hidden * e;                                            // down_output
  bytes += t * cfg.hidden * e;                                            // output
  bytes += static_cast<double>(cfg.num_experts) * plan.gemm1_n * cfg.hidden * e;      // w13
  bytes += static_cast<double>(cfg.num_experts) * cfg.hidden * cfg.intermediate * e;  // w2
  return bytes / (1024.0 * 1024.0 * 1024.0);
}

// Random but plausible routing: each token picks top_k distinct experts.
inline std::vector<int32_t> make_topk_ids(Case const& cfg, uint32_t seed) {
  std::vector<int32_t> ids(static_cast<std::size_t>(cfg.tokens) * cfg.top_k);
  std::mt19937 gen(seed);
  std::vector<int32_t> pool(cfg.num_experts);
  std::iota(pool.begin(), pool.end(), 0);
  for (int t = 0; t < cfg.tokens; ++t) {
    for (int k = 0; k < cfg.top_k; ++k) {
      std::uniform_int_distribution<int> dist(k, cfg.num_experts - 1);
      std::swap(pool[k], pool[dist(gen)]);
      ids[static_cast<std::size_t>(t) * cfg.top_k + k] = pool[k];
    }
  }
  return ids;
}

inline std::vector<float> make_topk_weights(Case const& cfg, uint32_t seed) {
  std::vector<float> weights(static_cast<std::size_t>(cfg.tokens) * cfg.top_k);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.05f, 1.0f);
  for (int t = 0; t < cfg.tokens; ++t) {
    float sum = 0.0f;
    for (int k = 0; k < cfg.top_k; ++k) {
      float v = dist(gen);
      weights[static_cast<std::size_t>(t) * cfg.top_k + k] = v;
      sum += v;
    }
    for (int k = 0; k < cfg.top_k; ++k) {
      weights[static_cast<std::size_t>(t) * cfg.top_k + k] /= sum;
    }
  }
  return weights;
}

// ===========================================================================
// Case runner
// ===========================================================================
inline bool run_case(sycl::queue& queue, Case cfg, Options const& options, std::size_t max_slm) {
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
  validate_case(cfg);
  Plan plan = make_plan(cfg, options.path, max_slm);

  if (options.path == PreprocessPath::kFused && plan.n > kFusedPreprocessWinTokens) {
    std::cout << "  " << cfg.name << " skip=fused-path-requires-n<="
              << kFusedPreprocessWinTokens << " (n=" << plan.n << ")\n";
    return true;
  }
  if (options.path == PreprocessPath::kFused && plan.fused_slm_over) {
    std::cerr << "  " << cfg.name << " error: the fused preprocess needs "
              << fused_preprocess_slm_bytes(plan.n, cfg.num_experts)
              << " B of SLM for E=" << cfg.num_experts << ", device reports " << max_slm
              << " B\n";
    return false;
  }
  if (plan.fused_slm_over) {
    std::cout << "  " << cfg.name << " note: E=" << cfg.num_experts
              << " exceeds the fused preprocess SLM budget; using the sort path\n";
  }

  double needed_gib = plan_bytes_gib(cfg, plan);
  std::cout << "  " << cfg.name << " T=" << cfg.tokens << " topk=" << cfg.top_k
            << " E=" << cfg.num_experts << " H=" << cfg.hidden << " I_p=" << cfg.intermediate
            << " n=" << plan.n << " BLOCK_M=" << plan.block_m << " BLOCK_N=" << plan.block_n
            << " BLOCK_K=" << plan.block_k << " blocks<=" << plan.max_num_blocks
            << " path=" << (plan.fused ? "fused" : "sort") << std::fixed << std::setprecision(2)
            << " device_GiB=" << needed_gib << std::defaultfloat << "\n";

  if (needed_gib > options.mem_budget_gb) {
    // Not a failure: I_p = 3072 with H = 6144 and E = 256 needs ~29 GiB of
    // weights, which is exactly why the model shards it over tensor parallel.
    std::cout << "    skip=OVER_MEM_BUDGET budget_GiB=" << options.mem_budget_gb
              << " (raise with --mem-budget-gb)\n";
    return true;
  }

  std::size_t slm_needed = (plan.block_m == kSmallMBlockSizeM)
                               ? grouped_gemm_slm_bytes<128, 128>()
                               : grouped_gemm_slm_bytes<256, 64>();
  if (slm_needed > max_slm) {
    std::cerr << "    error: grouped GEMM needs " << slm_needed << " B of SLM, device reports "
              << max_slm << " B\n";
    return false;
  }

  int n = plan.n;
  int T = cfg.tokens;
  int E = cfg.num_experts;
  int H = cfg.hidden;
  int Ip = cfg.intermediate;
  int num_chunks = ceil_div(n, kSortChunk);

  std::vector<int32_t> host_topk_ids = make_topk_ids(cfg, 1234u);
  std::vector<float> host_topk_weights = make_topk_weights(cfg, 5678u);

  constexpr uint32_t kHiddenSeed = 11u;
  constexpr uint32_t kW13Seed = 23u;
  constexpr uint32_t kW2Seed = 37u;
  constexpr float kHiddenScale = 1.0f;
  constexpr float kWeightScale = 0.05f;

  // Every device allocation happens inside this scope so an out-of-memory case
  // is reported as a skip rather than taking the whole run down (the card is
  // shared, so the free-memory headroom is not ours to predict).
  DeviceBuffer<int32_t> d_topk_ids;
  DeviceBuffer<float> d_topk_weights;
  DeviceBuffer<int32_t> d_src2dst;
  DeviceBuffer<int32_t> d_src2dst_alt;
  DeviceBuffer<int32_t> d_reorder_i32;
  DeviceBuffer<int16_t> d_reorder_i16;
  DeviceBuffer<int64_t> d_reorder_ids;
  DeviceBuffer<int32_t> d_counts;
  DeviceBuffer<int32_t> d_token_offs;
  DeviceBuffer<int32_t> d_block_offs;
  DeviceBuffer<int32_t> d_schedule;
  DeviceBuffer<int32_t> d_counts_alt;
  DeviceBuffer<int32_t> d_token_offs_alt;
  DeviceBuffer<int32_t> d_block_offs_alt;
  DeviceBuffer<int32_t> d_schedule_alt;
  DeviceBuffer<int32_t> d_block_hist;
  DeviceBuffer<int32_t> d_block_rel;
  DeviceBuffer<int32_t> d_expert_total;
  DeviceBuffer<int32_t> d_expert_base;
  DeviceBuffer<Element> d_hidden;
  DeviceBuffer<Element> d_gateup_input;
  DeviceBuffer<Element> d_gateup_output;
  DeviceBuffer<Element> d_down_input;
  DeviceBuffer<Element> d_down_output;
  DeviceBuffer<Element> d_output;
  DeviceBuffer<Element> d_w13;
  DeviceBuffer<Element> d_w2;
  try {
    d_topk_ids = DeviceBuffer<int32_t>(queue, host_topk_ids.size());
    d_topk_weights = DeviceBuffer<float>(queue, host_topk_weights.size());
    d_src2dst = DeviceBuffer<int32_t>(queue, n);
    d_src2dst_alt = DeviceBuffer<int32_t>(queue, n);
    d_reorder_i32 = DeviceBuffer<int32_t>(queue, n);
    d_reorder_i16 = DeviceBuffer<int16_t>(queue, n);
    d_reorder_ids = DeviceBuffer<int64_t>(queue, n);
    d_counts = DeviceBuffer<int32_t>(queue, E);
    d_token_offs = DeviceBuffer<int32_t>(queue, E + 1);
    d_block_offs = DeviceBuffer<int32_t>(queue, E + 1);
    d_schedule = DeviceBuffer<int32_t>(queue, plan.max_num_blocks);
    d_counts_alt = DeviceBuffer<int32_t>(queue, E);
    d_token_offs_alt = DeviceBuffer<int32_t>(queue, E + 1);
    d_block_offs_alt = DeviceBuffer<int32_t>(queue, E + 1);
    d_schedule_alt = DeviceBuffer<int32_t>(queue, plan.max_num_blocks);
    d_block_hist = DeviceBuffer<int32_t>(queue, static_cast<std::size_t>(num_chunks) * E);
    d_block_rel = DeviceBuffer<int32_t>(queue, static_cast<std::size_t>(num_chunks) * E);
    d_expert_total = DeviceBuffer<int32_t>(queue, E);
    d_expert_base = DeviceBuffer<int32_t>(queue, E);
    d_hidden = DeviceBuffer<Element>(queue, static_cast<std::size_t>(T) * H);
    d_gateup_input = DeviceBuffer<Element>(queue, static_cast<std::size_t>(n) * H);
    d_gateup_output = DeviceBuffer<Element>(queue, static_cast<std::size_t>(n) * plan.gemm1_n);
    d_down_input = DeviceBuffer<Element>(queue, static_cast<std::size_t>(n) * Ip);
    d_down_output = DeviceBuffer<Element>(queue, static_cast<std::size_t>(n) * H);
    d_output = DeviceBuffer<Element>(queue, static_cast<std::size_t>(T) * H);
    d_w13 = DeviceBuffer<Element>(queue, static_cast<std::size_t>(E) * plan.gemm1_n * H);
    d_w2 = DeviceBuffer<Element>(queue, static_cast<std::size_t>(E) * H * Ip);
  } catch (std::bad_alloc const&) {
    std::cout << "    skip=DEVICE_ALLOC_FAILED (needed about " << needed_gib << " GiB)\n";
    return true;
  }

  d_topk_ids.copy_from(host_topk_ids);
  d_topk_weights.copy_from(host_topk_weights);

  fill_hash(queue, d_hidden.get(), d_hidden.count, kHiddenSeed, kHiddenScale);
  fill_hash(queue, d_w13.get(), d_w13.count, kW13Seed, kWeightScale);
  fill_hash(queue, d_w2.get(), d_w2.count, kW2Seed, kWeightScale).wait();

  MetaBuffers meta_buf;
  meta_buf.num_tokens_per_expert = d_counts.get();
  meta_buf.expert_token_offs = d_token_offs.get();
  meta_buf.expert_block_offs = d_block_offs.get();
  meta_buf.expert_block_schedule = d_schedule.get();

  // --- one full pipeline pass, stage by stage ---
  // Events are collected and only queried after the queue drains: reading
  // profiling info on an in-flight event blocks the host, which would insert a
  // round-trip between every stage and make the timed runs behave differently
  // from the untimed warmup.
  std::vector<sycl::event> stage_events;
  auto run_pipeline = [&](std::vector<StageTime>* times) {
    stage_events.clear();
    auto record = [&](char const* name, sycl::event const& ev, double bytes, double ops) {
      if (times == nullptr) {
        return;
      }
      StageTime st;
      st.name = name;
      st.bytes = bytes;
      st.ops = ops;
      times->push_back(st);
      stage_events.push_back(ev);
    };

    if (plan.fused) {
      // Stage 1 produces everything; run_moe_preprocess then skips stage 3.
      auto ev = launch_fused_preprocess(queue, d_topk_ids.get(), d_src2dst.get(),
                                        d_reorder_i32.get(), meta_buf, n, E, plan.block_m,
                                        plan.max_num_blocks);
      record("1_fused_moe_preprocess", ev, 0.0, 0.0);
    } else {
      // Stage 2: stable sort + get_src2dst.
      auto e1 =
          launch_sort_histogram(queue, d_topk_ids.get(), d_block_hist.get(), n, E, num_chunks);
      record("2a_sort_histogram", e1, 0.0, 0.0);
      auto e2 = launch_sort_block_scan(queue, d_block_hist.get(), d_block_rel.get(),
                                      d_expert_total.get(), E, num_chunks);
      record("2b_sort_block_scan", e2, 0.0, 0.0);
      auto e3 = launch_sort_expert_base(queue, d_expert_total.get(), d_expert_base.get(), E);
      record("2c_sort_expert_base", e3, 0.0, 0.0);
      auto e4 = launch_sort_scatter(queue, d_topk_ids.get(), d_block_rel.get(),
                                    d_expert_base.get(), d_reorder_i16.get(),
                                    d_reorder_ids.get(), n, E, num_chunks);
      record("2d_sort_scatter", e4, 0.0, 0.0);
      auto e5 = launch_get_src2dst(queue, d_reorder_ids.get(), d_src2dst.get(), n);
      record("2e_get_src2dst", e5, 0.0, 0.0);

      // Stage 3: compute_grouped_gemm_metadata (sort path only, per the model).
      auto e6 = launch_expert_offsets<int16_t>(queue, d_reorder_i16.get(), d_token_offs.get(),
                                               n, E);
      record("3a_expert_offsets", e6, 0.0, 0.0);
      auto e7 = launch_expert_counts(queue, d_token_offs.get(), d_counts.get(), E);
      record("3b_expert_counts", e7, 0.0, 0.0);
      auto e8 = launch_memset_block_metadata(queue, d_counts.get(), d_block_offs.get(),
                                             d_schedule.get(), plan.max_num_blocks, E,
                                             plan.block_m);
      record("3c_memset_block_metadata", e8, 0.0, 0.0);
      auto e9 = launch_block_metadata(queue, d_counts.get(), d_block_offs.get(),
                                      d_schedule.get(), E, plan.block_m);
      record("3d_block_metadata", e9, 0.0, 0.0);
    }

    // Stage 4: pre_reorder
    {
      auto ev = launch_pre_reorder(queue, d_hidden.get(), d_gateup_input.get(), d_src2dst.get(),
                                   T, cfg.top_k, H);
      double bytes = (static_cast<double>(T) + n) * H * sizeof(Element);
      record("4_pre_reorder", ev, bytes, 0.0);
    }

    // Stage 5a: grouped GEMM 1
    {
      auto ev = launch_grouped_gemm(queue, d_gateup_input.get(), d_w13.get(),
                                    d_gateup_output.get(), d_counts.get(), d_token_offs.get(),
                                    d_block_offs.get(), d_schedule.get(), E, plan.gemm1_n,
                                    plan.gemm1_k, plan.max_num_blocks, plan.block_m);
      double ops = 2.0 * n * plan.gemm1_n * plan.gemm1_k;
      record("5a_grouped_gemm1", ev, 0.0, ops);
    }

    // Stage 5b: silu_and_mul
    {
      auto ev = launch_silu_and_mul(queue, d_gateup_output.get(), d_down_input.get(), n, Ip,
                                    options.interleaved);
      double bytes = static_cast<double>(n) * 3.0 * Ip * sizeof(Element);
      record("5b_silu_and_mul", ev, bytes, 0.0);
    }

    // Stage 5c: grouped GEMM 2
    {
      auto ev = launch_grouped_gemm(queue, d_down_input.get(), d_w2.get(), d_down_output.get(),
                                    d_counts.get(), d_token_offs.get(), d_block_offs.get(),
                                    d_schedule.get(), E, plan.gemm2_n, plan.gemm2_k,
                                    plan.max_num_blocks, plan.block_m);
      double ops = 2.0 * n * plan.gemm2_n * plan.gemm2_k;
      record("5c_grouped_gemm2", ev, 0.0, ops);
    }

    // Stage 6: post_reorder
    {
      auto ev = launch_post_reorder(queue, d_down_output.get(), d_output.get(), d_src2dst.get(),
                                    d_topk_weights.get(), T, cfg.top_k, H);
      double bytes = (static_cast<double>(n) + T) * H * sizeof(Element);
      record("6_post_reorder", ev, bytes, 0.0);
    }

    queue.wait();
    if (times != nullptr) {
      for (std::size_t i = 0; i < times->size(); ++i) {
        (*times)[i].ms = event_ms(stage_events[i]);
      }
    }
  };

  bool passed = true;

  if (options.verify) {
    run_pipeline(nullptr);
    queue.wait();

    HostMeta meta = reference_preprocess(host_topk_ids, E, plan.block_m, plan.max_num_blocks);

    // --- metadata: exact, in full, on every case ---
    std::vector<int32_t> got_src2dst(n);
    std::vector<int32_t> got_counts(E);
    std::vector<int32_t> got_token_offs(E + 1);
    std::vector<int32_t> got_block_offs(E + 1);
    std::vector<int32_t> got_schedule(plan.max_num_blocks);
    d_src2dst.copy_to(got_src2dst);
    d_counts.copy_to(got_counts);
    d_token_offs.copy_to(got_token_offs);
    d_block_offs.copy_to(got_block_offs);
    d_schedule.copy_to(got_schedule);

    std::string message;
    bool meta_ok = check_exact("src2dst", got_src2dst, meta.src2dst, message);
    if (meta_ok) {
      if (plan.fused) {
        std::vector<int32_t> got_ids(n);
        d_reorder_i32.copy_to(got_ids);
        meta_ok = check_exact("reorder_topk_ids(i32)", got_ids, meta.reorder_topk_ids, message);
      } else {
        std::vector<int16_t> got_ids(n);
        std::vector<int16_t> expected_ids(n);
        for (int i = 0; i < n; ++i) {
          expected_ids[i] = static_cast<int16_t>(meta.reorder_topk_ids[i]);
        }
        d_reorder_i16.copy_to(got_ids);
        meta_ok = check_exact("reorder_topk_ids(i16)", got_ids, expected_ids, message);
        if (meta_ok) {
          std::vector<int64_t> got_perm(n);
          d_reorder_ids.copy_to(got_perm);
          meta_ok = check_exact("reorder_ids", got_perm, meta.reorder_ids, message);
        }
      }
    }
    if (meta_ok) {
      meta_ok = check_exact("num_tokens_per_expert", got_counts, meta.num_tokens_per_expert,
                            message);
    }
    if (meta_ok) {
      meta_ok = check_exact("expert_token_offs", got_token_offs, meta.expert_token_offs, message);
    }
    if (meta_ok) {
      meta_ok = check_exact("expert_block_offs", got_block_offs, meta.expert_block_offs, message);
    }
    if (meta_ok) {
      meta_ok = check_exact("expert_block_schedule", got_schedule, meta.expert_block_schedule,
                            message);
    }
    if (!meta_ok) {
      std::cerr << "    verify=FAIL metadata: " << message << "\n";
      passed = false;
    } else {
      std::cout << "    verify_metadata=PASS total_blocks=" << meta.total_blocks << "\n";
    }

    std::vector<int> token_sample = sample_indices(T, kMaxVerifyRows);
    std::vector<int> row_sample = sample_indices(n, kMaxVerifyRows);

    // --- pre_reorder: bitwise (the triton kernel is a pure copy for bf16) ---
    {
      std::vector<int> dst_rows;
      dst_rows.reserve(token_sample.size() * cfg.top_k);
      for (int t : token_sample) {
        for (int k = 0; k < cfg.top_k; ++k) {
          dst_rows.push_back(meta.src2dst[static_cast<std::size_t>(t) * cfg.top_k + k]);
        }
      }
      std::vector<Element> got = d_gateup_input.read_rows(dst_rows, static_cast<std::size_t>(H));
      bool ok = true;
      std::size_t bad_row = 0;
      int bad_col = 0;
      for (std::size_t r = 0; r < dst_rows.size() && ok; ++r) {
        int token = token_sample[r / cfg.top_k];
        for (int i = 0; i < H; ++i) {
          Element expected =
              hash_bf16(static_cast<uint64_t>(token) * H + i, kHiddenSeed, kHiddenScale);
          if (got[r * static_cast<std::size_t>(H) + i].raw() != expected.raw()) {
            ok = false;
            bad_row = static_cast<std::size_t>(dst_rows[r]);
            bad_col = i;
            break;
          }
        }
      }
      if (!ok) {
        std::cerr << "    verify=FAIL pre_reorder row=" << bad_row << " col=" << bad_col << "\n";
        passed = false;
      } else {
        std::cout << "    verify_pre_reorder=PASS (bitwise, " << dst_rows.size() << " rows)\n";
      }
    }

    // --- silu_and_mul: reference taken from the device gateup output ---
    std::vector<Element> gateup_rows =
        d_gateup_output.read_rows(row_sample, static_cast<std::size_t>(plan.gemm1_n));
    {
      std::vector<Element> down_rows =
          d_down_input.read_rows(row_sample, static_cast<std::size_t>(Ip));
      double scale = 0.0;
      for (Element v : down_rows) {
        scale = std::max(scale, std::abs(static_cast<double>(bf16_to_float(v))));
      }
      double abs_floor = 1.0e-3 * std::max(scale, 1.0e-3);
      FloatCheck chk;
      for (std::size_t r = 0; r < row_sample.size(); ++r) {
        Element const* src = gateup_rows.data() + r * static_cast<std::size_t>(plan.gemm1_n);
        for (int col = 0; col < Ip; ++col) {
          float gate = options.interleaved ? bf16_to_float(src[2 * col]) : bf16_to_float(src[col]);
          float up = options.interleaved ? bf16_to_float(src[2 * col + 1])
                                         : bf16_to_float(src[col + Ip]);
          float expected = gate * (1.0f / (1.0f + std::exp(-gate))) * up;
          accumulate_check(chk,
                           static_cast<double>(bf16_to_float(
                               down_rows[r * static_cast<std::size_t>(Ip) + col])),
                           static_cast<double>(expected), 8.0e-3, abs_floor,
                           r * static_cast<std::size_t>(Ip) + col);
        }
      }
      if (!chk.passed) {
        std::cerr << "    verify=FAIL silu_and_mul max_abs=" << chk.max_abs
                  << " max_rel=" << chk.max_rel << " at " << chk.index << "\n";
        passed = false;
      } else {
        std::cout << "    verify_silu_and_mul=PASS max_rel=" << chk.max_rel << " ("
                  << chk.compared << " elements)\n";
      }
    }

    // --- post_reorder: reference taken from the device down output ---
    {
      std::vector<int> dst_rows;
      dst_rows.reserve(token_sample.size() * cfg.top_k);
      for (int t : token_sample) {
        for (int k = 0; k < cfg.top_k; ++k) {
          dst_rows.push_back(meta.src2dst[static_cast<std::size_t>(t) * cfg.top_k + k]);
        }
      }
      std::vector<Element> down_rows =
          d_down_output.read_rows(dst_rows, static_cast<std::size_t>(H));
      std::vector<Element> out_rows = d_output.read_rows(token_sample, static_cast<std::size_t>(H));
      double scale = 0.0;
      for (Element v : out_rows) {
        scale = std::max(scale, std::abs(static_cast<double>(bf16_to_float(v))));
      }
      double abs_floor = 1.0e-3 * std::max(scale, 1.0e-3);
      FloatCheck chk;
      for (std::size_t s = 0; s < token_sample.size(); ++s) {
        int t = token_sample[s];
        for (int i = 0; i < H; ++i) {
          float sum = 0.0f;
          for (int k = 0; k < cfg.top_k; ++k) {
            std::size_t row = s * static_cast<std::size_t>(cfg.top_k) + k;
            sum += bf16_to_float(down_rows[row * static_cast<std::size_t>(H) + i]) *
                   host_topk_weights[static_cast<std::size_t>(t) * cfg.top_k + k];
          }
          accumulate_check(
              chk,
              static_cast<double>(bf16_to_float(out_rows[s * static_cast<std::size_t>(H) + i])),
              static_cast<double>(sum), 8.0e-3, abs_floor, s * static_cast<std::size_t>(H) + i);
        }
      }
      if (!chk.passed) {
        std::cerr << "    verify=FAIL post_reorder max_abs=" << chk.max_abs
                  << " max_rel=" << chk.max_rel << " at " << chk.index << "\n";
        passed = false;
      } else {
        std::cout << "    verify_post_reorder=PASS max_rel=" << chk.max_rel << " ("
                  << chk.compared << " elements)\n";
      }
    }

    // --- grouped GEMM: sampled CPU reference for both GEMMs ---
    if (cfg.verify_gemm) {
      auto check_gemm = [&](char const* label, DeviceBuffer<Element> const& d_A,
                            DeviceBuffer<Element> const& d_C, uint32_t b_seed, int N, int K,
                            std::vector<Element> const* c_rows_cached) {
        std::vector<Element> a_rows = d_A.read_rows(row_sample, static_cast<std::size_t>(K));
        std::vector<Element> c_rows_local;
        std::vector<Element> const* c_rows = c_rows_cached;
        if (c_rows == nullptr) {
          c_rows_local = d_C.read_rows(row_sample, static_cast<std::size_t>(N));
          c_rows = &c_rows_local;
        }
        std::vector<int> col_sample = sample_indices(N, kMaxVerifyCols);
        double scale = 0.0;
        for (Element v : *c_rows) {
          scale = std::max(scale, std::abs(static_cast<double>(bf16_to_float(v))));
        }
        double abs_floor = 1.0e-3 * std::max(scale, 1.0e-3);
        FloatCheck chk;
        for (std::size_t r = 0; r < row_sample.size(); ++r) {
          int expert = meta.reorder_topk_ids[static_cast<std::size_t>(row_sample[r])];
          int64_t b_base = static_cast<int64_t>(expert) * N * K;
          Element const* a_row = a_rows.data() + r * static_cast<std::size_t>(K);
          for (int col : col_sample) {
            float acc = 0.0f;
            int64_t b_row = b_base + static_cast<int64_t>(col) * K;
            for (int k = 0; k < K; ++k) {
              acc += bf16_to_float(a_row[k]) *
                     bf16_to_float(hash_bf16(static_cast<uint64_t>(b_row + k), b_seed,
                                             kWeightScale));
            }
            accumulate_check(
                chk,
                static_cast<double>(bf16_to_float((*c_rows)[r * static_cast<std::size_t>(N) + col])),
                static_cast<double>(acc), 2.0e-2, abs_floor,
                r * static_cast<std::size_t>(N) + col);
          }
        }
        if (!chk.passed) {
          std::cerr << "    verify=FAIL " << label << " max_abs=" << chk.max_abs
                    << " max_rel=" << chk.max_rel << " at " << chk.index << "\n";
          return false;
        }
        std::cout << "    verify_" << label << "=PASS max_rel=" << chk.max_rel << " ("
                  << chk.compared << " outputs)\n";
        return true;
      };
      passed &= check_gemm("grouped_gemm1", d_gateup_input, d_gateup_output, kW13Seed,
                           plan.gemm1_n, plan.gemm1_k, &gateup_rows);
      passed &= check_gemm("grouped_gemm2", d_down_input, d_down_output, kW2Seed, plan.gemm2_n,
                           plan.gemm2_k, nullptr);
    } else {
      std::cout << "    verify_grouped_gemm=SKIP (disabled for this case)\n";
    }

    // --- cross-check: the two preprocess implementations must agree bitwise ---
    if (plan.fused && options.path == PreprocessPath::kAuto) {
      MetaBuffers alt;
      alt.num_tokens_per_expert = d_counts_alt.get();
      alt.expert_token_offs = d_token_offs_alt.get();
      alt.expert_block_offs = d_block_offs_alt.get();
      alt.expert_block_schedule = d_schedule_alt.get();
      launch_sort_histogram(queue, d_topk_ids.get(), d_block_hist.get(), n, E, num_chunks);
      launch_sort_block_scan(queue, d_block_hist.get(), d_block_rel.get(), d_expert_total.get(),
                             E, num_chunks);
      launch_sort_expert_base(queue, d_expert_total.get(), d_expert_base.get(), E);
      launch_sort_scatter(queue, d_topk_ids.get(), d_block_rel.get(), d_expert_base.get(),
                          d_reorder_i16.get(), d_reorder_ids.get(), n, E, num_chunks);
      launch_get_src2dst(queue, d_reorder_ids.get(), d_src2dst_alt.get(), n);
      launch_expert_offsets<int16_t>(queue, d_reorder_i16.get(), alt.expert_token_offs, n, E);
      launch_expert_counts(queue, alt.expert_token_offs, alt.num_tokens_per_expert, E);
      launch_memset_block_metadata(queue, alt.num_tokens_per_expert, alt.expert_block_offs,
                                   alt.expert_block_schedule, plan.max_num_blocks, E,
                                   plan.block_m);
      launch_block_metadata(queue, alt.num_tokens_per_expert, alt.expert_block_offs,
                            alt.expert_block_schedule, E, plan.block_m)
          .wait();

      std::vector<int32_t> alt_src2dst(n);
      std::vector<int16_t> alt_ids(n);
      std::vector<int32_t> alt_counts(E);
      std::vector<int32_t> alt_token_offs(E + 1);
      std::vector<int32_t> alt_block_offs(E + 1);
      std::vector<int32_t> alt_schedule(plan.max_num_blocks);
      d_src2dst_alt.copy_to(alt_src2dst);
      d_reorder_i16.copy_to(alt_ids);
      d_counts_alt.copy_to(alt_counts);
      d_token_offs_alt.copy_to(alt_token_offs);
      d_block_offs_alt.copy_to(alt_block_offs);
      d_schedule_alt.copy_to(alt_schedule);
      std::vector<int16_t> expected_ids(n);
      for (int i = 0; i < n; ++i) {
        expected_ids[i] = static_cast<int16_t>(meta.reorder_topk_ids[i]);
      }
      std::string cross;
      bool cross_ok = check_exact("src2dst", alt_src2dst, meta.src2dst, cross) &&
                      check_exact("reorder_topk_ids", alt_ids, expected_ids, cross) &&
                      check_exact("num_tokens_per_expert", alt_counts,
                                  meta.num_tokens_per_expert, cross) &&
                      check_exact("expert_token_offs", alt_token_offs, meta.expert_token_offs,
                                  cross) &&
                      check_exact("expert_block_offs", alt_block_offs, meta.expert_block_offs,
                                  cross) &&
                      check_exact("expert_block_schedule", alt_schedule,
                                  meta.expert_block_schedule, cross);
      if (!cross_ok) {
        std::cerr << "    verify=FAIL fused/sort cross-check: " << cross << "\n";
        passed = false;
      } else {
        std::cout << "    verify_fused_vs_sort=PASS (bit-identical)\n";
      }
    }
  }

  if (options.benchmark) {
    // BMG ramps 1200 -> 2400 MHz over roughly 2 s, so a handful of warmup
    // launches on a small shape would report the ramp instead of the kernel.
    auto warmup_start = std::chrono::steady_clock::now();
    int warmup_done = 0;
    while (warmup_done < options.warmup ||
           (warmup_done < kMaxWarmupIterations &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      warmup_start)
                    .count() < kClockRampMs)) {
      run_pipeline(nullptr);
      queue.wait();
      ++warmup_done;
    }

    std::vector<StageTime> accum;
    int iterations = std::max(options.iterations, 1);
    for (int i = 0; i < iterations; ++i) {
      std::vector<StageTime> times;
      run_pipeline(&times);
      queue.wait();
      if (accum.empty()) {
        accum = times;
      } else {
        if (accum.size() != times.size()) {
          throw std::runtime_error("stage count changed between iterations");
        }
        for (std::size_t s = 0; s < times.size(); ++s) {
          accum[s].ms += times[s].ms;
        }
      }
    }
    double total_ms = 0.0;
    std::vector<double> stage_gbps(accum.size(), 0.0);
    std::vector<double> stage_tops(accum.size(), 0.0);
    for (std::size_t s = 0; s < accum.size(); ++s) {
      StageTime& st = accum[s];
      st.ms /= iterations;
      total_ms += st.ms;
      if (st.bytes > 0.0 && st.ms > 0.0) {
        stage_gbps[s] = (st.bytes / kBytesPerGB) / (st.ms * 1.0e-3);
      }
      if (st.ops > 0.0 && st.ms > 0.0) {
        stage_tops[s] = (st.ops / kOpsPerTOP) / (st.ms * 1.0e-3);
      }
      std::cout << "    stage " << std::setw(26) << std::left << st.name << std::right
                << std::fixed << std::setprecision(4) << " avg_ms=" << std::setw(10) << st.ms;
      if (stage_gbps[s] > 0.0) {
        std::cout << std::setprecision(2) << " GBps=" << std::setw(8) << stage_gbps[s];
      }
      if (stage_tops[s] > 0.0) {
        std::cout << std::setprecision(3) << " TOPS=" << std::setw(8) << stage_tops[s];
      }
      std::cout << std::defaultfloat << "\n";
    }
    std::cout << std::fixed << std::setprecision(4)
              << "    pipeline_kernel_ms=" << total_ms << " warmup_iterations=" << warmup_done
              << std::defaultfloat << "\n";

    double gbps_target = cfg.target_gbps * options.perf_threshold_scale;
    double tops_target = cfg.target_tops * options.perf_threshold_scale;
    if (gbps_target > 0.0 || tops_target > 0.0) {
      for (std::size_t s = 0; s < accum.size(); ++s) {
        if (gbps_target > 0.0 && stage_gbps[s] > 0.0 && stage_gbps[s] < gbps_target) {
          std::cerr << "    perf=FAIL stage=" << accum[s].name << " GBps=" << stage_gbps[s]
                    << " target=" << gbps_target << "\n";
          passed = false;
        }
        if (tops_target > 0.0 && stage_tops[s] > 0.0 && stage_tops[s] < tops_target) {
          std::cerr << "    perf=FAIL stage=" << accum[s].name << " TOPS=" << stage_tops[s]
                    << " target=" << tops_target << "\n";
          passed = false;
        }
      }
    }
  }

  return passed;
}

inline void print_usage(char const* name) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=quick|inkling|perf    Built-in suite (default quick)\n"
      << "  --shape=T=<int>,topk=<int>,E=<int>,H=<int>,I=<int>[,verify_gemm=0|1]\n"
      << "  --dtype=bf16                  Element dtype (bf16 only; the Inkling routed path"
         " is bf16)\n"
      << "  --preprocess=auto|fused|sort  Force a preprocess path (default auto, following the"
         " model's n<=1024 rule)\n"
      << "  --interleaved=0|1             w13 gate/up interleaving"
         " (inference_moe_w13_interleaved, default 1)\n"
      << "  --iterations=<int>            Timed pipeline iterations (default 5)\n"
      << "  --warmup=<int>                Minimum warmup pipeline iterations (default 2; the"
         " loop also runs for at least the BMG clock ramp)\n"
      << "  --verify=0|1                  Run the CPU references (default 1)\n"
      << "  --benchmark=0|1               Run per-stage profiling-event timing (default 1)\n"
      << "  --perf-threshold-scale=<f>    Scale every perf gate (default 1.0)\n"
      << "  --mem-budget-gb=<f>           Skip cases needing more device memory (default 12)\n";
}

}  // namespace cutlass::examples::bmg_moe_reorder_grouped_gemm

int main(int argc, char const** argv) {
  namespace moe = cutlass::examples::bmg_moe_reorder_grouped_gemm;

  moe::Options options;
  std::string path_name = "auto";
  try {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) {
      moe::print_usage(argv[0]);
      return 0;
    }
    cmd.get_cmd_line_argument("suite", options.suite, std::string("quick"));
    cmd.get_cmd_line_argument("shape", options.shape, std::string(""));
    cmd.get_cmd_line_argument("dtype", options.dtype, std::string("bf16"));
    cmd.get_cmd_line_argument("preprocess", path_name, std::string("auto"));
    cmd.get_cmd_line_argument("iterations", options.iterations, 5);
    cmd.get_cmd_line_argument("warmup", options.warmup, 2);
    cmd.get_cmd_line_argument("perf-threshold-scale", options.perf_threshold_scale, 1.0);
    cmd.get_cmd_line_argument("mem-budget-gb", options.mem_budget_gb, 12.0);
    int verify_int = 1;
    int benchmark_int = 1;
    int interleaved_int = 1;
    cmd.get_cmd_line_argument("verify", verify_int, 1);
    cmd.get_cmd_line_argument("benchmark", benchmark_int, 1);
    cmd.get_cmd_line_argument("interleaved", interleaved_int, 1);
    options.verify = verify_int != 0;
    options.benchmark = benchmark_int != 0;
    options.interleaved = interleaved_int != 0;

    if (options.dtype != "bf16") {
      std::cerr << "Only --dtype=bf16 is supported: the Inkling routed-expert path this example"
                   " mirrors is bf16 (quantized layers take a different code path).\n";
      return -1;
    }
    if (!moe::parse_path(path_name, options.path)) {
      std::cerr << "Unknown --preprocess value: " << path_name << "\n";
      return -1;
    }
    if (options.iterations < 0 || options.warmup < 0) {
      std::cerr << "iterations and warmup must be non-negative\n";
      return -1;
    }
    if (options.perf_threshold_scale <= 0.0 || options.mem_budget_gb <= 0.0) {
      std::cerr << "perf-threshold-scale and mem-budget-gb must be positive\n";
      return -1;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<moe::Case> cases;
  if (!options.shape.empty()) {
    moe::Case cfg;
    cfg.name = "custom";
    if (!moe::parse_shape(options.shape, cfg)) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    cases.push_back(cfg);
  } else {
    cases = moe::make_suite(options.suite);
    if (cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = moe::make_queue();
    std::size_t max_slm =
        static_cast<std::size_t>(queue.get_device().get_info<sycl::info::device::local_mem_size>());
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>()
              << " local_mem_size=" << max_slm << "\n";
    std::cout << "25_bmg_moe_reorder_grouped_gemm: Inkling MoE dispatch pipeline"
                 " (preprocess -> sort/src2dst -> metadata -> pre_reorder -> grouped GEMM"
                 " -> post_reorder)\n";
    std::cout << "Suite=" << options.suite << " preprocess=" << moe::path_text(options.path)
              << " interleaved=" << (options.interleaved ? 1 : 0)
              << " iterations=" << options.iterations << " warmup=" << options.warmup
              << " verify=" << (options.verify ? 1 : 0)
              << " benchmark=" << (options.benchmark ? 1 : 0)
              << " mem_budget_gb=" << options.mem_budget_gb << "\n";

    bool all_passed = true;
    for (moe::Case const& cfg : cases) {
      // A single bad case is reported and the sweep continues: aborting would
      // throw away the results of every case already run.
      try {
        all_passed &= moe::run_case(queue, cfg, options, max_slm);
      } catch (std::exception const& e) {
        std::cerr << "  " << cfg.name << " error: " << e.what() << "\n";
        all_passed = false;
      }
    }
    if (!all_passed) {
      std::cerr << "FAILED\n";
    } else {
      std::cout << "PASSED\n";
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
