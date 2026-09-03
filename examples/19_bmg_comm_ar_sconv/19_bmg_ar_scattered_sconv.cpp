/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling 06.3 fused {reduce-scatter/all-gather -> causal SConv}.
 *
 * The upstream CUDA kernel reduces each rank-owned channel shard, applies
 * causal SConv locally, then multicasts post-conv shard values so every rank
 * sees the gathered [T, H] output. This SYCL example models that dataflow with
 * world-size partial buffers on one BMG device:
 *
 *   phase A/B in one kernel:
 *     xred[t,d] = all_reduce(partial[:,t,d])
 *     y[t,d] = act(xred[t,d] * weight[d,W-1] +
 *                  sum_{k<W-1} tap(t,k,d) * weight[d,k]) + xred when residual
 *     out[rank,t,d] = y[t,d] for every rank
 *
 *   phase C (cache update + prefix-cache track):
 *     new_cache[slot] = last W-1 rows of concat(old_cache[slot], xred[seq])
 *     new_cache[track_dst[b]] = tracked rows (gathered, or the updated window)
 *
 *   phase D (optional fused tail):
 *     residual = out_local + residual ; norm_out = rmsnorm(residual) * gamma
 *
 * The cache update is a second kernel because standalone SYCL does not provide
 * the CUDA grid-wide remote barrier used by the production fused kernel before
 * its phase-3 cache writes. Keeping it as a local scratch consumer preserves
 * correctness while exposing the same memory traffic and shape constraints.
 * The fused add+RMSNorm tail is a third launch for the same reason (upstream it
 * runs after the exit barrier inside the one kernel).
 *
 * Modes mirrored from ``inkling_ar_scattered_sconv`` (see
 * python/sglang/kernels/jit/csrc/inkling/inkling_ar_scattered_sconv.cuh):
 *
 *   SCATTERED (default): the conv-state cache is this rank's [slots, W-1, Hc]
 *   shard; phase 3 sources its rows from the local reduced scratch.
 *
 *   FULL-WIDTH (``full_update``, upstream ar_sconv_fullwidth_fused): the cache
 *   is the REPLICATED [slots, W-1, H] tensor and the input region is the full
 *   [T, H] row, of which this rank owns columns [rank*Hc, (rank+1)*Hc) =
 *   ``cache_col0``. The conv still runs column-sharded, but phase 3 updates and
 *   tracks ALL H cache columns on every rank, re-reducing the (few) window rows
 *   full-width so the replicated cache stays coherent for full-width consumers.
 *
 *   TRACK (``track_rows`` / ``track_mask`` / ``track_dst`` /
 *   ``track_from_cache``): prefix-cache tracking scatters W-1 conv rows into a
 *   tracking slot of the same pool. Extend gathers the rows named by
 *   ``track_rows`` out of the reduced stream; decode (``track_from_cache``)
 *   snapshots the post-update conv window instead.
 *
 *   NORM (``norm_gamma`` / ``norm_residual`` / ``norm_out`` / ``norm_eps``):
 *   the fused add+RMSNorm tail consumes the gathered full-width OUT rows.
 *   Because this example models one rank's shard of the collective, the peer
 *   columns of that rank's OUT view are pre-seeded on the host with the values
 *   the peers' multicast stores would have delivered; only this rank's shard
 *   columns are produced by the kernel.
 *
 * Roofline: W=4, TP4 does about 4 rank loads plus 4 tap FMAs per output and
 * writes the gathered result to every rank. The operation-to-byte ratio stays
 * below 1 FLOP/B, so this is memory-bound; performance reports effective GB/s.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include "cutlass/util/command_line.h"
#include "19_bmg_comm_ar_sconv_common.hpp"

#include <numeric>

namespace cutlass::examples::comm_ar_sconv {

constexpr int kThreads = 256;
// Largest W-1 the phase-3 register window supports (upstream W1 == 3).
constexpr int kMaxW1 = 7;

struct Options {
  std::string suite = "quick";
  DType dtype = DType::kAll;
  int iterations = 5;
  bool verify = true;
  double perf_threshold_scale = 1.0;
};

struct CaseConfig {
  std::string name;
  int world = 4;
  int batch = 1;
  int tokens_per_seq = 1;
  int D = 0;  // per-rank channel shard Hc = hidden / world
  int W = 4;
  bool varied_lengths = false;
  bool use_silu = true;
  bool use_residual = true;
  bool include_empty = false;
  bool include_false_masks = false;
  double min_gbps = 0.0;
  // Modes added on top of the baseline scattered path (see the file header).
  int rank = 0;
  bool full_update = false;
  bool track = false;
  bool track_from_cache = false;
  bool norm = false;
};

// hidden_size: the example's D is the per-rank shard Hc, so H = Hc * world.
inline int case_hidden(CaseConfig const& cfg) {
  return cfg.D * cfg.world;
}

// Full-width mode reads the whole [T, H] input row (this rank's shard sits at
// column rank*Hc); scattered mode only ever touches the shard, so the example
// keeps the input compacted to [T, Hc] there.
inline int case_in_stride(CaseConfig const& cfg) {
  return cfg.full_update ? case_hidden(cfg) : cfg.D;
}

inline int case_in_col0(CaseConfig const& cfg) {
  return cfg.full_update ? cfg.rank * cfg.D : 0;
}

// The fused norm tail reads the gathered full-width OUT row, so the OUT region
// has to carry all H columns; without it only the shard columns are ever read.
inline int case_out_stride(CaseConfig const& cfg) {
  return cfg.norm ? case_hidden(cfg) : cfg.D;
}

inline int case_out_col0(CaseConfig const& cfg) {
  return cfg.norm ? cfg.rank * cfg.D : 0;
}

// Replicated [slots, W-1, H] cache in full-width mode, [slots, W-1, Hc] shard
// otherwise.
inline int case_cache_cols(CaseConfig const& cfg) {
  return cfg.full_update ? case_hidden(cfg) : cfg.D;
}

inline int case_cache_col0(CaseConfig const& cfg) {
  return cfg.full_update ? cfg.rank * cfg.D : 0;
}

template <typename Element_>
struct ScatteredSconvParams {
  using Element = Element_;

  Element const* __restrict__ partials;
  Element const* __restrict__ cache;
  Element* __restrict__ scratch;
  Element* __restrict__ out;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ cache_mask;
  int32_t const* __restrict__ cu;
  int32_t const* __restrict__ si;
  Element const* __restrict__ weight;
  int world;
  int T;
  int D;
  int W;
  int batch;
  int in_stride;
  int in_col0;
  int out_stride;
  int out_col0;
  int cache_stride_slot;
  int cache_stride_w;
  int cache_col0;
  int use_silu;
  int use_residual;
};

template <typename Element_>
struct CacheUpdateParams {
  using Element = Element_;

  Element const* __restrict__ partials;  // full-width re-reduce source
  Element const* __restrict__ scratch;
  Element const* __restrict__ old_cache;
  Element* __restrict__ new_cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ has_initial_state;
  int32_t const* __restrict__ cu;
  int32_t const* __restrict__ track_rows;  // [batch, W-1] or nullptr
  uint8_t const* __restrict__ track_mask;  // [batch]      or nullptr
  int32_t const* __restrict__ track_dst;   // [batch]      or nullptr
  int world;
  int T;
  int D;
  int W;
  int batch;
  int in_stride;
  int upd_cols;  // H when full_update, Hc otherwise
  int cache_stride_slot;
  int cache_stride_w;
  int full_update;
  int track_from_cache;
};

template <typename Element_>
struct NormTailParams {
  using Element = Element_;

  Element const* __restrict__ out_local;  // this rank's [T, H] OUT view
  Element const* __restrict__ gamma;      // [H]
  Element* __restrict__ residual;         // [T, H] in/out
  Element* __restrict__ norm_out;         // [T, H]
  int T;
  int H;
  float eps;
};

template <typename Element>
class ArScatteredSconvKernel;

template <typename Element>
class ArScatteredReduceScratchKernel;

template <typename Element>
class ArScatteredSconvFromScratchKernel;

template <typename Element>
class ArScatteredReduceScratchPack4Kernel;

template <typename Element>
class ArScatteredSconvFromScratchPack4Kernel;

template <typename Element>
class ScatteredCacheUpdateKernel;

template <typename Element>
class ScatteredNormTailKernel;

template <typename Element>
CUTLASS_DEVICE
float reduce_partial(ScatteredSconvParams<Element> const& params, int t, int d) {
  float acc = 0.0f;
  for (int r = 0; r < params.world; ++r) {
    acc += element_to_float(
        params.partials[(static_cast<std::size_t>(r) * params.T + t) * params.in_stride + params.in_col0 + d]);
  }
  return element_to_float(Element(acc));
}

template <typename Element>
CUTLASS_DEVICE
Element element_from_pack4(uint64_t raw, int lane) {
  return Element::bitcast(static_cast<uint16_t>(raw >> (16 * lane)));
}

template <typename Element>
CUTLASS_DEVICE
uint64_t load_pack4(Element const* ptr) {
  return *reinterpret_cast<uint64_t const*>(ptr);
}

template <typename Element>
CUTLASS_DEVICE
void store_pack4(Element* ptr, uint64_t raw) {
  *reinterpret_cast<uint64_t*>(ptr) = raw;
}

template <typename Element>
CUTLASS_DEVICE
uint64_t pack4_from_floats(float const (&values)[4]) {
  uint64_t raw = 0;
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    raw |= static_cast<uint64_t>(Element(values[v]).raw()) << (16 * v);
  }
  return raw;
}

template <typename Element>
sycl::event launch_scattered_sconv(sycl::queue& q, ScatteredSconvParams<Element> const& params) {
  int total = params.T * params.D;
  if (total == 0) {
    return sycl::event{};
  }
  // Every row stride and column offset in play is D, rank*D or D*world, so a
  // 4-element-aligned shard width aligns all of the 4-wide accesses below.
  if ((params.D % 4) == 0) {
    int pack_total = params.T * (params.D / 4);
    int pack_global = ceil_div(pack_total, kThreads) * kThreads;
    auto reduce = q.parallel_for<ArScatteredReduceScratchPack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(pack_global), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= pack_total) {
            return;
          }
          int pack = linear % (params.D / 4);
          int t = linear / (params.D / 4);
          int d0 = pack * 4;
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int r = 0; r < params.world; ++r) {
            uint64_t raw = load_pack4(
                params.partials +
                (static_cast<std::size_t>(r) * params.T + t) * params.in_stride + params.in_col0 + d0);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[v] += element_to_float(element_from_pack4<Element>(raw, v));
            }
          }
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            acc[v] = element_to_float(Element(acc[v]));
          }
          store_pack4(params.scratch + static_cast<std::size_t>(t) * params.D + d0, pack4_from_floats<Element>(acc));
        });
    return q.parallel_for<ArScatteredSconvFromScratchPack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(pack_global), sycl::range<1>(kThreads)),
        reduce,
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= pack_total) {
            return;
          }
          int pack = linear % (params.D / 4);
          int t = linear / (params.D / 4);
          int d0 = pack * 4;
          int s = params.si[t];
          int bos = params.cu[s];
          int width_minus_one = params.W - 1;
          int slot = params.cache_indices[s] == kPadSlot ? 0 : params.cache_indices[s];
          bool use_cache = params.cache_indices[s] != kPadSlot && params.cache_mask[s] != 0;

          uint64_t xraw = load_pack4(params.scratch + static_cast<std::size_t>(t) * params.D + d0);
          float xcur[4];
          float acc[4];
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            xcur[v] = element_to_float(element_from_pack4<Element>(xraw, v));
            acc[v] = xcur[v] *
                element_to_float(params.weight[static_cast<std::size_t>(d0 + v) * params.W + width_minus_one]);
          }
          for (int k = 0; k < width_minus_one; ++k) {
            int pos = t - (width_minus_one - k);
            uint64_t tap_raw = 0;
            if (pos >= bos) {
              tap_raw = load_pack4(params.scratch + static_cast<std::size_t>(pos) * params.D + d0);
            } else if (use_cache) {
              int prow = pos - bos + width_minus_one;
              if (prow >= 0 && prow < width_minus_one) {
                std::size_t cache_off = static_cast<std::size_t>(slot) * params.cache_stride_slot +
                    static_cast<std::size_t>(prow) * params.cache_stride_w + params.cache_col0 + d0;
                tap_raw = load_pack4(params.cache + cache_off);
              }
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              float tap = element_to_float(element_from_pack4<Element>(tap_raw, v));
              acc[v] += tap * element_to_float(params.weight[static_cast<std::size_t>(d0 + v) * params.W + k]);
            }
          }
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            if (params.use_silu) {
              acc[v] = silu(acc[v]);
            }
            if (params.use_residual) {
              acc[v] += xcur[v];
            }
          }
          uint64_t yraw = pack4_from_floats<Element>(acc);
          for (int r = 0; r < params.world; ++r) {
            store_pack4(
                params.out + (static_cast<std::size_t>(r) * params.T + t) * params.out_stride + params.out_col0 + d0,
                yraw);
          }
        });
  }
  int global = ceil_div(total, kThreads) * kThreads;
  auto reduce = q.parallel_for<ArScatteredReduceScratchKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int d = linear % params.D;
        int t = linear / params.D;
        params.scratch[static_cast<std::size_t>(t) * params.D + d] = Element(reduce_partial(params, t, d));
      });
  return q.parallel_for<ArScatteredSconvFromScratchKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
      reduce,
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int d = linear % params.D;
        int t = linear / params.D;
        int s = params.si[t];
        int bos = params.cu[s];
        int width_minus_one = params.W - 1;
        int slot = params.cache_indices[s] == kPadSlot ? 0 : params.cache_indices[s];
        bool use_cache = params.cache_indices[s] != kPadSlot && params.cache_mask[s] != 0;

        float xcur = element_to_float(params.scratch[static_cast<std::size_t>(t) * params.D + d]);
        float acc = xcur * element_to_float(params.weight[static_cast<std::size_t>(d) * params.W + width_minus_one]);
        for (int k = 0; k < width_minus_one; ++k) {
          int pos = t - (width_minus_one - k);
          float tap = 0.0f;
          if (pos >= bos) {
            tap = element_to_float(params.scratch[static_cast<std::size_t>(pos) * params.D + d]);
          } else if (use_cache) {
            int prow = pos - bos + width_minus_one;
            if (prow >= 0 && prow < width_minus_one) {
              std::size_t cache_off = static_cast<std::size_t>(slot) * params.cache_stride_slot +
                  static_cast<std::size_t>(prow) * params.cache_stride_w + params.cache_col0 + d;
              tap = element_to_float(params.cache[cache_off]);
            }
          }
          acc += tap * element_to_float(params.weight[static_cast<std::size_t>(d) * params.W + k]);
        }
        if (params.use_silu) {
          acc = silu(acc);
        }
        if (params.use_residual) {
          acc += xcur;
        }
        Element y(acc);
        for (int r = 0; r < params.world; ++r) {
          params.out[(static_cast<std::size_t>(r) * params.T + t) * params.out_stride + params.out_col0 + d] = y;
        }
      });
}

// Full-width phase-3 source: upstream re-ld_reduces the window/track rows out
// of the pristine multicast input because the replicated cache spans columns
// this rank's local scratch never held.
template <typename Element>
CUTLASS_DEVICE
Element reduce_full_partial(CacheUpdateParams<Element> const& params, int row, int col) {
  float acc = 0.0f;
  for (int r = 0; r < params.world; ++r) {
    acc += element_to_float(params.partials[(static_cast<std::size_t>(r) * params.T + row) * params.in_stride + col]);
  }
  return Element(acc);
}

// Phase 3: fused conv-state update + prefix-cache track. One work-item owns all
// W-1 rows of one (sequence, channel) pair, mirroring the upstream
// load-all-then-store register window that the from-cache track path reuses.
template <typename Element>
sycl::event launch_cache_update(sycl::queue& q, CacheUpdateParams<Element> const& params) {
  int width_minus_one = params.W - 1;
  int total = params.batch * params.upd_cols;
  if (total == 0 || width_minus_one <= 0) {
    return sycl::event{};
  }
  int global = ceil_div(total, kThreads) * kThreads;
  return q.parallel_for<ScatteredCacheUpdateKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int c = linear % params.upd_cols;
        int b = linear / params.upd_cols;
        int slot = params.cache_indices[b];
        int start = params.cu[b];
        int end = params.cu[b + 1];
        int qlen = end - start;
        bool updated = slot != kPadSlot && qlen > 0;

        Element window[kMaxW1];
        if (updated) {
          std::size_t cbase = static_cast<std::size_t>(slot) * params.cache_stride_slot + c;
          // The shifted-prefix rows are only consumed when the sequence is
          // shorter than the window (qlen < W-1), which never happens at the
          // shipped extend shapes -- keep the loads off that path.
          Element old_rows[kMaxW1];
          bool need_old = qlen < width_minus_one && params.has_initial_state[b] != 0;
          if (need_old) {
            for (int w = 0; w < width_minus_one; ++w) {
              old_rows[w] = params.old_cache[cbase + static_cast<std::size_t>(w) * params.cache_stride_w];
            }
          }
          for (int w = 0; w < width_minus_one; ++w) {
            int seq_row = qlen - width_minus_one + w;
            Element nv(0.0f);
            if (seq_row >= 0) {
              int row = start + seq_row;
              nv = params.full_update ? reduce_full_partial(params, row, c)
                                      : params.scratch[static_cast<std::size_t>(row) * params.D + c];
            } else if (need_old) {
              nv = old_rows[w + qlen];
            }
            window[w] = nv;
            params.new_cache[cbase + static_cast<std::size_t>(w) * params.cache_stride_w] = nv;
          }
        }

        if (params.track_mask != nullptr && params.track_mask[b] != 0) {
          std::size_t dbase =
              static_cast<std::size_t>(params.track_dst[b]) * params.cache_stride_slot + c;
          if (params.track_from_cache) {
            // Decode: snapshot the post-update conv window. Upstream guards the
            // snapshot on the same `updated` predicate, so a tracked sequence
            // whose conv slot is PAD leaves its tracking slot untouched.
            if (updated) {
              for (int w = 0; w < width_minus_one; ++w) {
                params.new_cache[dbase + static_cast<std::size_t>(w) * params.cache_stride_w] = window[w];
              }
            }
          } else {
            // Extend: gather the named rows out of the (re-)reduced stream.
            for (int w = 0; w < width_minus_one; ++w) {
              int row = params.track_rows[static_cast<std::size_t>(b) * width_minus_one + w];
              Element nv = params.full_update ? reduce_full_partial(params, row, c)
                                              : params.scratch[static_cast<std::size_t>(row) * params.D + c];
              params.new_cache[dbase + static_cast<std::size_t>(w) * params.cache_stride_w] = nv;
            }
          }
        }
      });
}

// Fused add+RMSNorm tail: one work-group per token row over the gathered
// full-width OUT row. residual' = OUT + residual (written back), then
// norm_out = residual' * rsqrt(mean(residual'^2) + eps) * gamma. Pass 2
// re-reads the just-written low-precision residual, as upstream does.
template <typename Element>
sycl::event launch_norm_tail(sycl::queue& q, NormTailParams<Element> const& params) {
  if (params.T == 0 || params.H == 0) {
    return sycl::event{};
  }
  int global = params.T * kThreads;
  return q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial_sums(sycl::range<1>(kThreads), cgh);
    cgh.parallel_for<ScatteredNormTailKernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int t = static_cast<int>(item.get_group(0));
          int lane = static_cast<int>(item.get_local_id(0));
          std::size_t base = static_cast<std::size_t>(t) * params.H;
          float ssq = 0.0f;
          for (int h = lane; h < params.H; h += kThreads) {
            float v = element_to_float(params.out_local[base + h]) + element_to_float(params.residual[base + h]);
            params.residual[base + h] = Element(v);
            ssq += v * v;
          }
          partial_sums[lane] = ssq;
          item.barrier(sycl::access::fence_space::local_space);
          for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
            if (lane < offset) {
              partial_sums[lane] += partial_sums[lane + offset];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float inv_rms = sycl::native::rsqrt(partial_sums[0] / static_cast<float>(params.H) + params.eps);
          for (int h = lane; h < params.H; h += kThreads) {
            float r = element_to_float(params.residual[base + h]);
            float gamma = element_to_float(params.gamma[h]);
            params.norm_out[base + h] = Element(r * inv_rms * gamma);
          }
        });
  });
}

template <typename Element>
struct HostTensors {
  std::vector<Element> partials;
  std::vector<Element> cache;
  std::vector<Element> new_cache;
  std::vector<Element> cache_ref;
  std::vector<Element> scratch;
  std::vector<Element> scratch_ref;
  std::vector<Element> out;
  std::vector<Element> out_seed;
  std::vector<Element> out_ref;
  std::vector<Element> weight;
  std::vector<Element> gamma;
  std::vector<Element> residual;
  std::vector<Element> residual_seed;
  std::vector<Element> residual_ref;
  std::vector<Element> norm_out;
  std::vector<Element> norm_ref;
  std::vector<int32_t> cache_indices;
  std::vector<uint8_t> cache_mask;
  std::vector<uint8_t> has_initial_state;
  std::vector<int32_t> cu;
  std::vector<int32_t> si;
  std::vector<int32_t> track_rows;
  std::vector<uint8_t> track_mask;
  std::vector<int32_t> track_dst;
  int T = 0;
  int slots = 0;
};

constexpr float kNormEps = 1.0e-5f;

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
  int width_minus_one = cfg.W - 1;
  int hidden = case_hidden(cfg);
  int cache_cols = case_cache_cols(cfg);
  h.cu.resize(cfg.batch + 1);
  h.cu[0] = 0;
  for (int b = 0; b < cfg.batch; ++b) {
    int len = cfg.tokens_per_seq;
    if (cfg.varied_lengths) {
      len = (b % 5 == 0) ? std::max(1, cfg.tokens_per_seq / 3) : (cfg.tokens_per_seq - (b % 3));
    }
    if (cfg.include_empty && b % 11 == 7) {
      len = 0;
    }
    h.cu[b + 1] = h.cu[b] + std::max(0, len);
  }
  h.T = h.cu.back();
  h.si.resize(h.T);
  for (int b = 0; b < cfg.batch; ++b) {
    for (int t = h.cu[b]; t < h.cu[b + 1]; ++t) {
      h.si[t] = b;
    }
  }
  // Tracking scatters into dedicated slots of the same pool, so the pool has to
  // hold both the per-sequence conv slots and one tracking slot per sequence.
  h.slots = 2 * cfg.batch + 5;
  std::size_t rows_in = static_cast<std::size_t>(h.T) * case_in_stride(cfg);
  std::size_t rows_out = static_cast<std::size_t>(h.T) * case_out_stride(cfg);
  h.partials.resize(static_cast<std::size_t>(cfg.world) * rows_in);
  h.cache.resize(static_cast<std::size_t>(h.slots) * width_minus_one * cache_cols);
  h.new_cache.resize(h.cache.size());
  h.cache_ref.resize(h.cache.size());
  h.scratch.resize(static_cast<std::size_t>(h.T) * cfg.D);
  h.scratch_ref.resize(h.scratch.size());
  h.out.resize(static_cast<std::size_t>(cfg.world) * rows_out);
  h.out_ref.resize(h.out.size());
  h.weight.resize(static_cast<std::size_t>(cfg.D) * cfg.W);
  h.cache_indices.resize(cfg.batch);
  h.cache_mask.resize(cfg.batch);
  h.has_initial_state.resize(cfg.batch);

  uint32_t seed = 20260722u + static_cast<uint32_t>(cfg.world * 97 + cfg.batch * 13 + cfg.D);
  fill_random(h.partials, seed, -0.50f, 0.50f);
  fill_random(h.cache, seed + 1, -0.30f, 0.30f);
  fill_random(h.weight, seed + 2, -0.35f, 0.35f);
  // Peer shard columns of this rank's OUT view: the values the peers' multicast
  // stores deliver. This rank's own shard columns are overwritten by the kernel.
  // Only the norm tail reads columns the kernel does not write, so the seed
  // copy (up to hundreds of MB at the perf shapes) is norm-only.
  if (cfg.norm) {
    fill_random(h.out, seed + 3, -0.40f, 0.40f);
    h.out_seed = h.out;
    h.out_ref = h.out;
  }
  h.new_cache = h.cache;
  h.cache_ref = h.cache;
  for (int b = 0; b < cfg.batch; ++b) {
    h.cache_indices[b] = (b % 13 == 5) ? kPadSlot : b;
    h.cache_mask[b] = static_cast<uint8_t>(!(cfg.include_false_masks && (b % 4 == 1)));
    h.has_initial_state[b] = static_cast<uint8_t>(b % 3 != 2);
  }

  if (cfg.track) {
    h.track_rows.assign(static_cast<std::size_t>(cfg.batch) * width_minus_one, 0);
    h.track_mask.resize(cfg.batch);
    h.track_dst.resize(cfg.batch);
    for (int b = 0; b < cfg.batch; ++b) {
      int qlen = h.cu[b + 1] - h.cu[b];
      // Empty sequences have no rows to gather and never update, so they are
      // untracked (the backend masks them the same way).
      h.track_mask[b] = static_cast<uint8_t>(qlen > 0 && (b % 3 != 1));
      h.track_dst[b] = cfg.batch + 5 + b;
      for (int w = 0; w < width_minus_one; ++w) {
        int row = h.cu[b + 1] - width_minus_one + w;
        h.track_rows[static_cast<std::size_t>(b) * width_minus_one + w] = std::max(h.cu[b], row);
      }
    }
  }

  if (cfg.norm) {
    std::size_t th = static_cast<std::size_t>(h.T) * hidden;
    h.gamma.resize(hidden);
    h.residual.resize(th);
    h.residual_ref.resize(th);
    h.norm_out.resize(th);
    h.norm_ref.resize(th);
    fill_random(h.gamma, seed + 4, 0.80f, 1.20f);
    fill_random(h.residual, seed + 5, -0.25f, 0.25f);
    h.residual_seed = h.residual;
  }
  return h;
}

template <typename Element>
float reduce_partial_host(CaseConfig const& cfg, HostTensors<Element> const& h, int t, int d) {
  float acc = 0.0f;
  int in_stride = case_in_stride(cfg);
  int in_col0 = case_in_col0(cfg);
  for (int r = 0; r < cfg.world; ++r) {
    acc += element_to_float(h.partials[(static_cast<std::size_t>(r) * h.T + t) * in_stride + in_col0 + d]);
  }
  return element_to_float(Element(acc));
}

template <typename Element>
Element reduce_full_partial_host(CaseConfig const& cfg, HostTensors<Element> const& h, int row, int col) {
  float acc = 0.0f;
  int in_stride = case_in_stride(cfg);
  for (int r = 0; r < cfg.world; ++r) {
    acc += element_to_float(h.partials[(static_cast<std::size_t>(r) * h.T + row) * in_stride + col]);
  }
  return Element(acc);
}

template <typename Element>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  int width_minus_one = cfg.W - 1;
  int hidden = case_hidden(cfg);
  int out_stride = case_out_stride(cfg);
  int out_col0 = case_out_col0(cfg);
  int cache_cols = case_cache_cols(cfg);
  int cache_col0 = case_cache_col0(cfg);
  std::size_t cache_stride_slot = static_cast<std::size_t>(width_minus_one) * cache_cols;

  for (int t = 0; t < h.T; ++t) {
    int b = h.si[t];
    int bos = h.cu[b];
    int slot = h.cache_indices[b] == kPadSlot ? 0 : h.cache_indices[b];
    bool use_cache = h.cache_indices[b] != kPadSlot && h.cache_mask[b] != 0;
    for (int d = 0; d < cfg.D; ++d) {
      float xcur = reduce_partial_host(cfg, h, t, d);
      h.scratch_ref[static_cast<std::size_t>(t) * cfg.D + d] = Element(xcur);
      float acc = xcur * element_to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + width_minus_one]);
      for (int k = 0; k < width_minus_one; ++k) {
        int pos = t - (width_minus_one - k);
        float tap = 0.0f;
        if (pos >= bos) {
          tap = reduce_partial_host(cfg, h, pos, d);
        } else if (use_cache) {
          int prow = pos - bos + width_minus_one;
          if (prow >= 0 && prow < width_minus_one) {
            tap = element_to_float(h.cache[static_cast<std::size_t>(slot) * cache_stride_slot +
                static_cast<std::size_t>(prow) * cache_cols + cache_col0 + d]);
          }
        }
        acc += tap * element_to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + k]);
      }
      if (cfg.use_silu) {
        acc = silu(acc);
      }
      if (cfg.use_residual) {
        acc += xcur;
      }
      Element y(acc);
      for (int r = 0; r < cfg.world; ++r) {
        h.out_ref[(static_cast<std::size_t>(r) * h.T + t) * out_stride + out_col0 + d] = y;
      }
    }
  }

  for (int b = 0; b < cfg.batch; ++b) {
    int slot = h.cache_indices[b];
    int start = h.cu[b];
    int qlen = h.cu[b + 1] - h.cu[b];
    bool updated = slot != kPadSlot && qlen > 0;
    for (int c = 0; c < cache_cols; ++c) {
      Element window[kMaxW1];
      if (updated) {
        for (int w = 0; w < width_minus_one; ++w) {
          int seq_row = qlen - width_minus_one + w;
          std::size_t dst = static_cast<std::size_t>(slot) * cache_stride_slot +
              static_cast<std::size_t>(w) * cache_cols + c;
          Element nv(0.0f);
          if (seq_row >= 0) {
            int row = start + seq_row;
            nv = cfg.full_update ? reduce_full_partial_host(cfg, h, row, c)
                                 : h.scratch_ref[static_cast<std::size_t>(row) * cfg.D + c];
          } else if (h.has_initial_state[b]) {
            nv = h.cache[static_cast<std::size_t>(slot) * cache_stride_slot +
                static_cast<std::size_t>(w + qlen) * cache_cols + c];
          }
          window[w] = nv;
          h.cache_ref[dst] = nv;
        }
      }
      if (cfg.track && h.track_mask[b] != 0) {
        std::size_t dbase = static_cast<std::size_t>(h.track_dst[b]) * cache_stride_slot + c;
        if (cfg.track_from_cache) {
          if (updated) {
            for (int w = 0; w < width_minus_one; ++w) {
              h.cache_ref[dbase + static_cast<std::size_t>(w) * cache_cols] = window[w];
            }
          }
        } else {
          for (int w = 0; w < width_minus_one; ++w) {
            int row = h.track_rows[static_cast<std::size_t>(b) * width_minus_one + w];
            Element nv = cfg.full_update ? reduce_full_partial_host(cfg, h, row, c)
                                         : h.scratch_ref[static_cast<std::size_t>(row) * cfg.D + c];
            h.cache_ref[dbase + static_cast<std::size_t>(w) * cache_cols] = nv;
          }
        }
      }
    }
  }

  if (cfg.norm) {
    std::size_t rank_base = static_cast<std::size_t>(cfg.rank) * h.T * out_stride;
    for (int t = 0; t < h.T; ++t) {
      double ssq = 0.0;
      std::size_t row = static_cast<std::size_t>(t) * hidden;
      for (int c = 0; c < hidden; ++c) {
        float v = element_to_float(h.out_ref[rank_base + row + c]) +
            element_to_float(h.residual_seed[row + c]);
        ssq += static_cast<double>(v) * static_cast<double>(v);
        h.residual_ref[row + c] = Element(v);
      }
      float inv_rms = 1.0f / std::sqrt(static_cast<float>(ssq / static_cast<double>(hidden)) + kNormEps);
      for (int c = 0; c < hidden; ++c) {
        float r = element_to_float(h.residual_ref[row + c]);
        h.norm_ref[row + c] = Element(r * inv_rms * element_to_float(h.gamma[c]));
      }
    }
  }
}

// Mode helpers for the case tables below.
CaseConfig with_modes(
    CaseConfig cfg,
    std::string const& suffix,
    int rank,
    bool full_update,
    bool track,
    bool track_from_cache,
    bool norm) {
  cfg.name += suffix;
  cfg.rank = rank;
  cfg.full_update = full_update;
  cfg.track = track;
  cfg.track_from_cache = track_from_cache;
  cfg.norm = norm;
  return cfg;
}

std::vector<CaseConfig> quick_suite() {
  std::vector<CaseConfig> cases{
      {"tiny_tp2_b2_l3_d8_w3", 2, 2, 3, 8, 3, false, false, true, false, true},
      {"inkling_tp4_b8_l128_d1536_w4", 4, 8, 128, 1536, 4, false, true, true, false, false},
      {"scattered_tp8_b16_l9_d192_w4", 8, 16, 9, 192, 4, true, true, true, false, true},
      {"tail_tp4_b5_l11_d193_w5", 4, 5, 11, 193, 5, true, false, false, true, true},
  };
  // One case per added mode on top of the baseline scattered dataflow.
  CaseConfig base{"quick_tp4_b6_l7_d64_w4", 4, 6, 7, 64, 4, true, true, true, true, true};
  cases.push_back(with_modes(base, "_full", 2, true, false, false, false));
  cases.push_back(with_modes(base, "_track_extend", 1, false, true, false, false));
  cases.push_back(with_modes(base, "_track_decode", 1, false, true, true, false));
  cases.push_back(with_modes(base, "_norm", 3, false, false, false, true));
  cases.push_back(with_modes(base, "_full_track_norm", 2, true, true, false, true));
  // Odd shard width forces the scalar (non-pack4) path through every mode; the
  // modes are also pushed one at a time so a scalar-path failure is attributable.
  CaseConfig odd{"quick_tp2_b3_l5_d33_w4", 2, 3, 5, 33, 4, true, false, true, false, true};
  cases.push_back(with_modes(odd, "_base", 1, false, false, false, false));
  cases.push_back(with_modes(odd, "_full", 1, true, false, false, false));
  cases.push_back(with_modes(odd, "_track_extend", 1, false, true, false, false));
  cases.push_back(with_modes(odd, "_track_decode", 1, false, true, true, false));
  cases.push_back(with_modes(odd, "_norm", 1, false, false, false, true));
  cases.push_back(with_modes(odd, "_full_track_norm", 1, true, true, false, true));
  return cases;
}

std::vector<CaseConfig> stress_suite() {
  std::vector<CaseConfig> cases{
      {"stress_tp1_b1_l1_d1_w2", 1, 1, 1, 1, 2, false, false, false, false, true},
      {"stress_tp2_b7_l5_d31_w3", 2, 7, 5, 31, 3, true, true, true, true, true},
      {"stress_tp4_b17_l13_d257_w4", 4, 17, 13, 257, 4, true, false, true, true, true},
      {"stress_tp8_b19_l7_d769_w5", 8, 19, 7, 769, 5, true, true, false, true, true},
  };
  // Same edge shapes with all three added modes engaged at once (empty
  // sequences, pad slots, false masks, W != 4).
  cases.push_back(with_modes(cases[0], "_modes", 0, true, true, true, true));
  cases.push_back(with_modes(cases[1], "_modes", 1, true, true, false, true));
  cases.push_back(with_modes(cases[2], "_modes", 3, true, true, true, true));
  cases.push_back(with_modes(cases[3], "_modes", 7, false, true, false, true));
  return cases;
}

// Inkling ground truth: sconv_kernel_size 4 (W-1 == 3), hidden_size 768
// (checkpoint) / 1536 (config defaults) / 6144 (production), TP 1/2/4/8 so
// Hc = hidden / world, draft_token_num 9 (production) / 3 (checkpoint).
std::vector<CaseConfig> inkling_suite() {
  std::vector<CaseConfig> cases;
  int const worlds[] = {1, 2, 4, 8};
  int const hiddens[] = {768, 1536, 6144};
  for (int hidden : hiddens) {
    for (int world : worlds) {
      int shard = hidden / world;
      int rank = world - 1;  // exercise a non-zero cache_col0 / column offset
      std::ostringstream tag;
      tag << "inkling_tp" << world << "_h" << hidden << "_d" << shard;
      // Target-verify band: batch 8 x draft_token_num 9, varied lengths and
      // false cache masks for the prefix-tap edges.
      CaseConfig base{tag.str(), world, 8, 9, shard, 4, true, true, true, false, true};
      cases.push_back(with_modes(base, "_base", rank, false, false, false, false));
      cases.push_back(with_modes(base, "_full", rank, true, false, false, false));
      cases.push_back(with_modes(base, "_track_extend", rank, false, true, false, false));
      cases.push_back(with_modes(base, "_track_decode", rank, false, true, true, false));
      cases.push_back(with_modes(base, "_norm", rank, false, false, false, true));
      cases.push_back(with_modes(base, "_full_track_norm", rank, true, true, false, true));
      // Checkpoint draft_token_num 3 at a decode-sized batch.
      CaseConfig draft3{tag.str() + "_l3", world, 16, 3, shard, 4, false, true, true, false, false};
      // Decode combination the backend actually runs: post-update-window track
      // plus the fused norm tail.
      cases.push_back(with_modes(draft3, "_track_decode_norm", rank, false, true, true, true));
    }
  }
  return cases;
}

std::vector<CaseConfig> perf_suite() {
  // ar_scattered_sconv fires only when --enable-scattered-sconv is on, so
  // sconv D is always hidden_size / tp. Cover both shipped configs at TP=2/4/8.
  //   config defaults hidden=1536 → D = 768/384 for TP=2/4
  //   production      hidden=6144 → D = 3072/1536/768 for TP=2/4/8
  std::vector<CaseConfig> cases{
      // Config defaults per-rank D across TP=2/4 at a max_prefill_tokens/8 chunk.
      {"perf_tp2_cfg_b64_l1024_d768_w4",  2, 64, 1024,  768, 4, false, true, true, false, false, 180.0},
      {"perf_tp4_cfg_b64_l1024_d384_w4",  4, 64, 1024,  384, 4, false, true, true, false, false, 150.0},

      // Production per-rank D across TP=2/4/8.
      {"perf_tp2_prod_b64_l1024_d3072_w4", 2, 64, 1024, 3072, 4, false, true, true, false, false, 180.0},
      {"perf_tp4_b64_l1024_d1536_w4", 4, 64, 1024, 1536, 4, false, true, true, false, false, 180.0},
      {"perf_tp8_b64_l1024_d768_w4", 8, 64, 1024, 768, 4, false, true, true, false, false, 150.0},

      // Larger token chunk (l=512, batch=128) at production TP=2/4.
      {"perf_tp2_prod_b128_l512_d3072_w4", 2, 128, 512, 3072, 4, false, true, true, false, false, 180.0},
      {"perf_tp4_b128_l512_d1536_w4", 4, 128, 512, 1536, 4, false, true, true, false, false, 180.0},
      // Same 64k-token shape at production TP=8 (D=hidden/tp=768).
      {"perf_tp8_prod_b128_l512_d768_w4", 8, 128, 512, 768, 4, false, true, true, false, false, 150.0},

      // Config defaults (hidden=1536) 64k-token shape across TP=2/4/8.
      // D = 1536/tp = 768, 384, 192.
      {"perf_tp2_cfg_b128_l512_d768_w4", 2, 128, 512, 768, 4, false, true, true, false, false, 180.0},
      {"perf_tp4_cfg_b128_l512_d384_w4", 4, 128, 512, 384, 4, false, true, true, false, false, 150.0},
      {"perf_tp8_cfg_b128_l512_d192_w4", 8, 128, 512, 192, 4, false, true, true, false, false, 140.0},

      // Chunked-prefill max chunk (b=1, l=16384 = max_prefill_tokens) at cfg
      // TP=2/4 to exercise the long-tokens-per-seq path.
      {"perf_prefill_cap_tp2_cfg_l16384_d768_w4", 2, 1, 16384, 768, 4, false, true, true, false, false, 220.0},
      {"perf_prefill_cap_tp4_cfg_l16384_d384_w4", 4, 1, 16384, 384, 4, false, true, true, false, false, 110.0},
  };

  // --- Added modes. All new gates are report-only (0.0): these bands have not
  // been calibrated on this part yet, and a guessed number would flake CI.
  // Baseline the modes with --perf-threshold-scale=0 before setting them.

  // full_update (upstream ar_sconv_fullwidth_fused, extend streaming band):
  // the input region is full-width [T, H], so T is held where world*T*H stays
  // inside a couple of GB.
  CaseConfig fu_cfg2{"perf_full_tp2_cfg_b64_l1024_d768_w4", 2, 64, 1024, 768, 4,
                     false, true, true, false, false, 0.0};
  cases.push_back(with_modes(fu_cfg2, "", 1, true, false, false, false));
  CaseConfig fu_cfg4{"perf_full_tp4_cfg_b64_l1024_d384_w4", 4, 64, 1024, 384, 4,
                     false, true, true, false, false, 0.0};
  cases.push_back(with_modes(fu_cfg4, "", 3, true, false, false, false));
  CaseConfig fu_cfg8{"perf_full_tp8_cfg_b64_l256_d192_w4", 8, 64, 256, 192, 4,
                     false, true, true, false, false, 0.0};
  cases.push_back(with_modes(fu_cfg8, "", 7, true, false, false, false));
  // Production hidden=6144 at the 8k-token streaming band.
  CaseConfig fu_prod4{"perf_full_tp4_prod_b64_l128_d1536_w4", 4, 64, 128, 1536, 4,
                      false, true, true, false, false, 0.0};
  cases.push_back(with_modes(fu_prod4, "", 3, true, false, false, false));
  CaseConfig fu_prod8{"perf_full_tp8_prod_b64_l128_d768_w4", 8, 64, 128, 768, 4,
                      false, true, true, false, false, 0.0};
  cases.push_back(with_modes(fu_prod8, "", 7, true, false, false, false));

  // Prefix-cache track over the existing extend bands (shard-width buffers).
  CaseConfig tr_prod4{"perf_track_tp4_prod_b64_l1024_d1536_w4", 4, 64, 1024, 1536, 4,
                      false, true, true, false, false, 0.0};
  cases.push_back(with_modes(tr_prod4, "", 3, false, true, false, false));
  CaseConfig tr_prod8{"perf_track_tp8_prod_b128_l512_d768_w4", 8, 128, 512, 768, 4,
                      false, true, true, false, false, 0.0};
  cases.push_back(with_modes(tr_prod8, "", 7, false, true, false, false));

  // Fused add+RMSNorm tail: the production call sites only fuse it at
  // decode/verify shapes (batch x draft_token_num), so gate it there.
  CaseConfig nm_prod4{"perf_norm_tp4_prod_b64_l9_d1536_w4", 4, 64, 9, 1536, 4,
                      false, true, true, false, false, 0.0};
  cases.push_back(with_modes(nm_prod4, "", 3, false, false, false, true));
  CaseConfig nm_prod8{"perf_norm_tp8_prod_b128_l9_d768_w4", 8, 128, 9, 768, 4,
                      false, true, true, false, false, 0.0};
  cases.push_back(with_modes(nm_prod8, "", 7, false, false, false, true));
  CaseConfig nm_cfg4{"perf_norm_track_decode_tp4_cfg_b64_l9_d384_w4", 4, 64, 9, 384, 4,
                     false, true, true, false, false, 0.0};
  cases.push_back(with_modes(nm_cfg4, "", 3, false, true, true, true));

  // All three modes together at a mid extend band.
  CaseConfig all4{"perf_full_track_norm_tp4_cfg_b64_l512_d384_w4", 4, 64, 512, 384, 4,
                  false, true, true, false, false, 0.0};
  cases.push_back(with_modes(all4, "", 3, true, true, false, true));
  return cases;
}

template <typename Element>
double effective_bytes(CaseConfig const& cfg, int T) {
  double td = static_cast<double>(T) * cfg.D;
  double w = static_cast<double>(cfg.world);
  double W = static_cast<double>(cfg.W);
  double elem = static_cast<double>(sizeof(Element));
  double hidden = static_cast<double>(case_hidden(cfg));
  double cache_cols = static_cast<double>(case_cache_cols(cfg));
  double window_rows = static_cast<double>(cfg.batch) * (cfg.W - 1);
  double partial_reads = td * w * elem;
  double scratch = td * (W + 1.0) * elem;
  double weight_reads = td * W * elem;
  double cache_prefix_reads = static_cast<double>(cfg.batch) * (cfg.W - 1) * cfg.D * elem;
  double gather_writes = td * w * elem;
  // Phase 3 reads the old window and writes the new one; full-width mode spans
  // the replicated H columns and re-reduces its source rows across ranks.
  double cache_update = window_rows * cache_cols * 2.0 * elem;
  if (cfg.full_update) {
    cache_update += window_rows * hidden * w * elem;
  }
  double track_bytes = 0.0;
  if (cfg.track) {
    track_bytes = window_rows * cache_cols * elem;  // tracking-slot writes
    if (!cfg.track_from_cache) {
      track_bytes += cfg.full_update ? window_rows * hidden * w * elem : window_rows * cache_cols * elem;
    }
  }
  // Tail: OUT row read, residual read + write, norm_out write, gamma.
  double norm_bytes = cfg.norm ? (static_cast<double>(T) * hidden * 4.0 * elem + hidden * elem) : 0.0;
  return partial_reads + scratch + weight_reads + cache_prefix_reads + gather_writes + cache_update +
      track_bytes + norm_bytes;
}

inline bool validate_case(CaseConfig const& cfg) {
  if (cfg.world <= 0 || cfg.D <= 0 || cfg.W < 2) {
    std::cerr << cfg.name << ": invalid world/D/W\n";
    return false;
  }
  if (cfg.W - 1 > kMaxW1) {
    std::cerr << cfg.name << ": W-1 exceeds kMaxW1=" << kMaxW1 << "\n";
    return false;
  }
  if (cfg.rank < 0 || cfg.rank >= cfg.world) {
    std::cerr << cfg.name << ": rank " << cfg.rank << " outside world " << cfg.world << "\n";
    return false;
  }
  if (cfg.track_from_cache && !cfg.track) {
    std::cerr << cfg.name << ": track_from_cache requires track\n";
    return false;
  }
  return true;
}

template <typename Element>
bool run_case(sycl::queue& q, CaseConfig const& cfg, Options const& options) {
  if (!validate_case(cfg)) {
    return false;
  }
  HostTensors<Element> h = initialize_case<Element>(cfg);
  // The CPU model is O(T * D * world * W); skip it when nothing consumes it so
  // the perf suite is not dominated by host work.
  if (options.verify) {
    reference_case(cfg, h);
  }

  int hidden = case_hidden(cfg);
  int cache_cols = case_cache_cols(cfg);

  DeviceBuffer<Element> d_partials(q, h.partials.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<Element> d_new_cache(q, h.new_cache.size());
  DeviceBuffer<Element> d_scratch(q, h.scratch.size());
  DeviceBuffer<Element> d_out(q, h.out.size());
  DeviceBuffer<Element> d_weight(q, h.weight.size());
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<uint8_t> d_cache_mask(q, h.cache_mask.size());
  DeviceBuffer<uint8_t> d_has_initial_state(q, h.has_initial_state.size());
  DeviceBuffer<int32_t> d_cu(q, h.cu.size());
  DeviceBuffer<int32_t> d_si(q, h.si.size());
  DeviceBuffer<int32_t> d_track_rows(q, h.track_rows.size());
  DeviceBuffer<uint8_t> d_track_mask(q, h.track_mask.size());
  DeviceBuffer<int32_t> d_track_dst(q, h.track_dst.size());
  DeviceBuffer<Element> d_gamma(q, h.gamma.size());
  DeviceBuffer<Element> d_residual(q, h.residual.size());
  DeviceBuffer<Element> d_norm_out(q, h.norm_out.size());

  d_partials.copy_from(h.partials);
  d_cache.copy_from(h.cache);
  d_new_cache.copy_from(h.new_cache);
  d_out.copy_from(h.out_seed);
  d_weight.copy_from(h.weight);
  d_cache_indices.copy_from(h.cache_indices);
  d_cache_mask.copy_from(h.cache_mask);
  d_has_initial_state.copy_from(h.has_initial_state);
  d_cu.copy_from(h.cu);
  d_si.copy_from(h.si);
  d_track_rows.copy_from(h.track_rows);
  d_track_mask.copy_from(h.track_mask);
  d_track_dst.copy_from(h.track_dst);
  d_gamma.copy_from(h.gamma);
  d_residual.copy_from(h.residual_seed);

  ScatteredSconvParams<Element> sconv_params{
      d_partials.get(),
      d_cache.get(),
      d_scratch.get(),
      d_out.get(),
      d_cache_indices.get(),
      d_cache_mask.get(),
      d_cu.get(),
      d_si.get(),
      d_weight.get(),
      cfg.world,
      h.T,
      cfg.D,
      cfg.W,
      cfg.batch,
      case_in_stride(cfg),
      case_in_col0(cfg),
      case_out_stride(cfg),
      case_out_col0(cfg),
      (cfg.W - 1) * cache_cols,
      cache_cols,
      case_cache_col0(cfg),
      cfg.use_silu ? 1 : 0,
      cfg.use_residual ? 1 : 0};
  CacheUpdateParams<Element> update_params{
      d_partials.get(),
      d_scratch.get(),
      d_cache.get(),
      d_new_cache.get(),
      d_cache_indices.get(),
      d_has_initial_state.get(),
      d_cu.get(),
      cfg.track ? d_track_rows.get() : nullptr,
      cfg.track ? d_track_mask.get() : nullptr,
      cfg.track ? d_track_dst.get() : nullptr,
      cfg.world,
      h.T,
      cfg.D,
      cfg.W,
      cfg.batch,
      case_in_stride(cfg),
      cache_cols,
      (cfg.W - 1) * cache_cols,
      cache_cols,
      cfg.full_update ? 1 : 0,
      cfg.track_from_cache ? 1 : 0};
  NormTailParams<Element> norm_params{
      d_out.get() + static_cast<std::size_t>(cfg.rank) * h.T * case_out_stride(cfg),
      d_gamma.get(),
      d_residual.get(),
      d_norm_out.get(),
      h.T,
      hidden,
      kNormEps};

  auto launch_kernels = [&]() {
    auto event = launch_scattered_sconv(q, sconv_params);
    event.wait();
    auto update = launch_cache_update(q, update_params);
    if (!cfg.norm) {
      return update;
    }
    update.wait();
    return launch_norm_tail(q, norm_params);
  };
  d_new_cache.copy_from(h.cache);
  launch_kernels().wait();

  bool passed = true;
  if (options.verify) {
    d_out.copy_to(h.out);
    d_scratch.copy_to(h.scratch);
    d_new_cache.copy_to(h.new_cache);
    std::string base = cfg.name + "/" + element_dtype_text<Element>();
    passed &= compare_vectors(base + "/out", h.out, h.out_ref, default_atol<Element>(), default_rtol<Element>());
    passed &= compare_vectors(base + "/scratch", h.scratch, h.scratch_ref, default_atol<Element>(), default_rtol<Element>());
    passed &= compare_vectors(base + "/cache", h.new_cache, h.cache_ref, default_atol<Element>(), default_rtol<Element>());
    if (cfg.norm) {
      d_residual.copy_to(h.residual);
      d_norm_out.copy_to(h.norm_out);
      passed &= compare_vectors(
          base + "/residual", h.residual, h.residual_ref, default_atol<Element>(), default_rtol<Element>());
      passed &= compare_vectors(
          base + "/norm_out", h.norm_out, h.norm_ref, default_atol<Element>(), default_rtol<Element>());
    }
  }

  // The cache update and the norm residual are in-place, so re-seed both
  // outside the timed region (upstream re-runs against fresh state too). The
  // residual then grows by one OUT row per timed repeat, which is harmless at
  // any sane --iterations (fp16 would only saturate after ~1e4 repeats).
  d_new_cache.copy_from(h.cache);
  d_residual.copy_from(h.residual_seed);
  double ms = time_ms(q, options.iterations, launch_kernels);
  double gbps = effective_bytes<Element>(cfg, h.T) / (ms * 1.0e6);
  std::string perf_label = cfg.name + "/" + element_dtype_text<Element>();
  passed &= check_min_gbps(perf_label, gbps, cfg.min_gbps, options.perf_threshold_scale);
  double min_gbps = scaled_min_gbps(cfg.min_gbps, options.perf_threshold_scale);
  std::cout << "[ar_scattered_sconv] " << std::left << std::setw(44) << cfg.name << " dtype=" << std::setw(4)
            << element_dtype_text<Element>() << " world=" << cfg.world << " rank=" << cfg.rank << " T=" << h.T
            << " D=" << cfg.D << " H=" << hidden << " W=" << cfg.W << " varied=" << bool_text(cfg.varied_lengths)
            << " full=" << bool_text(cfg.full_update) << " track=" << bool_text(cfg.track)
            << (cfg.track ? (cfg.track_from_cache ? "(cache)" : "(rows)") : "") << " norm=" << bool_text(cfg.norm)
            << " time_ms=" << std::fixed << std::setprecision(4) << ms << " eff_GBps=" << std::setprecision(2)
            << gbps << " min_GBps=" << min_gbps << " " << (passed ? "PASSED" : "FAILED") << "\n";
  return passed;
}

template <typename Element>
bool run_typed(sycl::queue& q, std::vector<CaseConfig> const& cases, Options const& options) {
  bool passed = true;
  for (auto const& cfg : cases) {
    passed &= run_case<Element>(q, cfg, options);
  }
  return passed;
}

void print_usage(char const* name) {
  std::cout << "Usage: " << name << " [options]\n"
            << "  --suite=<quick|stress|inkling|perf>\n"
            << "  --dtype=<all|bf16|fp16>\n"
            << "  --iterations=<int>\n"
            << "  --verify=<0|1>\n"
            << "  --perf-threshold-scale=<float> (0 disables perf thresholds)\n";
}

}  // namespace cutlass::examples::comm_ar_sconv

int main(int argc, char const** argv) {
  using namespace cutlass::examples::comm_ar_sconv;

  cutlass::CommandLine cmd(argc, argv);
  Options options;
  cmd.get_cmd_line_argument("suite", options.suite, options.suite);
  cmd.get_cmd_line_argument("iterations", options.iterations, options.iterations);
  cmd.get_cmd_line_argument("perf-threshold-scale", options.perf_threshold_scale, options.perf_threshold_scale);
  int verify = options.verify ? 1 : 0;
  cmd.get_cmd_line_argument("verify", verify, verify);
  options.verify = verify != 0;
  std::string dtype_arg = dtype_text(options.dtype);
  cmd.get_cmd_line_argument("dtype", dtype_arg, dtype_arg);
  if (!parse_dtype(dtype_arg, options.dtype)) {
    std::cerr << "Unknown dtype: " << dtype_arg << "\n";
    print_usage(argv[0]);
    return -1;
  }
  if (cmd.check_cmd_line_flag("help")) {
    print_usage(argv[0]);
    return 0;
  }

  std::vector<CaseConfig> cases;
  if (options.suite == "quick") {
    cases = quick_suite();
  } else if (options.suite == "stress") {
    cases = stress_suite();
  } else if (options.suite == "inkling") {
    cases = inkling_suite();
  } else if (options.suite == "perf") {
    cases = perf_suite();
  } else {
    std::cerr << "Unknown suite: " << options.suite << "\n";
    print_usage(argv[0]);
    return -1;
  }

  sycl::queue q{sycl::gpu_selector_v};
  print_device(q);

  bool passed = true;
  if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
    passed &= run_typed<cutlass::bfloat16_t>(q, cases, options);
  }
  if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
    passed &= run_typed<cutlass::half_t>(q, cases, options);
  }
  return passed ? 0 : -1;
}
