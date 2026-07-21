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
 *   phase C:
 *     new_cache[slot] = last W-1 rows of concat(old_cache[slot], xred[seq])
 *
 * The cache update is a second kernel because standalone SYCL does not provide
 * the CUDA grid-wide remote barrier used by the production fused kernel before
 * its phase-3 cache writes. Keeping it as a local scratch consumer preserves
 * correctness while exposing the same memory traffic and shape constraints.
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

struct Options {
  std::string suite = "quick";
  DType dtype = DType::kAll;
  int iterations = 5;
  bool verify = true;
};

struct CaseConfig {
  std::string name;
  int world = 4;
  int batch = 1;
  int tokens_per_seq = 1;
  int D = 0;
  int W = 4;
  bool varied_lengths = false;
  bool use_silu = true;
  bool use_residual = true;
  bool include_empty = false;
  bool include_false_masks = false;
};

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
  int cache_stride_slot;
  int cache_stride_w;
  int use_silu;
  int use_residual;
};

template <typename Element_>
struct CacheUpdateParams {
  using Element = Element_;

  Element const* __restrict__ scratch;
  Element const* __restrict__ old_cache;
  Element* __restrict__ new_cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ has_initial_state;
  int32_t const* __restrict__ cu;
  int T;
  int D;
  int W;
  int batch;
  int cache_stride_slot;
  int cache_stride_w;
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
CUTLASS_DEVICE
float reduce_partial(ScatteredSconvParams<Element> const& params, int t, int d) {
  float acc = 0.0f;
  for (int r = 0; r < params.world; ++r) {
    acc += element_to_float(params.partials[(static_cast<std::size_t>(r) * params.T + t) * params.D + d]);
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
            uint64_t raw = load_pack4(params.partials + (static_cast<std::size_t>(r) * params.T + t) * params.D + d0);
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
                    static_cast<std::size_t>(prow) * params.cache_stride_w + d0;
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
            store_pack4(params.out + (static_cast<std::size_t>(r) * params.T + t) * params.D + d0, yraw);
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
                  static_cast<std::size_t>(prow) * params.cache_stride_w + d;
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
          params.out[(static_cast<std::size_t>(r) * params.T + t) * params.D + d] = y;
        }
      });
}

template <typename Element>
sycl::event launch_cache_update(sycl::queue& q, CacheUpdateParams<Element> const& params) {
  int width_minus_one = params.W - 1;
  int total = params.batch * width_minus_one * params.D;
  if (total == 0) {
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
        int d = linear % params.D;
        int tmp = linear / params.D;
        int w = tmp % width_minus_one;
        int b = tmp / width_minus_one;
        int slot = params.cache_indices[b];
        if (slot == kPadSlot) {
          return;
        }
        int start = params.cu[b];
        int end = params.cu[b + 1];
        int qlen = end - start;
        if (qlen <= 0) {
          return;
        }
        std::size_t dst = static_cast<std::size_t>(slot) * params.cache_stride_slot +
            static_cast<std::size_t>(w) * params.cache_stride_w + d;
        int seq_row = qlen - width_minus_one + w;
        if (seq_row >= 0) {
          params.new_cache[dst] = params.scratch[static_cast<std::size_t>(start + seq_row) * params.D + d];
        } else if (params.has_initial_state[b] != 0) {
          int old_w = w + qlen;
          params.new_cache[dst] = params.old_cache[static_cast<std::size_t>(slot) * params.cache_stride_slot +
              static_cast<std::size_t>(old_w) * params.cache_stride_w + d];
        } else {
          params.new_cache[dst] = Element(0.0f);
        }
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
  std::vector<Element> out_ref;
  std::vector<Element> weight;
  std::vector<int32_t> cache_indices;
  std::vector<uint8_t> cache_mask;
  std::vector<uint8_t> has_initial_state;
  std::vector<int32_t> cu;
  std::vector<int32_t> si;
  int T = 0;
  int slots = 0;
};

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
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
  h.slots = cfg.batch + 5;
  std::size_t td = static_cast<std::size_t>(h.T) * cfg.D;
  h.partials.resize(static_cast<std::size_t>(cfg.world) * td);
  h.cache.resize(static_cast<std::size_t>(h.slots) * (cfg.W - 1) * cfg.D);
  h.new_cache.resize(h.cache.size());
  h.cache_ref.resize(h.cache.size());
  h.scratch.resize(td);
  h.scratch_ref.resize(td);
  h.out.resize(static_cast<std::size_t>(cfg.world) * td);
  h.out_ref.resize(h.out.size());
  h.weight.resize(static_cast<std::size_t>(cfg.D) * cfg.W);
  h.cache_indices.resize(cfg.batch);
  h.cache_mask.resize(cfg.batch);
  h.has_initial_state.resize(cfg.batch);

  uint32_t seed = 20260722u + static_cast<uint32_t>(cfg.world * 97 + cfg.batch * 13 + cfg.D);
  fill_random(h.partials, seed, -0.50f, 0.50f);
  fill_random(h.cache, seed + 1, -0.30f, 0.30f);
  fill_random(h.weight, seed + 2, -0.35f, 0.35f);
  h.new_cache = h.cache;
  h.cache_ref = h.cache;
  for (int b = 0; b < cfg.batch; ++b) {
    h.cache_indices[b] = (b % 13 == 5) ? kPadSlot : b;
    h.cache_mask[b] = static_cast<uint8_t>(!(cfg.include_false_masks && (b % 4 == 1)));
    h.has_initial_state[b] = static_cast<uint8_t>(b % 3 != 2);
  }
  return h;
}

template <typename Element>
float reduce_partial_host(CaseConfig const& cfg, HostTensors<Element> const& h, int t, int d) {
  float acc = 0.0f;
  for (int r = 0; r < cfg.world; ++r) {
    acc += element_to_float(h.partials[(static_cast<std::size_t>(r) * h.T + t) * cfg.D + d]);
  }
  return element_to_float(Element(acc));
}

template <typename Element>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  int width_minus_one = cfg.W - 1;
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
            tap = element_to_float(h.cache[static_cast<std::size_t>(slot) * width_minus_one * cfg.D +
                static_cast<std::size_t>(prow) * cfg.D + d]);
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
        h.out_ref[(static_cast<std::size_t>(r) * h.T + t) * cfg.D + d] = y;
      }
    }
  }

  for (int b = 0; b < cfg.batch; ++b) {
    int slot = h.cache_indices[b];
    if (slot == kPadSlot) {
      continue;
    }
    int start = h.cu[b];
    int qlen = h.cu[b + 1] - h.cu[b];
    if (qlen <= 0) {
      continue;
    }
    for (int w = 0; w < width_minus_one; ++w) {
      int seq_row = qlen - width_minus_one + w;
      for (int d = 0; d < cfg.D; ++d) {
        std::size_t dst = static_cast<std::size_t>(slot) * width_minus_one * cfg.D +
            static_cast<std::size_t>(w) * cfg.D + d;
        if (seq_row >= 0) {
          h.cache_ref[dst] = h.scratch_ref[static_cast<std::size_t>(start + seq_row) * cfg.D + d];
        } else if (h.has_initial_state[b]) {
          int old_w = w + qlen;
          h.cache_ref[dst] = h.cache[static_cast<std::size_t>(slot) * width_minus_one * cfg.D +
              static_cast<std::size_t>(old_w) * cfg.D + d];
        } else {
          h.cache_ref[dst] = Element(0.0f);
        }
      }
    }
  }
}

std::vector<CaseConfig> quick_suite() {
  return {
      {"tiny_tp2_b2_l3_d8_w3", 2, 2, 3, 8, 3, false, false, true, false, true},
      {"inkling_tp4_b8_l128_d1536_w4", 4, 8, 128, 1536, 4, false, true, true, false, false},
      {"scattered_tp8_b16_l9_d192_w4", 8, 16, 9, 192, 4, true, true, true, false, true},
      {"tail_tp4_b5_l11_d193_w5", 4, 5, 11, 193, 5, true, false, false, true, true},
  };
}

std::vector<CaseConfig> stress_suite() {
  return {
      {"stress_tp1_b1_l1_d1_w2", 1, 1, 1, 1, 2, false, false, false, false, true},
      {"stress_tp2_b7_l5_d31_w3", 2, 7, 5, 31, 3, true, true, true, true, true},
      {"stress_tp4_b17_l13_d257_w4", 4, 17, 13, 257, 4, true, false, true, true, true},
      {"stress_tp8_b19_l7_d769_w5", 8, 19, 7, 769, 5, true, true, false, true, true},
  };
}

std::vector<CaseConfig> perf_suite() {
  return {
      {"perf_tp4_b64_l1024_d1536_w4", 4, 64, 1024, 1536, 4, false, true, true, false, false},
      {"perf_tp8_b64_l1024_d768_w4", 8, 64, 1024, 768, 4, false, true, true, false, false},
      {"perf_tp4_b128_l512_d3072_w4", 4, 128, 512, 3072, 4, false, true, true, false, false},
  };
}

template <typename Element>
double effective_bytes(CaseConfig const& cfg, int T) {
  double td = static_cast<double>(T) * cfg.D;
  double w = static_cast<double>(cfg.world);
  double W = static_cast<double>(cfg.W);
  double elem = static_cast<double>(sizeof(Element));
  double partial_reads = td * w * elem;
  double scratch = td * (W + 1.0) * elem;
  double weight_reads = td * W * elem;
  double cache_prefix_reads = static_cast<double>(cfg.batch) * (cfg.W - 1) * cfg.D * elem;
  double gather_writes = td * w * elem;
  double cache_update = static_cast<double>(cfg.batch) * (cfg.W - 1) * cfg.D * 2.0 * elem;
  return partial_reads + scratch + weight_reads + cache_prefix_reads + gather_writes + cache_update;
}

template <typename Element>
bool run_case(sycl::queue& q, CaseConfig const& cfg, Options const& options) {
  HostTensors<Element> h = initialize_case<Element>(cfg);
  reference_case(cfg, h);

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

  d_partials.copy_from(h.partials);
  d_cache.copy_from(h.cache);
  d_new_cache.copy_from(h.new_cache);
  d_weight.copy_from(h.weight);
  d_cache_indices.copy_from(h.cache_indices);
  d_cache_mask.copy_from(h.cache_mask);
  d_has_initial_state.copy_from(h.has_initial_state);
  d_cu.copy_from(h.cu);
  d_si.copy_from(h.si);

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
      (cfg.W - 1) * cfg.D,
      cfg.D,
      cfg.use_silu ? 1 : 0,
      cfg.use_residual ? 1 : 0};
  CacheUpdateParams<Element> update_params{
      d_scratch.get(),
      d_cache.get(),
      d_new_cache.get(),
      d_cache_indices.get(),
      d_has_initial_state.get(),
      d_cu.get(),
      h.T,
      cfg.D,
      cfg.W,
      cfg.batch,
      (cfg.W - 1) * cfg.D,
      cfg.D};

  std::vector<Element> initial_new_cache = h.cache;
  auto launch_kernels = [&]() {
    auto event = launch_scattered_sconv(q, sconv_params);
    event.wait();
    return launch_cache_update(q, update_params);
  };
  d_new_cache.copy_from(initial_new_cache);
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
  }

  d_new_cache.copy_from(initial_new_cache);
  double ms = time_ms(q, options.iterations, launch_kernels);
  double gbps = effective_bytes<Element>(cfg, h.T) / (ms * 1.0e6);
  std::cout << "[ar_scattered_sconv] " << std::left << std::setw(34) << cfg.name << " dtype=" << std::setw(4)
            << element_dtype_text<Element>() << " world=" << cfg.world << " T=" << h.T << " D=" << cfg.D
            << " W=" << cfg.W << " varied=" << bool_text(cfg.varied_lengths)
            << " time_ms=" << std::fixed << std::setprecision(4) << ms << " eff_GBps=" << std::setprecision(2)
            << gbps << " " << (passed ? "PASSED" : "FAILED") << "\n";
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
            << "  --suite=<quick|stress|perf>\n"
            << "  --dtype=<all|bf16|fp16>\n"
            << "  --iterations=<int>\n"
            << "  --verify=<0|1>\n";
}

}  // namespace cutlass::examples::comm_ar_sconv

int main(int argc, char const** argv) {
  using namespace cutlass::examples::comm_ar_sconv;

  cutlass::CommandLine cmd(argc, argv);
  Options options;
  cmd.get_cmd_line_argument("suite", options.suite, options.suite);
  cmd.get_cmd_line_argument("iterations", options.iterations, options.iterations);
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
