/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling 06.2 fused decode {all-reduce -> SConv cache update -> add+RMSNorm}.
 *
 * The CUDA source fuses the v5 push all-reduce epilogue with decode
 * fused_decode_update and fused_add_rmsnorm. This SYCL example keeps the same
 * operation order on BMG using multi-rank buffers in one process:
 *
 *   xb[t,d] = round(sum_r round(partial[r,t,d] + shared[r,t,d]))
 *   y[t,d]  = act(sum_{iw<W-1} cache[slot,iw,d] * mask * weight[d,iw]
 *                 + xb[t,d] * weight[d,W-1]) + xb[t,d] when residual is enabled
 *   residual_out[t,d] = residual_in[t,d] + y[t,d]
 *   hs_out[t,d] = residual_out[t,d] * gamma[d] / sqrt(mean_d(residual_out^2)+eps)
 *
 * Valid cache slots are shifted left and append xb. Invalid/pad slots and false
 * cache masks preserve the no-cache semantics of the upstream decode path.
 *
 * Roofline: at W=4, TP4 performs a few adds/FMA and one row reduction while
 * streaming partials, cache, weights, residual, norm weight, two outputs, and
 * cache updates. Arithmetic intensity is well below 1 FLOP/B for Inkling decode
 * widths, so the optimization target is sustained memory bandwidth plus one
 * launch instead of separate AR, SConv, and RMSNorm launches.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include "cutlass/util/command_line.h"
#include "19_bmg_comm_ar_sconv_common.hpp"

namespace cutlass::examples::comm_ar_sconv {

constexpr int kThreads = 256;
constexpr int kPackThreads = 128;
constexpr int kSmallPackThreads = 64;

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
  int T = 0;
  int D = 0;
  int W = 4;
  bool use_silu = true;
  bool use_residual = true;
  bool use_shared = false;
  bool include_pad_slots = false;
  bool include_false_masks = false;
  double min_gbps = 0.0;
};

template <typename Element_>
struct FusedDecodeParams {
  using Element = Element_;

  Element const* __restrict__ partials;
  Element const* __restrict__ shared;
  Element const* __restrict__ residual_in;
  Element* __restrict__ residual_out;
  Element* __restrict__ hs_out;
  Element* __restrict__ cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ cache_mask;
  Element const* __restrict__ weight;
  Element const* __restrict__ norm_weight;
  int world;
  int T;
  int D;
  int W;
  int cache_stride_slot;
  int cache_stride_w;
  float eps;
  int use_silu;
  int use_residual;
  int use_shared;
};

template <typename Element>
class ArFusedDecodeKernel;

template <typename Element>
class ArFusedDecodePack4Kernel;

template <typename Element, int World, int PackThreads>
class ArFusedDecodeW4SiluResidualSharedPack4Kernel;

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
CUTLASS_DEVICE
float reduce_rank_partials(FusedDecodeParams<Element> const& params, int t, int d) {
  float acc = 0.0f;
  for (int r = 0; r < params.world; ++r) {
    std::size_t off = (static_cast<std::size_t>(r) * params.T + t) * params.D + d;
    float value = element_to_float(params.partials[off]);
    if (params.use_shared) {
      value += element_to_float(params.shared[off]);
      value = element_to_float(Element(value));
    }
    acc += value;
  }
  return element_to_float(Element(acc));
}

template <typename Element>
CUTLASS_DEVICE
void reduce_rank_partials_pack4(FusedDecodeParams<Element> const& params, int t, int d0, float (&out)[4]) {
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    out[v] = 0.0f;
  }
  for (int r = 0; r < params.world; ++r) {
    std::size_t off = (static_cast<std::size_t>(r) * params.T + t) * params.D + d0;
    uint64_t raw = load_pack4(params.partials + off);
    uint64_t shared_raw = 0;
    if (params.use_shared) {
      shared_raw = load_pack4(params.shared + off);
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      float value = element_to_float(element_from_pack4<Element>(raw, v));
      if (params.use_shared) {
        value += element_to_float(element_from_pack4<Element>(shared_raw, v));
        value = element_to_float(Element(value));
      }
      out[v] += value;
    }
  }
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    out[v] = element_to_float(Element(out[v]));
  }
}

template <typename Element, int World, int PackThreads>
sycl::event launch_fused_decode_w4_silu_residual_shared_pack4(
    sycl::queue& q,
    FusedDecodeParams<Element> const& params) {
  int global = params.T * PackThreads;
  return q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial_sums(sycl::range<1>(PackThreads), cgh);
    cgh.parallel_for<ArFusedDecodeW4SiluResidualSharedPack4Kernel<Element, World, PackThreads>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(PackThreads)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          sycl::sub_group sg = item.get_sub_group();
          int t = static_cast<int>(item.get_group(0));
          int lane = static_cast<int>(item.get_local_id(0));
          int sg_lane = static_cast<int>(sg.get_local_id());
          int sg_id = static_cast<int>(sg.get_group_id());
          int sg_size = static_cast<int>(sg.get_local_range()[0]);
          int sg_count = (PackThreads + sg_size - 1) / sg_size;
          int cache_index = params.cache_indices[t];
          bool valid_slot = cache_index != kPadSlot;
          int slot = valid_slot ? cache_index : 0;
          bool use_cache = valid_slot && params.cache_mask[t] != 0;
          int pack_D = params.D / 4;
          float ssq = 0.0f;

#pragma unroll 4
          for (int pack = lane; pack < pack_D; pack += PackThreads) {
            int d0 = pack * 4;
            float xb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int r = 0; r < World; ++r) {
              std::size_t off = (static_cast<std::size_t>(r) * params.T + t) * params.D + d0;
              uint64_t raw = load_pack4(params.partials + off);
              uint64_t shared_raw = load_pack4(params.shared + off);
#pragma unroll
              for (int v = 0; v < 4; ++v) {
                float value = element_to_float(element_from_pack4<Element>(raw, v)) +
                    element_to_float(element_from_pack4<Element>(shared_raw, v));
                xb[v] += element_to_float(Element(value));
              }
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              xb[v] = element_to_float(Element(xb[v]));
            }

            float acc[4];
            uint64_t weight_raw[4];
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              weight_raw[v] = load_pack4(params.weight + static_cast<std::size_t>(d0 + v) * 4);
              acc[v] = xb[v] * element_to_float(element_from_pack4<Element>(weight_raw[v], 3));
            }
            if (use_cache) {
              std::size_t cache_base = static_cast<std::size_t>(slot) * params.cache_stride_slot + d0;
#pragma unroll
              for (int iw = 0; iw < 3; ++iw) {
                uint64_t tap_raw =
                    load_pack4(params.cache + cache_base + static_cast<std::size_t>(iw) * params.cache_stride_w);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                  float tap = element_to_float(element_from_pack4<Element>(tap_raw, v));
                  float wt = element_to_float(element_from_pack4<Element>(weight_raw[v], iw));
                  acc[v] += tap * wt;
                }
              }
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[v] = silu(acc[v]) + xb[v];
            }

            uint64_t residual_in_raw = load_pack4(params.residual_in + static_cast<std::size_t>(t) * params.D + d0);
            float residual[4];
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              residual[v] = element_to_float(element_from_pack4<Element>(residual_in_raw, v)) + acc[v];
              ssq += residual[v] * residual[v];
            }
            uint64_t residual_raw = pack4_from_floats<Element>(residual);
            store_pack4(params.residual_out + static_cast<std::size_t>(t) * params.D + d0, residual_raw);

            if (valid_slot) {
              std::size_t cache_base = static_cast<std::size_t>(slot) * params.cache_stride_slot + d0;
              uint64_t row1 = load_pack4(params.cache + cache_base + params.cache_stride_w);
              uint64_t row2 =
                  load_pack4(params.cache + cache_base + static_cast<std::size_t>(2) * params.cache_stride_w);
              store_pack4(params.cache + cache_base, row1);
              store_pack4(params.cache + cache_base + params.cache_stride_w, row2);
              store_pack4(
                  params.cache + cache_base + static_cast<std::size_t>(2) * params.cache_stride_w,
                  pack4_from_floats<Element>(xb));
            }
          }

          float sg_ssq = sycl::reduce_over_group(sg, ssq, sycl::plus<float>());
          if (sg_lane == 0) {
            partial_sums[sg_id] = sg_ssq;
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (lane == 0) {
            float reduced = 0.0f;
            for (int i = 0; i < sg_count; ++i) {
              reduced += partial_sums[i];
            }
            partial_sums[0] = reduced;
          }
          item.barrier(sycl::access::fence_space::local_space);
          float inv_rms = sycl::native::rsqrt(partial_sums[0] / static_cast<float>(params.D) + params.eps);
#pragma unroll 4
          for (int pack = lane; pack < pack_D; pack += PackThreads) {
            int d0 = pack * 4;
            uint64_t residual_raw = load_pack4(params.residual_out + static_cast<std::size_t>(t) * params.D + d0);
            float hs[4];
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              float residual = element_to_float(element_from_pack4<Element>(residual_raw, v));
              float gamma = element_to_float(params.norm_weight[d0 + v]);
              hs[v] = residual * inv_rms * gamma;
            }
            store_pack4(params.hs_out + static_cast<std::size_t>(t) * params.D + d0, pack4_from_floats<Element>(hs));
          }
        });
  });
}

template <typename Element>
sycl::event launch_fused_decode(sycl::queue& q, FusedDecodeParams<Element> const& params) {
  if (params.T == 0 || params.D == 0) {
    return sycl::event{};
  }
  if ((params.D % 4) == 0) {
    if (params.W == 4 && params.use_silu && params.use_residual && params.use_shared) {
      if (params.world == 4) {
        if (params.D >= 4096) {
          return launch_fused_decode_w4_silu_residual_shared_pack4<Element, 4, kThreads>(q, params);
        }
        return launch_fused_decode_w4_silu_residual_shared_pack4<Element, 4, kPackThreads>(q, params);
      }
      if (params.world == 8) {
        if (params.D >= 4096) {
          return launch_fused_decode_w4_silu_residual_shared_pack4<Element, 8, kThreads>(q, params);
        }
        if (params.D <= 1536) {
          return launch_fused_decode_w4_silu_residual_shared_pack4<Element, 8, kSmallPackThreads>(q, params);
        }
        return launch_fused_decode_w4_silu_residual_shared_pack4<Element, 8, kPackThreads>(q, params);
      }
    }
    int pack_threads = params.D >= 4096 ? kThreads : kPackThreads;
    int pack_D_host = params.D / 4;
    int global = params.T * pack_threads;
    return q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> partial_sums(sycl::range<1>(pack_threads), cgh);
      sycl::local_accessor<uint64_t, 1> residual_packs(sycl::range<1>(pack_D_host), cgh);
      cgh.parallel_for<ArFusedDecodePack4Kernel<Element>>(
          sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(pack_threads)),
          [=](sycl::nd_item<1> item) {
            sycl::sub_group sg = item.get_sub_group();
            int t = static_cast<int>(item.get_group(0));
            int lane = static_cast<int>(item.get_local_id(0));
            int sg_lane = static_cast<int>(sg.get_local_id());
            int sg_id = static_cast<int>(sg.get_group_id());
            int sg_size = static_cast<int>(sg.get_local_range()[0]);
            int sg_count = (pack_threads + sg_size - 1) / sg_size;
            int cache_index = params.cache_indices[t];
            bool valid_slot = cache_index != kPadSlot;
            int slot = valid_slot ? cache_index : 0;
            bool use_cache = valid_slot && params.cache_mask[t] != 0;
            int width_minus_one = params.W - 1;
            int pack_D = params.D / 4;
            float ssq = 0.0f;

            for (int pack = lane; pack < pack_D; pack += pack_threads) {
              int d0 = pack * 4;
              float xb[4];
              reduce_rank_partials_pack4(params, t, d0, xb);
              float acc[4];
#pragma unroll
              for (int v = 0; v < 4; ++v) {
                acc[v] = xb[v] *
                    element_to_float(params.weight[static_cast<std::size_t>(d0 + v) * params.W + width_minus_one]);
              }
              if (use_cache) {
                std::size_t cache_base = static_cast<std::size_t>(slot) * params.cache_stride_slot + d0;
                for (int iw = 0; iw < width_minus_one; ++iw) {
                  uint64_t tap_raw =
                      load_pack4(params.cache + cache_base + static_cast<std::size_t>(iw) * params.cache_stride_w);
#pragma unroll
                  for (int v = 0; v < 4; ++v) {
                    float tap = element_to_float(element_from_pack4<Element>(tap_raw, v));
                    float wt = element_to_float(params.weight[static_cast<std::size_t>(d0 + v) * params.W + iw]);
                    acc[v] += tap * wt;
                  }
                }
              }
              if (params.use_silu) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                  acc[v] = silu(acc[v]);
                }
              }
              if (params.use_residual) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                  acc[v] += xb[v];
                }
              }

              uint64_t residual_in_raw = load_pack4(params.residual_in + static_cast<std::size_t>(t) * params.D + d0);
              float residual[4];
#pragma unroll
              for (int v = 0; v < 4; ++v) {
                residual[v] = element_to_float(element_from_pack4<Element>(residual_in_raw, v)) + acc[v];
                ssq += residual[v] * residual[v];
              }
              uint64_t residual_raw = pack4_from_floats<Element>(residual);
              residual_packs[pack] = residual_raw;
              store_pack4(params.residual_out + static_cast<std::size_t>(t) * params.D + d0, residual_raw);

              if (valid_slot) {
                std::size_t cache_base = static_cast<std::size_t>(slot) * params.cache_stride_slot + d0;
                for (int iw = 0; iw < width_minus_one - 1; ++iw) {
                  uint64_t next =
                      load_pack4(params.cache + cache_base + static_cast<std::size_t>(iw + 1) * params.cache_stride_w);
                  store_pack4(params.cache + cache_base + static_cast<std::size_t>(iw) * params.cache_stride_w, next);
                }
                store_pack4(
                    params.cache + cache_base + static_cast<std::size_t>(width_minus_one - 1) * params.cache_stride_w,
                    pack4_from_floats<Element>(xb));
              }
            }

            float sg_ssq = sycl::reduce_over_group(sg, ssq, sycl::plus<float>());
            if (sg_lane == 0) {
              partial_sums[sg_id] = sg_ssq;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (lane == 0) {
              float reduced = 0.0f;
              for (int i = 0; i < sg_count; ++i) {
                reduced += partial_sums[i];
              }
              partial_sums[0] = reduced;
            }
            item.barrier(sycl::access::fence_space::local_space);
            float inv_rms = sycl::native::rsqrt(partial_sums[0] / static_cast<float>(params.D) + params.eps);
            for (int pack = lane; pack < pack_D; pack += pack_threads) {
              int d0 = pack * 4;
              uint64_t residual_raw = residual_packs[pack];
              float hs[4];
#pragma unroll
              for (int v = 0; v < 4; ++v) {
                float residual = element_to_float(element_from_pack4<Element>(residual_raw, v));
                float gamma = element_to_float(params.norm_weight[d0 + v]);
                hs[v] = residual * inv_rms * gamma;
              }
              store_pack4(params.hs_out + static_cast<std::size_t>(t) * params.D + d0, pack4_from_floats<Element>(hs));
            }
          });
    });
  }
  int global = params.T * kThreads;
  return q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial_sums(sycl::range<1>(kThreads), cgh);
    cgh.parallel_for<ArFusedDecodeKernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int t = static_cast<int>(item.get_group(0));
          int lane = static_cast<int>(item.get_local_id(0));
          int cache_index = params.cache_indices[t];
          bool valid_slot = cache_index != kPadSlot;
          int slot = valid_slot ? cache_index : 0;
          bool use_cache = valid_slot && params.cache_mask[t] != 0;
          int width_minus_one = params.W - 1;
          float ssq = 0.0f;

          for (int d = lane; d < params.D; d += kThreads) {
            float xb = reduce_rank_partials(params, t, d);
            float acc = xb * element_to_float(params.weight[static_cast<std::size_t>(d) * params.W + width_minus_one]);
            if (use_cache) {
              std::size_t cache_base = static_cast<std::size_t>(slot) * params.cache_stride_slot + d;
              for (int iw = 0; iw < width_minus_one; ++iw) {
                float tap =
                    element_to_float(params.cache[cache_base + static_cast<std::size_t>(iw) * params.cache_stride_w]);
                float wt = element_to_float(params.weight[static_cast<std::size_t>(d) * params.W + iw]);
                acc += tap * wt;
              }
            }
            if (params.use_silu) {
              acc = silu(acc);
            }
            if (params.use_residual) {
              acc += xb;
            }
            float residual = element_to_float(params.residual_in[static_cast<std::size_t>(t) * params.D + d]) + acc;
            params.residual_out[static_cast<std::size_t>(t) * params.D + d] = Element(residual);
            ssq += residual * residual;

            if (valid_slot) {
              std::size_t cache_base = static_cast<std::size_t>(slot) * params.cache_stride_slot + d;
              for (int iw = 0; iw < width_minus_one - 1; ++iw) {
                params.cache[cache_base + static_cast<std::size_t>(iw) * params.cache_stride_w] =
                    params.cache[cache_base + static_cast<std::size_t>(iw + 1) * params.cache_stride_w];
              }
              if (width_minus_one > 0) {
                params.cache[cache_base + static_cast<std::size_t>(width_minus_one - 1) * params.cache_stride_w] =
                    Element(xb);
              }
            }
          }

          partial_sums[lane] = ssq;
          item.barrier(sycl::access::fence_space::local_space);
          for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
            if (lane < offset) {
              partial_sums[lane] += partial_sums[lane + offset];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float inv_rms = sycl::native::rsqrt(partial_sums[0] / static_cast<float>(params.D) + params.eps);
          for (int d = lane; d < params.D; d += kThreads) {
            float residual = element_to_float(params.residual_out[static_cast<std::size_t>(t) * params.D + d]);
            float gamma = element_to_float(params.norm_weight[d]);
            params.hs_out[static_cast<std::size_t>(t) * params.D + d] = Element(residual * inv_rms * gamma);
          }
        });
  });
}

template <typename Element>
struct HostTensors {
  std::vector<Element> partials;
  std::vector<Element> shared;
  std::vector<Element> residual_in;
  std::vector<Element> residual_out;
  std::vector<Element> residual_ref;
  std::vector<Element> hs_out;
  std::vector<Element> hs_ref;
  std::vector<Element> cache;
  std::vector<Element> cache_ref;
  std::vector<Element> weight;
  std::vector<Element> norm_weight;
  std::vector<int32_t> cache_indices;
  std::vector<uint8_t> cache_mask;
  int slots = 0;
};

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
  h.slots = cfg.T + 3;
  std::size_t td = static_cast<std::size_t>(cfg.T) * cfg.D;
  h.partials.resize(static_cast<std::size_t>(cfg.world) * td);
  h.shared.resize(static_cast<std::size_t>(cfg.world) * td);
  h.residual_in.resize(td);
  h.residual_out.resize(td);
  h.residual_ref.resize(td);
  h.hs_out.resize(td);
  h.hs_ref.resize(td);
  h.cache.resize(static_cast<std::size_t>(h.slots) * (cfg.W - 1) * cfg.D);
  h.cache_ref.resize(h.cache.size());
  h.weight.resize(static_cast<std::size_t>(cfg.D) * cfg.W);
  h.norm_weight.resize(cfg.D);
  h.cache_indices.resize(cfg.T);
  h.cache_mask.resize(cfg.T);

  uint32_t seed = 20260720u + static_cast<uint32_t>(cfg.world * 101 + cfg.T * 7 + cfg.D);
  fill_random(h.partials, seed, -0.45f, 0.45f);
  fill_random(h.shared, seed + 1, -0.15f, 0.15f);
  fill_random(h.residual_in, seed + 2, -0.25f, 0.25f);
  fill_random(h.cache, seed + 3, -0.30f, 0.30f);
  fill_random(h.weight, seed + 4, -0.35f, 0.35f);
  fill_random(h.norm_weight, seed + 5, 0.80f, 1.20f);
  h.cache_ref = h.cache;

  for (int t = 0; t < cfg.T; ++t) {
    bool pad = cfg.include_pad_slots && (t % 7 == 3);
    h.cache_indices[t] = pad ? kPadSlot : t;
    h.cache_mask[t] = static_cast<uint8_t>(!(cfg.include_false_masks && (t % 5 == 2)));
  }
  return h;
}

template <typename Element>
float folded_partial_host(CaseConfig const& cfg, HostTensors<Element> const& h, int t, int d) {
  float acc = 0.0f;
  for (int r = 0; r < cfg.world; ++r) {
    std::size_t off = (static_cast<std::size_t>(r) * cfg.T + t) * cfg.D + d;
    float value = element_to_float(h.partials[off]);
    if (cfg.use_shared) {
      value += element_to_float(h.shared[off]);
      value = element_to_float(Element(value));
    }
    acc += value;
  }
  return element_to_float(Element(acc));
}

template <typename Element>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  int width_minus_one = cfg.W - 1;
  for (int t = 0; t < cfg.T; ++t) {
    float ssq = 0.0f;
    int cache_index = h.cache_indices[t];
    bool valid_slot = cache_index != kPadSlot;
    int slot = valid_slot ? cache_index : 0;
    bool use_cache = valid_slot && h.cache_mask[t] != 0;
    std::vector<float> reduced(cfg.D);
    for (int d = 0; d < cfg.D; ++d) {
      float xb = folded_partial_host(cfg, h, t, d);
      reduced[d] = xb;
      float acc = xb * element_to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + width_minus_one]);
      if (use_cache) {
        std::size_t cache_base = static_cast<std::size_t>(slot) * width_minus_one * cfg.D + d;
        for (int iw = 0; iw < width_minus_one; ++iw) {
          acc += element_to_float(h.cache_ref[cache_base + static_cast<std::size_t>(iw) * cfg.D]) *
              element_to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + iw]);
        }
      }
      if (cfg.use_silu) {
        acc = silu(acc);
      }
      if (cfg.use_residual) {
        acc += xb;
      }
      float residual = element_to_float(h.residual_in[static_cast<std::size_t>(t) * cfg.D + d]) + acc;
      h.residual_ref[static_cast<std::size_t>(t) * cfg.D + d] = Element(residual);
      ssq += residual * residual;
    }
    float inv_rms = 1.0f / std::sqrt(ssq / static_cast<float>(cfg.D) + 1.0e-5f);
    for (int d = 0; d < cfg.D; ++d) {
      float residual = element_to_float(h.residual_ref[static_cast<std::size_t>(t) * cfg.D + d]);
      float gamma = element_to_float(h.norm_weight[d]);
      h.hs_ref[static_cast<std::size_t>(t) * cfg.D + d] = Element(residual * inv_rms * gamma);
    }
    if (valid_slot) {
      for (int d = 0; d < cfg.D; ++d) {
        std::size_t cache_base = static_cast<std::size_t>(slot) * width_minus_one * cfg.D + d;
        for (int iw = 0; iw < width_minus_one - 1; ++iw) {
          h.cache_ref[cache_base + static_cast<std::size_t>(iw) * cfg.D] =
              h.cache_ref[cache_base + static_cast<std::size_t>(iw + 1) * cfg.D];
        }
        h.cache_ref[cache_base + static_cast<std::size_t>(width_minus_one - 1) * cfg.D] = Element(reduced[d]);
      }
    }
  }
}

std::vector<CaseConfig> quick_suite() {
  return {
      {"tiny_reference_tp2_t2_d7_w3", 2, 2, 7, 3, false, true, false, true, true},
      {"inkling_decode_tp4_t16_d1536_w4", 4, 16, 1536, 4, true, true, true, false, false},
      {"kv_decode_tp4_t16_d512_w4", 4, 16, 512, 4, false, false, false, true, false},
      {"tail_decode_tp8_t9_d193_w5", 8, 9, 193, 5, true, true, true, true, true},
  };
}

std::vector<CaseConfig> stress_suite() {
  return {
      {"stress_tp1_t1_d1_w2", 1, 1, 1, 2, false, false, false, true, true},
      {"stress_tp2_t5_d31_w3", 2, 5, 31, 3, true, true, true, true, true},
      {"stress_tp4_t17_d257_w4", 4, 17, 257, 4, true, false, true, true, true},
      {"stress_tp8_t33_d769_w5", 8, 33, 769, 5, false, true, false, true, true},
  };
}

std::vector<CaseConfig> perf_suite() {
  // Real Inkling decode AR+sconv per-rank shapes: sconv D is the full
  // hidden_size for the non-scattered path (kept the same across TP), or
  // hidden_size / tp when --enable-scattered-sconv is on. Cover both configs
  // at TP=2/4/8 so no rank population is missing a perf gate.
  return {
      // Non-scattered decode path: D == hidden_size (same across TP).
      {"perf_tp2_t128_d1536_w4", 2, 128, 1536, 4, true, true, true, false, false, 120.0},
      {"perf_tp4_t128_d1536_w4", 4, 128, 1536, 4, true, true, true, false, false, 250.0},
      {"perf_tp2_t128_d6144_w4", 2, 128, 6144, 4, true, true, true, false, false, 300.0},
      {"perf_tp4_t128_d6144_w4", 4, 128, 6144, 4, true, true, true, false, false, 300.0},
      {"perf_tp8_t128_d6144_w4", 8, 128, 6144, 4, true, true, true, false, false, 220.0},

      // Scattered decode path: D = hidden_size / tp.
      // Config defaults hidden=1536: 768, 384 for TP=2, 4.
      {"perf_tp2_scattered_cfg_d768_w4", 2, 128,  768, 4, true, true, true, false, false, 100.0},
      {"perf_tp4_scattered_cfg_d384_w4", 4, 128,  384, 4, true, true, true, false, false, 90.0},
      // Production hidden=6144: 3072, 1536, 768 for TP=2, 4, 8.
      {"perf_tp2_scattered_prod_d3072_w4", 2, 128, 3072, 4, true, true, true, false, false, 220.0},
      {"perf_tp4_scattered_prod_d1536_w4", 4, 128, 1536, 4, true, true, true, false, false, 250.0},
      {"perf_tp8_scattered_prod_d768_w4",  8, 128,  768, 4, true, true, true, false, false, 180.0},
      {"perf_tp8_t256_d768_w4", 8, 256, 768, 4, true, true, true, false, false, 280.0},
      {"perf_tp8_t256_d1536_w4", 8, 256, 1536, 4, true, true, true, false, false, 350.0},

      // Target-verify band: T = batch * draft_token_num (144 at bs=16, Q=9),
      // full-hidden decode fusion (non-scattered path). Cache indices/masks
      // exercised to model the real target-verify batch mix.
      {"perf_tp2_verify_t144_d1536_w4", 2, 144, 1536, 4, true, true, true, true, true, 180.0},
      {"perf_tp4_verify_t144_d1536_w4", 4, 144, 1536, 4, true, true, true, true, true, 260.0},
      {"perf_tp8_verify_t144_d1536_w4", 8, 144, 1536, 4, true, true, true, true, true, 280.0},
      {"perf_tp2_verify_t144_d6144_w4", 2, 144, 6144, 4, true, true, true, true, true, 300.0},
      {"perf_tp4_verify_t144_d6144_w4", 4, 144, 6144, 4, true, true, true, true, true, 300.0},
      {"perf_tp8_verify_t144_d6144_w4", 8, 144, 6144, 4, true, true, true, true, true, 240.0},

      // Max-decode band T=96 (the JIT's _INKLING_AR_FUSED_MAX_TOKENS) for
      // non-scattered decode at both configs.
      {"perf_tp2_decode_t96_d1536_w4", 2, 96, 1536, 4, true, true, true, false, false, 130.0},
      {"perf_tp4_decode_t96_d1536_w4", 4, 96, 1536, 4, true, true, true, false, false, 200.0},
      {"perf_tp8_decode_t96_d1536_w4", 8, 96, 1536, 4, true, true, true, false, false, 190.0},
      {"perf_tp2_decode_t96_d6144_w4", 2, 96, 6144, 4, true, true, true, false, false, 260.0},
      {"perf_tp4_decode_t96_d6144_w4", 4, 96, 6144, 4, true, true, true, false, false, 280.0},
      {"perf_tp8_decode_t96_d6144_w4", 8, 96, 6144, 4, true, true, true, false, false, 190.0},
  };
}

template <typename Element>
double effective_bytes(CaseConfig const& cfg) {
  double td = static_cast<double>(cfg.T) * cfg.D;
  double w = static_cast<double>(cfg.world);
  double W = static_cast<double>(cfg.W);
  double shared = cfg.use_shared ? 1.0 : 0.0;
  double elem = static_cast<double>(sizeof(Element));
  double partial_reads = td * w * (1.0 + shared) * elem;
  double cache_tap_reads = td * std::max(cfg.W - 1, 0) * elem;
  double weight_reads = td * W * elem;
  double residual_reads = td * elem;
  double norm_reads = td * 2.0 * elem;
  double output_writes = td * 2.0 * elem;
  double cache_update = td * (2.0 * std::max(cfg.W - 2, 0) + (cfg.W > 1 ? 1.0 : 0.0)) * elem;
  return partial_reads + cache_tap_reads + weight_reads + residual_reads + norm_reads + output_writes +
      cache_update;
}

template <typename Element>
bool run_case(sycl::queue& q, CaseConfig const& cfg, Options const& options) {
  HostTensors<Element> h = initialize_case<Element>(cfg);
  reference_case(cfg, h);

  DeviceBuffer<Element> d_partials(q, h.partials.size());
  DeviceBuffer<Element> d_shared(q, h.shared.size());
  DeviceBuffer<Element> d_residual_in(q, h.residual_in.size());
  DeviceBuffer<Element> d_residual_out(q, h.residual_out.size());
  DeviceBuffer<Element> d_hs_out(q, h.hs_out.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<Element> d_weight(q, h.weight.size());
  DeviceBuffer<Element> d_norm_weight(q, h.norm_weight.size());
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<uint8_t> d_cache_mask(q, h.cache_mask.size());

  d_partials.copy_from(h.partials);
  d_shared.copy_from(h.shared);
  d_residual_in.copy_from(h.residual_in);
  d_cache.copy_from(h.cache);
  d_weight.copy_from(h.weight);
  d_norm_weight.copy_from(h.norm_weight);
  d_cache_indices.copy_from(h.cache_indices);
  d_cache_mask.copy_from(h.cache_mask);

  FusedDecodeParams<Element> params{
      d_partials.get(),
      cfg.use_shared ? d_shared.get() : nullptr,
      d_residual_in.get(),
      d_residual_out.get(),
      d_hs_out.get(),
      d_cache.get(),
      d_cache_indices.get(),
      d_cache_mask.get(),
      d_weight.get(),
      d_norm_weight.get(),
      cfg.world,
      cfg.T,
      cfg.D,
      cfg.W,
      (cfg.W - 1) * cfg.D,
      cfg.D,
      1.0e-5f,
      cfg.use_silu ? 1 : 0,
      cfg.use_residual ? 1 : 0,
      cfg.use_shared ? 1 : 0};

  std::vector<Element> initial_cache = h.cache;
  auto launch_kernel = [&]() {
    return launch_fused_decode(q, params);
  };
  d_cache.copy_from(initial_cache);
  launch_kernel().wait();

  bool passed = true;
  if (options.verify) {
    d_residual_out.copy_to(h.residual_out);
    d_hs_out.copy_to(h.hs_out);
    d_cache.copy_to(h.cache);
    std::string base = cfg.name + "/" + element_dtype_text<Element>();
    passed &= compare_vectors(
        base + "/residual", h.residual_out, h.residual_ref, default_atol<Element>(), default_rtol<Element>());
    passed &= compare_vectors(
        base + "/hs", h.hs_out, h.hs_ref, default_atol<Element>() * 2.0f, default_rtol<Element>() * 2.0f);
    passed &= compare_vectors(base + "/cache", h.cache, h.cache_ref, default_atol<Element>(), default_rtol<Element>());
  }

  d_cache.copy_from(initial_cache);
  double ms = time_ms(q, options.iterations, launch_kernel);
  double gbps = effective_bytes<Element>(cfg) / (ms * 1.0e6);
  std::string perf_label = cfg.name + "/" + element_dtype_text<Element>();
  passed &= check_min_gbps(perf_label, gbps, cfg.min_gbps, options.perf_threshold_scale);
  double min_gbps = scaled_min_gbps(cfg.min_gbps, options.perf_threshold_scale);
  std::cout << "[ar_fused_decode] " << std::left << std::setw(32) << cfg.name << " dtype=" << std::setw(4)
            << element_dtype_text<Element>() << " world=" << cfg.world << " T=" << cfg.T << " D=" << cfg.D
            << " W=" << cfg.W << " silu=" << bool_text(cfg.use_silu)
            << " residual=" << bool_text(cfg.use_residual) << " shared=" << bool_text(cfg.use_shared)
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
            << "  --suite=<quick|stress|perf>\n"
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
