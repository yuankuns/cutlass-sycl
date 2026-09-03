/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling 06.2 fused {all-reduce -> SConv -> add+RMSNorm}: both the decode and
 * the target-verify (speculative decode) kernels.
 *
 * The CUDA source (inkling_ar_fused_decode.cuh) exports TWO kernels off the
 * same v5 push all-reduce epilogue seam:
 *
 *   inkling_ar_sconv_norm         -- decode: one row per sequence, the conv taps
 *                                    come from the per-sequence conv cache and
 *                                    the cache is shift-updated in place.
 *   inkling_ar_sconv_norm_verify  -- target verify: `draft_token_num` (q) rows
 *                                    per sequence, so the causal conv runs along
 *                                    the DRAFT-TOKEN axis (token j of a sequence
 *                                    sees taps j-1, j-2, ... of the same
 *                                    sequence, falling back to the cache prefix
 *                                    only where the window runs off the front).
 *                                    The working cache is READ-ONLY; the
 *                                    per-position windows go to inter_out
 *                                    (save_intermediate_conv_windows), which
 *                                    update_conv_state_after_mtp_verify later
 *                                    folds back into the cache.
 *
 * This SYCL example keeps the same operation order on BMG using multi-rank
 * buffers in one process (CUDA multimem and in-kernel cross-GPU barriers do not
 * exist here; SYCL event ordering substitutes).
 *
 *   xb[t,d] = round(sum_r round(partial[r,t,d] + shared[r,t,d]))
 *
 * decode (cache_indices/cache_mask are per-ROW, [T]):
 *   y[t,d]  = act(sum_{iw<W-1} cache[slot,iw,d] * mask * weight[d,iw]
 *                 + xb[t,d] * weight[d,W-1]) + xb[t,d] when residual is enabled
 *   Valid cache slots are shifted left and append xb. Invalid/pad slots and
 *   false cache masks preserve the no-cache semantics of the upstream kernel.
 *
 * verify (cache_indices/cache_mask are per-SEQUENCE, [B]; T = B*q):
 *   seq = t/q, tq = t%q, bos = seq*q
 *   tap(iw) = xb[t-(W-1)+iw,d]              if t-(W-1)+iw >= bos
 *           = mask * cache[slot,tq+iw,d]    otherwise (cache prefix)
 *   y[t,d]  = act(sum_{iw<W-1} tap(iw) * weight[d,iw]
 *                 + xb[t,d] * weight[d,W-1]) + xb[t,d] when residual is enabled
 *   inter_out[seq,tq,w,d] = cache[slot,tq+1+w,d]  if tq+1+w < W-1
 *                         = xb[bos+tq+1+w-(W-1),d] otherwise
 *   (raw copies, no mask gating -- and nothing is written for pad sequences)
 *
 * Both branches then share (verify rounds y to the element type first, matching
 * the CUDA kernel's `yb`; the decode branch keeps its fp32 y):
 *   residual_out[t,d] = residual_in[t,d] + y[t,d]
 *   hs_out[t,d] = residual_out[t,d] * gamma[d] / sqrt(mean_d(residual_out^2)+eps)
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
  // draft_token_num (q). 0 selects the decode branch (inkling_ar_sconv_norm);
  // q > 0 selects the target-verify branch (inkling_ar_sconv_norm_verify), which
  // needs T == B * q with B sequences of q consecutive draft tokens.
  int draft = 0;
};

// The verify branch keeps the conv window (W-1 taps) in registers; W-1 <= 4
// covers every W the suites use (the model ships sconv_kernel_size == 4).
constexpr int kMaxTaps = 4;

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

// ---------------------------------------------------------------------------
// Target-verify branch (inkling_ar_sconv_norm_verify).
// ---------------------------------------------------------------------------

template <typename Element_>
struct FusedVerifyParams {
  using Element = Element_;

  Element const* __restrict__ partials;
  Element const* __restrict__ shared;
  Element const* __restrict__ residual_in;
  Element* __restrict__ residual_out;
  Element* __restrict__ hs_out;
  Element const* __restrict__ cache;  // [slots, W-1, D], READ-ONLY at verify
  int32_t const* __restrict__ cache_indices;  // [B] per-sequence slot
  uint8_t const* __restrict__ cache_mask;     // [B] per-sequence prefix gate
  Element const* __restrict__ weight;
  Element const* __restrict__ norm_weight;
  Element* __restrict__ inter_out;  // [B, q, W-1, D]
  int world;
  int T;
  int D;
  int W;
  int q;
  int cache_stride_slot;
  int cache_stride_w;
  int inter_stride_b;
  int inter_stride_t;
  int inter_stride_w;
  float eps;
  int use_silu;
  int use_residual;
  int use_shared;
};

template <typename Element, int Pack>
class ArFusedVerifyKernel;

// Pack == 4 uses one 8B (bf16x4) access per lane; Pack == 1 is the scalar
// fallback for D that is not a multiple of 4.
template <typename Element, int Pack>
CUTLASS_DEVICE
void load_vec(Element const* ptr, float (&out)[Pack]) {
  if constexpr (Pack == 4) {
    uint64_t raw = load_pack4(ptr);
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      out[v] = element_to_float(element_from_pack4<Element>(raw, v));
    }
  } else {
#pragma unroll
    for (int v = 0; v < Pack; ++v) {
      out[v] = element_to_float(ptr[v]);
    }
  }
}

template <typename Element, int Pack>
CUTLASS_DEVICE
void store_vec(Element* ptr, float const (&in)[Pack]) {
  if constexpr (Pack == 4) {
    store_pack4(ptr, pack4_from_floats<Element>(in));
  } else {
#pragma unroll
    for (int v = 0; v < Pack; ++v) {
      ptr[v] = Element(in[v]);
    }
  }
}

// Re-reduce row `t` out of the staging buffer, exactly as the CUDA verify kernel
// does for the cross-token conv taps (any block can rebuild any token's reduced
// row, so no cross-block dependency is needed).
template <typename Element, int Pack>
CUTLASS_DEVICE
void reduce_verify_row(FusedVerifyParams<Element> const& params, int t, int d0, float (&out)[Pack]) {
#pragma unroll
  for (int v = 0; v < Pack; ++v) {
    out[v] = 0.0f;
  }
  for (int r = 0; r < params.world; ++r) {
    std::size_t off = (static_cast<std::size_t>(r) * params.T + t) * params.D + d0;
    float value[Pack];
    load_vec<Element, Pack>(params.partials + off, value);
    if (params.use_shared) {
      float sh[Pack];
      load_vec<Element, Pack>(params.shared + off, sh);
#pragma unroll
      for (int v = 0; v < Pack; ++v) {
        value[v] = element_to_float(Element(value[v] + sh[v]));
      }
    }
#pragma unroll
    for (int v = 0; v < Pack; ++v) {
      out[v] += value[v];
    }
  }
#pragma unroll
  for (int v = 0; v < Pack; ++v) {
    out[v] = element_to_float(Element(out[v]));
  }
}

template <typename Element, int Pack>
sycl::event launch_fused_verify(sycl::queue& q, FusedVerifyParams<Element> const& params, int threads) {
  int global = params.T * threads;
  return q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial_sums(sycl::range<1>(threads), cgh);
    cgh.parallel_for<ArFusedVerifyKernel<Element, Pack>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(threads)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          sycl::sub_group sg = item.get_sub_group();
          int t = static_cast<int>(item.get_group(0));
          int lane = static_cast<int>(item.get_local_id(0));
          int sg_lane = static_cast<int>(sg.get_local_id());
          int sg_id = static_cast<int>(sg.get_group_id());
          int sg_size = static_cast<int>(sg.get_local_range()[0]);
          int sg_count = (threads + sg_size - 1) / sg_size;

          int width_minus_one = params.W - 1;
          int seq = t / params.q;
          int tq = t - seq * params.q;
          int bos = seq * params.q;
          int cache_index = params.cache_indices[seq];
          bool valid_slot = cache_index != kPadSlot;
          int slot = valid_slot ? cache_index : 0;
          float cm = (valid_slot && params.cache_mask[seq] != 0) ? 1.0f : 0.0f;
          int num_packs = params.D / Pack;
          float ssq = 0.0f;

          for (int pk = lane; pk < num_packs; pk += threads) {
            int d0 = pk * Pack;

            // Cache prefix rows (read-only) and the reduced rows this token's
            // window needs: its own plus up to W-1 in-sequence predecessors.
            // Only the first W-1 draft positions of a sequence can reach off the
            // front of the sequence, so later tokens skip the prefix entirely
            // (the CUDA kernel loads it unconditionally; on BMG this is ~4% of
            // the traffic at the shipped q=9, W=4 shape).
            bool need_prefix = tq < width_minus_one;
            float prefix[kMaxTaps][Pack] = {};
            std::size_t cache_base = static_cast<std::size_t>(slot) * params.cache_stride_slot + d0;
            if (need_prefix) {
              for (int iw = 0; iw < width_minus_one; ++iw) {
                load_vec<Element, Pack>(
                    params.cache + cache_base + static_cast<std::size_t>(iw) * params.cache_stride_w, prefix[iw]);
              }
            }
            float xb[Pack];
            reduce_verify_row<Element, Pack>(params, t, d0, xb);
            float neighbor[kMaxTaps][Pack];
#pragma unroll
            for (int j = 0; j < kMaxTaps; ++j) {
#pragma unroll
              for (int v = 0; v < Pack; ++v) {
                neighbor[j][v] = 0.0f;
              }
            }
            for (int j = 1; j <= width_minus_one; ++j) {
              int n = t - j;
              if (n >= bos) {
                reduce_verify_row<Element, Pack>(params, n, d0, neighbor[j - 1]);
              }
            }

            // Causal conv along the draft-token axis (ascending tap order).
            float y[Pack];
#pragma unroll
            for (int v = 0; v < Pack; ++v) {
              float acc = 0.0f;
              for (int iw = 0; iw < width_minus_one; ++iw) {
                int shifted = t - width_minus_one + iw;
                float tap = (shifted >= bos) ? neighbor[width_minus_one - 1 - iw][v] : cm * prefix[tq + iw][v];
                acc += tap * element_to_float(params.weight[static_cast<std::size_t>(d0 + v) * params.W + iw]);
              }
              acc += xb[v] *
                  element_to_float(params.weight[static_cast<std::size_t>(d0 + v) * params.W + width_minus_one]);
              if (params.use_silu) {
                acc = silu(acc);
              }
              if (params.use_residual) {
                acc += xb[v];
              }
              y[v] = acc;
            }

            // save_intermediate_conv_windows: the window after draft position tq
            // is raw copies of {cache prefix rows | reduced x rows}, ungated.
            if (valid_slot) {
              Element* op = params.inter_out + static_cast<std::size_t>(seq) * params.inter_stride_b +
                  static_cast<std::size_t>(tq) * params.inter_stride_t + d0;
              for (int w = 0; w < width_minus_one; ++w) {
                int position = tq + 1 + w;
                float const* src;
                if (position < width_minus_one) {
                  src = prefix[position];
                } else {
                  int g = bos + position - width_minus_one;
                  src = (g == t) ? xb : neighbor[t - g - 1];
                }
                float val[Pack];
#pragma unroll
                for (int v = 0; v < Pack; ++v) {
                  val[v] = src[v];
                }
                store_vec<Element, Pack>(op + static_cast<std::size_t>(w) * params.inter_stride_w, val);
              }
            }

            // Residual add (the CUDA kernel rounds y to the element type first).
            float residual_in_vec[Pack];
            load_vec<Element, Pack>(params.residual_in + static_cast<std::size_t>(t) * params.D + d0, residual_in_vec);
            float residual[Pack];
#pragma unroll
            for (int v = 0; v < Pack; ++v) {
              residual[v] = element_to_float(Element(y[v])) + residual_in_vec[v];
              ssq += residual[v] * residual[v];
            }
            store_vec<Element, Pack>(params.residual_out + static_cast<std::size_t>(t) * params.D + d0, residual);
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
          // Second pass: CUDA still holds r in registers (one block owns one
          // token); the two-pass SYCL shape re-reads the rounded residual_out
          // instead, so hs_out carries one extra element-rounding of r. The CPU
          // reference rounds identically -- see reference_verify_case.
          for (int pk = lane; pk < num_packs; pk += threads) {
            int d0 = pk * Pack;
            float residual[Pack];
            load_vec<Element, Pack>(params.residual_out + static_cast<std::size_t>(t) * params.D + d0, residual);
            float hs[Pack];
#pragma unroll
            for (int v = 0; v < Pack; ++v) {
              hs[v] = residual[v] * inv_rms * element_to_float(params.norm_weight[d0 + v]);
            }
            store_vec<Element, Pack>(params.hs_out + static_cast<std::size_t>(t) * params.D + d0, hs);
          }
        });
  });
}

template <typename Element>
sycl::event launch_fused_verify(sycl::queue& q, FusedVerifyParams<Element> const& params) {
  if (params.T == 0 || params.D == 0) {
    return sycl::event{};
  }
  if ((params.D % 4) == 0) {
    int threads = params.D >= 4096 ? kThreads : (params.D <= 1024 ? kSmallPackThreads : kPackThreads);
    return launch_fused_verify<Element, 4>(q, params, threads);
  }
  return launch_fused_verify<Element, 1>(q, params, kThreads);
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
  // Verify branch only: the per-position conv windows
  // (save_intermediate_conv_windows output), [B, q, W-1, D].
  std::vector<Element> inter_out;
  std::vector<Element> inter_ref;
  int slots = 0;
  int batch = 0;  // sequences: T at decode, T/q at verify
};

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
  // Verify indexes the cache per SEQUENCE, decode per ROW.
  h.batch = cfg.draft > 0 ? cfg.T / cfg.draft : cfg.T;
  h.slots = h.batch + 3;
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
  h.cache_indices.resize(h.batch);
  h.cache_mask.resize(h.batch);
  if (cfg.draft > 0) {
    h.inter_out.resize(static_cast<std::size_t>(h.batch) * cfg.draft * (cfg.W - 1) * cfg.D);
    h.inter_ref.resize(h.inter_out.size());
  }

  uint32_t seed = 20260720u + static_cast<uint32_t>(cfg.world * 101 + cfg.T * 7 + cfg.D + cfg.draft * 13);
  fill_random(h.partials, seed, -0.45f, 0.45f);
  fill_random(h.shared, seed + 1, -0.15f, 0.15f);
  fill_random(h.residual_in, seed + 2, -0.25f, 0.25f);
  fill_random(h.cache, seed + 3, -0.30f, 0.30f);
  fill_random(h.weight, seed + 4, -0.35f, 0.35f);
  fill_random(h.norm_weight, seed + 5, 0.80f, 1.20f);
  h.cache_ref = h.cache;
  if (cfg.draft > 0) {
    // Pre-filled with junk so the reference can prove pad sequences are never
    // written (the CUDA kernel skips inter_out for cache_indices == -1).
    fill_random(h.inter_out, seed + 6, -2.0f, 2.0f);
    h.inter_ref = h.inter_out;
  }

  // The pad/false-mask patterns are per SEQUENCE, so a small batch would never
  // hit `b % 7 == 3` / `b % 5 == 2`; fall back to the last/first sequence there
  // so tiny cases still cover the pad-slot and no-cache paths.
  for (int b = 0; b < h.batch; ++b) {
    bool pad = cfg.include_pad_slots && (b % 7 == 3 || (h.batch < 4 && b + 1 == h.batch));
    bool false_mask = cfg.include_false_masks && (b % 5 == 2 || (h.batch < 3 && b == 0));
    h.cache_indices[b] = pad ? kPadSlot : b;
    h.cache_mask[b] = static_cast<uint8_t>(!false_mask);
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

// CPU reference for the target-verify branch. The cache is read-only here; the
// per-position windows land in inter_ref instead.
template <typename Element>
void reference_verify_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  int width_minus_one = cfg.W - 1;
  int q = cfg.draft;
  std::size_t td = static_cast<std::size_t>(cfg.T) * cfg.D;

  // The reduced (all-reduced, element-rounded) rows -- the conv taps read them
  // across tokens, so materialize them all first.
  std::vector<float> reduced(td);
  for (int t = 0; t < cfg.T; ++t) {
    for (int d = 0; d < cfg.D; ++d) {
      reduced[static_cast<std::size_t>(t) * cfg.D + d] = folded_partial_host(cfg, h, t, d);
    }
  }

  for (int t = 0; t < cfg.T; ++t) {
    int seq = t / q;
    int tq = t - seq * q;
    int bos = seq * q;
    int cache_index = h.cache_indices[seq];
    bool valid_slot = cache_index != kPadSlot;
    int slot = valid_slot ? cache_index : 0;
    float cm = (valid_slot && h.cache_mask[seq] != 0) ? 1.0f : 0.0f;
    float ssq = 0.0f;
    for (int d = 0; d < cfg.D; ++d) {
      float xb = reduced[static_cast<std::size_t>(t) * cfg.D + d];
      std::size_t cache_base = static_cast<std::size_t>(slot) * width_minus_one * cfg.D + d;
      float acc = 0.0f;
      for (int iw = 0; iw < width_minus_one; ++iw) {
        int shifted = t - width_minus_one + iw;
        float tap;
        if (shifted >= bos) {
          tap = reduced[static_cast<std::size_t>(shifted) * cfg.D + d];
        } else {
          tap = cm *
              element_to_float(h.cache_ref[cache_base + static_cast<std::size_t>(tq + iw) * cfg.D]);
        }
        acc += tap * element_to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + iw]);
      }
      acc += xb * element_to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + width_minus_one]);
      if (cfg.use_silu) {
        acc = silu(acc);
      }
      if (cfg.use_residual) {
        acc += xb;
      }
      float residual = element_to_float(Element(acc)) +
          element_to_float(h.residual_in[static_cast<std::size_t>(t) * cfg.D + d]);
      h.residual_ref[static_cast<std::size_t>(t) * cfg.D + d] = Element(residual);
      ssq += residual * residual;

      if (valid_slot) {
        std::size_t inter_base = ((static_cast<std::size_t>(seq) * q + tq) * width_minus_one) * cfg.D + d;
        for (int w = 0; w < width_minus_one; ++w) {
          int position = tq + 1 + w;
          float value;
          if (position < width_minus_one) {
            value = element_to_float(h.cache_ref[cache_base + static_cast<std::size_t>(position) * cfg.D]);
          } else {
            int g = bos + position - width_minus_one;
            value = reduced[static_cast<std::size_t>(g) * cfg.D + d];
          }
          h.inter_ref[inter_base + static_cast<std::size_t>(w) * cfg.D] = Element(value);
        }
      }
    }
    float inv_rms = 1.0f / std::sqrt(ssq / static_cast<float>(cfg.D) + 1.0e-5f);
    for (int d = 0; d < cfg.D; ++d) {
      float residual = element_to_float(h.residual_ref[static_cast<std::size_t>(t) * cfg.D + d]);
      float gamma = element_to_float(h.norm_weight[d]);
      h.hs_ref[static_cast<std::size_t>(t) * cfg.D + d] = Element(residual * inv_rms * gamma);
    }
  }
}

std::vector<CaseConfig> quick_suite() {
  return {
      {"tiny_reference_tp2_t2_d7_w3", 2, 2, 7, 3, false, true, false, true, true},
      {"inkling_decode_tp4_t16_d1536_w4", 4, 16, 1536, 4, true, true, true, false, false},
      {"kv_decode_tp4_t16_d512_w4", 4, 16, 512, 4, false, false, false, true, false},
      {"tail_decode_tp8_t9_d193_w5", 8, 9, 193, 5, true, true, true, true, true},
      // Target-verify branch (draft > 0): the conv runs along the draft-token
      // axis. activation=None / use_residual=true is what the model calls.
      {"tiny_verify_tp2_b2_q3_d7_w3", 2, 6, 7, 3, false, true, false, true, true, 0.0, 3},
      {"inkling_verify_tp4_b4_q9_d1536_w4", 4, 36, 1536, 4, false, true, true, false, false, 0.0, 9},
      {"inkling_verify_tp8_b8_q3_d768_w4", 8, 24, 768, 4, false, true, true, true, true, 0.0, 3},
  };
}

std::vector<CaseConfig> stress_suite() {
  return {
      {"stress_tp1_t1_d1_w2", 1, 1, 1, 2, false, false, false, true, true},
      {"stress_tp2_t5_d31_w3", 2, 5, 31, 3, true, true, true, true, true},
      {"stress_tp4_t17_d257_w4", 4, 17, 257, 4, true, false, true, true, true},
      {"stress_tp8_t33_d769_w5", 8, 33, 769, 5, false, true, false, true, true},
      // Verify branch corners: q == 1 (window degenerates to the cache prefix,
      // i.e. decode without the cache write-back), silu on, odd D and W == 5.
      {"stress_verify_tp1_b4_q1_d31_w3", 1, 4, 31, 3, false, true, false, true, true, 0.0, 1},
      // B >= 4 so the per-sequence pad-slot pattern fires on the 4-tap path.
      {"stress_verify_tp2_b8_q5_d129_w5", 2, 40, 129, 5, true, true, true, true, true, 0.0, 5},
      {"stress_verify_tp4_b5_q9_d257_w4", 4, 45, 257, 4, false, false, true, true, true, 0.0, 9},
      {"stress_verify_tp8_b7_q3_d192_w4", 8, 21, 192, 4, true, true, false, true, true, 0.0, 3},
  };
}

// Shapes the Inkling model actually runs through inkling_ar_sconv_norm_verify:
// sconv_kernel_size == 4 (W-1 == 3 taps), bf16, activation=None,
// use_residual=true; draft_token_num = num_nextn_predict_layers + 1, i.e. 9 for
// the production config (8 MTP layers) and 3 for the shipped checkpoint (2).
// D is either the full hidden_size (non-scattered path: 768 checkpoint, 1536
// defaults, 6144 production) or the TP shard hidden_size/tp.
std::vector<CaseConfig> inkling_suite() {
  return {
      // draft_token_num = 9, full hidden_size, all TP ranks.
      {"verify_tp1_b16_q9_d1536", 1, 144, 1536, 4, false, true, false, true, true, 0.0, 9},
      {"verify_tp2_b16_q9_d1536", 2, 144, 1536, 4, false, true, true, true, true, 0.0, 9},
      {"verify_tp4_b16_q9_d1536", 4, 144, 1536, 4, false, true, true, true, true, 0.0, 9},
      {"verify_tp8_b16_q9_d1536", 8, 144, 1536, 4, false, true, true, true, true, 0.0, 9},
      {"verify_tp1_b16_q9_d768_ckpt", 1, 144, 768, 4, false, true, false, true, true, 0.0, 9},
      {"verify_tp4_b16_q9_d6144_prod", 4, 144, 6144, 4, false, true, true, true, true, 0.0, 9},
      {"verify_tp8_b16_q9_d6144_prod", 8, 144, 6144, 4, false, true, true, true, true, 0.0, 9},
      // draft_token_num = 3 (checkpoint num_nextn_predict_layers = 2).
      {"verify_tp1_b16_q3_d768_ckpt", 1, 48, 768, 4, false, true, false, true, true, 0.0, 3},
      {"verify_tp2_b16_q3_d1536", 2, 48, 1536, 4, false, true, true, true, true, 0.0, 3},
      {"verify_tp4_b96_q3_d1536", 4, 288, 1536, 4, false, true, true, true, true, 0.0, 3},
      {"verify_tp8_b96_q3_d6144_prod", 8, 288, 6144, 4, false, true, true, true, true, 0.0, 3},
      // TP shards, D = hidden_size / tp. Production hidden 6144.
      {"verify_tp2_shard_prod_d3072_q9", 2, 144, 3072, 4, false, true, true, true, true, 0.0, 9},
      {"verify_tp4_shard_prod_d1536_q9", 4, 144, 1536, 4, false, true, true, false, false, 0.0, 9},
      {"verify_tp8_shard_prod_d768_q9", 8, 144, 768, 4, false, true, true, true, true, 0.0, 9},
      // Config-default hidden 1536.
      {"verify_tp2_shard_cfg_d768_q9", 2, 144, 768, 4, false, true, true, true, true, 0.0, 9},
      {"verify_tp4_shard_cfg_d384_q9", 4, 144, 384, 4, false, true, true, true, true, 0.0, 9},
      {"verify_tp8_shard_cfg_d192_q9", 8, 144, 192, 4, false, true, true, true, true, 0.0, 9},
      // Checkpoint hidden 768.
      {"verify_tp2_shard_ckpt_d384_q3", 2, 48, 384, 4, false, true, false, true, true, 0.0, 3},
      {"verify_tp4_shard_ckpt_d192_q3", 4, 48, 192, 4, false, true, false, true, true, 0.0, 3},
      {"verify_tp8_shard_ckpt_d96_q3", 8, 48, 96, 4, false, true, false, true, true, 0.0, 3},
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

      // Target-verify band (inkling_ar_sconv_norm_verify): T = B * q with the
      // causal conv along the draft-token axis and the cache read-only, NOT a
      // decode band with T = 144 rows (which is what these cases used to model).
      // activation=None / use_residual=true is what the model calls. The extra
      // cross-token re-reduces make this a different roofline from decode, so
      // the gates are set at ~60% of the slowest of two B60 runs x {bf16, fp16}
      // (fp16 is the slower dtype here and varies ~10% run to run).
      {"perf_verify_tp2_b16_q9_d1536", 2, 144, 1536, 4, false, true, true, true, true, 130.0, 9},
      {"perf_verify_tp4_b16_q9_d1536", 4, 144, 1536, 4, false, true, true, true, true, 150.0, 9},
      {"perf_verify_tp8_b16_q9_d1536", 8, 144, 1536, 4, false, true, true, true, true, 165.0, 9},
      {"perf_verify_tp2_b16_q9_d6144", 2, 144, 6144, 4, false, true, true, true, true, 160.0, 9},
      {"perf_verify_tp4_b16_q9_d6144", 4, 144, 6144, 4, false, true, true, true, true, 160.0, 9},
      {"perf_verify_tp8_b16_q9_d6144", 8, 144, 6144, 4, false, true, true, true, true, 165.0, 9},
      {"perf_verify_tp4_b96_q3_d1536", 4, 288, 1536, 4, false, true, true, true, true, 145.0, 3},
      {"perf_verify_tp8_b16_q3_d768", 8, 48, 768, 4, false, true, true, true, true, 26.0, 3},
      {"perf_verify_tp2_shard_prod_d3072_q9", 2, 144, 3072, 4, false, true, true, true, true, 140.0, 9},
      {"perf_verify_tp8_shard_prod_d768_q9", 8, 144, 768, 4, false, true, true, true, true, 90.0, 9},

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
double effective_bytes_verify(CaseConfig const& cfg) {
  double td = static_cast<double>(cfg.T) * cfg.D;
  double w = static_cast<double>(cfg.world);
  double shared = cfg.use_shared ? 1.0 : 0.0;
  double elem = static_cast<double>(sizeof(Element));
  int width_minus_one = std::max(cfg.W - 1, 0);
  // Every token re-reduces its own row plus each in-sequence predecessor inside
  // the conv window: sum over draft positions of min(tq, W-1) extra rows.
  double extra_rows = 0.0;
  int batch = cfg.T / cfg.draft;
  for (int tq = 0; tq < cfg.draft; ++tq) {
    extra_rows += std::min(tq, width_minus_one);
  }
  extra_rows *= static_cast<double>(batch);
  double staging_reads = (td + extra_rows * cfg.D) * w * (1.0 + shared) * elem;
  // Only the first W-1 draft positions of a sequence read the cache prefix.
  double cache_prefix_reads = static_cast<double>(batch) * std::min(cfg.draft, width_minus_one) * width_minus_one *
      cfg.D * elem;
  double weight_reads = td * static_cast<double>(cfg.W) * elem;
  double residual_reads = td * elem;
  double norm_reads = td * 2.0 * elem;  // gamma + the residual_out re-read
  double output_writes = td * 2.0 * elem;
  double inter_writes = td * width_minus_one * elem;
  return staging_reads + cache_prefix_reads + weight_reads + residual_reads + norm_reads + output_writes +
      inter_writes;
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
bool run_verify_case(sycl::queue& q, CaseConfig const& cfg, Options const& options) {
  if (cfg.T % cfg.draft != 0) {
    std::cerr << cfg.name << ": T (" << cfg.T << ") must be a multiple of draft_token_num (" << cfg.draft << ")\n";
    return false;
  }
  if (cfg.W - 1 > kMaxTaps) {
    std::cerr << cfg.name << ": W-1 exceeds kMaxTaps (" << kMaxTaps << ")\n";
    return false;
  }
  HostTensors<Element> h = initialize_case<Element>(cfg);
  reference_verify_case(cfg, h);

  DeviceBuffer<Element> d_partials(q, h.partials.size());
  DeviceBuffer<Element> d_shared(q, h.shared.size());
  DeviceBuffer<Element> d_residual_in(q, h.residual_in.size());
  DeviceBuffer<Element> d_residual_out(q, h.residual_out.size());
  DeviceBuffer<Element> d_hs_out(q, h.hs_out.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<Element> d_weight(q, h.weight.size());
  DeviceBuffer<Element> d_norm_weight(q, h.norm_weight.size());
  DeviceBuffer<Element> d_inter_out(q, h.inter_out.size());
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

  std::vector<Element> initial_inter = h.inter_out;
  d_inter_out.copy_from(initial_inter);

  int width_minus_one = cfg.W - 1;
  FusedVerifyParams<Element> params{
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
      d_inter_out.get(),
      cfg.world,
      cfg.T,
      cfg.D,
      cfg.W,
      cfg.draft,
      width_minus_one * cfg.D,
      cfg.D,
      cfg.draft * width_minus_one * cfg.D,
      width_minus_one * cfg.D,
      cfg.D,
      1.0e-5f,
      cfg.use_silu ? 1 : 0,
      cfg.use_residual ? 1 : 0,
      cfg.use_shared ? 1 : 0};

  auto launch_kernel = [&]() {
    return launch_fused_verify(q, params);
  };
  launch_kernel().wait();

  bool passed = true;
  if (options.verify) {
    d_residual_out.copy_to(h.residual_out);
    d_hs_out.copy_to(h.hs_out);
    d_inter_out.copy_to(h.inter_out);
    d_cache.copy_to(h.cache);
    std::string base = cfg.name + "/" + element_dtype_text<Element>();
    passed &= compare_vectors(
        base + "/residual", h.residual_out, h.residual_ref, default_atol<Element>(), default_rtol<Element>());
    passed &= compare_vectors(
        base + "/hs", h.hs_out, h.hs_ref, default_atol<Element>() * 2.0f, default_rtol<Element>() * 2.0f);
    // The verify kernel must leave the working conv cache untouched.
    passed &= compare_vectors(base + "/cache", h.cache, h.cache_ref, 0.0f, 0.0f);
    // inter_out is a raw copy of rows this kernel already holds -- bit-exact.
    passed &= compare_vectors(base + "/inter", h.inter_out, h.inter_ref, 0.0f, 0.0f);
  }

  double ms = time_ms(q, options.iterations, launch_kernel);
  double gbps = effective_bytes_verify<Element>(cfg) / (ms * 1.0e6);
  std::string perf_label = cfg.name + "/" + element_dtype_text<Element>();
  passed &= check_min_gbps(perf_label, gbps, cfg.min_gbps, options.perf_threshold_scale);
  double min_gbps = scaled_min_gbps(cfg.min_gbps, options.perf_threshold_scale);
  std::cout << "[ar_fused_verify] " << std::left << std::setw(38) << cfg.name << " dtype=" << std::setw(4)
            << element_dtype_text<Element>() << " world=" << cfg.world << " B=" << h.batch << " q=" << cfg.draft
            << " T=" << cfg.T << " D=" << cfg.D << " W=" << cfg.W << " silu=" << bool_text(cfg.use_silu)
            << " residual=" << bool_text(cfg.use_residual) << " shared=" << bool_text(cfg.use_shared)
            << " time_ms=" << std::fixed << std::setprecision(4) << ms << " eff_GBps=" << std::setprecision(2)
            << gbps << " min_GBps=" << min_gbps << " " << (passed ? "PASSED" : "FAILED") << "\n";
  return passed;
}

template <typename Element>
bool run_typed(sycl::queue& q, std::vector<CaseConfig> const& cases, Options const& options) {
  bool passed = true;
  for (auto const& cfg : cases) {
    if (cfg.draft > 0) {
      passed &= run_verify_case<Element>(q, cfg, options);
    } else {
      passed &= run_case<Element>(q, cfg, options);
    }
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
