/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "21_bmg_quantization_common.hpp"

namespace quant = cutlass::examples::bmg_quantization;

namespace {

template <typename T, int N>
using NativeVec = T __attribute__((ext_vector_type(N)));

struct Nvfp4Case {
  std::string name;
  int m = 1;
  int n = quant::kNvfp4GroupSize;
  double target_gbps = 0.0;
  // Number of expert slabs stacked in dim 0. The MoE weight tensors are
  // w13: (E, 2I/P, H) and w2: (E, H, I/P) -- see InklingNvfp4MoEMethod
  // .create_weights and ModelOptNvFp4FusedMoEMethod.create_weights. The block
  // scale surface restarts (row padding + swizzle) at every expert, so E is a
  // real layout parameter and not just a row multiplier. experts == 1 means a
  // plain 2-D linear weight shard.
  int experts = 1;
  // InklingModelOptNvfp4Config.dim1_group_size: 1 when scales_2d is false
  // (block scales group only along the last axis) and 16 when scales_2d is true
  // (block_sizes carries a "-2" entry, so scales group over a 16x16 tile).
  // dim2_group_size is always the NVFP4 group size 16.
  //
  // dim1_group_size is read in exactly one place -- InklingNvfp4MoEMethod
  // .create_weights, which asserts its layer is an InklingBatchDenseMLP, i.e.
  // the shared experts. Every other quantized weight goes through upstream
  // ModelOpt, whose scale params have no dim1 grouping:
  //   ModelOptFp4LinearMethod:      weight_scale (out_per_part, in_per_part/16)
  //   ModelOptNvFp4FusedMoEMethod:  w13_weight_scale (E, 2I/P, H/16)
  //                                 w2_weight_scale  (E, H,    I/P/16)
  // So scale_dim1_group == 16 is only reachable for shared-expert slabs.
  int scale_dim1_group = 1;
  // True for the fused gate+up (w13) slab. Whether its per-expert global scale2
  // widens to (E, 2) depends on which method owns the layer -- see
  // scale2_columns() below.
  bool fused_w13 = false;
};

// Columns of the per-expert global weight scale ("scale2") param.
//
// The checkpoint always stores (E,). InklingForCausalLM._ckpt_scale_to_modelopt
// widens it only when the destination param is 2-D, and the only 2-D scale2
// param is ModelOpt's w13_weight_scale_2, registered as (E, 2) for gated layers
// by ModelOptNvFp4FusedMoEMethod.create_weights (column 0 feeds the gate half's
// alpha, column 1 the up half's -- see _compute_gemm1_alphas).
//
// InklingNvfp4MoEMethod -- the shared-expert method, and the only consumer of
// dim1_group_size -- registers w13_scale2 as torch.empty(num_experts), i.e.
// (E,). So scale_dim1_group == 16 implies the un-widened (E,) shape even for
// the fused w13 slab, and the two settings are not independent.
int scale2_columns(Nvfp4Case const& cfg) {
  return (cfg.fused_w13 && cfg.scale_dim1_group == 1) ? 2 : 1;
}

template <typename Element>
struct Nvfp4Params {
  Element const* __restrict x = nullptr;
  uint8_t* __restrict packed = nullptr;
  uint8_t* __restrict scales = nullptr;
  int m = 0;
  int n = 0;
  int groups = 0;
  int rounded_groups = 0;
  float global_scale = 1.0f;
  float scale_factor = 1.0f;
  uint8_t const* __restrict raw_scale_code_lut = nullptr;
  float const* __restrict output_scale_lut = nullptr;
  uint64_t const* __restrict raw_scale_output_lut = nullptr;
};

template <typename Element>
struct Nvfp4BatchItem {
  Element const* __restrict x = nullptr;
  uint8_t* __restrict packed = nullptr;
  uint8_t* __restrict scales = nullptr;
  int m = 0;
  int n = 0;
  int groups = 0;
  int rounded_groups = 0;
  int total_groups = 0;
  int group_offset = 0;
  int row_offset = 0;
  float global_scale = 1.0f;
  float scale_factor = 1.0f;
  uint64_t const* __restrict raw_scale_output_lut = nullptr;
};

constexpr int kNvfp4OneDGroupThreshold = 64;

template <typename Element>
class Nvfp4LayoutKernel1D;

template <typename Element>
class Nvfp4LayoutBatchedKernel;

template <typename Element, int Groups, int GroupsPerItem>
class Nvfp4LayoutBatchedStaticGroupTileKernel;

template <typename Element, int Groups, int GroupsPerItem>
class Nvfp4LayoutKernel2DStaticGroupTile;

template <typename Element>
class Nvfp4LayoutKernel2D;

template <typename Element, int Groups>
class Nvfp4LayoutKernel2DStaticGroups;

template <int Groups>
CUTLASS_DEVICE int nvfp4_swizzled_scale_quad_index(int row, int group) {
  constexpr int kGroups4 = Groups / 4;
  int row_block = row >> 7;
  int row_rem = row & 127;
  int e = row_rem >> 5;
  int d = row_rem & 31;
  int c = group >> 2;
  return ((row_block * kGroups4 + c) * 32 + d) * 16 + (e << 2);
}

template <typename Element, int StaticGroups = 0, bool StoreScale = true, int StaticScaleGroups = StaticGroups>
CUTLASS_DEVICE uint8_t process_nvfp4_group_impl(Nvfp4Params<Element> const& params, int row, int group) {
  int col0 = group * quant::kNvfp4GroupSize;
  int cols = StaticGroups > 0 ? StaticGroups * quant::kNvfp4GroupSize : params.n;
  int input_base = row * cols + col0;
  int rounded_groups = StaticScaleGroups == 0 ? params.rounded_groups : StaticScaleGroups;

  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t> ||
                std::is_same_v<Element, cutlass::half_t>) {
    using RawWords = NativeVec<uint64_t, quant::kNvfp4GroupSize / 4>;
    Element const* input_ptr = static_cast<Element const*>(__builtin_assume_aligned(params.x + input_base, 32));
    RawWords raw_words = *reinterpret_cast<RawWords const*>(input_ptr);
    uint16_t max_abs_raw = 0;
#pragma unroll
    for (int word = 0; word < quant::kNvfp4GroupSize / 4; ++word) {
      uint64_t raw = raw_words[word];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint16_t bits = static_cast<uint16_t>(raw >> (16 * j));
        uint16_t abs_bits = quant::raw16_abs_bits(bits);
        max_abs_raw = abs_bits > max_abs_raw ? abs_bits : max_abs_raw;
      }
    }

    uint8_t scale_byte = 0;
    float output_scale = 0.0f;
    if constexpr (StaticGroups >= 48) {
      uint64_t packed_scale = params.raw_scale_output_lut[max_abs_raw];
      scale_byte = static_cast<uint8_t>(packed_scale);
      output_scale = sycl::bit_cast<float>(static_cast<uint32_t>(packed_scale >> 32));
    } else {
      if constexpr (std::is_same_v<Element, cutlass::half_t>) {
        if (params.raw_scale_output_lut != nullptr) {
          uint64_t packed_scale = params.raw_scale_output_lut[max_abs_raw];
          scale_byte = static_cast<uint8_t>(packed_scale);
          output_scale = sycl::bit_cast<float>(static_cast<uint32_t>(packed_scale >> 32));
        } else {
          float max_abs = quant::raw16_to_float<Element>(max_abs_raw);
          float raw_scale = max_abs * params.scale_factor;
          auto scale = quant::e4m3fn_encode_positive_with_inv_decode(raw_scale);
          scale_byte = scale.code;
          output_scale = params.global_scale * scale.inv_decoded;
        }
      } else {
        float max_abs = quant::raw16_to_float<Element>(max_abs_raw);
        float raw_scale = max_abs * params.scale_factor;
        auto scale = quant::e4m3fn_encode_positive_with_inv_decode(raw_scale);
        scale_byte = scale.code;
        output_scale = params.global_scale * scale.inv_decoded;
      }
    }

    if constexpr (StoreScale) {
      int scale_idx = 0;
      if constexpr (StaticScaleGroups > 0) {
        scale_idx = nvfp4_swizzled_scale_quad_index<StaticScaleGroups>(row, group) + (group & 3);
      } else {
        scale_idx = quant::nvfp4_swizzled_scale_index(row, group, rounded_groups);
      }
      params.scales[scale_idx] = scale_byte;
    }

    constexpr bool kOrderedQuant = false;
    uint32_t packed_lo =
        quant::quantize_e2m1_raw_word_pairs<Element, kOrderedQuant>(raw_words[0], output_scale) |
        (quant::quantize_e2m1_raw_word_pairs<Element, kOrderedQuant>(raw_words[1], output_scale) << 16);
    uint32_t packed_hi =
        quant::quantize_e2m1_raw_word_pairs<Element, kOrderedQuant>(raw_words[2], output_scale) |
        (quant::quantize_e2m1_raw_word_pairs<Element, kOrderedQuant>(raw_words[3], output_scale) << 16);
    NativeVec<uint32_t, 2> packed_words;
    packed_words[0] = packed_lo;
    packed_words[1] = packed_hi;
    int packed_base = (row * cols + col0) / 2;
    uint8_t* packed_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(params.packed + packed_base, 8));
    *reinterpret_cast<NativeVec<uint32_t, 2>*>(packed_ptr) = packed_words;
    return scale_byte;
  }

  float values[quant::kNvfp4GroupSize];
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < quant::kNvfp4GroupSize; ++i) {
    float v = quant::to_float(params.x[input_base + i]);
    values[i] = v;
    max_abs = sycl::fmax(max_abs, quant::abs_f(v));
  }

  float raw_scale = max_abs * params.scale_factor;
  auto scale = quant::e4m3fn_encode_positive_with_inv_decode(raw_scale);
  uint8_t scale_byte = scale.code;
  float output_scale = params.global_scale * scale.inv_decoded;

  if constexpr (StoreScale) {
    int scale_idx = 0;
    if constexpr (StaticScaleGroups > 0) {
      scale_idx = nvfp4_swizzled_scale_quad_index<StaticScaleGroups>(row, group) + (group & 3);
    } else {
      scale_idx = quant::nvfp4_swizzled_scale_index(row, group, rounded_groups);
    }
    params.scales[scale_idx] = scale_byte;
  }

  uint32_t packed_lo = 0;
  uint32_t packed_hi = 0;
#pragma unroll
  for (int i = 0; i < quant::kNvfp4GroupSize; i += 2) {
    float scaled0 = values[i] * output_scale;
    float scaled1 = values[i + 1] * output_scale;
    int pair = i / 2;
    uint32_t shifted_pair = quant::quantize_e2m1_pair(scaled0, scaled1) << (8 * (pair & 3));
    if (pair < 4) {
      packed_lo |= shifted_pair;
    } else {
      packed_hi |= shifted_pair;
    }
  }
  NativeVec<uint32_t, 2> packed_words;
  packed_words[0] = packed_lo;
  packed_words[1] = packed_hi;
  int packed_base = (row * cols + col0) / 2;
  *reinterpret_cast<NativeVec<uint32_t, 2>*>(params.packed + packed_base) = packed_words;
  return scale_byte;
}

template <typename Element, int StaticGroups = 0, int StaticScaleGroups = StaticGroups>
CUTLASS_DEVICE void process_nvfp4_group(Nvfp4Params<Element> const& params, int row, int group) {
  (void)process_nvfp4_group_impl<Element, StaticGroups, true, StaticScaleGroups>(params, row, group);
}

template <typename Element>
CUTLASS_DEVICE void process_nvfp4_group_static_if_known(
    Nvfp4Params<Element> const& params,
    int row,
    int group) {
  if (params.groups == 12) {
    process_nvfp4_group<Element, 12>(params, row, group);
  } else if (params.groups == 24) {
    process_nvfp4_group<Element, 24>(params, row, group);
  } else if (params.groups == 48) {
    process_nvfp4_group<Element, 48>(params, row, group);
  } else if (params.groups == 96) {
    process_nvfp4_group<Element, 96>(params, row, group);
  } else if (params.groups == 192) {
    process_nvfp4_group<Element, 192>(params, row, group);
  } else if (params.groups == 256) {
    process_nvfp4_group<Element, 256>(params, row, group);
  } else if (params.groups == 384) {
    process_nvfp4_group<Element, 384>(params, row, group);
  } else {
    process_nvfp4_group(params, row, group);
  }
}

template <typename Element, int Groups, int GroupsPerItem>
CUTLASS_DEVICE void process_nvfp4_static_group_tile_store(
    Nvfp4Params<Element> const& params,
    int row,
    int group_tile) {
  int group = group_tile * GroupsPerItem;
  int static_scale_idx = nvfp4_swizzled_scale_quad_index<Groups>(row, group);
#pragma unroll
  for (int i = 0; i < GroupsPerItem; i += 4) {
    uint8_t scale0 = process_nvfp4_group_impl<Element, Groups, false, Groups>(params, row, group + i);
    uint8_t scale1 = process_nvfp4_group_impl<Element, Groups, false, Groups>(params, row, group + i + 1);
    uint8_t scale2 = process_nvfp4_group_impl<Element, Groups, false, Groups>(params, row, group + i + 2);
    uint8_t scale3 = process_nvfp4_group_impl<Element, Groups, false, Groups>(params, row, group + i + 3);
    uint32_t scale_quad = static_cast<uint32_t>(scale0) |
        (static_cast<uint32_t>(scale1) << 8) |
        (static_cast<uint32_t>(scale2) << 16) |
        (static_cast<uint32_t>(scale3) << 24);
    int scale_idx = static_scale_idx;
    static_scale_idx += 512;
    uint8_t* scale_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(params.scales + scale_idx, 4));
    *reinterpret_cast<uint32_t*>(scale_ptr) = scale_quad;
  }
}

template <typename Element>
sycl::event launch_nvfp4_layout_1d(sycl::queue& queue, Nvfp4Params<Element> const& params) {
  int total_groups = params.m * params.groups;
  int global = quant::round_up(total_groups, quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4LayoutKernel1D<Element>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          int global_group = static_cast<int>(item.get_global_id(0));
          if (global_group >= total_groups) {
            return;
          }
          int row = global_group / params.groups;
          int group = global_group - row * params.groups;
          process_nvfp4_group(params, row, group);
        });
  });
}

template <typename Element>
sycl::event launch_nvfp4_layout_batched(
    sycl::queue& queue,
    Nvfp4BatchItem<Element> const* items,
    int item_count,
    int total_groups) {
  int global = quant::round_up(total_groups, quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4LayoutBatchedKernel<Element>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          int global_group = static_cast<int>(item.get_global_id(0));
          if (global_group >= total_groups) {
            return;
          }

          int item_idx = 0;
          int upper = item_count;
          while (item_idx + 1 < upper) {
            int mid = (item_idx + upper) >> 1;
            if (global_group >= items[mid].group_offset) {
              item_idx = mid;
            } else {
              upper = mid;
            }
          }

          Nvfp4BatchItem<Element> desc = items[item_idx];
          int local_group = global_group - desc.group_offset;
          if (local_group >= desc.total_groups) {
            return;
          }

          Nvfp4Params<Element> params;
          params.x = desc.x;
          params.packed = desc.packed;
          params.scales = desc.scales;
          params.m = desc.m;
          params.n = desc.n;
          params.groups = desc.groups;
          params.rounded_groups = desc.rounded_groups;
          params.global_scale = desc.global_scale;
          params.scale_factor = desc.scale_factor;
          params.raw_scale_code_lut = nullptr;
          params.output_scale_lut = nullptr;
          params.raw_scale_output_lut = desc.raw_scale_output_lut;

          int row = local_group / desc.groups;
          int group = local_group - row * desc.groups;
          process_nvfp4_group_static_if_known(params, row, group);
        });
  });
}

template <typename Element, int Groups, int GroupsPerItem>
sycl::event launch_nvfp4_layout_batched_2d_static_group_tile(
    sycl::queue& queue,
    Nvfp4BatchItem<Element> const* items,
    int const* row_to_item,
    int item_count,
    int total_rows,
    int row_base,
    float global_scale,
    float scale_factor,
    uint64_t const* raw_scale_output_lut,
    int row_block) {
  static_assert(GroupsPerItem % 4 == 0,
                "Batched static group tile stores scales in 4-byte swizzled quads");
  constexpr int kGroupTiles = Groups / GroupsPerItem;
  int rows_global = quant::round_up(total_rows, row_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4LayoutBatchedStaticGroupTileKernel<Element, Groups, GroupsPerItem>>(
        sycl::nd_range<2>(
            sycl::range<2>(
                static_cast<std::size_t>(rows_global),
                static_cast<std::size_t>(kGroupTiles)),
            sycl::range<2>(
                static_cast<std::size_t>(row_block),
                static_cast<std::size_t>(kGroupTiles))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int global_row = static_cast<int>(item.get_global_id(0));
          int group_tile = static_cast<int>(item.get_global_id(1));
          if (global_row >= total_rows) {
            return;
          }

          int item_idx = row_to_item[global_row];
          Nvfp4BatchItem<Element> desc = items[item_idx];
          int row = global_row + row_base - desc.row_offset;

          Nvfp4Params<Element> params;
          params.x = desc.x;
          params.packed = desc.packed;
          params.scales = desc.scales;
          params.m = desc.m;
          params.n = desc.n;
          params.groups = desc.groups;
          params.rounded_groups = desc.rounded_groups;
          params.global_scale = global_scale;
          params.scale_factor = scale_factor;
          params.raw_scale_code_lut = nullptr;
          params.output_scale_lut = nullptr;
          params.raw_scale_output_lut = raw_scale_output_lut;

          process_nvfp4_static_group_tile_store<Element, Groups, GroupsPerItem>(params, row, group_tile);
        });
  });
}

template <typename Element, int Groups, int GroupsPerItem, int ScaleGroups = Groups>
sycl::event launch_nvfp4_layout_2d_static_group_tile(
    sycl::queue& queue,
    Nvfp4Params<Element> const& params,
    int row_block) {
  static_assert(GroupsPerItem == 1 || GroupsPerItem == 2 || GroupsPerItem % 4 == 0,
                "GroupsPerItem must preserve 1-, 2-, or 4-scale swizzle adjacency");
  constexpr int kGroupTiles = Groups / GroupsPerItem;
  int rows_global = quant::round_up(params.m, row_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4LayoutKernel2DStaticGroupTile<Element, Groups, GroupsPerItem>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(rows_global), static_cast<std::size_t>(kGroupTiles)),
            sycl::range<2>(static_cast<std::size_t>(row_block), static_cast<std::size_t>(kGroupTiles))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0));
          if (row >= params.m) {
            return;
          }
          int group_tile = static_cast<int>(item.get_global_id(1));
          int group = group_tile * GroupsPerItem;
          int static_scale_idx = nvfp4_swizzled_scale_quad_index<ScaleGroups>(row, group);
          if constexpr (GroupsPerItem == 1) {
            uint8_t scale0 = process_nvfp4_group_impl<Element, Groups, false, ScaleGroups>(params, row, group);
            params.scales[static_scale_idx + (group & 3)] = scale0;
          } else if constexpr (GroupsPerItem == 2) {
            uint8_t scale0 = process_nvfp4_group_impl<Element, Groups, false, ScaleGroups>(params, row, group);
            uint8_t scale1 = process_nvfp4_group_impl<Element, Groups, false, ScaleGroups>(params, row, group + 1);
            uint16_t scale_pair = static_cast<uint16_t>(scale0) |
                (static_cast<uint16_t>(scale1) << 8);
            uint8_t* scale_ptr = static_cast<uint8_t*>(
                __builtin_assume_aligned(params.scales + static_scale_idx + (group & 3), 2));
            *reinterpret_cast<uint16_t*>(scale_ptr) = scale_pair;
          } else {
#pragma unroll
            for (int i = 0; i < GroupsPerItem; i += 4) {
              uint8_t scale0 = process_nvfp4_group_impl<Element, Groups, false, ScaleGroups>(params, row, group + i);
              uint8_t scale1 = process_nvfp4_group_impl<Element, Groups, false, ScaleGroups>(params, row, group + i + 1);
              uint8_t scale2 = process_nvfp4_group_impl<Element, Groups, false, ScaleGroups>(params, row, group + i + 2);
              uint8_t scale3 = process_nvfp4_group_impl<Element, Groups, false, ScaleGroups>(params, row, group + i + 3);
              uint32_t scale_quad = static_cast<uint32_t>(scale0) |
                  (static_cast<uint32_t>(scale1) << 8) |
                  (static_cast<uint32_t>(scale2) << 16) |
                  (static_cast<uint32_t>(scale3) << 24);
              int scale_idx = static_scale_idx;
              static_scale_idx += 512;
              uint8_t* scale_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(params.scales + scale_idx, 4));
              *reinterpret_cast<uint32_t*>(scale_ptr) = scale_quad;
            }
          }
        });
  });
}

template <typename Element, int Groups>
sycl::event launch_nvfp4_layout_2d_static(sycl::queue& queue, Nvfp4Params<Element> const& params) {
  int group_block = quant::choose_group_block(Groups);
  int groups_global = quant::round_up(Groups, group_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4LayoutKernel2DStaticGroups<Element, Groups>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(params.m), static_cast<std::size_t>(groups_global)),
            sycl::range<2>(1, static_cast<std::size_t>(group_block))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0));
          int group = static_cast<int>(item.get_global_id(1));
          process_nvfp4_group<Element, Groups>(params, row, group);
        });
  });
}

template <typename Element>
sycl::event launch_nvfp4_layout(sycl::queue& queue, Nvfp4Params<Element> const& params) {
  if (params.groups < kNvfp4OneDGroupThreshold) {
    if (params.groups == 48) {
      return launch_nvfp4_layout_2d_static_group_tile<Element, 48, 4>(queue, params, 8);
    }
    if (params.groups == 24) {
      if constexpr (std::is_same_v<Element, cutlass::half_t>) {
        return launch_nvfp4_layout_2d_static_group_tile<Element, 24, 4>(queue, params, 16);
      }
      return launch_nvfp4_layout_2d_static_group_tile<Element, 24, 2>(queue, params, 16);
    }
    if (params.groups == 12) {
      return launch_nvfp4_layout_2d_static_group_tile<Element, 12, 2>(queue, params, 16);
    }
    if (params.groups == 6) {
      return launch_nvfp4_layout_1d(queue, params);
    }
    return launch_nvfp4_layout_1d(queue, params);
  }

  if (params.groups == 96) {
    return launch_nvfp4_layout_2d_static_group_tile<Element, 96, 1>(queue, params, 2);
  }
  if (params.groups == 192) {
    return launch_nvfp4_layout_2d_static<Element, 192>(queue, params);
  }
  if (params.groups == 384) {
    return launch_nvfp4_layout_2d_static<Element, 384>(queue, params);
  }

  int group_block = quant::choose_group_block(params.groups);
  int groups_global = quant::round_up(params.groups, group_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4LayoutKernel2D<Element>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(params.m), static_cast<std::size_t>(groups_global)),
            sycl::range<2>(1, static_cast<std::size_t>(group_block))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0));
          int group = static_cast<int>(item.get_global_id(1));
          if (group >= params.groups) {
            return;
          }
          process_nvfp4_group(params, row, group);
        });
  });
}

// ---------------------------------------------------------------------------
// Per-expert slabs and the two scales_2d block-scale layouts.
//
// InklingModelOptNvfp4Config (models/inkling_common/quantization/config.py)
// derives two group sizes from the checkpoint's weight-quantizer block_sizes:
//
//   dim2_group_size = group_size = 16                (always; the "-1" entry)
//   dim1_group_size = 16 if scales_2d else 1         (the optional "-2" entry)
//
// InklingNvfp4MoEMethod.create_weights -- the shared-expert
// (InklingBatchDenseMLP) method, which it asserts on entry, and the only reader
// of dim1_group_size -- then registers, per expert,
//   w13_scale: (ceil_div(2I/P, dim1_group_size), ceil_div(H,  16))
//   w2_scale:  (ceil_div(H,    dim1_group_size), ceil_div(I/P, 16))
//
// so scales_2d=false gives one FP8 E4M3FN scale per 1x16 row group and
// scales_2d=true gives one per 16x16 tile. Routed experts and every quantized
// linear go through upstream ModelOpt instead, which has no dim1 grouping, so
// the 2-D-group form is reachable only for the shared experts (E = 2).
//
// Scale surface layout per expert:
//   dim1_group == 1  -> the ModelOpt/TRT-LLM swizzle this example already
//                       models: rows padded to 128, groups padded to 4.
//   dim1_group == 16 -> plain row-major (ceil_div(m,16), groups) exactly as
//                       create_weights registers it. swizzle_blockscale is
//                       defined over a (rows, cols/16) surface and is only
//                       applied to the 1-D-group form (see
//                       InklingBatchDenseMLP._build_fp4_linears, which
//                       reshapes w13_scale assuming dim1 == 2I/P), so the
//                       2-D-group form stays unswizzled here.
// ---------------------------------------------------------------------------

template <typename Element, int RowsPerScale>
class Nvfp4SlabKernel;

class Nvfp4Scale2WidenKernel;

template <typename Element>
struct Nvfp4SlabParams {
  Element const* __restrict x = nullptr;
  uint8_t* __restrict packed = nullptr;
  uint8_t* __restrict scales = nullptr;
  int m = 0;
  int n = 0;
  int groups = 0;
  int scale_rows = 0;      // ceil_div(m, RowsPerScale)
  int rounded_groups = 0;  // only used by the swizzled dim1_group == 1 layout
  int experts = 1;
  int64_t input_expert_stride = 0;
  int64_t packed_expert_stride = 0;
  int64_t scale_expert_stride = 0;
  float global_scale = 1.0f;
  float scale_factor = 1.0f;
};

template <typename Element, int RowsPerScale>
sycl::event launch_nvfp4_slab(sycl::queue& queue, Nvfp4SlabParams<Element> const& params) {
  int64_t blocks_per_expert = static_cast<int64_t>(params.scale_rows) * params.groups;
  int64_t total = blocks_per_expert * params.experts;
  constexpr int kWorkgroup = 128;
  std::size_t global = static_cast<std::size_t>((total + kWorkgroup - 1) / kWorkgroup) *
      static_cast<std::size_t>(kWorkgroup);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4SlabKernel<Element, RowsPerScale>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item) {
          int64_t idx = static_cast<int64_t>(item.get_global_linear_id());
          if (idx >= total) {
            return;
          }
          int expert = static_cast<int>(idx / blocks_per_expert);
          int rem = static_cast<int>(idx - static_cast<int64_t>(expert) * blocks_per_expert);
          int scale_row = rem / params.groups;
          int group = rem - scale_row * params.groups;

          int row0 = scale_row * RowsPerScale;
          int rows = RowsPerScale;
          if constexpr (RowsPerScale > 1) {
            int remaining = params.m - row0;
            rows = remaining < RowsPerScale ? remaining : RowsPerScale;
          }
          int col0 = group * quant::kNvfp4GroupSize;

          Element const* __restrict xe = params.x + expert * params.input_expert_stride;
          float max_abs = 0.0f;
          for (int r = 0; r < rows; ++r) {
            Element const* __restrict row_ptr =
                xe + static_cast<int64_t>(row0 + r) * params.n + col0;
            for (int i = 0; i < quant::kNvfp4GroupSize; ++i) {
              float v = quant::abs_f(quant::to_float(row_ptr[i]));
              max_abs = v > max_abs ? v : max_abs;
            }
          }

          // Use the exact-reciprocal encoder rather than a device fp32 divide:
          // icpx compiles the device divide to an approximation, which differs
          // from the host reference's global_scale / decoded by 1 ULP and flips
          // E2M1 codes at a rounding boundary. The 1-D fast path above uses the
          // same helper for the same reason.
          auto scale = quant::e4m3fn_encode_positive_with_inv_decode(max_abs * params.scale_factor);
          uint8_t scale_byte = scale.code;
          float output_scale = params.global_scale * scale.inv_decoded;

          int64_t scale_idx = expert * params.scale_expert_stride;
          if constexpr (RowsPerScale == 1) {
            scale_idx += quant::nvfp4_swizzled_scale_index(row0, group, params.rounded_groups);
          } else {
            scale_idx += static_cast<int64_t>(scale_row) * params.groups + group;
          }
          params.scales[scale_idx] = scale_byte;

          uint8_t* __restrict pe = params.packed + expert * params.packed_expert_stride;
          for (int r = 0; r < rows; ++r) {
            Element const* __restrict row_ptr =
                xe + static_cast<int64_t>(row0 + r) * params.n + col0;
            int64_t packed_base = (static_cast<int64_t>(row0 + r) * params.n + col0) / 2;
            for (int i = 0; i < quant::kNvfp4GroupSize; i += 2) {
              float v0 = quant::clamp_f(quant::to_float(row_ptr[i]) * output_scale,
                                        -quant::kE2M1Max,
                                        quant::kE2M1Max);
              float v1 = quant::clamp_f(quant::to_float(row_ptr[i + 1]) * output_scale,
                                        -quant::kE2M1Max,
                                        quant::kE2M1Max);
              pe[packed_base + i / 2] = quant::pack_e2m1_pair(quant::quantize_e2m1_code(v0),
                                                              quant::quantize_e2m1_code(v1));
            }
          }
        });
  });
}

// Per-expert global ("scale2") widening, mirroring
// InklingForCausalLM._ckpt_scale_to_modelopt: the checkpoint stores one FP32
// weight scale per expert, shape (E,). ModelOpt's fused w13 param is (E, 2)
// -- column 0 feeds the gate half, column 1 the up half (see
// _compute_gemm1_alphas in layers/quantization/modelopt_quant.py) -- while w2
// keeps (E,). The activation side stores a single global amax which becomes
// input_scale = amax / (448.0 * 6.0) broadcast over the param shape.
struct Nvfp4Scale2Params {
  float const* __restrict ckpt_scale2 = nullptr;  // (E,)
  float* __restrict widened = nullptr;            // (E, cols) with cols in {1, 2}
  float* __restrict alphas = nullptr;             // (E, cols)
  int experts = 0;
  int cols = 1;
  float input_scale = 1.0f;
};

sycl::event launch_nvfp4_scale2_widen(sycl::queue& queue, Nvfp4Scale2Params const& params) {
  std::size_t total = static_cast<std::size_t>(params.experts) * params.cols;
  constexpr int kWorkgroup = 64;
  std::size_t global = ((total + kWorkgroup - 1) / kWorkgroup) * kWorkgroup;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Nvfp4Scale2WidenKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item) {
          std::size_t idx = item.get_global_linear_id();
          if (idx >= total) {
            return;
          }
          int expert = static_cast<int>(idx) / params.cols;
          float v = params.ckpt_scale2[expert];
          params.widened[idx] = v;
          params.alphas[idx] = params.input_scale * v;
        });
  });
}

std::vector<Nvfp4Case> quick_suite() {
  return {
      {"tiny_reference_layout", 3, 16, 0.0},
      {"row_tail_scale_tail", 90, 48, 0.0},
      {"aligned_128x64", 128, 64, 0.0},
      {"padded_150x80", 150, 80, 0.0},
      // scales_2d=true (dim1_group_size=16): one FP8 scale per 16x16 tile.
      // Shared-expert-shaped (experts=2) because that is the only method that
      // reads dim1_group_size. m=3 and m=90 leave a partial trailing tile,
      // which the 2-D reduction must clamp to the live rows -- synthetic row
      // counts, since every real shared-expert slab dim (2I/P and H) is a
      // multiple of 16, so the clamp is unreachable from the shipped shapes.
      {"tiny_2d_group_tile", 3, 16, 0.0, 2, 16},
      {"row_tail_2d_group_tile", 90, 48, 0.0, 2, 16},
      {"aligned_2d_group_tile", 128, 64, 0.0, 2, 16},
      // Two expert slabs: the scale surface (and its row padding) restarts per
      // expert, so a 2-slab case is the smallest shape that catches a missing
      // per-expert stride. fused_w13=true drives the scale2 model: (E, 2) on
      // the 1-D (ModelOpt FusedMoE) path, (E,) on the 2-D (shared-expert) one.
      {"two_expert_slabs_1d", 48, 32, 0.0, 2, 1, true},
      {"two_expert_slabs_2d", 48, 32, 0.0, 2, 16, true},
  };
}

std::vector<Nvfp4Case> inkling_suite() {
  // ModelOpt NVFP4 packs Inkling weight tensors. Typical shard shapes:
  //   column-parallel (up/gate/qkv): rows = out_features/tp, cols = hidden
  //   row-parallel   (down/o_proj):  rows = out_features,    cols = hidden/tp
  // cfg-defaults (hidden=1536, intermediate=768) and production (hidden=6144,
  // intermediate=6144 — inferred from the row_down cols below) are both covered
  // across TP=1/2/4/8. NVFP4 group size is 16, so cols must divide 16 —
  // hidden/tp and 2*intermediate/tp for both configs satisfy this.
  //
  // Coverage note: Inkling's dense MLP wires gate_up as
  // MergedColumnParallelLinear(hidden, [intermediate]*2), so the fused packed
  // out_features is 2*intermediate. Per-rank shard rows = 2*intermediate/tp.
  // The "col_gate_fused" family covers that layout. The older "col_gate"
  // cases stay for regression parity (they were sized as a single gate slab
  // at rows = intermediate/tp for cfg / hidden/tp for prod). MoE routed
  // experts (InklingMoE.experts) quantize per-expert slabs of the same
  // [2*I/moe_tp, H] / [H, I/moe_tp] shape — the kernel is per-row, so a
  // single expert's rows quantize identically to a linear with matching
  // shape. The col_gate_fused and row_down cases below cover both dense and
  // per-expert layouts. The "qkvr" cases cover the attention-side fused
  // MergedColumnParallelLinear (q/k/v/r) whose rows include the K/V
  // replication term max(num_kv_heads, tp); TP=8 at both configs (nkv=4)
  // activates that replication branch — a shape family the mlp
  // col/row/gate_up cases do not reach.
  return {
      // Legacy quick-inkling cases (kept for regression parity).
      {"cfg_hidden_w13", 1024, 1536, 0.0},
      {"cfg_hidden_w2", 1536, 4096, 0.0},
      {"prod_tp8_hidden", 2048, 6144, 0.0},
      {"prod_tp4_intermediate", 4096, 6144, 0.0},
      // Config-defaults (hidden=1536, intermediate=768) — column-parallel gate/up
      // pre-fusion (rows = intermediate/tp = 768/tp), kept for regression parity.
      {"cfg_h1536_tp1_col_gate",    768, 1536, 0.0},
      {"cfg_h1536_tp2_col_gate",    384, 1536, 0.0},
      {"cfg_h1536_tp4_col_gate",    192, 1536, 0.0},
      {"cfg_h1536_tp8_col_gate",     96, 1536, 0.0},
      // Config-defaults fused gate_up (rows = 2*intermediate/tp). This is also
      // the MoE routed-expert w13 per-expert slab shape at moe_tp=tp.
      {"cfg_h1536_tp1_col_gate_fused", 1536, 1536, 0.0},
      {"cfg_h1536_tp2_col_gate_fused",  768, 1536, 0.0},
      {"cfg_h1536_tp4_col_gate_fused",  384, 1536, 0.0},
      {"cfg_h1536_tp8_col_gate_fused",  192, 1536, 0.0},
      // Config-defaults row-parallel down: rows=hidden(unshard), cols=intermediate/tp.
      // Also the MoE routed-expert w2 per-expert slab shape at moe_tp=tp.
      {"cfg_h1536_tp1_row_down",   1536,  768, 0.0},
      {"cfg_h1536_tp2_row_down",   1536,  384, 0.0},
      {"cfg_h1536_tp4_row_down",   1536,  192, 0.0},
      {"cfg_h1536_tp8_row_down",   1536,   96, 0.0},
      // Production (hidden=6144) column-parallel gate/up pre-fusion @
      // intermediate=6144 (rows = intermediate/tp = 6144/tp).
      {"prod_h6144_tp1_col_gate",  6144, 6144, 0.0},
      {"prod_h6144_tp2_col_gate",  3072, 6144, 0.0},
      {"prod_h6144_tp4_col_gate",  1536, 6144, 0.0},
      {"prod_h6144_tp8_col_gate",   768, 6144, 0.0},
      // Production fused gate_up (rows = 2*intermediate/tp = 12288/tp). Also
      // matches the MoE routed w13 per-expert slab at moe_tp=tp.
      {"prod_h6144_tp2_col_gate_fused", 6144, 6144, 0.0},
      {"prod_h6144_tp4_col_gate_fused", 3072, 6144, 0.0},
      {"prod_h6144_tp8_col_gate_fused", 1536, 6144, 0.0},
      // Production row-parallel down (also MoE w2 per-expert at moe_tp=tp).
      {"prod_h6144_tp2_row_down",  6144, 3072, 0.0},
      {"prod_h6144_tp4_row_down",  6144, 1536, 0.0},
      {"prod_h6144_tp8_row_down",  6144,  768, 0.0},
      // Fused attention QKVR (attn.py:269 output_sizes) is a
      // MergedColumnParallelLinear whose per-rank rows are
      //   (head_dim*num_heads + 2*head_dim*max(num_kv_heads,tp)
      //    + d_rel*num_heads) / tp
      // and cols = hidden_size. The max(num_kv_heads, tp) term makes TP=8
      // (num_kv_heads=4) exercise the KV-replication branch — a shard shape
      // that neither the gate/up nor the down cases above cover. d_rel=16
      // means the row count is already 16-multiple, so NVFP4 group=16
      // divisibility holds.
      // Config-defaults (nh=12, nkv=4, hd=128, d_rel=16, hidden=1536):
      //   rows = (128*12 + 2*128*max(4,tp) + 16*12)/tp
      //        = TP=1:2752, TP=2:1376, TP=4:688, TP=8:472 (kv replicated).
      {"cfg_h1536_tp1_qkvr", 2752, 1536, 0.0},
      {"cfg_h1536_tp2_qkvr", 1376, 1536, 0.0},
      {"cfg_h1536_tp4_qkvr",  688, 1536, 0.0},
      {"cfg_h1536_tp8_qkvr",  472, 1536, 0.0},
      // Production (nh=48, nkv=4, hd=128, d_rel=16, hidden=6144):
      //   rows = TP=1:7936, TP=2:3968, TP=4:1984, TP=8:1120 (kv replicated).
      {"prod_h6144_tp1_qkvr", 7936, 6144, 0.0},
      {"prod_h6144_tp2_qkvr", 3968, 6144, 0.0},
      {"prod_h6144_tp4_qkvr", 1984, 6144, 0.0},
      {"prod_h6144_tp8_qkvr", 1120, 6144, 0.0},
      // No scales_2d cases here on purpose: the linears run
      // ModelOptFp4LinearMethod, whose weight_scale is
      // (output_size_per_partition, input_size_per_partition / group_size) --
      // ungrouped in dim 0 regardless of InklingModelOptNvfp4Config.scales_2d.
      // dim1_group_size reaches only the shared-expert slabs, covered by the
      // "moe" suite.
  };
}

// MoE expert-weight slabs. Shapes come straight from
// ModelOptNvFp4FusedMoEMethod.create_weights (routed experts) and
// InklingNvfp4MoEMethod.create_weights (shared experts) with
// intermediate_size_per_partition = I/P:
//   w13: (E, 2I/P, H)   packed uint8 (E, 2I/P, H/2)
//   w2:  (E, H,  I/P)   packed uint8 (E, H,  I/P/2)
// E = n_routed_experts = 256 for the routed FusedMoE and
// E = n_shared_experts = 2 for the InklingBatchDenseMLP shared experts.
// I is 384 in the shipped checkpoint's routed experts and 3072 in production
// (dense_intermediate_size); H is hidden_size in {768, 1536, 6144}; P is the
// MoE TP degree in {1, 2, 4, 8}.
//
// The full (E, rows, cols) tensors are far larger than device memory at
// E=256 (e.g. 256 * 6144 * 6144 bf16 = 18 GiB), so the runner quantizes a
// bounded prefix of the expert dim (see kSlabInputByteBudget) while the
// per-expert global scale2 widening below and audit_moe_expert_shapes() cover
// the full E. Every case is a single kernel launch over its expert prefix, so
// the per-expert scale-surface restart is exercised, not just a row multiplier.
std::vector<Nvfp4Case> moe_suite() {
  return {
      // Routed experts, checkpoint intermediate_size = 384 (E = 256).
      {"moe_routed_i384_h1536_tp1_w13", 768, 1536, 0.0, 256, 1, true},
      {"moe_routed_i384_h1536_tp8_w13",  96, 1536, 0.0, 256, 1, true},
      {"moe_routed_i384_h1536_tp1_w2",  1536, 384, 0.0, 256, 1, false},
      {"moe_routed_i384_h1536_tp8_w2",  1536,  48, 0.0, 256, 1, false},
      {"moe_routed_i384_h768_tp4_w13",  192,  768, 0.0, 256, 1, true},
      {"moe_routed_i384_h768_tp4_w2",   768,   96, 0.0, 256, 1, false},
      {"moe_routed_i384_h6144_tp2_w13", 384, 6144, 0.0, 256, 1, true},
      {"moe_routed_i384_h6144_tp2_w2", 6144,  192, 0.0, 256, 1, false},
      // Routed experts, production intermediate_size = 3072 (E = 256).
      {"moe_routed_i3072_h6144_tp8_w13",  768, 6144, 0.0, 256, 1, true},
      {"moe_routed_i3072_h6144_tp8_w2",  6144,  384, 0.0, 256, 1, false},
      {"moe_routed_i3072_h1536_tp4_w13", 1536, 1536, 0.0, 256, 1, true},
      {"moe_routed_i3072_h1536_tp4_w2",  1536,  768, 0.0, 256, 1, false},
      // Shared experts (InklingBatchDenseMLP, E = 2, I = 3072).
      {"moe_shared_i3072_h1536_tp2_w13", 3072, 1536, 0.0, 2, 1, true},
      {"moe_shared_i3072_h1536_tp2_w2",  1536, 1536, 0.0, 2, 1, false},
      {"moe_shared_i3072_h6144_tp4_w13", 1536, 6144, 0.0, 2, 1, true},
      {"moe_shared_i3072_h6144_tp4_w2",  6144,  768, 0.0, 2, 1, false},
      {"moe_shared_i3072_h768_tp8_w13",   768,  768, 0.0, 2, 1, true},
      {"moe_shared_i3072_h768_tp8_w2",    768,  384, 0.0, 2, 1, false},
      // The same shared-expert slabs under scales_2d=true (16x16-tile block
      // scales). Only the shared experts appear here: dim1_group_size is read
      // solely by InklingNvfp4MoEMethod.create_weights, which asserts its layer
      // is an InklingBatchDenseMLP, so no E=256 slab can carry this layout.
      // fused_w13 stays meaningful (it is still the gate+up slab) but scale2
      // does not widen on this path -- see scale2_columns().
      {"moe_shared_i384_h1536_tp1_w13_s2d",   768, 1536, 0.0, 2, 16, true},
      {"moe_shared_i384_h1536_tp1_w2_s2d",   1536,  384, 0.0, 2, 16, false},
      {"moe_shared_i3072_h1536_tp2_w13_s2d", 3072, 1536, 0.0, 2, 16, true},
      {"moe_shared_i3072_h1536_tp2_w2_s2d",  1536, 1536, 0.0, 2, 16, false},
      {"moe_shared_i3072_h6144_tp4_w13_s2d", 1536, 6144, 0.0, 2, 16, true},
      {"moe_shared_i3072_h6144_tp4_w2_s2d",  6144,  768, 0.0, 2, 16, false},
      {"moe_shared_i3072_h768_tp8_w13_s2d",   768,  768, 0.0, 2, 16, true},
      {"moe_shared_i3072_h768_tp8_w2_s2d",    768,  384, 0.0, 2, 16, false},
  };
}

std::vector<Nvfp4Case> inkling_batched_suite() {
  return {
      {"cfg_hidden_w13", 1024, 1536, 0.0},
      {"cfg_h1536_tp1_col_gate", 768, 1536, 0.0},
      {"cfg_h1536_tp2_col_gate", 384, 1536, 0.0},
      {"cfg_h1536_tp4_col_gate", 192, 1536, 0.0},
      {"cfg_h1536_tp8_col_gate", 96, 1536, 0.0},
      {"cfg_h1536_tp1_col_gate_fused", 1536, 1536, 0.0},
      {"cfg_h1536_tp2_col_gate_fused", 768, 1536, 0.0},
      {"cfg_h1536_tp4_col_gate_fused", 384, 1536, 0.0},
      {"cfg_h1536_tp8_col_gate_fused", 192, 1536, 0.0},
      {"cfg_h1536_tp1_qkvr", 2752, 1536, 0.0},
      {"cfg_h1536_tp2_qkvr", 1376, 1536, 0.0},
      {"cfg_h1536_tp4_qkvr", 688, 1536, 0.0},
      {"cfg_h1536_tp8_qkvr", 472, 1536, 0.0},
  };
}

std::vector<Nvfp4Case> perf_suite() {
  // Perf sweep exercises production hidden=6144 sharded across TP=1/2/4/8 for
  // both column-parallel (rows=out_features/tp, cols=hidden) and row-parallel
  // (rows=hidden, cols=intermediate/tp) weight shapes, plus the fused gate_up
  // (rows = 2*intermediate/tp = 12288/tp). GB/s gates left at 0 for the new
  // shard shapes until an Inkling-specific baseline is captured.
  return {
      {"perf_4096x6144", 4096, 6144, 120.0},
      {"perf_8192x6144", 8192, 6144, 120.0},
      // Column-parallel gate/up shard at production hidden=6144 (pre-fusion).
      {"perf_prod_col_tp1_6144x6144", 6144, 6144, 0.0},
      {"perf_prod_col_tp2_3072x6144", 3072, 6144, 0.0},
      {"perf_prod_col_tp4_1536x6144", 1536, 6144, 0.0},
      {"perf_prod_col_tp8_768x6144",   768, 6144, 0.0},
      // Fused gate_up at production intermediate=6144 (rows = 2*I/tp).
      {"perf_prod_col_fused_tp2_6144x6144", 6144, 6144, 0.0},
      {"perf_prod_col_fused_tp4_3072x6144", 3072, 6144, 0.0},
      {"perf_prod_col_fused_tp8_1536x6144", 1536, 6144, 0.0},
      // Row-parallel down shard at production hidden=6144.
      {"perf_prod_row_tp2_6144x3072", 6144, 3072, 0.0},
      {"perf_prod_row_tp4_6144x1536", 6144, 1536, 0.0},
      {"perf_prod_row_tp8_6144x768",  6144,  768, 0.0},
      // Fused attention QKVR shards at production hidden=6144 across TP=1/2/4/8;
      // rows follow (hd*nh + 2*hd*max(nkv,tp) + drel*nh)/tp with nh=48, nkv=4,
      // hd=128, drel=16 → 7936/3968/1984/1120. TP=8 activates the K/V
      // replication branch (tp > num_kv_heads=4).
      {"perf_prod_qkvr_tp1_7936x6144", 7936, 6144, 0.0},
      {"perf_prod_qkvr_tp2_3968x6144", 3968, 6144, 0.0},
      {"perf_prod_qkvr_tp4_1984x6144", 1984, 6144, 0.0},
      {"perf_prod_qkvr_tp8_1120x6144", 1120, 6144, 0.0},
      // scales_2d=true (16x16-tile block scales) on the shared-expert slabs,
      // the only place dim1_group_size is read. Report-only (target 0.0): this
      // is a different kernel from the tuned 1-D path, so its GB/s has no
      // calibrated baseline yet. Do not guess a gate here -- see
      // 17_bmg_relative_attention_backend for the same convention.
      //
      // No 1-D-group slab cases here: for dim1_group_size == 1 the per-expert
      // surface is byte-identical to what the tuned launch_nvfp4_layout already
      // produces at the same (m, n), and it is those linear cases above that
      // carry the calibrated numbers. A 1-D slab perf case would only measure
      // the simple verification kernel and could not be compared against them.
      {"perf_s2d_shared_i3072_h6144_tp1_w13", 6144, 6144, 0.0, 2, 16, true},
      {"perf_s2d_shared_i3072_h6144_tp4_w13", 1536, 6144, 0.0, 2, 16, true},
      {"perf_s2d_shared_i3072_h6144_tp4_w2",  6144,  768, 0.0, 2, 16, false},
      {"perf_s2d_shared_i3072_h1536_tp1_w13", 6144, 1536, 0.0, 2, 16, true},
  };
}

std::vector<Nvfp4Case> make_suite(std::string const& suite) {
  if (suite == "quick") {
    return quick_suite();
  }
  if (suite == "inkling") {
    return inkling_suite();
  }
  if (suite == "perf") {
    return perf_suite();
  }
  if (suite == "moe") {
    return moe_suite();
  }
  return {};
}

bool parse_nv_shape(std::string const& text, Nvfp4Case& cfg) {
  if (text.empty()) {
    return true;
  }
  for (std::string const& item : quant::split(text, ',')) {
    auto eq = item.find('=');
    if (eq == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, eq);
    std::string value = item.substr(eq + 1);
    if (key == "name") {
      cfg.name = value;
    } else if (key == "m") {
      cfg.m = std::stoi(value);
    } else if (key == "n") {
      cfg.n = std::stoi(value);
    } else if (key == "target_gbps" || key == "target-gbps") {
      cfg.target_gbps = std::stod(value);
    } else if (key == "experts" || key == "e") {
      cfg.experts = std::stoi(value);
    } else if (key == "scales_2d" || key == "scales-2d") {
      // Mirrors InklingModelOptNvfp4Config: scales_2d selects
      // dim1_group_size = group_size (16) instead of 1.
      cfg.scale_dim1_group = std::stoi(value) != 0 ? quant::kNvfp4GroupSize : 1;
    } else if (key == "dim1_group" || key == "dim1-group") {
      cfg.scale_dim1_group = std::stoi(value);
    } else if (key == "fused_w13" || key == "fused-w13") {
      cfg.fused_w13 = std::stoi(value) != 0;
    } else {
      return false;
    }
  }
  return true;
}

void validate_case(Nvfp4Case& cfg) {
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
  if (cfg.m <= 0 || cfg.n <= 0) {
    throw std::invalid_argument("m and n must be positive");
  }
  if (cfg.n % quant::kNvfp4GroupSize != 0) {
    throw std::invalid_argument("n must be divisible by NVFP4 group size 16");
  }
  if (cfg.experts <= 0) {
    throw std::invalid_argument("experts must be positive");
  }
  // InklingModelOptNvfp4Config only ever sets dim1_group_size to 1 or to
  // group_size, and rejects any group_size other than 16.
  if (cfg.scale_dim1_group != 1 && cfg.scale_dim1_group != quant::kNvfp4GroupSize) {
    throw std::invalid_argument("scale_dim1_group must be 1 (scales_2d=false) or 16 (scales_2d=true)");
  }
}

template <typename Element>
void seed_edge_values(std::vector<Element>& input) {
  float values[] = {
      0.0f, 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f,
      -0.25f, -0.75f, -1.25f, -1.75f, -2.5f, -3.5f, -5.0f, 6.0f};
  int count = std::min<int>(static_cast<int>(input.size()), 16);
  for (int i = 0; i < count; ++i) {
    input[i] = quant::from_float<Element>(values[i]);
  }
}

template <typename Element>
std::vector<uint8_t> make_raw_scale_code_lut(float scale_factor) {
  std::vector<uint8_t> lut(65536, 0);
  for (int raw = 0; raw < 65536; ++raw) {
    uint16_t raw_abs = quant::raw16_abs_bits(static_cast<uint16_t>(raw));
    float max_abs = quant::to_float(Element::bitcast(raw_abs));
    lut[static_cast<std::size_t>(raw)] = quant::e4m3fn_encode_positive(max_abs * scale_factor);
  }
  return lut;
}

std::vector<float> make_output_scale_lut(float global_scale) {
  std::vector<float> lut(256, 0.0f);
  for (int code = 0; code < 256; ++code) {
    float decoded = quant::e4m3fn_decode(static_cast<uint8_t>(code));
    lut[static_cast<std::size_t>(code)] = decoded == 0.0f ? 0.0f : global_scale / decoded;
  }
  return lut;
}

std::vector<uint64_t> make_raw_scale_output_lut(
    std::vector<uint8_t> const& raw_scale_code_lut,
    std::vector<float> const& output_scale_lut) {
  std::vector<uint64_t> lut(65536, 0);
  for (int raw = 0; raw < 65536; ++raw) {
    uint8_t scale_byte = raw_scale_code_lut[static_cast<std::size_t>(raw)];
    uint32_t output_bits = sycl::bit_cast<uint32_t>(output_scale_lut[scale_byte]);
    lut[static_cast<std::size_t>(raw)] = static_cast<uint64_t>(scale_byte) |
        (static_cast<uint64_t>(output_bits) << 32);
  }
  return lut;
}

template <typename Element>
void nvfp4_reference(
    Nvfp4Case const& cfg,
    std::vector<Element> const& input,
    float global_scale,
    float scale_factor,
    std::vector<uint8_t>& packed,
    std::vector<uint8_t>& scales) {
  int groups = cfg.n / quant::kNvfp4GroupSize;
  int rounded_m = quant::round_up(cfg.m, 128);
  int rounded_groups = quant::round_up(groups, 4);
  packed.assign(static_cast<std::size_t>(cfg.m) * cfg.n / 2, 0);
  scales.assign(static_cast<std::size_t>(rounded_m) * rounded_groups, 0);

  for (int row = 0; row < cfg.m; ++row) {
    for (int group = 0; group < groups; ++group) {
      int col0 = group * quant::kNvfp4GroupSize;
      std::size_t input_base = static_cast<std::size_t>(row) * cfg.n + col0;
      float max_abs = 0.0f;
      for (int i = 0; i < quant::kNvfp4GroupSize; ++i) {
        max_abs = std::max(max_abs, std::fabs(quant::to_float(input[input_base + i])));
      }

      uint8_t scale_byte = quant::e4m3fn_encode(max_abs * scale_factor);
      float encoded_scale = quant::e4m3fn_decode(scale_byte);
      float output_scale = encoded_scale == 0.0f ? 0.0f : global_scale / encoded_scale;
      int scale_idx = quant::nvfp4_swizzled_scale_index(row, group, rounded_groups);
      scales[scale_idx] = scale_byte;

      std::size_t packed_base = (static_cast<std::size_t>(row) * cfg.n + col0) / 2;
      for (int i = 0; i < quant::kNvfp4GroupSize; i += 2) {
        float v0 = quant::clamp_f(quant::to_float(input[input_base + i]) * output_scale,
                                  -quant::kE2M1Max,
                                  quant::kE2M1Max);
        float v1 = quant::clamp_f(quant::to_float(input[input_base + i + 1]) * output_scale,
                                  -quant::kE2M1Max,
                                  quant::kE2M1Max);
        packed[packed_base + i / 2] =
            quant::pack_e2m1_pair(quant::quantize_e2m1_code(v0), quant::quantize_e2m1_code(v1));
      }
    }
  }
}

template <typename Element>
bool run_case_for_dtype(sycl::queue& queue, Nvfp4Case cfg, quant::Options const& options) {
  validate_case(cfg);

  int groups = cfg.n / quant::kNvfp4GroupSize;
  int rounded_m = quant::round_up(cfg.m, 128);
  int rounded_groups = quant::round_up(groups, 4);
  std::size_t input_count = static_cast<std::size_t>(cfg.m) * cfg.n;
  std::size_t packed_count = input_count / 2;
  std::size_t scale_count = static_cast<std::size_t>(rounded_m) * rounded_groups;

  std::vector<Element> h_input = quant::make_input<Element>(input_count, 20260217u, -4.0f, 4.0f);
  seed_edge_values(h_input);
  float amax = quant::max_abs_host(h_input);
  float global_scale = amax > 0.0f ? (quant::kE4M3FnMax * quant::kE2M1Max / amax) : 1.0f;
  float scale_factor = global_scale / quant::kE2M1Max;

  quant::DeviceBuffer<Element> d_input(queue, input_count);
  quant::DeviceBuffer<uint8_t> d_packed(queue, packed_count);
  quant::DeviceBuffer<uint8_t> d_scales(queue, scale_count);
  constexpr bool kCanUseScaleLuts = std::is_same_v<Element, cutlass::bfloat16_t> ||
      std::is_same_v<Element, cutlass::half_t>;
  bool use_scale_luts = kCanUseScaleLuts &&
      (groups >= 48 ||
       (std::is_same_v<Element, cutlass::half_t> && groups >= 12));
  bool use_combined_scale_lut = use_scale_luts;
  quant::DeviceBuffer<uint8_t> d_raw_scale_code_lut(queue, (use_scale_luts && !use_combined_scale_lut) ? 65536 : 1);
  quant::DeviceBuffer<float> d_output_scale_lut(queue, (use_scale_luts && !use_combined_scale_lut) ? 256 : 1);
  quant::DeviceBuffer<uint64_t> d_raw_scale_output_lut(queue, use_combined_scale_lut ? 65536 : 1);
  d_input.copy_from(h_input);
  if constexpr (kCanUseScaleLuts) {
    if (use_scale_luts) {
      std::vector<uint8_t> h_raw_scale_code_lut = make_raw_scale_code_lut<Element>(scale_factor);
      std::vector<float> h_output_scale_lut = make_output_scale_lut(global_scale);
      if (use_combined_scale_lut) {
        std::vector<uint64_t> h_raw_scale_output_lut =
            make_raw_scale_output_lut(h_raw_scale_code_lut, h_output_scale_lut);
        d_raw_scale_output_lut.copy_from(h_raw_scale_output_lut);
      } else {
        d_raw_scale_code_lut.copy_from(h_raw_scale_code_lut);
        d_output_scale_lut.copy_from(h_output_scale_lut);
      }
    }
  }

  Nvfp4Params<Element> params;
  params.x = d_input.get();
  params.packed = d_packed.get();
  params.scales = d_scales.get();
  params.m = cfg.m;
  params.n = cfg.n;
  params.groups = groups;
  params.rounded_groups = rounded_groups;
  params.global_scale = global_scale;
  params.scale_factor = scale_factor;
  params.raw_scale_code_lut = (use_scale_luts && !use_combined_scale_lut) ? d_raw_scale_code_lut.get() : nullptr;
  params.output_scale_lut = (use_scale_luts && !use_combined_scale_lut) ? d_output_scale_lut.get() : nullptr;
  params.raw_scale_output_lut = use_combined_scale_lut ? d_raw_scale_output_lut.get() : nullptr;

  auto launch = [&]() {
    return launch_nvfp4_layout<Element>(queue, params);
  };

  bool passed = true;
  if (options.verify) {
    d_packed.zero();
    d_scales.zero();
    launch().wait();

    std::vector<uint8_t> h_packed(packed_count);
    std::vector<uint8_t> h_scales(scale_count);
    d_packed.copy_to(h_packed);
    d_scales.copy_to(h_scales);

    std::vector<uint8_t> ref_packed;
    std::vector<uint8_t> ref_scales;
    nvfp4_reference(cfg, h_input, global_scale, scale_factor, ref_packed, ref_scales);

    quant::ByteCompareResult packed_cmp = quant::compare_bytes(h_packed, ref_packed);
    quant::ByteCompareResult scale_cmp = quant::compare_bytes(h_scales, ref_scales);
    if (!packed_cmp.passed || !scale_cmp.passed) {
      std::cerr << "  [FAIL] dtype=" << quant::element_dtype_text<Element>()
                << " case=" << cfg.name << "\n";
      quant::print_byte_compare("packed", packed_cmp);
      quant::print_byte_compare("scales", scale_cmp);
      passed = false;
    }
  }

  double mean_ms = 0.0;
  double gbps = 0.0;
  if (options.benchmark) {
    mean_ms = quant::benchmark_ms(launch, options.warmup, options.iterations);
    double moved_bytes = static_cast<double>(input_count * sizeof(Element) + packed_count + cfg.m * groups);
    gbps = quant::effective_gbps(moved_bytes, mean_ms);
    double target = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
    if (target > 0.0 && gbps < target && moved_bytes >= quant::kMinSustainedTargetBytes) {
      passed = false;
    }
  }

  std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] dtype=" << quant::element_dtype_text<Element>()
            << " case=" << cfg.name
            << " m=" << cfg.m
            << " n=" << cfg.n
            << " groups=" << groups
            << " global_scale=" << std::fixed << std::setprecision(3) << global_scale;
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

// ---------------------------------------------------------------------------
// Per-expert slab path (experts > 1 and/or scales_2d).
// ---------------------------------------------------------------------------

// Cap on the input bytes materialized for one slab case. The routed MoE tensors
// are (256, 2I/P, H), which never fits on one card, so the runner quantizes a
// bounded prefix of the expert dim. At least two experts are always run so the
// per-expert scale-surface restart stays covered.
constexpr std::size_t kSlabInputByteBudget = 256ull * 1024ull * 1024ull;

struct Nvfp4SlabLayout {
  int groups = 0;
  int rows_per_scale = 1;
  int scale_rows = 0;
  int rounded_groups = 0;
  int64_t input_expert_stride = 0;
  int64_t packed_expert_stride = 0;
  int64_t scale_expert_stride = 0;
};

Nvfp4SlabLayout make_slab_layout(Nvfp4Case const& cfg) {
  Nvfp4SlabLayout layout;
  layout.groups = cfg.n / quant::kNvfp4GroupSize;
  layout.rows_per_scale = cfg.scale_dim1_group;
  layout.scale_rows = quant::ceil_div(cfg.m, layout.rows_per_scale);
  layout.rounded_groups = quant::round_up(layout.groups, 4);
  layout.input_expert_stride = static_cast<int64_t>(cfg.m) * cfg.n;
  layout.packed_expert_stride = layout.input_expert_stride / 2;
  layout.scale_expert_stride = layout.rows_per_scale == 1
      // ModelOpt/TRT-LLM swizzle: rows padded to 128, groups padded to 4.
      ? static_cast<int64_t>(quant::round_up(cfg.m, 128)) * layout.rounded_groups
      // scales_2d: plain row-major (ceil_div(m,16), groups), as registered by
      // InklingNvfp4MoEMethod.create_weights.
      : static_cast<int64_t>(layout.scale_rows) * layout.groups;
  return layout;
}

template <typename Element>
void nvfp4_slab_reference(
    Nvfp4Case const& cfg,
    Nvfp4SlabLayout const& layout,
    int experts_run,
    std::vector<Element> const& input,
    float global_scale,
    float scale_factor,
    std::vector<uint8_t>& packed,
    std::vector<uint8_t>& scales) {
  packed.assign(static_cast<std::size_t>(layout.packed_expert_stride) * experts_run, 0);
  scales.assign(static_cast<std::size_t>(layout.scale_expert_stride) * experts_run, 0);

  for (int expert = 0; expert < experts_run; ++expert) {
    int64_t input_base_e = static_cast<int64_t>(expert) * layout.input_expert_stride;
    int64_t packed_base_e = static_cast<int64_t>(expert) * layout.packed_expert_stride;
    int64_t scale_base_e = static_cast<int64_t>(expert) * layout.scale_expert_stride;
    for (int scale_row = 0; scale_row < layout.scale_rows; ++scale_row) {
      int row0 = scale_row * layout.rows_per_scale;
      int rows = std::min(layout.rows_per_scale, cfg.m - row0);
      for (int group = 0; group < layout.groups; ++group) {
        int col0 = group * quant::kNvfp4GroupSize;
        float max_abs = 0.0f;
        for (int r = 0; r < rows; ++r) {
          int64_t base = input_base_e + static_cast<int64_t>(row0 + r) * cfg.n + col0;
          for (int i = 0; i < quant::kNvfp4GroupSize; ++i) {
            max_abs = std::max(max_abs, std::fabs(quant::to_float(input[base + i])));
          }
        }

        uint8_t scale_byte = quant::e4m3fn_encode(max_abs * scale_factor);
        float encoded_scale = quant::e4m3fn_decode(scale_byte);
        float output_scale = encoded_scale == 0.0f ? 0.0f : global_scale / encoded_scale;

        int64_t scale_idx = scale_base_e;
        if (layout.rows_per_scale == 1) {
          scale_idx += quant::nvfp4_swizzled_scale_index(row0, group, layout.rounded_groups);
        } else {
          scale_idx += static_cast<int64_t>(scale_row) * layout.groups + group;
        }
        scales[static_cast<std::size_t>(scale_idx)] = scale_byte;

        for (int r = 0; r < rows; ++r) {
          int64_t base = input_base_e + static_cast<int64_t>(row0 + r) * cfg.n + col0;
          int64_t out_base = packed_base_e +
              (static_cast<int64_t>(row0 + r) * cfg.n + col0) / 2;
          for (int i = 0; i < quant::kNvfp4GroupSize; i += 2) {
            float v0 = quant::clamp_f(quant::to_float(input[base + i]) * output_scale,
                                      -quant::kE2M1Max,
                                      quant::kE2M1Max);
            float v1 = quant::clamp_f(quant::to_float(input[base + i + 1]) * output_scale,
                                      -quant::kE2M1Max,
                                      quant::kE2M1Max);
            packed[static_cast<std::size_t>(out_base + i / 2)] =
                quant::pack_e2m1_pair(quant::quantize_e2m1_code(v0), quant::quantize_e2m1_code(v1));
          }
        }
      }
    }
  }
}

// Verify the loader's per-expert global-scale ("scale2") handling for the full
// expert count. Mirrors InklingForCausalLM._ckpt_scale_to_modelopt:
//   * the checkpoint stores one FP32 weight scale per expert, shape (E,);
//   * ModelOpt's fused w13 param is (E, 2) -- column 0 becomes the gate half's
//     alpha and column 1 the up half's (see _compute_gemm1_alphas) -- so the
//     loader broadcasts the checkpoint column into both;
//   * w2, and the shared-expert method's w13_scale2, keep (E,) (see
//     scale2_columns());
//   * input_amax -> input_scale = amax / (448.0 * 6.0), broadcast to the shape.
//
// This is dtype-independent, so run_cases calls it once per case rather than
// once per element type.
bool verify_scale2_widening(sycl::queue& queue, Nvfp4Case const& cfg) {
  int experts = cfg.experts;
  int cols = scale2_columns(cfg);

  std::vector<float> h_ckpt(static_cast<std::size_t>(experts));
  std::mt19937 gen(0x51A1E2u + static_cast<uint32_t>(experts));
  std::uniform_real_distribution<float> dist(0.005f, 0.25f);
  for (float& v : h_ckpt) {
    v = dist(gen);
  }
  float const input_amax = 12.5f;
  // input_scale = amax / (448.0 * 6.0) = amax / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX).
  float const input_scale = input_amax / (quant::kE4M3FnMax * quant::kE2M1Max);

  std::size_t widened_count = static_cast<std::size_t>(experts) * cols;
  quant::DeviceBuffer<float> d_ckpt(queue, h_ckpt.size());
  quant::DeviceBuffer<float> d_widened(queue, widened_count);
  quant::DeviceBuffer<float> d_alphas(queue, widened_count);
  d_ckpt.copy_from(h_ckpt);
  d_widened.zero();
  d_alphas.zero();

  Nvfp4Scale2Params params;
  params.ckpt_scale2 = d_ckpt.get();
  params.widened = d_widened.get();
  params.alphas = d_alphas.get();
  params.experts = experts;
  params.cols = cols;
  params.input_scale = input_scale;
  launch_nvfp4_scale2_widen(queue, params).wait();

  std::vector<float> h_widened(widened_count);
  std::vector<float> h_alphas(widened_count);
  d_widened.copy_to(h_widened);
  d_alphas.copy_to(h_alphas);

  std::size_t mismatches = 0;
  for (int e = 0; e < experts; ++e) {
    for (int c = 0; c < cols; ++c) {
      std::size_t idx = static_cast<std::size_t>(e) * cols + c;
      float expected_scale = h_ckpt[static_cast<std::size_t>(e)];
      float expected_alpha = input_scale * expected_scale;
      if (h_widened[idx] != expected_scale ||
          std::fabs(h_alphas[idx] - expected_alpha) >
              1e-6f * std::fabs(expected_alpha)) {
        if (mismatches == 0) {
          std::cerr << "  [FAIL] scale2 widening mismatch at expert=" << e
                    << " col=" << c << " got_scale=" << h_widened[idx]
                    << " want_scale=" << expected_scale
                    << " got_alpha=" << h_alphas[idx]
                    << " want_alpha=" << expected_alpha << "\n";
        }
        ++mismatches;
      }
    }
  }
  if (mismatches != 0) {
    std::cerr << "  [FAIL] scale2 widening: " << mismatches << " of " << widened_count
              << " entries wrong\n";
    return false;
  }
  // The widened columns must be equal: the checkpoint has a single per-expert
  // scale, so ModelOpt's gate and up alphas coincide for Inkling.
  if (cols == 2) {
    for (int e = 0; e < experts; ++e) {
      std::size_t idx = static_cast<std::size_t>(e) * 2;
      if (h_widened[idx] != h_widened[idx + 1]) {
        std::cerr << "  [FAIL] scale2 (E,2) columns differ at expert=" << e << "\n";
        return false;
      }
    }
  }
  std::cout << "  [PASS] scale2 case=" << cfg.name
            << " ckpt=(" << experts << ") -> param=(" << experts;
  if (cols == 2) {
    std::cout << ",2)  (ModelOpt fused w13: gate|up)";
  } else {
    std::cout << ")   (no widening on this path)";
  }
  std::cout << " input_scale=" << std::scientific << std::setprecision(4) << input_scale
            << std::defaultfloat << "\n";
  return true;
}

template <typename Element>
bool run_slab_for_dtype(sycl::queue& queue, Nvfp4Case cfg, quant::Options const& options) {
  validate_case(cfg);
  Nvfp4SlabLayout layout = make_slab_layout(cfg);

  std::size_t slab_bytes = static_cast<std::size_t>(layout.input_expert_stride) * sizeof(Element);
  int experts_run = cfg.experts;
  if (slab_bytes > 0) {
    std::size_t max_experts = kSlabInputByteBudget / slab_bytes;
    if (max_experts < 1) {
      max_experts = 1;
    }
    if (static_cast<std::size_t>(experts_run) > max_experts) {
      experts_run = static_cast<int>(max_experts);
    }
  }
  // Always keep at least two slabs when the tensor has them, so a dropped
  // per-expert stride cannot pass.
  if (cfg.experts >= 2 && experts_run < 2) {
    experts_run = 2;
  }

  std::size_t input_count = static_cast<std::size_t>(layout.input_expert_stride) * experts_run;
  std::size_t packed_count = static_cast<std::size_t>(layout.packed_expert_stride) * experts_run;
  std::size_t scale_count = static_cast<std::size_t>(layout.scale_expert_stride) * experts_run;

  std::vector<Element> h_input = quant::make_input<Element>(input_count, 20260903u, -4.0f, 4.0f);
  seed_edge_values(h_input);
  float amax = quant::max_abs_host(h_input);
  float global_scale = amax > 0.0f ? (quant::kE4M3FnMax * quant::kE2M1Max / amax) : 1.0f;
  float scale_factor = global_scale / quant::kE2M1Max;

  quant::DeviceBuffer<Element> d_input(queue, input_count);
  quant::DeviceBuffer<uint8_t> d_packed(queue, packed_count);
  quant::DeviceBuffer<uint8_t> d_scales(queue, scale_count);
  d_input.copy_from(h_input);

  Nvfp4SlabParams<Element> params;
  params.x = d_input.get();
  params.packed = d_packed.get();
  params.scales = d_scales.get();
  params.m = cfg.m;
  params.n = cfg.n;
  params.groups = layout.groups;
  params.scale_rows = layout.scale_rows;
  params.rounded_groups = layout.rounded_groups;
  params.experts = experts_run;
  params.input_expert_stride = layout.input_expert_stride;
  params.packed_expert_stride = layout.packed_expert_stride;
  params.scale_expert_stride = layout.scale_expert_stride;
  params.global_scale = global_scale;
  params.scale_factor = scale_factor;

  bool const two_d = layout.rows_per_scale != 1;
  auto launch = [&]() {
    return two_d ? launch_nvfp4_slab<Element, quant::kNvfp4GroupSize>(queue, params)
                 : launch_nvfp4_slab<Element, 1>(queue, params);
  };

  bool passed = true;
  if (options.verify) {
    d_packed.zero();
    d_scales.zero();
    launch().wait();

    std::vector<uint8_t> h_packed(packed_count);
    std::vector<uint8_t> h_scales(scale_count);
    d_packed.copy_to(h_packed);
    d_scales.copy_to(h_scales);

    std::vector<uint8_t> ref_packed;
    std::vector<uint8_t> ref_scales;
    nvfp4_slab_reference(cfg, layout, experts_run, h_input, global_scale, scale_factor,
                         ref_packed, ref_scales);

    quant::ByteCompareResult packed_cmp = quant::compare_bytes(h_packed, ref_packed);
    quant::ByteCompareResult scale_cmp = quant::compare_bytes(h_scales, ref_scales);
    if (!packed_cmp.passed || !scale_cmp.passed) {
      std::cerr << "  [FAIL] dtype=" << quant::element_dtype_text<Element>()
                << " case=" << cfg.name << "\n";
      quant::print_byte_compare("packed", packed_cmp);
      quant::print_byte_compare("scales", scale_cmp);
      passed = false;
    }
  }

  double mean_ms = 0.0;
  double gbps = 0.0;
  if (options.benchmark) {
    mean_ms = quant::benchmark_ms(launch, options.warmup, options.iterations);
    double moved_bytes = static_cast<double>(input_count * sizeof(Element)) +
        static_cast<double>(packed_count) +
        static_cast<double>(experts_run) * layout.scale_rows * layout.groups;
    gbps = quant::effective_gbps(moved_bytes, mean_ms);
    double target = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
    if (target > 0.0 && gbps < target && moved_bytes >= quant::kMinSustainedTargetBytes) {
      passed = false;
    }
  }

  std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] dtype=" << quant::element_dtype_text<Element>()
            << " case=" << cfg.name
            << " experts=" << cfg.experts
            << " experts_run=" << experts_run
            << " m=" << cfg.m
            << " n=" << cfg.n
            << " groups=" << layout.groups
            << " dim1_group=" << layout.rows_per_scale
            << " scale_rows=" << layout.scale_rows
            << " scale_layout=" << (two_d ? "linear_2d" : "swizzled_1d")
            << " global_scale=" << std::fixed << std::setprecision(3) << global_scale;
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

// Host-only shape table for the full MoE expert-weight space. The GPU cases can
// only materialize a prefix of the expert dim, so this prints -- and checks the
// divisibility invariants of -- the shape algebra for every shipped (E, I, H, P).
//
// The two expert families take different quant methods, and therefore different
// scale layouts:
//   routed (E = n_routed_experts = 256) -> ModelOptNvFp4FusedMoEMethod:
//       block scales (E, rows, cols/16); w13 scale2 (E, 2), w2 scale2 (E,);
//       dim1_group_size is not read, so scales_2d has no effect.
//   shared (E = n_shared_experts = 2)   -> InklingNvfp4MoEMethod:
//       block scales (E, ceil_div(rows, dim1_group_size), cols/16), i.e. both
//       scales_2d settings; scale2 stays (E,) for w13 and w2 alike.
//
// The divisibility checks below all hold for the shipped grid; they are guards
// against a future shape edit (e.g. a P that does not divide I, or a hidden
// size that is not a multiple of 16), not live failures.
bool audit_moe_expert_shapes() {
  struct Family {
    char const* label;
    int experts;
    bool scales_2d_reachable;
    bool scale2_widens;
  };
  Family const families[] = {
      {"routed", 256, false, true},
      {"shared", 2, true, false},
  };
  int const intermediates[] = {384, 3072};  // checkpoint routed, production
  int const hiddens[] = {768, 1536, 6144};
  int const tps[] = {1, 2, 4, 8};

  std::cout << "MoE expert-weight shape table (w13: (E, 2I/P, H), w2: (E, H, I/P))\n";
  bool passed = true;
  int checked = 0;
  for (Family const& family : families) {
    int experts = family.experts;
    for (int intermediate : intermediates) {
      for (int hidden : hiddens) {
        for (int tp : tps) {
          ++checked;
          if (intermediate % tp != 0) {
            std::cerr << "  [FAIL] I=" << intermediate << " not divisible by P=" << tp << "\n";
            passed = false;
            continue;
          }
          int ipp = intermediate / tp;
          int w13_rows = 2 * ipp;
          int w13_cols = hidden;
          int w2_rows = hidden;
          int w2_cols = ipp;
          // The last axis is the NVFP4-grouped one for both tensors.
          if (w13_cols % quant::kNvfp4GroupSize != 0 || w2_cols % quant::kNvfp4GroupSize != 0) {
            std::cerr << "  [FAIL] " << family.label << " I=" << intermediate << " H=" << hidden
                      << " P=" << tp << ": grouped axis not a multiple of 16 ("
                      << w13_cols << ", " << w2_cols << ")\n";
            passed = false;
            continue;
          }
          // Packed E2M1 halves the last axis.
          int64_t w13_packed = static_cast<int64_t>(experts) * w13_rows * (w13_cols / 2);
          int64_t w2_packed = static_cast<int64_t>(experts) * w2_rows * (w2_cols / 2);
          std::cout << "  " << family.label << " E=" << experts << " I=" << intermediate
                    << " H=" << hidden << " P=" << tp
                    << " w13=(" << experts << "," << w13_rows << "," << w13_cols << ")"
                    << " w2=(" << experts << "," << w2_rows << "," << w2_cols << ")"
                    << " packed_MiB=" << ((w13_packed + w2_packed) >> 20);
          std::cout << " scales2d=0:w13=(" << experts << "," << w13_rows
                    << "," << (w13_cols / quant::kNvfp4GroupSize) << ")"
                    << ",w2=(" << experts << "," << w2_rows
                    << "," << (w2_cols / quant::kNvfp4GroupSize) << ")";
          if (family.scales_2d_reachable) {
            int dim1 = quant::kNvfp4GroupSize;
            std::cout << " scales2d=1:w13=(" << experts << "," << quant::ceil_div(w13_rows, dim1)
                      << "," << (w13_cols / quant::kNvfp4GroupSize) << ")"
                      << ",w2=(" << experts << "," << quant::ceil_div(w2_rows, dim1)
                      << "," << (w2_cols / quant::kNvfp4GroupSize) << ")";
          } else {
            std::cout << " scales2d=1:n/a";
          }
          std::cout << " scale2:w13=(" << experts << (family.scale2_widens ? ",2)" : ")")
                    << ",w2=(" << experts << ")\n";
        }
      }
    }
  }
  std::cout << "  shape table: " << checked << " (family,I,H,P) combinations, "
            << (passed ? "all invariants hold" : "FAILURES above") << "\n";
  return passed;
}

bool needs_slab_path(Nvfp4Case const& cfg) {
  return cfg.experts != 1 || cfg.scale_dim1_group != 1;
}

template <typename Element>
bool run_batched_for_dtype(
    sycl::queue& queue,
    std::vector<Nvfp4Case> cases,
    quant::Options const& options) {
  std::vector<std::vector<Element>> h_inputs;
  std::vector<float> global_scales;
  std::vector<float> scale_factors;
  std::vector<quant::DeviceBuffer<Element>> d_inputs;
  std::vector<quant::DeviceBuffer<uint8_t>> d_packed;
  std::vector<quant::DeviceBuffer<uint8_t>> d_scales;
  std::vector<quant::DeviceBuffer<uint64_t>> d_raw_scale_output_luts;
  std::vector<std::size_t> packed_counts;
  std::vector<std::size_t> scale_counts;
  std::vector<Nvfp4BatchItem<Element>> h_items;
  std::vector<int> h_row_to_item;

  h_inputs.reserve(cases.size());
  global_scales.reserve(cases.size());
  scale_factors.reserve(cases.size());
  d_inputs.reserve(cases.size());
  d_packed.reserve(cases.size());
  d_scales.reserve(cases.size());
  d_raw_scale_output_luts.reserve(cases.size());
  packed_counts.reserve(cases.size());
  scale_counts.reserve(cases.size());
  h_items.reserve(cases.size());

  int total_groups = 0;
  int total_rows = 0;
  double moved_bytes = 0.0;
  int lut_cases = 0;
  bool all_groups_96 = true;
  for (Nvfp4Case& cfg : cases) {
    validate_case(cfg);

    int groups = cfg.n / quant::kNvfp4GroupSize;
    int rounded_m = quant::round_up(cfg.m, 128);
    int rounded_groups = quant::round_up(groups, 4);
    std::size_t input_count = static_cast<std::size_t>(cfg.m) * cfg.n;
    std::size_t packed_count = input_count / 2;
    std::size_t scale_count = static_cast<std::size_t>(rounded_m) * rounded_groups;
    all_groups_96 = all_groups_96 && groups == 96;

    h_inputs.push_back(quant::make_input<Element>(
        input_count,
        static_cast<uint32_t>(20260217u + h_inputs.size()),
        -4.0f,
        4.0f));
    seed_edge_values(h_inputs.back());
    float amax = quant::max_abs_host(h_inputs.back());
    float global_scale = amax > 0.0f ? (quant::kE4M3FnMax * quant::kE2M1Max / amax) : 1.0f;
    float scale_factor = global_scale / quant::kE2M1Max;
    global_scales.push_back(global_scale);
    scale_factors.push_back(scale_factor);

    d_inputs.emplace_back(queue, input_count);
    d_packed.emplace_back(queue, packed_count);
    d_scales.emplace_back(queue, scale_count);
    d_inputs.back().copy_from(h_inputs.back());

    constexpr bool kCanUseScaleLuts = std::is_same_v<Element, cutlass::bfloat16_t> ||
        std::is_same_v<Element, cutlass::half_t>;
    bool use_scale_luts = kCanUseScaleLuts &&
        (groups >= 48 ||
         (std::is_same_v<Element, cutlass::half_t> && groups >= 12));
    uint64_t const* raw_scale_output_lut = nullptr;
    if constexpr (kCanUseScaleLuts) {
      if (use_scale_luts) {
        std::vector<uint8_t> h_raw_scale_code_lut = make_raw_scale_code_lut<Element>(scale_factor);
        std::vector<float> h_output_scale_lut = make_output_scale_lut(global_scale);
        std::vector<uint64_t> h_raw_scale_output_lut =
            make_raw_scale_output_lut(h_raw_scale_code_lut, h_output_scale_lut);
        d_raw_scale_output_luts.emplace_back(queue, h_raw_scale_output_lut.size());
        d_raw_scale_output_luts.back().copy_from(h_raw_scale_output_lut);
        raw_scale_output_lut = d_raw_scale_output_luts.back().get();
        ++lut_cases;
      }
    }

    Nvfp4BatchItem<Element> item;
    item.x = d_inputs.back().get();
    item.packed = d_packed.back().get();
    item.scales = d_scales.back().get();
    item.m = cfg.m;
    item.n = cfg.n;
    item.groups = groups;
    item.rounded_groups = rounded_groups;
    item.total_groups = cfg.m * groups;
    item.group_offset = total_groups;
    item.row_offset = total_rows;
    item.global_scale = global_scale;
    item.scale_factor = scale_factor;
    item.raw_scale_output_lut = raw_scale_output_lut;
    h_items.push_back(item);

    total_groups += item.total_groups;
    int padded_rows = quant::round_up(cfg.m, 8);
    int item_idx = static_cast<int>(h_items.size()) - 1;
    for (int row = 0; row < padded_rows; ++row) {
      h_row_to_item.push_back(item_idx);
    }
    total_rows += padded_rows;
    moved_bytes += static_cast<double>(input_count * sizeof(Element) + packed_count + cfg.m * groups);
    packed_counts.push_back(packed_count);
    scale_counts.push_back(scale_count);
  }

  quant::DeviceBuffer<Nvfp4BatchItem<Element>> d_items(queue, h_items.size());
  d_items.copy_from(h_items);

  int split_row = total_rows;
  if (all_groups_96) {
    for (std::size_t i = 0; i < cases.size(); ++i) {
      if (cases[i].name.find("_qkvr") != std::string::npos) {
        split_row = h_items[i].row_offset;
        break;
      }
    }
  }
  std::vector<int> h_row_to_item_first;
  std::vector<int> h_row_to_item_second;
  if (all_groups_96 && split_row > 0 && split_row < total_rows) {
    h_row_to_item_first.assign(h_row_to_item.begin(), h_row_to_item.begin() + split_row);
    h_row_to_item_second.assign(h_row_to_item.begin() + split_row, h_row_to_item.end());
  } else {
    h_row_to_item_first = h_row_to_item;
  }
  quant::DeviceBuffer<int> d_row_to_item_first(queue, h_row_to_item_first.size());
  d_row_to_item_first.copy_from(h_row_to_item_first);
  quant::DeviceBuffer<int> d_row_to_item_second(queue, h_row_to_item_second.size());
  d_row_to_item_second.copy_from(h_row_to_item_second);

  auto launch = [&]() -> quant::EventBundle {
    if (all_groups_96) {
      sycl::event first = launch_nvfp4_layout_batched_2d_static_group_tile<Element, 96, 4>(
          queue,
          d_items.get(),
          d_row_to_item_first.get(),
          static_cast<int>(h_items.size()),
          split_row,
          0,
          h_items.empty() ? 1.0f : h_items.front().global_scale,
          h_items.empty() ? 1.0f : h_items.front().scale_factor,
          d_raw_scale_output_luts.empty() ? nullptr : d_raw_scale_output_luts.front().get(),
          8);
      if (!h_row_to_item_second.empty()) {
        sycl::event second = launch_nvfp4_layout_batched_2d_static_group_tile<Element, 96, 4>(
            queue,
            d_items.get(),
            d_row_to_item_second.get(),
            static_cast<int>(h_items.size()),
            total_rows - split_row,
            split_row,
            h_items.empty() ? 1.0f : h_items.front().global_scale,
            h_items.empty() ? 1.0f : h_items.front().scale_factor,
            d_raw_scale_output_luts.empty() ? nullptr : d_raw_scale_output_luts.front().get(),
            8);
        return quant::EventBundle(first, second);
      }
      return quant::EventBundle(first);
    }
    return quant::EventBundle(launch_nvfp4_layout_batched<Element>(
        queue,
        d_items.get(),
        static_cast<int>(h_items.size()),
        total_groups));
  };

  bool passed = true;
  if (options.verify) {
    for (auto& packed : d_packed) {
      packed.zero();
    }
    for (auto& scales : d_scales) {
      scales.zero();
    }
    launch().wait();

    for (std::size_t case_idx = 0; case_idx < cases.size(); ++case_idx) {
      std::vector<uint8_t> h_packed(packed_counts[case_idx]);
      std::vector<uint8_t> h_scales(scale_counts[case_idx]);
      d_packed[case_idx].copy_to(h_packed);
      d_scales[case_idx].copy_to(h_scales);

      std::vector<uint8_t> ref_packed;
      std::vector<uint8_t> ref_scales;
      nvfp4_reference(
          cases[case_idx],
          h_inputs[case_idx],
          global_scales[case_idx],
          scale_factors[case_idx],
          ref_packed,
          ref_scales);

      quant::ByteCompareResult packed_cmp = quant::compare_bytes(h_packed, ref_packed);
      quant::ByteCompareResult scale_cmp = quant::compare_bytes(h_scales, ref_scales);
      if (!packed_cmp.passed || !scale_cmp.passed) {
        std::cerr << "  [FAIL] dtype=" << quant::element_dtype_text<Element>()
                  << " batched_case=" << cases[case_idx].name << "\n";
        quant::print_byte_compare("packed", packed_cmp);
        quant::print_byte_compare("scales", scale_cmp);
        passed = false;
      }
    }
  }

  double mean_ms = 0.0;
  double gbps = 0.0;
  if (options.benchmark) {
    mean_ms = quant::benchmark_ms(launch, options.warmup, options.iterations);
    gbps = quant::effective_gbps(moved_bytes, mean_ms);
    if (options.target_gbps_set && gbps < options.target_gbps) {
      passed = false;
    }
  }

  std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] dtype=" << quant::element_dtype_text<Element>()
            << " case=inkling_batched_nv"
            << " cases=" << cases.size()
            << " lut_cases=" << lut_cases
            << " total_groups=" << total_groups
            << " global_scale=" << std::fixed << std::setprecision(3)
            << (global_scales.empty() ? 0.0f : global_scales.front());
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

template <typename Element>
bool run_batched_g96_contiguous_for_dtype(
    sycl::queue& queue,
    std::vector<Nvfp4Case> cases,
    quant::Options const& options) {
  int n = cases.front().n;
  int groups = n / quant::kNvfp4GroupSize;
  int case_count = static_cast<int>(cases.size());

  std::vector<Nvfp4Case> buckets;
  auto add_bucket = [&](int begin, int end, char const* name) {
    if (begin >= end) {
      return;
    }
    int rows = 0;
    for (int i = begin; i < end; ++i) {
      rows += cases[i].m;
    }
    buckets.push_back({name, rows, n, 0.0});
  };
  add_bucket(0, case_count, "g96_bucket_all");

  std::vector<std::vector<Element>> h_inputs;
  std::vector<float> global_scales;
  std::vector<float> scale_factors;
  std::vector<quant::DeviceBuffer<Element>> d_inputs;
  std::vector<quant::DeviceBuffer<uint8_t>> d_packed;
  std::vector<quant::DeviceBuffer<uint8_t>> d_scales;
  std::vector<quant::DeviceBuffer<uint64_t>> d_raw_scale_output_luts;
  std::vector<Nvfp4Params<Element>> params_vec;
  std::vector<std::size_t> packed_counts;
  std::vector<std::size_t> scale_counts;
  h_inputs.reserve(buckets.size());
  global_scales.reserve(buckets.size());
  scale_factors.reserve(buckets.size());
  d_inputs.reserve(buckets.size());
  d_packed.reserve(buckets.size());
  d_scales.reserve(buckets.size());
  d_raw_scale_output_luts.reserve(buckets.size());
  params_vec.reserve(buckets.size());
  packed_counts.reserve(buckets.size());
  scale_counts.reserve(buckets.size());

  double moved_bytes = 0.0;
  int total_rows = 0;
  int lut_buckets = 0;
  int case_idx = 0;
  for (std::size_t bucket_idx = 0; bucket_idx < buckets.size(); ++bucket_idx) {
    Nvfp4Case const& bucket = buckets[bucket_idx];
    std::size_t input_count = static_cast<std::size_t>(bucket.m) * bucket.n;
    std::size_t packed_count = input_count / 2;
    int rounded_m = quant::round_up(bucket.m, 128);
    int rounded_groups = quant::round_up(groups, 4);
    std::size_t scale_count = static_cast<std::size_t>(rounded_m) * rounded_groups;

    std::vector<Element> h_input;
    h_input.reserve(input_count);
    int bucket_end = case_count;
    for (; case_idx < bucket_end; ++case_idx) {
      std::size_t case_input_count = static_cast<std::size_t>(cases[case_idx].m) * cases[case_idx].n;
      std::vector<Element> case_input = quant::make_input<Element>(
          case_input_count,
          static_cast<uint32_t>(20260217u + case_idx),
          -4.0f,
          4.0f);
      seed_edge_values(case_input);
      h_input.insert(h_input.end(), case_input.begin(), case_input.end());
    }
    h_inputs.push_back(std::move(h_input));

    float amax = quant::max_abs_host(h_inputs.back());
    float global_scale = amax > 0.0f ? (quant::kE4M3FnMax * quant::kE2M1Max / amax) : 1.0f;
    float scale_factor = global_scale / quant::kE2M1Max;
    global_scales.push_back(global_scale);
    scale_factors.push_back(scale_factor);

    d_inputs.emplace_back(queue, input_count);
    d_packed.emplace_back(queue, packed_count);
    d_scales.emplace_back(queue, scale_count);
    d_inputs.back().copy_from(h_inputs.back());

    uint64_t const* raw_scale_output_lut = nullptr;
    constexpr bool kCanUseScaleLuts = std::is_same_v<Element, cutlass::bfloat16_t> ||
        std::is_same_v<Element, cutlass::half_t>;
    if constexpr (kCanUseScaleLuts) {
      std::vector<uint8_t> h_raw_scale_code_lut = make_raw_scale_code_lut<Element>(scale_factor);
      std::vector<float> h_output_scale_lut = make_output_scale_lut(global_scale);
      std::vector<uint64_t> h_raw_scale_output_lut =
          make_raw_scale_output_lut(h_raw_scale_code_lut, h_output_scale_lut);
      d_raw_scale_output_luts.emplace_back(queue, h_raw_scale_output_lut.size());
      d_raw_scale_output_luts.back().copy_from(h_raw_scale_output_lut);
      raw_scale_output_lut = d_raw_scale_output_luts.back().get();
      ++lut_buckets;
    }

    Nvfp4Params<Element> params;
    params.x = d_inputs.back().get();
    params.packed = d_packed.back().get();
    params.scales = d_scales.back().get();
    params.m = bucket.m;
    params.n = bucket.n;
    params.groups = groups;
    params.rounded_groups = rounded_groups;
    params.global_scale = global_scale;
    params.scale_factor = scale_factor;
    params.raw_scale_code_lut = nullptr;
    params.output_scale_lut = nullptr;
    params.raw_scale_output_lut = raw_scale_output_lut;
    params_vec.push_back(params);

    moved_bytes += static_cast<double>(input_count * sizeof(Element) + packed_count + bucket.m * groups);
    total_rows += bucket.m;
    packed_counts.push_back(packed_count);
    scale_counts.push_back(scale_count);
  }

  auto launch = [&]() -> quant::EventBundle {
    sycl::event first = launch_nvfp4_layout<Element>(queue, params_vec[0]);
    if (params_vec.size() > 1) {
      sycl::event second = launch_nvfp4_layout<Element>(queue, params_vec[1]);
      return quant::EventBundle(first, second);
    }
    return quant::EventBundle(first);
  };

  bool passed = true;
  if (options.verify) {
    for (auto& packed : d_packed) {
      packed.zero();
    }
    for (auto& scales : d_scales) {
      scales.zero();
    }
    launch().wait();

    for (std::size_t bucket_idx = 0; bucket_idx < buckets.size(); ++bucket_idx) {
      std::vector<uint8_t> h_packed(packed_counts[bucket_idx]);
      std::vector<uint8_t> h_scales(scale_counts[bucket_idx]);
      d_packed[bucket_idx].copy_to(h_packed);
      d_scales[bucket_idx].copy_to(h_scales);

      std::vector<uint8_t> ref_packed;
      std::vector<uint8_t> ref_scales;
      nvfp4_reference(
          buckets[bucket_idx],
          h_inputs[bucket_idx],
          global_scales[bucket_idx],
          scale_factors[bucket_idx],
          ref_packed,
          ref_scales);

      quant::ByteCompareResult packed_cmp = quant::compare_bytes(h_packed, ref_packed);
      quant::ByteCompareResult scale_cmp = quant::compare_bytes(h_scales, ref_scales);
      if (!packed_cmp.passed || !scale_cmp.passed) {
        std::cerr << "  [FAIL] dtype=" << quant::element_dtype_text<Element>()
                  << " bucket=" << buckets[bucket_idx].name << "\n";
        quant::print_byte_compare("packed", packed_cmp);
        quant::print_byte_compare("scales", scale_cmp);
        passed = false;
      }
    }
  }

  double mean_ms = 0.0;
  double gbps = 0.0;
  if (options.benchmark) {
    mean_ms = quant::benchmark_ms(launch, options.warmup, options.iterations);
    gbps = quant::effective_gbps(moved_bytes, mean_ms);
    if (options.target_gbps_set && gbps < options.target_gbps) {
      passed = false;
    }
  }

  std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] dtype=" << quant::element_dtype_text<Element>()
            << " case=inkling_batched_nv_contiguous"
            << " cases=" << cases.size()
            << " buckets=" << buckets.size()
            << " lut_buckets=" << lut_buckets
            << " total_rows=" << total_rows
            << " groups=" << groups
            << " global_scale=" << std::fixed << std::setprecision(3)
            << (global_scales.empty() ? 0.0f : global_scales.front());
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

bool run_batched_cases(
    sycl::queue& queue,
    std::vector<Nvfp4Case> const& cases,
    quant::Options const& options) {
  auto all_g96_same_n = [&]() {
    if (cases.empty()) {
      return false;
    }
    int n = cases.front().n;
    for (Nvfp4Case const& cfg : cases) {
      if (cfg.n != n || cfg.n / quant::kNvfp4GroupSize != 96) {
        return false;
      }
    }
    return true;
  };

  bool all_passed = true;
  if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFloat) {
    all_passed &= all_g96_same_n()
        ? run_batched_g96_contiguous_for_dtype<float>(queue, cases, options)
        : run_batched_for_dtype<float>(queue, cases, options);
  }
  if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kBf16) {
    all_passed &= all_g96_same_n()
        ? run_batched_g96_contiguous_for_dtype<cutlass::bfloat16_t>(queue, cases, options)
        : run_batched_for_dtype<cutlass::bfloat16_t>(queue, cases, options);
  }
  if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFp16) {
    all_passed &= all_g96_same_n()
        ? run_batched_g96_contiguous_for_dtype<cutlass::half_t>(queue, cases, options)
        : run_batched_for_dtype<cutlass::half_t>(queue, cases, options);
  }
  return all_passed;
}

bool run_cases(sycl::queue& queue, std::vector<Nvfp4Case> const& cases, quant::Options const& options) {
  bool all_passed = true;
  for (Nvfp4Case cfg : cases) {
    bool const slab = needs_slab_path(cfg);
    // The per-expert global scale is FP32 regardless of the weight dtype, so
    // check it once per case rather than inside the dtype loop below.
    if (slab && options.verify && cfg.experts > 1) {
      all_passed &= verify_scale2_widening(queue, cfg);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFloat) {
      all_passed &= slab ? run_slab_for_dtype<float>(queue, cfg, options)
                         : run_case_for_dtype<float>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kBf16) {
      all_passed &= slab ? run_slab_for_dtype<cutlass::bfloat16_t>(queue, cfg, options)
                         : run_case_for_dtype<cutlass::bfloat16_t>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFp16) {
      all_passed &= slab ? run_slab_for_dtype<cutlass::half_t>(queue, cfg, options)
                         : run_case_for_dtype<cutlass::half_t>(queue, cfg, options);
    }
  }
  return all_passed;
}

}  // namespace

int main(int argc, char const** argv) {
  quant::Options options;
  try {
    options = quant::parse_common_options(argc, argv);
    if (options.help) {
      std::cout << "21_bmg_nvfp4_layout: ModelOpt NVFP4 E2M1 pack plus swizzled FP8 scales\n\n";
      quant::print_common_usage(
          argv[0],
          "quick|inkling|perf|moe|inkling_batched",
          "m=<int>,n=<int>,experts=<int>,scales_2d=<0|1>,fused_w13=<0|1>");
      std::cout << "\n"
                << "  experts     number of expert slabs in dim 0 (MoE w13/w2); 1 = plain linear\n"
                << "  scales_2d   InklingModelOptNvfp4Config.scales_2d: 0 -> dim1_group_size=1\n"
                << "              (one FP8 scale per 1x16 row group, ModelOpt swizzle),\n"
                << "              1 -> dim1_group_size=16 (one per 16x16 tile, row-major)\n"
                << "  fused_w13   the slab is the fused gate+up weight, whose per-expert\n"
                << "              global scale2 the loader widens from (E,) to (E, 2)\n";
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<Nvfp4Case> cases;
  bool run_batched = false;
  if (!options.shape.empty()) {
    Nvfp4Case cfg;
    cfg.name = "custom_nvfp4";
    if (!parse_nv_shape(options.shape, cfg)) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    cases.push_back(cfg);
  } else if (options.suite == "inkling_batched" || options.suite == "inkling-batched") {
    cases = inkling_batched_suite();
    run_batched = true;
  } else {
    cases = make_suite(options.suite);
    if (cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = quant::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "21_bmg_nvfp4_layout: group=16 packed E2M1, FP8 E4M3FN scales, ModelOpt swizzle\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << quant::dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << quant::bool_text(options.verify)
              << " benchmark=" << quant::bool_text(options.benchmark) << "\n";

    bool passed = true;
    if (options.suite == "moe" && options.shape.empty()) {
      passed &= audit_moe_expert_shapes();
    }
    passed &= run_batched
        ? run_batched_cases(queue, cases, options)
        : run_cases(queue, cases, options);
    return passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
