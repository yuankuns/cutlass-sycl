/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "21_bmg_quantization_common.hpp"

namespace quant = cutlass::examples::bmg_quantization;

namespace {

enum class ScaleLayout {
  kRowMajor,
  kColumnMajor
};

struct Mxfp4Case {
  std::string name;
  int rows = 1;
  int cols = quant::kMxfp4GroupSize;
  ScaleLayout layout = ScaleLayout::kRowMajor;
  float eps = 1.0e-10f;
  double target_gbps = 0.0;
};

template <typename Element>
struct Mxfp4Params {
  Element const* __restrict x = nullptr;
  uint8_t* __restrict packed = nullptr;
  uint8_t* __restrict scales = nullptr;
  int rows = 0;
  int cols = 0;
  int groups = 0;
  int total_groups = 0;
  float eps = 1.0e-10f;
  int eps_exp = -34;
  bool column_major_scales = false;
};

template <typename Element>
struct Mxfp4BatchItem {
  Element const* __restrict x = nullptr;
  uint8_t* __restrict packed = nullptr;
  uint8_t* __restrict scales = nullptr;
  int rows = 0;
  int cols = 0;
  int groups = 0;
  int total_groups = 0;
  int group_offset = 0;
  float eps = 1.0e-10f;
  int eps_exp = -34;
};

struct Mxfp4ScaleTransposeBatchItem {
  uint8_t const* __restrict row_scales = nullptr;
  uint8_t* __restrict column_scales = nullptr;
  int rows = 0;
  int groups = 0;
  int tile_count = 0;
  int tile_offset = 0;
};

template <typename Element>
class Mxfp4MappingKernel1D;

template <typename Element>
class Mxfp4MappingBatchedRowMajorKernel;

class Mxfp4ScaleTransposeBatchedKernel;

template <typename Element, int Groups>
class Mxfp4MappingKernel1DStaticGroups;

template <typename Element>
class Mxfp4MappingKernel2D;

template <typename Element>
class Mxfp4MappingKernelTiled2D;

template <typename Element, int Groups>
class Mxfp4MappingKernelTiled2DStaticGroups;

template <typename Element, bool EvenRows>
class Mxfp4MappingKernelColumnMajorRowPairTiled2D;

template <typename Element, int Groups>
class Mxfp4MappingKernel2DStaticGroups;

template <bool RowsMultiple8>
class Mxfp4ScaleTransposeKernel;

template <typename Element, bool StoreScale, int StaticGroups = 0>
CUTLASS_DEVICE uint8_t process_mxfp4_group_impl(Mxfp4Params<Element> const& params, int row, int group) {
  int col0 = group * quant::kMxfp4GroupSize;
  int groups = StaticGroups > 0 ? StaticGroups : params.groups;
  int cols = StaticGroups > 0 ? StaticGroups * quant::kMxfp4GroupSize : params.cols;
  int base = row * cols + col0;
  int packed_base = (row * cols) / 2 + group * (quant::kMxfp4GroupSize / 2);

  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t> ||
                std::is_same_v<Element, cutlass::half_t>) {
    using RawWords = sycl::vec<uint64_t, quant::kMxfp4GroupSize / 4>;
    Element const* input_ptr = static_cast<Element const*>(__builtin_assume_aligned(params.x + base, 64));
    RawWords raw_words = *reinterpret_cast<RawWords const*>(input_ptr);
    uint16_t local_absmax_raw = 0;
#pragma unroll
    for (int word = 0; word < quant::kMxfp4GroupSize / 4; ++word) {
      uint64_t raw = raw_words[word];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint16_t bits = static_cast<uint16_t>(raw >> (16 * j));
        uint16_t abs_bits = quant::raw16_abs_bits(bits);
        local_absmax_raw = abs_bits > local_absmax_raw ? abs_bits : local_absmax_raw;
      }
    }
    int eps_exp = params.eps_exp;
    int raw_exp = quant::raw16_floor_log2_positive<Element>(local_absmax_raw);
    int max_exp = raw_exp;
    if (local_absmax_raw == 0 || raw_exp <= eps_exp) {
      float local_absmax = sycl::fmax(params.eps, quant::raw16_to_float<Element>(local_absmax_raw));
      max_exp = quant::floor_log2_positive(local_absmax);
    }

    int shared_exp = max_exp - 2;
    shared_exp = quant::clamp_exponent_to_ue8m0(shared_exp);
    float inv_scale = quant::pow2_int(-shared_exp);
    uint8_t* packed_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(params.packed + packed_base, 16));

    uint32_t packed_0 =
        quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[0], inv_scale) |
        (quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[1], inv_scale) << 16);
    uint32_t packed_1 =
        quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[2], inv_scale) |
        (quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[3], inv_scale) << 16);
    uint32_t packed_2 =
        quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[4], inv_scale) |
        (quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[5], inv_scale) << 16);
    uint32_t packed_3 =
        quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[6], inv_scale) |
        (quant::quantize_e2m1_raw_word_pairs<Element>(raw_words[7], inv_scale) << 16);
    sycl::vec<uint32_t, 4> packed_words;
    packed_words[0] = packed_0;
    packed_words[1] = packed_1;
    packed_words[2] = packed_2;
    packed_words[3] = packed_3;
    *reinterpret_cast<sycl::vec<uint32_t, 4>*>(packed_ptr) = packed_words;

    uint8_t scale_byte = quant::encode_ue8m0_exponent(shared_exp);
    if constexpr (StoreScale) {
      int scale_idx = params.column_major_scales ? group * params.rows + row : row * groups + group;
      params.scales[scale_idx] = scale_byte;
    }
    return scale_byte;
  }

  float values[quant::kMxfp4GroupSize];
  float local_absmax = params.eps;
#pragma unroll
  for (int i = 0; i < quant::kMxfp4GroupSize; ++i) {
    float value = quant::to_float(params.x[base + i]);
    values[i] = value;
    local_absmax = sycl::fmax(local_absmax, quant::abs_f(value));
  }

  int shared_exp = quant::floor_log2_positive(local_absmax) - 2;
  shared_exp = quant::clamp_exponent_to_ue8m0(shared_exp);
  float inv_scale = quant::pow2_int(-shared_exp);

  uint32_t packed_0 = 0;
  uint32_t packed_1 = 0;
  uint32_t packed_2 = 0;
  uint32_t packed_3 = 0;
#pragma unroll
  for (int i = 0; i < quant::kMxfp4GroupSize; i += 2) {
    float scaled0 = values[i] * inv_scale;
    float scaled1 = values[i + 1] * inv_scale;
    uint32_t packed_pair = quant::quantize_e2m1_pair(scaled0, scaled1);
    int pair = i / 2;
    uint32_t shifted_pair = packed_pair << (8 * (pair & 3));
    if (pair < 4) {
      packed_0 |= shifted_pair;
    } else if (pair < 8) {
      packed_1 |= shifted_pair;
    } else if (pair < 12) {
      packed_2 |= shifted_pair;
    } else {
      packed_3 |= shifted_pair;
    }
  }
  sycl::vec<uint32_t, 4> packed_words;
  packed_words[0] = packed_0;
  packed_words[1] = packed_1;
  packed_words[2] = packed_2;
  packed_words[3] = packed_3;
  *reinterpret_cast<sycl::vec<uint32_t, 4>*>(params.packed + packed_base) = packed_words;

  uint8_t scale_byte = quant::encode_ue8m0_exponent(shared_exp);
  if constexpr (StoreScale) {
    int scale_idx = params.column_major_scales ? group * params.rows + row : row * groups + group;
    params.scales[scale_idx] = scale_byte;
  }
  return scale_byte;
}

template <typename Element, int StaticGroups = 0>
CUTLASS_DEVICE void process_mxfp4_group(Mxfp4Params<Element> const& params, int row, int group) {
  (void)process_mxfp4_group_impl<Element, true, StaticGroups>(params, row, group);
}

template <typename Element>
CUTLASS_DEVICE void process_mxfp4_group_static_if_known(
    Mxfp4Params<Element> const& params,
    int row,
    int group) {
  if (params.groups == 6) {
    process_mxfp4_group<Element, 6>(params, row, group);
  } else if (params.groups == 12) {
    process_mxfp4_group<Element, 12>(params, row, group);
  } else if (params.groups == 24) {
    process_mxfp4_group<Element, 24>(params, row, group);
  } else if (params.groups == 48) {
    process_mxfp4_group<Element, 48>(params, row, group);
  } else if (params.groups == 96) {
    process_mxfp4_group<Element, 96>(params, row, group);
  } else if (params.groups == 192) {
    process_mxfp4_group<Element, 192>(params, row, group);
  } else {
    process_mxfp4_group(params, row, group);
  }
}

template <typename Element>
sycl::event launch_mxfp4_mapping_tiled_2d(
    sycl::queue& queue,
    Mxfp4Params<Element> const& params,
    int row_block) {
  int rows_global = quant::round_up(params.rows, row_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingKernelTiled2D<Element>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(rows_global), static_cast<std::size_t>(params.groups)),
            sycl::range<2>(static_cast<std::size_t>(row_block), static_cast<std::size_t>(params.groups))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0));
          if (row >= params.rows) {
            return;
          }
          int group = static_cast<int>(item.get_global_id(1));
          process_mxfp4_group(params, row, group);
        });
  });
}

template <typename Element, int Groups>
sycl::event launch_mxfp4_mapping_tiled_2d_static(
    sycl::queue& queue,
    Mxfp4Params<Element> const& params,
    int row_block) {
  int rows_global = quant::round_up(params.rows, row_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingKernelTiled2DStaticGroups<Element, Groups>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(rows_global), static_cast<std::size_t>(Groups)),
            sycl::range<2>(static_cast<std::size_t>(row_block), static_cast<std::size_t>(Groups))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0));
          if (row >= params.rows) {
            return;
          }
          int group = static_cast<int>(item.get_global_id(1));
          process_mxfp4_group<Element, Groups>(params, row, group);
        });
  });
}

template <typename Element>
sycl::event launch_mxfp4_mapping_1d(sycl::queue& queue, Mxfp4Params<Element> const& params) {
  int global = quant::round_up(params.total_groups, quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingKernel1D<Element>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          int global_group = static_cast<int>(item.get_global_id(0));
          if (global_group >= params.total_groups) {
            return;
          }
          int row = global_group / params.groups;
          int group = global_group - row * params.groups;
          process_mxfp4_group(params, row, group);
        });
  });
}

template <typename Element>
sycl::event launch_mxfp4_mapping_batched_row_major(
    sycl::queue& queue,
    Mxfp4BatchItem<Element> const* items,
    int item_count,
    int total_groups) {
  int global = quant::round_up(total_groups, quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingBatchedRowMajorKernel<Element>>(
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

          Mxfp4BatchItem<Element> desc = items[item_idx];
          int local_group = global_group - desc.group_offset;
          if (local_group >= desc.total_groups) {
            return;
          }

          Mxfp4Params<Element> params;
          params.x = desc.x;
          params.packed = desc.packed;
          params.scales = desc.scales;
          params.rows = desc.rows;
          params.cols = desc.cols;
          params.groups = desc.groups;
          params.total_groups = desc.total_groups;
          params.eps = desc.eps;
          params.eps_exp = desc.eps_exp;
          params.column_major_scales = false;

          int row = local_group / desc.groups;
          int group = local_group - row * desc.groups;
          process_mxfp4_group_static_if_known(params, row, group);
        });
  });
}

template <typename Element, int Groups>
sycl::event launch_mxfp4_mapping_1d_static(sycl::queue& queue, Mxfp4Params<Element> const& params) {
  int total_groups = params.rows * Groups;
  int global = quant::round_up(total_groups, quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingKernel1DStaticGroups<Element, Groups>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          int global_group = static_cast<int>(item.get_global_id(0));
          if (global_group >= total_groups) {
            return;
          }
          int row = global_group / Groups;
          int group = global_group - row * Groups;
          process_mxfp4_group(params, row, group);
        });
  });
}

template <typename Element, bool EvenRows>
sycl::event launch_mxfp4_mapping_column_major_row_pair_tiled_2d(
    sycl::queue& queue,
    Mxfp4Params<Element> const& params,
    int row_pair_block) {
  int row_factor = 2;
  int row_pairs = quant::ceil_div(params.rows, row_factor);
  int row_pairs_global = quant::round_up(row_pairs, row_pair_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingKernelColumnMajorRowPairTiled2D<Element, EvenRows>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(row_pairs_global), static_cast<std::size_t>(params.groups)),
            sycl::range<2>(static_cast<std::size_t>(row_pair_block), static_cast<std::size_t>(params.groups))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0)) * row_factor;
          if (row >= params.rows) {
            return;
          }
          int group = static_cast<int>(item.get_global_id(1));
          uint8_t scale0 = process_mxfp4_group_impl<Element, false>(params, row, group);
          int scale_idx = group * params.rows + row;
          if constexpr (EvenRows) {
            uint8_t scale1 = process_mxfp4_group_impl<Element, false>(params, row + 1, group);
            uint16_t scale_pair = static_cast<uint16_t>(scale0) |
                static_cast<uint16_t>(scale1) << 8;
            *reinterpret_cast<uint16_t*>(params.scales + scale_idx) = scale_pair;
          } else {
            if (row + 1 < params.rows) {
              uint8_t scale1 = process_mxfp4_group_impl<Element, false>(params, row + 1, group);
              params.scales[scale_idx] = scale0;
              params.scales[scale_idx + 1] = scale1;
            } else {
              params.scales[scale_idx] = scale0;
            }
          }
        });
  });
}

template <bool RowsMultiple8>
sycl::event launch_mxfp4_scale_transpose(
    sycl::queue& queue,
    uint8_t const* __restrict row_scales,
    uint8_t* __restrict column_scales,
    int rows,
    int groups) {
  constexpr int kRowsPerItem = RowsMultiple8 ? 8 : 4;
  int group_block = quant::choose_group_block(groups);
  int row_tile_block = std::max(1, quant::kDefaultBlock / group_block);
  int row_tiles = quant::ceil_div(rows, kRowsPerItem);
  int row_tiles_global = quant::round_up(row_tiles, row_tile_block);
  int groups_global = quant::round_up(groups, group_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4ScaleTransposeKernel<RowsMultiple8>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(row_tiles_global), static_cast<std::size_t>(groups_global)),
            sycl::range<2>(static_cast<std::size_t>(row_tile_block), static_cast<std::size_t>(group_block))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0)) * kRowsPerItem;
          int group = static_cast<int>(item.get_global_id(1));
          if (group >= groups || row >= rows) {
            return;
          }

          int dst_idx = group * rows + row;
          int src_idx = row * groups + group;
          if constexpr (RowsMultiple8) {
            uint8_t scale0 = row_scales[src_idx];
            uint8_t scale1 = row_scales[src_idx + groups];
            uint8_t scale2 = row_scales[src_idx + 2 * groups];
            uint8_t scale3 = row_scales[src_idx + 3 * groups];
            uint8_t scale4 = row_scales[src_idx + 4 * groups];
            uint8_t scale5 = row_scales[src_idx + 5 * groups];
            uint8_t scale6 = row_scales[src_idx + 6 * groups];
            uint8_t scale7 = row_scales[src_idx + 7 * groups];
            uint32_t scale_quad_lo = static_cast<uint32_t>(scale0) |
                (static_cast<uint32_t>(scale1) << 8) |
                (static_cast<uint32_t>(scale2) << 16) |
                (static_cast<uint32_t>(scale3) << 24);
            uint32_t scale_quad_hi = static_cast<uint32_t>(scale4) |
                (static_cast<uint32_t>(scale5) << 8) |
                (static_cast<uint32_t>(scale6) << 16) |
                (static_cast<uint32_t>(scale7) << 24);
            uint64_t scale_oct = static_cast<uint64_t>(scale_quad_lo) |
                (static_cast<uint64_t>(scale_quad_hi) << 32);
            uint8_t* dst_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(column_scales + dst_idx, 8));
            *reinterpret_cast<uint64_t*>(dst_ptr) = scale_oct;
          } else {
#pragma unroll
            for (int i = 0; i < kRowsPerItem; ++i) {
              if (row + i < rows) {
                column_scales[dst_idx + i] = row_scales[(row + i) * groups + group];
              }
            }
          }
        });
  });
}

sycl::event launch_mxfp4_scale_transpose_batched(
    sycl::queue& queue,
    Mxfp4ScaleTransposeBatchItem const* items,
    int item_count,
    int total_tiles) {
  constexpr int kRowsPerItem = 8;
  int global = quant::round_up(total_tiles, quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4ScaleTransposeBatchedKernel>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          int global_tile = static_cast<int>(item.get_global_id(0));
          if (global_tile >= total_tiles) {
            return;
          }

          int item_idx = 0;
          int upper = item_count;
          while (item_idx + 1 < upper) {
            int mid = (item_idx + upper) >> 1;
            if (global_tile >= items[mid].tile_offset) {
              item_idx = mid;
            } else {
              upper = mid;
            }
          }

          Mxfp4ScaleTransposeBatchItem desc = items[item_idx];
          int local_tile = global_tile - desc.tile_offset;
          if (local_tile >= desc.tile_count) {
            return;
          }

          int row_tile = local_tile / desc.groups;
          int group = local_tile - row_tile * desc.groups;
          int row = row_tile * kRowsPerItem;
          int dst_idx = group * desc.rows + row;
          int src_idx = row * desc.groups + group;
          if (row + kRowsPerItem <= desc.rows) {
            uint8_t scale0 = desc.row_scales[src_idx];
            uint8_t scale1 = desc.row_scales[src_idx + desc.groups];
            uint8_t scale2 = desc.row_scales[src_idx + 2 * desc.groups];
            uint8_t scale3 = desc.row_scales[src_idx + 3 * desc.groups];
            uint8_t scale4 = desc.row_scales[src_idx + 4 * desc.groups];
            uint8_t scale5 = desc.row_scales[src_idx + 5 * desc.groups];
            uint8_t scale6 = desc.row_scales[src_idx + 6 * desc.groups];
            uint8_t scale7 = desc.row_scales[src_idx + 7 * desc.groups];
            uint32_t scale_quad_lo = static_cast<uint32_t>(scale0) |
                (static_cast<uint32_t>(scale1) << 8) |
                (static_cast<uint32_t>(scale2) << 16) |
                (static_cast<uint32_t>(scale3) << 24);
            uint32_t scale_quad_hi = static_cast<uint32_t>(scale4) |
                (static_cast<uint32_t>(scale5) << 8) |
                (static_cast<uint32_t>(scale6) << 16) |
                (static_cast<uint32_t>(scale7) << 24);
            uint64_t scale_oct = static_cast<uint64_t>(scale_quad_lo) |
                (static_cast<uint64_t>(scale_quad_hi) << 32);
            uint8_t* dst_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(desc.column_scales + dst_idx, 8));
            *reinterpret_cast<uint64_t*>(dst_ptr) = scale_oct;
          } else {
#pragma unroll
            for (int i = 0; i < kRowsPerItem; ++i) {
              if (row + i < desc.rows) {
                desc.column_scales[dst_idx + i] = desc.row_scales[(row + i) * desc.groups + group];
              }
            }
          }
        });
  });
}

template <typename Element, int Groups>
sycl::event launch_mxfp4_mapping_2d_static(sycl::queue& queue, Mxfp4Params<Element> const& params) {
  int group_block = quant::choose_group_block(Groups);
  int groups_global = quant::round_up(Groups, group_block);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingKernel2DStaticGroups<Element, Groups>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(params.rows), static_cast<std::size_t>(groups_global)),
            sycl::range<2>(1, static_cast<std::size_t>(group_block))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0));
          int group = static_cast<int>(item.get_global_id(1));
          if (group >= Groups) {
            return;
          }
          process_mxfp4_group<Element, Groups>(params, row, group);
        });
  });
}

template <typename Element>
sycl::event launch_mxfp4_mapping(sycl::queue& queue, Mxfp4Params<Element> const& params) {
  if (params.groups <= 8) {
    if (params.groups == 6) {
      return launch_mxfp4_mapping_1d_static<Element, 6>(queue, params);
    }
    return launch_mxfp4_mapping_1d(queue, params);
  }

  if (params.groups < 128) {
    int row_block = quant::choose_row_block_for_group_tile(params.groups);
    if (params.column_major_scales) {
      if (params.groups == 24) {
        row_block = 2;
      } else if (params.groups == 48) {
        return launch_mxfp4_mapping_tiled_2d<Element>(queue, params, row_block);
      }
      if ((params.rows & 1) == 0) {
        return launch_mxfp4_mapping_column_major_row_pair_tiled_2d<Element, true>(queue, params, row_block);
      }
      return launch_mxfp4_mapping_column_major_row_pair_tiled_2d<Element, false>(queue, params, row_block);
    }
    if (params.groups == 48) {
      return launch_mxfp4_mapping_tiled_2d_static<Element, 48>(queue, params, row_block);
    }
    if (params.groups == 24) {
      return launch_mxfp4_mapping_tiled_2d_static<Element, 24>(queue, params, row_block);
    }
    if (params.groups == 12) {
      return launch_mxfp4_mapping_tiled_2d_static<Element, 12>(queue, params, row_block);
    }
    if (params.groups == 96) {
      return launch_mxfp4_mapping_tiled_2d_static<Element, 96>(queue, params, row_block);
    }
    return launch_mxfp4_mapping_tiled_2d<Element>(queue, params, row_block);
  }

  int group_block = quant::choose_group_block(params.groups);
  int groups_global = quant::round_up(params.groups, group_block);

  if (!params.column_major_scales) {
    if (params.groups == 192) {
      return launch_mxfp4_mapping_2d_static<Element, 192>(queue, params);
    }
    if (params.groups == 384) {
      return launch_mxfp4_mapping_2d_static<Element, 384>(queue, params);
    }
  }

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp4MappingKernel2D<Element>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(params.rows), static_cast<std::size_t>(groups_global)),
            sycl::range<2>(1, static_cast<std::size_t>(group_block))),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          int row = static_cast<int>(item.get_global_id(0));
          int group = static_cast<int>(item.get_global_id(1));
          if (group >= params.groups) {
            return;
          }
          process_mxfp4_group(params, row, group);
        });
  });
}

char const* layout_text(ScaleLayout layout) {
  return layout == ScaleLayout::kColumnMajor ? "column" : "row";
}

bool parse_layout(std::string const& text, ScaleLayout& layout) {
  if (text == "row" || text == "row_major" || text == "row-major") {
    layout = ScaleLayout::kRowMajor;
    return true;
  }
  if (text == "column" || text == "col" || text == "column_major" || text == "column-major") {
    layout = ScaleLayout::kColumnMajor;
    return true;
  }
  return false;
}

std::vector<Mxfp4Case> quick_suite() {
  return {
      {"single_group_midpoints", 1, 32, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"small_column_interleave", 2, 96, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"row_tail_37x160", 37, 160, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"aligned_128x256_column", 128, 256, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
  };
}

std::vector<Mxfp4Case> inkling_suite() {
  // MXFP4 per-token activation quantization runs at two call sites in the
  // Inkling forward path:
  //   1) hidden state before an MoE/dense gate_up GEMM:
  //        rows = T tokens, cols = hidden (non-scattered) or hidden/tp
  //        (scattered sconv).
  //   2) post-SiLU-and-mul, before down_proj/w2:
  //        rows = T tokens, cols = intermediate/tp (cfg: 768/tp;
  //        prod: 6144/tp — see get_hidden_dim's down_proj branch and the
  //        FUSE_SILU_AND_MUL kernel path in xpu_per_token_group_quant_fp4).
  // T bands: decode T=96, target-verify T=144 = bs*draft_token_num,
  // chunk T=4096, prefill cap T=16384. MXFP4 group size is 32, so cols must
  // be a multiple of 32; both hidden/tp and intermediate/tp at the two
  // shipped hidden sizes {1536, 6144} for TP={1,2,4,8} satisfy this.
  return {
      // Legacy cases (kept for regression parity).
      {"decode_128x256", 128, 256, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prefill_256x2048_column", 256, 2048, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_hidden_2048x6144", 2048, 6144, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_hidden_2048x7168_column", 2048, 7168, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      // Config-defaults hidden_size=1536, TP-shard decode (T=96) sweep.
      {"cfg_h1536_tp1_decode_96x1536",  96, 1536, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"cfg_h1536_tp2_decode_96x768",   96,  768, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"cfg_h1536_tp4_decode_96x384",   96,  384, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"cfg_h1536_tp8_decode_96x192",   96,  192, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      // Config-defaults target-verify band T=144 (TP=1 covers unshard).
      {"cfg_h1536_tp1_verify_144x1536", 144, 1536, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp2_verify_144x768", 144,  768, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp4_verify_144x384", 144,  384, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp8_verify_144x192", 144,  192, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      // Config-defaults prefill-chunk T=4096 across TP=1/2/4/8.
      {"cfg_h1536_tp1_chunk_4096x1536", 4096, 1536, ScaleLayout::kRowMajor,  1.0e-10f, 0.0},
      {"cfg_h1536_tp2_chunk_4096x768", 4096,  768, ScaleLayout::kRowMajor,   1.0e-10f, 0.0},
      {"cfg_h1536_tp4_chunk_4096x384", 4096,  384, ScaleLayout::kRowMajor,   1.0e-10f, 0.0},
      {"cfg_h1536_tp8_chunk_4096x192", 4096,  192, ScaleLayout::kRowMajor,   1.0e-10f, 0.0},
      // Config-defaults pre-downproj activation (cols = intermediate/tp = 768/tp).
      // Covers the FUSE_SILU_AND_MUL activation-quant call site; the kernel
      // itself is separate from mapping, but the layout it emits is identical.
      {"cfg_h1536_tp1_predown_96x768",   96,  768, ScaleLayout::kRowMajor,   1.0e-10f, 0.0},
      {"cfg_h1536_tp2_predown_96x384",   96,  384, ScaleLayout::kRowMajor,   1.0e-10f, 0.0},
      {"cfg_h1536_tp4_predown_96x192",   96,  192, ScaleLayout::kRowMajor,   1.0e-10f, 0.0},
      {"cfg_h1536_tp8_predown_96x96",    96,   96, ScaleLayout::kRowMajor,   1.0e-10f, 0.0},
      // Production hidden_size=6144 TP sweep.
      {"prod_h6144_tp1_decode_96x6144", 96, 6144, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"prod_h6144_tp2_decode_96x3072", 96, 3072, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"prod_h6144_tp4_decode_96x1536", 96, 1536, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"prod_h6144_tp8_decode_96x768",  96,  768, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      // Production target-verify (TP=1 covers unshard).
      {"prod_h6144_tp1_verify_144x6144", 144, 6144, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp2_verify_144x3072", 144, 3072, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp4_verify_144x1536", 144, 1536, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp8_verify_144x768",  144,  768, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      // Production prefill-chunk T=4096 across TP=1/2/4/8.
      {"prod_h6144_tp1_chunk_4096x6144", 4096, 6144, ScaleLayout::kRowMajor,  1.0e-10f, 0.0},
      {"prod_h6144_tp2_chunk_4096x3072", 4096, 3072, ScaleLayout::kRowMajor,  1.0e-10f, 0.0},
      {"prod_h6144_tp4_chunk_4096x1536", 4096, 1536, ScaleLayout::kRowMajor,  1.0e-10f, 0.0},
      {"prod_h6144_tp8_chunk_4096x768",  4096,  768, ScaleLayout::kRowMajor,  1.0e-10f, 0.0},
      // Production pre-downproj activation (cols = intermediate/tp = 6144/tp).
      {"prod_h6144_tp2_predown_96x3072", 96, 3072, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"prod_h6144_tp4_predown_96x1536", 96, 1536, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"prod_h6144_tp8_predown_96x768",  96,  768, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
  };
}

std::vector<Mxfp4Case> perf_suite() {
  // Perf sweep exercises production hidden=6144 sharded across TP=1/2/4/8
  // at the prefill-cap band T=16384 (max_prefill_tokens), plus the cfg-defaults
  // hidden=1536 counterpart. GB/s gates left at 0 for new shard shapes until
  // an Inkling baseline is captured.
  return {
      {"perf_4096x6144", 4096, 6144, ScaleLayout::kRowMajor, 1.0e-10f, 120.0},
      {"perf_8192x8192_column", 8192, 8192, ScaleLayout::kColumnMajor, 1.0e-10f, 120.0},
      // Production prefill-cap T=16384 activation quantization, TP-shard hidden.
      {"perf_prod_prefill_tp1_16384x6144", 16384, 6144, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"perf_prod_prefill_tp2_16384x3072", 16384, 3072, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"perf_prod_prefill_tp4_16384x1536", 16384, 1536, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"perf_prod_prefill_tp8_16384x768",  16384,  768, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      // Column-major scale layout at the same shards for the alternate output view.
      {"perf_prod_prefill_col_tp4_16384x1536", 16384, 1536, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"perf_prod_prefill_col_tp8_16384x768",  16384,  768, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      // Config-defaults prefill-cap T=16384 across TP=1/2/4/8.
      {"perf_cfg_prefill_tp1_16384x1536", 16384, 1536, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"perf_cfg_prefill_tp2_16384x768", 16384,  768, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"perf_cfg_prefill_tp4_16384x384", 16384,  384, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
      {"perf_cfg_prefill_tp8_16384x192", 16384,  192, ScaleLayout::kRowMajor,    1.0e-10f, 0.0},
  };
}

std::vector<Mxfp4Case> inkling_batched_suite() {
  return {
      {"cfg_h1536_tp1_decode_96x1536",  96, 1536, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp2_decode_96x768",   96,  768, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp4_decode_96x384",   96,  384, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp8_decode_96x192",   96,  192, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp1_verify_144x1536", 144, 1536, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp2_verify_144x768",  144,  768, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp4_verify_144x384",  144,  384, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp8_verify_144x192",  144,  192, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp1_predown_96x768",  96,  768, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp2_predown_96x384",  96,  384, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp4_predown_96x192",  96,  192, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"cfg_h1536_tp8_predown_96x96",   96,   96, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp1_decode_96x6144", 96, 6144, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp2_decode_96x3072", 96, 3072, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp4_decode_96x1536", 96, 1536, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp8_decode_96x768",  96,  768, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp1_verify_144x6144", 144, 6144, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp2_verify_144x3072", 144, 3072, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp4_verify_144x1536", 144, 1536, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp8_verify_144x768",  144,  768, ScaleLayout::kColumnMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp2_predown_96x3072", 96, 3072, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp4_predown_96x1536", 96, 1536, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
      {"prod_h6144_tp8_predown_96x768",  96,  768, ScaleLayout::kRowMajor, 1.0e-10f, 0.0},
  };
}

std::vector<Mxfp4Case> make_suite(std::string const& suite) {
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

bool parse_mx_shape(std::string const& text, Mxfp4Case& cfg) {
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
    } else if (key == "rows" || key == "m") {
      cfg.rows = std::stoi(value);
    } else if (key == "cols" || key == "n" || key == "k") {
      cfg.cols = std::stoi(value);
    } else if (key == "layout" || key == "scales") {
      if (!parse_layout(value, cfg.layout)) {
        return false;
      }
    } else if (key == "eps") {
      cfg.eps = std::stof(value);
    } else if (key == "target_gbps" || key == "target-gbps") {
      cfg.target_gbps = std::stod(value);
    } else {
      return false;
    }
  }
  return true;
}

void validate_case(Mxfp4Case& cfg) {
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
  if (cfg.rows <= 0 || cfg.cols <= 0) {
    throw std::invalid_argument("rows and cols must be positive");
  }
  if (cfg.cols % quant::kMxfp4GroupSize != 0) {
    throw std::invalid_argument("cols must be divisible by MXFP4 group size 32");
  }
  if (cfg.eps <= 0.0f) {
    cfg.eps = 1.0e-10f;
  }
}

template <typename Element>
void seed_edge_values(std::vector<Element>& input) {
  float values[] = {
      0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f,
      -0.25f, -0.75f, -1.25f, -1.75f, -2.5f, -3.5f, -5.0f,
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f, 0.0f, 0.0f, 0.0f};
  int count = std::min<int>(static_cast<int>(input.size()), quant::kMxfp4GroupSize);
  for (int i = 0; i < count; ++i) {
    input[i] = quant::from_float<Element>(values[i]);
  }
}

template <typename Element>
void mxfp4_reference(
    Mxfp4Case const& cfg,
    std::vector<Element> const& input,
    std::vector<uint8_t>& packed,
    std::vector<uint8_t>& scales) {
  int groups = cfg.cols / quant::kMxfp4GroupSize;
  packed.assign(static_cast<std::size_t>(cfg.rows) * cfg.cols / 2, 0);
  scales.assign(static_cast<std::size_t>(cfg.rows) * groups, 0);

  for (int row = 0; row < cfg.rows; ++row) {
    for (int group = 0; group < groups; ++group) {
      int col0 = group * quant::kMxfp4GroupSize;
      std::size_t base = static_cast<std::size_t>(row) * cfg.cols + col0;
      float max_abs = cfg.eps;
      for (int i = 0; i < quant::kMxfp4GroupSize; ++i) {
        max_abs = std::max(max_abs, std::fabs(quant::to_float(input[base + i])));
      }
      int shared_exp = quant::floor_log2_positive(max_abs) - 2;
      shared_exp = quant::clamp_exponent_to_ue8m0(shared_exp);
      float inv_scale = quant::pow2_int(-shared_exp);

      int scale_idx = cfg.layout == ScaleLayout::kColumnMajor ? group * cfg.rows + row : row * groups + group;
      scales[scale_idx] = quant::encode_ue8m0_exponent(shared_exp);

      std::size_t packed_base = (static_cast<std::size_t>(row) * cfg.cols) / 2 +
          static_cast<std::size_t>(group) * (quant::kMxfp4GroupSize / 2);
      for (int i = 0; i < quant::kMxfp4GroupSize; i += 2) {
        uint8_t q0 = quant::quantize_e2m1_code(quant::to_float(input[base + i]) * inv_scale);
        uint8_t q1 = quant::quantize_e2m1_code(quant::to_float(input[base + i + 1]) * inv_scale);
        packed[packed_base + i / 2] = quant::pack_e2m1_pair(q0, q1);
      }
    }
  }
}

template <typename Element>
bool run_case_for_dtype(sycl::queue& queue, Mxfp4Case cfg, quant::Options const& options) {
  validate_case(cfg);

  int groups = cfg.cols / quant::kMxfp4GroupSize;
  std::size_t input_count = static_cast<std::size_t>(cfg.rows) * cfg.cols;
  std::size_t packed_count = input_count / 2;
  std::size_t scale_count = static_cast<std::size_t>(cfg.rows) * groups;

  std::vector<Element> h_input = quant::make_input<Element>(input_count, 20260303u, -4.0f, 4.0f);
  seed_edge_values(h_input);

  quant::DeviceBuffer<Element> d_input(queue, input_count);
  quant::DeviceBuffer<uint8_t> d_packed(queue, packed_count);
  quant::DeviceBuffer<uint8_t> d_scales(queue, scale_count);
  bool column_major_scales = cfg.layout == ScaleLayout::kColumnMajor;
  bool transpose_column_scales = column_major_scales;
  quant::DeviceBuffer<uint8_t> d_row_scales_tmp(queue, transpose_column_scales ? scale_count : 1);
  d_input.copy_from(h_input);

  Mxfp4Params<Element> params;
  params.x = d_input.get();
  params.packed = d_packed.get();
  params.scales = transpose_column_scales ? d_row_scales_tmp.get() : d_scales.get();
  params.rows = cfg.rows;
  params.cols = cfg.cols;
  params.groups = groups;
  params.total_groups = cfg.rows * groups;
  params.eps = cfg.eps;
  params.eps_exp = quant::floor_log2_positive(cfg.eps);
  params.column_major_scales = false;

  auto launch = [&]() -> quant::EventBundle {
    sycl::event quant_event = launch_mxfp4_mapping<Element>(queue, params);
    if (transpose_column_scales) {
      sycl::event transpose_event = (cfg.rows & 7) == 0
          ? launch_mxfp4_scale_transpose<true>(
                queue, d_row_scales_tmp.get(), d_scales.get(), cfg.rows, groups)
          : launch_mxfp4_scale_transpose<false>(
                queue, d_row_scales_tmp.get(), d_scales.get(), cfg.rows, groups);
      return quant::EventBundle(quant_event, transpose_event);
    }
    return quant::EventBundle(quant_event);
  };

  bool passed = true;
  if (options.verify) {
    d_packed.zero();
    d_scales.zero();
    if (transpose_column_scales) {
      d_row_scales_tmp.zero();
    }
    launch().wait();

    std::vector<uint8_t> h_packed(packed_count);
    std::vector<uint8_t> h_scales(scale_count);
    d_packed.copy_to(h_packed);
    d_scales.copy_to(h_scales);

    std::vector<uint8_t> ref_packed;
    std::vector<uint8_t> ref_scales;
    mxfp4_reference(cfg, h_input, ref_packed, ref_scales);

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
    double moved_bytes = static_cast<double>(input_count * sizeof(Element) + packed_count + scale_count);
    gbps = quant::effective_gbps(moved_bytes, mean_ms);
    double target = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
    if (target > 0.0 && gbps < target && moved_bytes >= quant::kMinSustainedTargetBytes) {
      passed = false;
    }
  }

  std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] dtype=" << quant::element_dtype_text<Element>()
            << " case=" << cfg.name
            << " rows=" << cfg.rows
            << " cols=" << cfg.cols
            << " groups=" << groups
            << " scale_layout=" << layout_text(cfg.layout);
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::fixed << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

template <typename Element>
bool run_batched_row_major_for_dtype(
    sycl::queue& queue,
    std::vector<Mxfp4Case> cases,
    quant::Options const& options) {
  std::vector<std::vector<Element>> h_inputs;
  std::vector<quant::DeviceBuffer<Element>> d_inputs;
  std::vector<quant::DeviceBuffer<uint8_t>> d_packed;
  std::vector<quant::DeviceBuffer<uint8_t>> d_scales;
  std::vector<quant::DeviceBuffer<uint8_t>> d_row_scales_tmp;
  std::vector<std::size_t> packed_counts;
  std::vector<std::size_t> scale_counts;
  std::vector<Mxfp4BatchItem<Element>> h_items;
  std::vector<Mxfp4ScaleTransposeBatchItem> h_transpose_items;

  h_inputs.reserve(cases.size());
  d_inputs.reserve(cases.size());
  d_packed.reserve(cases.size());
  d_scales.reserve(cases.size());
  d_row_scales_tmp.reserve(cases.size());
  packed_counts.reserve(cases.size());
  scale_counts.reserve(cases.size());
  h_items.reserve(cases.size());
  h_transpose_items.reserve(cases.size());

  int total_groups = 0;
  int total_scale_tiles = 0;
  int column_major_cases = 0;
  double moved_bytes = 0.0;
  for (Mxfp4Case& cfg : cases) {
    validate_case(cfg);

    int groups = cfg.cols / quant::kMxfp4GroupSize;
    std::size_t input_count = static_cast<std::size_t>(cfg.rows) * cfg.cols;
    std::size_t packed_count = input_count / 2;
    std::size_t scale_count = static_cast<std::size_t>(cfg.rows) * groups;

    h_inputs.push_back(quant::make_input<Element>(
        input_count,
        static_cast<uint32_t>(20260303u + h_inputs.size()),
        -4.0f,
        4.0f));
    seed_edge_values(h_inputs.back());

    d_inputs.emplace_back(queue, input_count);
    d_packed.emplace_back(queue, packed_count);
    d_scales.emplace_back(queue, scale_count);
    d_inputs.back().copy_from(h_inputs.back());

    Mxfp4BatchItem<Element> item;
    item.x = d_inputs.back().get();
    item.packed = d_packed.back().get();
    item.scales = d_scales.back().get();
    if (cfg.layout == ScaleLayout::kColumnMajor) {
      d_row_scales_tmp.emplace_back(queue, scale_count);
      item.scales = d_row_scales_tmp.back().get();

      Mxfp4ScaleTransposeBatchItem transpose_item;
      transpose_item.row_scales = d_row_scales_tmp.back().get();
      transpose_item.column_scales = d_scales.back().get();
      transpose_item.rows = cfg.rows;
      transpose_item.groups = groups;
      transpose_item.tile_count = quant::ceil_div(cfg.rows, 8) * groups;
      transpose_item.tile_offset = total_scale_tiles;
      h_transpose_items.push_back(transpose_item);
      total_scale_tiles += transpose_item.tile_count;
      ++column_major_cases;
    }
    item.rows = cfg.rows;
    item.cols = cfg.cols;
    item.groups = groups;
    item.total_groups = cfg.rows * groups;
    item.group_offset = total_groups;
    item.eps = cfg.eps;
    item.eps_exp = quant::floor_log2_positive(cfg.eps);
    h_items.push_back(item);

    total_groups += item.total_groups;
    moved_bytes += static_cast<double>(input_count * sizeof(Element) + packed_count + scale_count);
    packed_counts.push_back(packed_count);
    scale_counts.push_back(scale_count);
  }

  quant::DeviceBuffer<Mxfp4BatchItem<Element>> d_items(queue, h_items.size());
  d_items.copy_from(h_items);
  quant::DeviceBuffer<Mxfp4ScaleTransposeBatchItem> d_transpose_items(queue, h_transpose_items.size());
  d_transpose_items.copy_from(h_transpose_items);

  auto launch = [&]() -> quant::EventBundle {
    sycl::event quant_event = launch_mxfp4_mapping_batched_row_major<Element>(
        queue,
        d_items.get(),
        static_cast<int>(h_items.size()),
        total_groups);
    if (!h_transpose_items.empty()) {
      sycl::event transpose_event = launch_mxfp4_scale_transpose_batched(
          queue,
          d_transpose_items.get(),
          static_cast<int>(h_transpose_items.size()),
          total_scale_tiles);
      return quant::EventBundle(quant_event, transpose_event);
    }
    return quant::EventBundle(quant_event);
  };

  bool passed = true;
  if (options.verify) {
    for (auto& packed : d_packed) {
      packed.zero();
    }
    for (auto& scales : d_scales) {
      scales.zero();
    }
    for (auto& row_scales : d_row_scales_tmp) {
      row_scales.zero();
    }
    launch().wait();

    for (std::size_t case_idx = 0; case_idx < cases.size(); ++case_idx) {
      std::vector<uint8_t> h_packed(packed_counts[case_idx]);
      std::vector<uint8_t> h_scales(scale_counts[case_idx]);
      d_packed[case_idx].copy_to(h_packed);
      d_scales[case_idx].copy_to(h_scales);

      std::vector<uint8_t> ref_packed;
      std::vector<uint8_t> ref_scales;
      mxfp4_reference(cases[case_idx], h_inputs[case_idx], ref_packed, ref_scales);

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
            << " case=inkling_batched_mx"
            << " cases=" << cases.size()
            << " column_cases=" << column_major_cases
            << " total_groups=" << total_groups;
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::fixed << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

bool run_batched_row_major_cases(
    sycl::queue& queue,
    std::vector<Mxfp4Case> const& cases,
    quant::Options const& options) {
  bool all_passed = true;
  if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFloat) {
    all_passed &= run_batched_row_major_for_dtype<float>(queue, cases, options);
  }
  if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kBf16) {
    all_passed &= run_batched_row_major_for_dtype<cutlass::bfloat16_t>(queue, cases, options);
  }
  if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFp16) {
    all_passed &= run_batched_row_major_for_dtype<cutlass::half_t>(queue, cases, options);
  }
  return all_passed;
}

bool run_cases(sycl::queue& queue, std::vector<Mxfp4Case> const& cases, quant::Options const& options) {
  bool all_passed = true;
  for (Mxfp4Case cfg : cases) {
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFloat) {
      all_passed &= run_case_for_dtype<float>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kBf16) {
      all_passed &= run_case_for_dtype<cutlass::bfloat16_t>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFp16) {
      all_passed &= run_case_for_dtype<cutlass::half_t>(queue, cfg, options);
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
      std::cout << "21_bmg_mxfp4_mapping: subgroup MXFP4 E2M1 pack plus UE8M0 scale mapping\n\n";
      quant::print_common_usage(
          argv[0],
          "quick|inkling|perf|inkling_batched",
          "rows=<int>,cols=<int>,layout=row|column,eps=<float>");
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<Mxfp4Case> cases;
  bool run_batched = false;
  if (!options.shape.empty()) {
    Mxfp4Case cfg;
    cfg.name = "custom_mxfp4";
    if (!parse_mx_shape(options.shape, cfg)) {
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
    std::cout << "21_bmg_mxfp4_mapping: group=32 packed E2M1, UE8M0 scales, row/column scale layout\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << quant::dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << quant::bool_text(options.verify)
              << " benchmark=" << quant::bool_text(options.benchmark) << "\n";

    bool passed = run_batched
        ? run_batched_row_major_cases(queue, cases, options)
        : run_cases(queue, cases, options);
    return passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
