/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Inkling activation quantizers for CUTLASS SYCL on BMG.

    IMPORTANT -- what the Inkling model actually runs:

      MXFP4 activation quantization (E2M1 payload + UE8M0 scales, group 32) is
      *not* used by the Inkling model. Inkling never quantizes activations to
      FP4; only weights are FP4 (NVFP4, see 21_bmg_nvfp4_layout.cpp). The MXFP4
      mapping below is retained as a standalone reference kernel because other
      work references it, but it does not mirror an Inkling op.

      Inkling uses exactly two activation quantizers, both implemented here:

      1) --mode=mxfp8_kv -- MXFP8 KV cache. K and V rows are quantized to
         float8_e4m3fn with one float8_e8m0fnu scale per 32 channels and
         scattered into the paged cache at loc[t]. At the production
         page_size == 128 the scale-factor buffer is the interleaved FA4
         BlockScaledBasicChunk layout
             (slots / 128, dkv / 128, 32, page_size / 32, 4)
         written by sglang's quant_store_kv_mxfp8 / store_sf_interleaved
         (python/sglang/kernels/ops/quantization/mxfp8_quant.py,
          mxfp8_interleave_sf.py) and read by FA4's block-scaled QK / V dequant.
         Q is quantized with the same per-32-channel rule but its scales stay
         flat per token (they ride q_descale), so this example only models the
         KV-cache scatter, which is where the layout is non-trivial.
         Example 15 (15_bmg_attn_prologue_mxfp8_store_tau.cpp) stores the same
         bytes from inside the fused attention prologue. It carries its own
         private copies of the scale rule and the interleaved offset rather than
         calling the shared helpers, so the agreement between the two examples is
         checked by eye, not by construction -- change one, change the other.

      2) --mode=fp8_pertensor -- static per-tensor activation scaling to E4M3.
         Two conventions ship, selected per case by amax_divisor:

           amax / 448          static_quant_fp8 (kernels/ops/quantization/
                               fp8_kernel.py), the plain per-tensor FP8-E4M3
                               quantizer: scale = amax / e4m3_max, then
                               q = e4m3(clamp(x * (1 / scale), +-448)).

           amax / (448 * 6)    input_scale for the NVFP4 experts
                               (models/inkling.py::_ckpt_scale_to_modelopt); 6
                               is the E2M1 max. This is the only per-tensor
                               activation scale Inkling itself derives -- it
                               ships no FP8 linear method. Its consumer is
                               fp4_quantize (modelopt_quant.py), which emits an
                               E2M1 payload plus swizzled per-16 E4M3 block
                               scales, NOT the E4M3 bytes stored here; that
                               payload and layout are 21_bmg_nvfp4_layout's.
                               So for these cases only the scale derivation is
                               one-for-one with Inkling; the byte store is
                               static_quant_fp8's, kept so both conventions are
                               measured by the same kernel. Note the extra
                               factor of 6 makes an E4M3 store saturate at
                               |x| > amax / 6.

    All three families are pure SYCL: no sycl::ext::intel::esimd. BMG has no
    native FP8 arithmetic here, so E4M3/E8M0 encoding is done in software and
    the kernels are byte-store kernels, not FP8 matmuls. Performance is reported
    as effective GB/s.
*/

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
  // NOTE: MXFP4 activation quantization is NOT an Inkling op -- see the file
  // header. The shapes below are the *hypothetical* per-token activation-quant
  // call sites (they follow Inkling's hidden/intermediate geometry so the
  // kernel is exercised at realistic sizes), kept for regression parity of this
  // reference kernel. The two quantizers Inkling really runs are covered by
  // --mode=mxfp8_kv and --mode=fp8_pertensor below.
  //
  // Shape provenance (geometry only):
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

// ===========================================================================
// Shared vector loads for the MXFP8 / FP8 paths.
// ===========================================================================

constexpr int kFp8VecElems = 8;
// Consecutive 8-element vectors handled by one work item in the per-tensor FP8
// kernel; see launch_fp8_per_tensor() for why one vector per item is too little.
constexpr int kFp8ChunksPerItem = 4;

template <typename Element>
CUTLASS_DEVICE void load_vec8(Element const* src, float (&out)[kFp8VecElems]) {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t> ||
                std::is_same_v<Element, cutlass::half_t>) {
    using RawWords = sycl::vec<uint64_t, 2>;
    Element const* aligned = static_cast<Element const*>(__builtin_assume_aligned(src, 16));
    RawWords words = *reinterpret_cast<RawWords const*>(aligned);
#pragma unroll
    for (int word = 0; word < 2; ++word) {
      uint64_t raw = words[word];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        out[word * 4 + j] = quant::raw16_to_float<Element>(static_cast<uint16_t>(raw >> (16 * j)));
      }
    }
  } else {
    using FloatWords = sycl::vec<float, kFp8VecElems>;
    Element const* aligned = static_cast<Element const*>(__builtin_assume_aligned(src, 32));
    FloatWords words = *reinterpret_cast<FloatWords const*>(aligned);
#pragma unroll
    for (int j = 0; j < kFp8VecElems; ++j) {
      out[j] = words[j];
    }
  }
}

// Pack eight already-scaled floats into eight E4M3 bytes with one 64-bit store.
CUTLASS_DEVICE void store_e4m3_vec8(float const (&scaled)[kFp8VecElems], uint8_t* dst) {
  uint64_t raw = 0;
#pragma unroll
  for (int j = 0; j < kFp8VecElems; ++j) {
    raw |= static_cast<uint64_t>(quant::e4m3fn_encode_signed(scaled[j])) << (8 * j);
  }
  *reinterpret_cast<uint64_t*>(dst) = raw;
}

// ===========================================================================
// MXFP8 KV cache (the Inkling activation quantizer).
//
//   K/V rows (tokens, dkv) bf16/fp16/fp32 -> paged float8_e4m3fn cache at
//   loc[t] plus float8_e8m0fnu scales in the interleaved FA4 layout
//   (slots / page_size, dkv / 128, 32, page_size / 32, 4).
//
// This mirrors sglang's quant_store_kv_mxfp8: one launch quantizes both K and V
// and scatters payload + scales.
//
// loc[t] < 0 marks a padded token, which this kernel skips. That is a deliberate
// divergence: sglang's _mxfp8_quant_store_qkv_kernel has no such guard and would
// scatter out of bounds for a negative loc (only its
// _mxfp8_v_cache_update_kernel masks on loc >= 0). The guard is what lets the
// suites run a fixed grid over padded tokens the way a CUDA-graph capture would,
// and --shape=neg_loc=0 covers the no-padding path.
// ===========================================================================

struct Mxfp8KvCase {
  std::string name;
  int tokens = 1;
  int dkv = 256;
  int slots = 1024;
  int page_size = 128;
  bool include_negative_loc = false;
  double target_gbps = 0.0;
};

template <typename Element>
struct Mxfp8KvParams {
  Element const* __restrict k = nullptr;
  Element const* __restrict v = nullptr;
  int64_t const* __restrict loc = nullptr;
  uint8_t* __restrict k_cache = nullptr;
  uint8_t* __restrict v_cache = nullptr;
  uint8_t* __restrict sfk = nullptr;
  uint8_t* __restrict sfv = nullptr;
  int tokens = 0;
  int dkv = 0;
  int page_size = 128;
  int blocks_per_token = 0;
  int total_blocks = 0;
};

template <typename Element>
class Mxfp8KvStoreKernel;

// Quantize one 32-channel MXFP8 group: writes 32 E4M3 payload bytes and returns
// the E8M0 scale byte.
template <typename Element>
CUTLASS_DEVICE uint8_t quantize_mxfp8_group(Element const* src, uint8_t* dst) {
  float values[quant::kMxfp8GroupSize];
#pragma unroll
  for (int chunk = 0; chunk < quant::kMxfp8GroupSize / kFp8VecElems; ++chunk) {
    float chunk_values[kFp8VecElems];
    load_vec8<Element>(src + chunk * kFp8VecElems, chunk_values);
#pragma unroll
    for (int j = 0; j < kFp8VecElems; ++j) {
      values[chunk * kFp8VecElems + j] = chunk_values[j];
    }
  }

  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < quant::kMxfp8GroupSize; ++i) {
    float ax = quant::abs_f(values[i]);
    amax = ax > amax ? ax : amax;
  }

  int descale_exponent = 0;
  uint8_t scale_byte = quant::mxfp8_scale_byte(amax, descale_exponent);

#pragma unroll
  for (int chunk = 0; chunk < quant::kMxfp8GroupSize / kFp8VecElems; ++chunk) {
    float scaled[kFp8VecElems];
#pragma unroll
    for (int j = 0; j < kFp8VecElems; ++j) {
      scaled[j] = quant::clamp_f(
          quant::scalb_f(values[chunk * kFp8VecElems + j], -descale_exponent),
          -quant::kE4M3FnMax,
          quant::kE4M3FnMax);
    }
    store_e4m3_vec8(scaled, dst + chunk * kFp8VecElems);
  }
  return scale_byte;
}

template <typename Element>
sycl::event launch_mxfp8_kv_store(sycl::queue& queue, Mxfp8KvParams<Element> const& params) {
  int global = quant::round_up(params.total_blocks, quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Mxfp8KvStoreKernel<Element>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          int gid = static_cast<int>(item.get_global_id(0));
          if (gid >= params.total_blocks) {
            return;
          }
          int blocks_per_row = 2 * params.blocks_per_token;
          int token = gid / blocks_per_row;
          int rem = gid - token * blocks_per_row;
          bool is_v = rem >= params.blocks_per_token;
          int block = is_v ? rem - params.blocks_per_token : rem;

          int64_t slot = params.loc[token];
          if (slot < 0) {
            return;
          }

          int channel = block * quant::kMxfp8GroupSize;
          Element const* src = (is_v ? params.v : params.k) +
              static_cast<int64_t>(token) * params.dkv + channel;
          uint8_t* dst = (is_v ? params.v_cache : params.k_cache) +
              slot * params.dkv + channel;
          uint8_t scale_byte = quantize_mxfp8_group<Element>(src, dst);

          uint8_t* sf_base = is_v ? params.sfv : params.sfk;
          sf_base[quant::mxfp8_interleaved_sf_offset(slot, channel, params.dkv, params.page_size)] =
              scale_byte;
        });
  });
}

// Host reference: identical arithmetic, straight loops, explicit index math.
template <typename Element>
void mxfp8_kv_reference(
    Mxfp8KvCase const& cfg,
    std::vector<Element> const& k,
    std::vector<Element> const& v,
    std::vector<int64_t> const& loc,
    std::size_t cache_bytes,
    std::size_t sf_bytes,
    std::vector<uint8_t>& ref_k_cache,
    std::vector<uint8_t>& ref_v_cache,
    std::vector<uint8_t>& ref_sfk,
    std::vector<uint8_t>& ref_sfv) {
  ref_k_cache.assign(cache_bytes, 0);
  ref_v_cache.assign(cache_bytes, 0);
  ref_sfk.assign(sf_bytes, 0);
  ref_sfv.assign(sf_bytes, 0);

  int blocks = cfg.dkv / quant::kMxfp8GroupSize;
  for (int token = 0; token < cfg.tokens; ++token) {
    int64_t slot = loc[static_cast<std::size_t>(token)];
    if (slot < 0) {
      continue;
    }
    for (int side = 0; side < 2; ++side) {
      std::vector<Element> const& src = side == 0 ? k : v;
      std::vector<uint8_t>& cache = side == 0 ? ref_k_cache : ref_v_cache;
      std::vector<uint8_t>& sf = side == 0 ? ref_sfk : ref_sfv;
      for (int block = 0; block < blocks; ++block) {
        int channel = block * quant::kMxfp8GroupSize;
        std::size_t base = static_cast<std::size_t>(token) * cfg.dkv + channel;
        float amax = 0.0f;
        for (int i = 0; i < quant::kMxfp8GroupSize; ++i) {
          amax = std::max(amax, std::fabs(quant::to_float(src[base + i])));
        }
        int descale_exponent = 0;
        uint8_t scale_byte = quant::mxfp8_scale_byte(amax, descale_exponent);

        std::size_t dst = static_cast<std::size_t>(slot) * cfg.dkv + channel;
        for (int i = 0; i < quant::kMxfp8GroupSize; ++i) {
          float scaled = quant::clamp_f(
              quant::scalb_f(quant::to_float(src[base + i]), -descale_exponent),
              -quant::kE4M3FnMax,
              quant::kE4M3FnMax);
          cache[dst + i] = quant::e4m3fn_encode_signed(scaled);
        }
        sf[static_cast<std::size_t>(
            quant::mxfp8_interleaved_sf_offset(slot, channel, cfg.dkv, cfg.page_size))] =
            scale_byte;
      }
    }
  }
}

void validate_mxfp8_kv_case(Mxfp8KvCase& cfg) {
  if (cfg.name.empty()) {
    cfg.name = "custom_mxfp8_kv";
  }
  if (cfg.tokens <= 0 || cfg.dkv <= 0 || cfg.slots <= 0) {
    throw std::invalid_argument("tokens, dkv and slots must be positive");
  }
  if (cfg.dkv % quant::kMxfp8HeadDim != 0) {
    throw std::invalid_argument("dkv must be a multiple of head_dim 128");
  }
  if (cfg.page_size <= 0 || cfg.page_size % quant::kMxfp8GroupSize != 0) {
    throw std::invalid_argument("page_size must be a positive multiple of 32");
  }
  if (cfg.slots % cfg.page_size != 0) {
    throw std::invalid_argument("slots must be a multiple of page_size");
  }
  if (cfg.tokens > cfg.slots) {
    throw std::invalid_argument("tokens must not exceed slots (one slot per token)");
  }
}

// Distinct destination slots, deterministically scrambled so the scatter is not
// a straight copy (real allocations hand out pages out of order).
std::vector<int64_t> make_kv_loc(Mxfp8KvCase const& cfg) {
  std::vector<int64_t> slots(static_cast<std::size_t>(cfg.slots));
  for (int i = 0; i < cfg.slots; ++i) {
    slots[static_cast<std::size_t>(i)] = i;
  }
  std::mt19937 gen(20260601u + static_cast<uint32_t>(cfg.tokens));
  std::shuffle(slots.begin(), slots.end(), gen);
  slots.resize(static_cast<std::size_t>(cfg.tokens));
  if (cfg.include_negative_loc) {
    for (int t = 3; t < cfg.tokens; t += 7) {
      slots[static_cast<std::size_t>(t)] = -1;
    }
  }
  return slots;
}

template <typename Element>
void seed_mxfp8_edge_values(std::vector<Element>& input, int dkv) {
  if (static_cast<int>(input.size()) < 3 * dkv || dkv < quant::kMxfp8GroupSize) {
    return;
  }
  // Group 0 of token 0: a huge amax forces the top E8M0 exponents and payload
  //                     saturation.
  // Group 0 of token 1: all zeros exercises the 1e-30 amax floor.
  // Group 0 of token 2: tiny values exercise E4M3 subnormals after scaling.
  for (int i = 0; i < quant::kMxfp8GroupSize; ++i) {
    input[static_cast<std::size_t>(i)] =
        quant::from_float<Element>(i == 0 ? 30000.0f : (i % 3 == 0 ? -1.5f : 0.75f));
    input[static_cast<std::size_t>(dkv + i)] = quant::from_float<Element>(0.0f);
    input[static_cast<std::size_t>(2 * dkv + i)] =
        quant::from_float<Element>((i % 2 == 0 ? 1.0f : -1.0f) * 1.0e-4f * (i + 1));
  }
}

template <typename Element>
bool run_mxfp8_kv_case_for_dtype(sycl::queue& queue, Mxfp8KvCase cfg, quant::Options const& options) {
  validate_mxfp8_kv_case(cfg);

  int heads = cfg.dkv / quant::kMxfp8HeadDim;
  int blocks_per_token = cfg.dkv / quant::kMxfp8GroupSize;
  int pages = cfg.slots / cfg.page_size;
  std::size_t row_count = static_cast<std::size_t>(cfg.tokens) * cfg.dkv;
  std::size_t cache_bytes = static_cast<std::size_t>(cfg.slots) * cfg.dkv;
  std::size_t sf_bytes = static_cast<std::size_t>(pages) * heads * quant::kMxfp8GroupSize *
      (cfg.page_size / quant::kMxfp8GroupSize) * quant::kMxfp8ScalesPerHead;

  std::vector<Element> h_k = quant::make_input<Element>(row_count, 20260601u, -4.0f, 4.0f);
  std::vector<Element> h_v = quant::make_input<Element>(row_count, 20260602u, -4.0f, 4.0f);
  seed_mxfp8_edge_values(h_k, cfg.dkv);
  seed_mxfp8_edge_values(h_v, cfg.dkv);
  std::vector<int64_t> h_loc = make_kv_loc(cfg);

  quant::DeviceBuffer<Element> d_k(queue, row_count);
  quant::DeviceBuffer<Element> d_v(queue, row_count);
  quant::DeviceBuffer<int64_t> d_loc(queue, h_loc.size());
  quant::DeviceBuffer<uint8_t> d_k_cache(queue, cache_bytes);
  quant::DeviceBuffer<uint8_t> d_v_cache(queue, cache_bytes);
  quant::DeviceBuffer<uint8_t> d_sfk(queue, sf_bytes);
  quant::DeviceBuffer<uint8_t> d_sfv(queue, sf_bytes);
  d_k.copy_from(h_k);
  d_v.copy_from(h_v);
  d_loc.copy_from(h_loc);

  Mxfp8KvParams<Element> params;
  params.k = d_k.get();
  params.v = d_v.get();
  params.loc = d_loc.get();
  params.k_cache = d_k_cache.get();
  params.v_cache = d_v_cache.get();
  params.sfk = d_sfk.get();
  params.sfv = d_sfv.get();
  params.tokens = cfg.tokens;
  params.dkv = cfg.dkv;
  params.page_size = cfg.page_size;
  params.blocks_per_token = blocks_per_token;
  params.total_blocks = cfg.tokens * 2 * blocks_per_token;

  auto launch = [&]() -> sycl::event { return launch_mxfp8_kv_store<Element>(queue, params); };

  bool passed = true;
  if (options.verify) {
    // Zero-init matters: unwritten scale bytes must stay 0, never 0xFF (E8M0 NaN).
    d_k_cache.zero();
    d_v_cache.zero();
    d_sfk.zero();
    d_sfv.zero();
    launch().wait();

    std::vector<uint8_t> h_k_cache(cache_bytes);
    std::vector<uint8_t> h_v_cache(cache_bytes);
    std::vector<uint8_t> h_sfk(sf_bytes);
    std::vector<uint8_t> h_sfv(sf_bytes);
    d_k_cache.copy_to(h_k_cache);
    d_v_cache.copy_to(h_v_cache);
    d_sfk.copy_to(h_sfk);
    d_sfv.copy_to(h_sfv);

    std::vector<uint8_t> ref_k_cache;
    std::vector<uint8_t> ref_v_cache;
    std::vector<uint8_t> ref_sfk;
    std::vector<uint8_t> ref_sfv;
    mxfp8_kv_reference(
        cfg, h_k, h_v, h_loc, cache_bytes, sf_bytes, ref_k_cache, ref_v_cache, ref_sfk, ref_sfv);

    quant::ByteCompareResult k_cmp = quant::compare_bytes(h_k_cache, ref_k_cache);
    quant::ByteCompareResult v_cmp = quant::compare_bytes(h_v_cache, ref_v_cache);
    quant::ByteCompareResult sfk_cmp = quant::compare_bytes(h_sfk, ref_sfk);
    quant::ByteCompareResult sfv_cmp = quant::compare_bytes(h_sfv, ref_sfv);
    if (!k_cmp.passed || !v_cmp.passed || !sfk_cmp.passed || !sfv_cmp.passed) {
      std::cerr << "  [FAIL] dtype=" << quant::element_dtype_text<Element>()
                << " mxfp8_kv_case=" << cfg.name << "\n";
      quant::print_byte_compare("k_cache", k_cmp);
      quant::print_byte_compare("v_cache", v_cmp);
      quant::print_byte_compare("sfk", sfk_cmp);
      quant::print_byte_compare("sfv", sfv_cmp);
      passed = false;
    }
  }

  double mean_ms = 0.0;
  double gbps = 0.0;
  if (options.benchmark) {
    mean_ms = quant::benchmark_ms(launch, options.warmup, options.iterations);
    // Touched bytes: K+V reads, K+V payload writes, K+V scale writes.
    double moved_bytes = 2.0 * static_cast<double>(row_count) * sizeof(Element) +
        2.0 * static_cast<double>(row_count) +
        2.0 * static_cast<double>(cfg.tokens) * blocks_per_token;
    gbps = quant::effective_gbps(moved_bytes, mean_ms);
    double target = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
    if (target > 0.0 && gbps < target && moved_bytes >= quant::kMinSustainedTargetBytes) {
      passed = false;
    }
  }

  std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] dtype=" << quant::element_dtype_text<Element>()
            << " mode=mxfp8_kv case=" << cfg.name
            << " tokens=" << cfg.tokens
            << " dkv=" << cfg.dkv
            << " kv_heads=" << heads
            << " slots=" << cfg.slots
            << " page=" << cfg.page_size;
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::fixed << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

// dkv = max(1, Nkv / TP) * head_dim with head_dim = 128. Verified Inkling
// configs: checkpoint (hidden 768, Nq 8, Nkv 2, swa_num_key_value_heads 4),
// defaults (hidden 1536, Nq 12, Nkv 4), production (hidden 6144, Nq 48, Nkv 4).
// GQA floors Nkv_local at 1, so TP >= Nkv all share one local KV head.
//
// head_dim is 128 in every one of those configs because the checkpoint sets it
// explicitly; it is not hidden_size / Nq (which would give 96 for the
// checkpoint). InklingConfig only falls back to hidden_size // num_attention_heads
// when head_dim is absent, and swa_head_dim defaults to head_dim, so the SWA
// layers are 128 too -- which is what makes them eligible for the fused MXFP8
// prologue at all (models/inkling.py skips any layer with head_dim != 128).
// The ckpt_swa_* cases differ from the ckpt_* ones only in Nkv
// (swa_num_key_value_heads=4 vs num_key_value_heads=2), hence dkv 512 vs 256;
// they coincide with the nkv4_* shapes by construction, and are named separately
// so the provenance of each shape stays readable.
std::vector<Mxfp8KvCase> mxfp8_kv_quick_suite() {
  return {
      // One page, one KV head: smallest shape that still spans the interleaved
      // (32, page_size/32, 4) scale tile.
      {"single_page_128x128", 128, 128, 128, 128, false, 0.0},
      // Two KV heads, scattered across pages, with padded (loc < 0) tokens.
      {"scatter_96x256_neg_loc", 96, 256, 1024, 128, true, 0.0},
      // Four KV heads (defaults / production Nkv=4 at TP=1).
      {"scatter_144x512", 144, 512, 2048, 128, false, 0.0},
      // Non-128 page size exercises the flat-chunk arithmetic
      // (sglang only interleaves at page_size==128, but the offset formula is
      // general in page_size and this pins that).
      {"page64_128x256", 128, 256, 1024, 64, false, 0.0},
  };
}

std::vector<Mxfp8KvCase> mxfp8_kv_inkling_suite() {
  return {
      // Checkpoint config, Nkv=2 -> dkv 256 at TP=1, 128 at TP>=2.
      {"ckpt_nkv2_tp1_decode_96x256", 96, 256, 8192, 128, false, 0.0},
      {"ckpt_nkv2_tp2_decode_96x128", 96, 128, 8192, 128, false, 0.0},
      {"ckpt_nkv2_tp4_decode_96x128", 96, 128, 8192, 128, false, 0.0},
      {"ckpt_nkv2_tp8_decode_96x128", 96, 128, 8192, 128, false, 0.0},
      // Checkpoint SWA layers, swa_num_key_value_heads=4 -> dkv 512 at TP=1.
      {"ckpt_swa_nkv4_tp1_decode_96x512", 96, 512, 8192, 128, false, 0.0},
      {"ckpt_swa_nkv4_tp2_decode_96x256", 96, 256, 8192, 128, false, 0.0},
      {"ckpt_swa_nkv4_tp4_decode_96x128", 96, 128, 8192, 128, false, 0.0},
      {"ckpt_swa_nkv4_tp8_decode_96x128", 96, 128, 8192, 128, false, 0.0},
      // Defaults (hidden 1536, Nkv 4) and production (hidden 6144, Nkv 4) share
      // the same KV geometry: dkv = 512 / 256 / 128 / 128 for TP = 1 / 2 / 4 / 8.
      {"nkv4_tp1_verify_144x512", 144, 512, 8192, 128, false, 0.0},
      {"nkv4_tp2_verify_144x256", 144, 256, 8192, 128, false, 0.0},
      {"nkv4_tp4_verify_144x128", 144, 128, 8192, 128, false, 0.0},
      {"nkv4_tp8_verify_144x128", 144, 128, 8192, 128, false, 0.0},
      // Prefill-chunk band with padded tokens (target-verify pads the batch).
      {"nkv4_tp1_chunk_4096x512", 4096, 512, 8192, 128, true, 0.0},
      {"nkv4_tp2_chunk_4096x256", 4096, 256, 8192, 128, true, 0.0},
      {"nkv4_tp4_chunk_4096x128", 4096, 128, 8192, 128, true, 0.0},
      // Wide-KV geometry: dkv = 6144 / TP, i.e. an MHA-style layer whose local
      // KV width tracks production hidden_size. Named in the KV-geometry audit;
      // covers head counts far above the GQA configs above.
      {"wide_tp1_512x6144", 512, 6144, 2048, 128, false, 0.0},
      {"wide_tp2_512x3072", 512, 3072, 2048, 128, false, 0.0},
      {"wide_tp4_512x1536", 512, 1536, 2048, 128, false, 0.0},
      {"wide_tp8_512x768", 512, 768, 2048, 128, false, 0.0},
  };
}

std::vector<Mxfp8KvCase> mxfp8_kv_perf_suite() {
  // Prefill-cap band T=16384 (max_prefill_tokens) at the shipped KV widths.
  // GB/s gates are 0.0 (report-only) until a BMG baseline is captured; a
  // guessed number would flake CI (see 17_bmg_relative_attention_backend).
  return {
      {"perf_nkv4_tp1_16384x512", 16384, 512, 32768, 128, false, 0.0},
      {"perf_nkv4_tp2_16384x256", 16384, 256, 32768, 128, false, 0.0},
      {"perf_nkv4_tp4_16384x128", 16384, 128, 32768, 128, false, 0.0},
      {"perf_ckpt_nkv2_tp1_16384x256", 16384, 256, 32768, 128, false, 0.0},
      {"perf_wide_tp1_4096x6144", 4096, 6144, 8192, 128, false, 0.0},
      {"perf_wide_tp2_4096x3072", 4096, 3072, 8192, 128, false, 0.0},
      {"perf_wide_tp4_4096x1536", 4096, 1536, 8192, 128, false, 0.0},
      {"perf_wide_tp8_4096x768", 4096, 768, 8192, 128, false, 0.0},
  };
}

std::vector<Mxfp8KvCase> make_mxfp8_kv_suite(std::string const& suite) {
  if (suite == "quick") {
    return mxfp8_kv_quick_suite();
  }
  if (suite == "inkling") {
    return mxfp8_kv_inkling_suite();
  }
  if (suite == "perf") {
    return mxfp8_kv_perf_suite();
  }
  return {};
}

bool parse_mxfp8_kv_shape(std::string const& text, Mxfp8KvCase& cfg) {
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
    } else if (key == "tokens" || key == "t" || key == "rows") {
      cfg.tokens = std::stoi(value);
    } else if (key == "dkv") {
      cfg.dkv = std::stoi(value);
    } else if (key == "slots") {
      cfg.slots = std::stoi(value);
    } else if (key == "page_size" || key == "page") {
      cfg.page_size = std::stoi(value);
    } else if (key == "neg_loc") {
      cfg.include_negative_loc = quant::parse_bool(value);
    } else if (key == "target_gbps" || key == "target-gbps") {
      cfg.target_gbps = std::stod(value);
    } else {
      return false;
    }
  }
  return true;
}

bool run_mxfp8_kv_cases(
    sycl::queue& queue,
    std::vector<Mxfp8KvCase> const& cases,
    quant::Options const& options) {
  bool all_passed = true;
  for (Mxfp8KvCase const& cfg : cases) {
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFloat) {
      all_passed &= run_mxfp8_kv_case_for_dtype<float>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kBf16) {
      all_passed &= run_mxfp8_kv_case_for_dtype<cutlass::bfloat16_t>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFp16) {
      all_passed &= run_mxfp8_kv_case_for_dtype<cutlass::half_t>(queue, cfg, options);
    }
  }
  return all_passed;
}

// ===========================================================================
// Static per-tensor activation scaling to E4M3.
//
//   input_scale = input_amax / amax_divisor       (loaded from the checkpoint)
//   q           = e4m3(clamp(x * (1 / input_scale), -448, 448))
//
// The reciprocal is formed once (as static_quant_fp8 does) and there is no
// reduction at all: the scale is static per linear, not per batch.
//
// amax_divisor selects the convention (see the file header for the full
// provenance):
//   kE4M3FnMax (448)               static_quant_fp8, the plain per-tensor FP8
//                                  quantizer -- payload and scale both belong to
//                                  this kernel.
//   kNvfp4InputScaleDivisor (2688) Inkling's NVFP4 expert input_scale. Only the
//                                  scale derivation is Inkling's here; the real
//                                  consumer (fp4_quantize) emits E2M1 plus
//                                  per-16 block scales, which is
//                                  21_bmg_nvfp4_layout's subject, so an E4M3
//                                  store under this divisor saturates above
//                                  input_amax / 6 by construction.
// ===========================================================================

struct Fp8PerTensorCase {
  std::string name;
  int rows = 1;
  int cols = 128;
  float input_amax = 24.0f;
  float amax_divisor = quant::kE4M3FnMax;
  double target_gbps = 0.0;
};

template <typename Element>
struct Fp8PerTensorParams {
  Element const* __restrict x = nullptr;
  uint8_t* __restrict q = nullptr;
  int64_t count = 0;
  int64_t vectors = 0;
  float inv_scale = 1.0f;
};

template <typename Element>
class Fp8PerTensorVecKernel;

template <typename Element>
class Fp8PerTensorScalarKernel;

template <typename Element>
sycl::event launch_fp8_per_tensor(sycl::queue& queue, Fp8PerTensorParams<Element> const& params) {
  if (params.vectors > 0) {
    // Each work item owns kFp8ChunksPerItem consecutive 8-element vectors. One
    // vector per item measured 2.7x slower than the MXFP4 kernel on identical
    // traffic: at 16 B loaded per item there is not enough memory-level
    // parallelism in flight per thread to cover the E4M3 encode.
    int64_t items = (params.vectors + kFp8ChunksPerItem - 1) / kFp8ChunksPerItem;
    int64_t global = quant::round_up(static_cast<int>(items), quant::kDefaultBlock);
    return queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<Fp8PerTensorVecKernel<Element>>(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<std::size_t>(global)),
              sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            int64_t first = static_cast<int64_t>(item.get_global_id(0)) * kFp8ChunksPerItem;
#pragma unroll
            for (int c = 0; c < kFp8ChunksPerItem; ++c) {
              int64_t vec = first + c;
              if (vec >= params.vectors) {
                break;
              }
              int64_t base = vec * kFp8VecElems;
              float values[kFp8VecElems];
              load_vec8<Element>(params.x + base, values);
#pragma unroll
              for (int j = 0; j < kFp8VecElems; ++j) {
                values[j] = quant::clamp_f(
                    values[j] * params.inv_scale, -quant::kE4M3FnMax, quant::kE4M3FnMax);
              }
              store_e4m3_vec8(values, params.q + base);
            }
          });
    });
  }

  int64_t global = quant::round_up(static_cast<int>(params.count), quant::kDefaultBlock);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Fp8PerTensorScalarKernel<Element>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(quant::kDefaultBlock))),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          int64_t idx = static_cast<int64_t>(item.get_global_id(0));
          if (idx >= params.count) {
            return;
          }
          float scaled = quant::clamp_f(
              quant::to_float(params.x[idx]) * params.inv_scale,
              -quant::kE4M3FnMax,
              quant::kE4M3FnMax);
          params.q[idx] = quant::e4m3fn_encode_signed(scaled);
        });
  });
}

template <typename Element>
void fp8_per_tensor_reference(
    std::vector<Element> const& x,
    float inv_scale,
    std::vector<uint8_t>& out) {
  out.assign(x.size(), 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    float scaled = quant::clamp_f(
        quant::to_float(x[i]) * inv_scale, -quant::kE4M3FnMax, quant::kE4M3FnMax);
    out[i] = quant::e4m3fn_encode_signed(scaled);
  }
}

void validate_fp8_per_tensor_case(Fp8PerTensorCase& cfg) {
  if (cfg.name.empty()) {
    cfg.name = "custom_fp8_pertensor";
  }
  if (cfg.rows <= 0 || cfg.cols <= 0) {
    throw std::invalid_argument("rows and cols must be positive");
  }
  if (!(cfg.input_amax > 0.0f)) {
    throw std::invalid_argument("input_amax must be positive");
  }
  if (!(cfg.amax_divisor > 0.0f)) {
    throw std::invalid_argument("amax_divisor must be positive");
  }
}

template <typename Element>
bool run_fp8_per_tensor_case_for_dtype(
    sycl::queue& queue,
    Fp8PerTensorCase cfg,
    quant::Options const& options) {
  validate_fp8_per_tensor_case(cfg);

  std::size_t count = static_cast<std::size_t>(cfg.rows) * cfg.cols;
  float input_scale = quant::per_tensor_input_scale_from_amax(cfg.input_amax, cfg.amax_divisor);
  float inv_scale = 1.0f / input_scale;

  std::vector<Element> h_x = quant::make_input<Element>(count, 20260603u, -4.0f, 4.0f);
  // Force saturation on both signs plus an exact-zero and a subnormal-after-
  // scaling value, so the E4M3 corner cases are covered at every shape.
  if (count >= 4) {
    h_x[0] = quant::from_float<Element>(cfg.input_amax);
    h_x[1] = quant::from_float<Element>(-cfg.input_amax);
    h_x[2] = quant::from_float<Element>(0.0f);
    h_x[3] = quant::from_float<Element>(input_scale * 0.001953125f);
  }

  quant::DeviceBuffer<Element> d_x(queue, count);
  quant::DeviceBuffer<uint8_t> d_q(queue, count);
  d_x.copy_from(h_x);

  Fp8PerTensorParams<Element> params;
  params.x = d_x.get();
  params.q = d_q.get();
  params.count = static_cast<int64_t>(count);
  params.vectors = (count % kFp8VecElems == 0) ? static_cast<int64_t>(count / kFp8VecElems) : 0;
  params.inv_scale = inv_scale;

  auto launch = [&]() -> sycl::event { return launch_fp8_per_tensor<Element>(queue, params); };

  bool passed = true;
  if (options.verify) {
    d_q.zero();
    launch().wait();

    std::vector<uint8_t> h_q(count);
    d_q.copy_to(h_q);

    std::vector<uint8_t> ref_q;
    fp8_per_tensor_reference(h_x, inv_scale, ref_q);
    quant::ByteCompareResult cmp = quant::compare_bytes(h_q, ref_q);
    if (!cmp.passed) {
      std::cerr << "  [FAIL] dtype=" << quant::element_dtype_text<Element>()
                << " fp8_pertensor_case=" << cfg.name << "\n";
      quant::print_byte_compare("q", cmp);
      passed = false;
    }
  }

  double mean_ms = 0.0;
  double gbps = 0.0;
  if (options.benchmark) {
    mean_ms = quant::benchmark_ms(launch, options.warmup, options.iterations);
    double moved_bytes = static_cast<double>(count) * (sizeof(Element) + 1);
    gbps = quant::effective_gbps(moved_bytes, mean_ms);
    double target = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
    if (target > 0.0 && gbps < target && moved_bytes >= quant::kMinSustainedTargetBytes) {
      passed = false;
    }
  }

  std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] dtype=" << quant::element_dtype_text<Element>()
            << " mode=fp8_pertensor case=" << cfg.name
            << " rows=" << cfg.rows
            << " cols=" << cfg.cols
            << " input_amax=" << std::fixed << std::setprecision(3) << cfg.input_amax
            << " amax_divisor=" << std::fixed << std::setprecision(1) << cfg.amax_divisor
            << " input_scale=" << std::scientific << std::setprecision(4) << input_scale
            << std::defaultfloat;
  if (options.benchmark) {
    std::cout << " mean_ms=" << std::fixed << std::setprecision(4) << mean_ms
              << " effective_gbps=" << std::setprecision(2) << gbps;
  }
  std::cout << "\n";
  return passed;
}

// Activation widths of the quantized linears. Column-parallel linears (qkv_proj,
// gate_up_proj) see the full hidden_size; row-parallel ones (o_proj, down_proj)
// see their shard, hidden/TP or intermediate/TP. hidden_size is 1536 (defaults)
// or 6144 (production); the checkpoint's own hidden_size is 768.
//
// kFp8Div    = static_quant_fp8's scale = amax / e4m3_max.
// kNvfp4Div  = Inkling's NVFP4 expert input_scale = amax / (e4m3_max * e2m1_max).
// Only the second is a scale Inkling itself derives; see the section comment
// above for why its payload is fp4_quantize's and not the bytes stored here.
constexpr float kFp8Div = quant::kE4M3FnMax;
constexpr float kNvfp4Div = quant::kNvfp4InputScaleDivisor;

std::vector<Fp8PerTensorCase> fp8_per_tensor_quick_suite() {
  return {
      {"single_row_1x768", 1, 768, 24.0f, kFp8Div, 0.0},
      {"tail_37x1536", 37, 1536, 24.0f, kFp8Div, 0.0},
      {"unaligned_cols_5x101", 5, 101, 24.0f, kFp8Div, 0.0},
      {"tight_amax_128x1536", 128, 1536, 4.0f, kFp8Div, 0.0},
      // Same shapes under the NVFP4 divisor, where everything above amax/6
      // saturates: exercises the saturating branch of the encoder at scale.
      {"nvfp4in_tail_37x1536", 37, 1536, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_unaligned_cols_5x101", 5, 101, 24.0f, kNvfp4Div, 0.0},
  };
}

std::vector<Fp8PerTensorCase> fp8_per_tensor_inkling_suite() {
  // The nvfp4in_* cases carry Inkling's own divisor; the fp8lin_* cases at the
  // end carry static_quant_fp8's, so both conventions are verified per dtype.
  return {
      // Checkpoint hidden_size=768 (column-parallel input is unsharded).
      {"nvfp4in_ckpt_h768_colparallel_96x768", 96, 768, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_ckpt_h768_rowparallel_tp2_96x384", 96, 384, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_ckpt_h768_rowparallel_tp4_96x192", 96, 192, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_ckpt_h768_rowparallel_tp8_96x96", 96, 96, 24.0f, kNvfp4Div, 0.0},
      // Defaults hidden_size=1536.
      {"nvfp4in_cfg_h1536_colparallel_decode_96x1536", 96, 1536, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_cfg_h1536_rowparallel_tp2_96x768", 96, 768, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_cfg_h1536_rowparallel_tp4_96x384", 96, 384, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_cfg_h1536_rowparallel_tp8_96x192", 96, 192, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_cfg_h1536_colparallel_verify_144x1536", 144, 1536, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_cfg_h1536_colparallel_chunk_4096x1536", 4096, 1536, 24.0f, kNvfp4Div, 0.0},
      // Production hidden_size=6144.
      {"nvfp4in_prod_h6144_colparallel_decode_96x6144", 96, 6144, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_prod_h6144_rowparallel_tp2_96x3072", 96, 3072, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_prod_h6144_rowparallel_tp4_96x1536", 96, 1536, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_prod_h6144_rowparallel_tp8_96x768", 96, 768, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_prod_h6144_colparallel_verify_144x6144", 144, 6144, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_prod_h6144_colparallel_chunk_4096x6144", 4096, 6144, 24.0f, kNvfp4Div, 0.0},
      // down_proj input = intermediate/TP; intermediate is 768 (defaults) and
      // 6144 (production) per the shipped configs.
      {"nvfp4in_cfg_downproj_tp2_96x384", 96, 384, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_prod_downproj_tp2_96x3072", 96, 3072, 24.0f, kNvfp4Div, 0.0},
      {"nvfp4in_prod_downproj_tp8_96x768", 96, 768, 24.0f, kNvfp4Div, 0.0},
      // static_quant_fp8's own convention across the three hidden_size values.
      {"fp8lin_ckpt_h768_96x768", 96, 768, 24.0f, kFp8Div, 0.0},
      {"fp8lin_cfg_h1536_96x1536", 96, 1536, 24.0f, kFp8Div, 0.0},
      {"fp8lin_prod_h6144_96x6144", 96, 6144, 24.0f, kFp8Div, 0.0},
      {"fp8lin_prod_h6144_chunk_4096x6144", 4096, 6144, 24.0f, kFp8Div, 0.0},
  };
}

std::vector<Fp8PerTensorCase> fp8_per_tensor_perf_suite() {
  // Prefill-cap band. Gates report-only (0.0) until a BMG baseline is captured.
  // The divisor changes no work, so only one case repeats under kNvfp4Div.
  return {
      {"perf_prod_prefill_tp1_16384x6144", 16384, 6144, 24.0f, kFp8Div, 0.0},
      {"perf_prod_prefill_tp2_16384x3072", 16384, 3072, 24.0f, kFp8Div, 0.0},
      {"perf_prod_prefill_tp4_16384x1536", 16384, 1536, 24.0f, kFp8Div, 0.0},
      {"perf_prod_prefill_tp8_16384x768", 16384, 768, 24.0f, kFp8Div, 0.0},
      {"perf_cfg_prefill_tp1_16384x1536", 16384, 1536, 24.0f, kFp8Div, 0.0},
      {"perf_cfg_prefill_tp4_16384x384", 16384, 384, 24.0f, kFp8Div, 0.0},
      {"perf_nvfp4in_prod_prefill_tp1_16384x6144", 16384, 6144, 24.0f, kNvfp4Div, 0.0},
  };
}

std::vector<Fp8PerTensorCase> make_fp8_per_tensor_suite(std::string const& suite) {
  if (suite == "quick") {
    return fp8_per_tensor_quick_suite();
  }
  if (suite == "inkling") {
    return fp8_per_tensor_inkling_suite();
  }
  if (suite == "perf") {
    return fp8_per_tensor_perf_suite();
  }
  return {};
}

bool parse_fp8_per_tensor_shape(std::string const& text, Fp8PerTensorCase& cfg) {
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
    } else if (key == "rows" || key == "m" || key == "tokens") {
      cfg.rows = std::stoi(value);
    } else if (key == "cols" || key == "n" || key == "k") {
      cfg.cols = std::stoi(value);
    } else if (key == "amax" || key == "input_amax") {
      cfg.input_amax = std::stof(value);
    } else if (key == "divisor" || key == "amax_divisor") {
      if (value == "fp8") {
        cfg.amax_divisor = kFp8Div;
      } else if (value == "nvfp4") {
        cfg.amax_divisor = kNvfp4Div;
      } else {
        cfg.amax_divisor = std::stof(value);
      }
    } else if (key == "target_gbps" || key == "target-gbps") {
      cfg.target_gbps = std::stod(value);
    } else {
      return false;
    }
  }
  return true;
}

bool run_fp8_per_tensor_cases(
    sycl::queue& queue,
    std::vector<Fp8PerTensorCase> const& cases,
    quant::Options const& options) {
  bool all_passed = true;
  for (Fp8PerTensorCase const& cfg : cases) {
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFloat) {
      all_passed &= run_fp8_per_tensor_case_for_dtype<float>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kBf16) {
      all_passed &= run_fp8_per_tensor_case_for_dtype<cutlass::bfloat16_t>(queue, cfg, options);
    }
    if (options.dtype == quant::DType::kAll || options.dtype == quant::DType::kFp16) {
      all_passed &= run_fp8_per_tensor_case_for_dtype<cutlass::half_t>(queue, cfg, options);
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
      std::cout << "21_bmg_mxfp4_mapping: Inkling activation quantizers -- MXFP8 KV cache and\n"
                << "static per-tensor E4M3 scaling (NVFP4 expert input_scale = amax/(448*6),\n"
                << "or static_quant_fp8's amax/448), plus the (non-Inkling) MXFP4 E2M1 + UE8M0\n"
                << "reference mapping.\n\n";
      quant::print_common_usage(
          argv[0],
          "quick|inkling|perf|inkling_batched",
          "mxfp4:         rows=<int>,cols=<int>,layout=row|column,eps=<float>\n"
          "                                mxfp8_kv:      tokens=<int>,dkv=<int>,slots=<int>,"
          "page_size=<int>,neg_loc=0|1\n"
          "                                fp8_pertensor: rows=<int>,cols=<int>,amax=<float>,"
          "divisor=fp8|nvfp4|<float>",
          /*show_mode=*/true);
      std::cout << "\nWith --shape, --mode selects which family the shape describes"
                << " (default mxfp4).\n";
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<Mxfp4Case> mxfp4_cases;
  std::vector<Mxfp8KvCase> mxfp8_kv_cases;
  std::vector<Fp8PerTensorCase> fp8_cases;
  bool run_batched = false;
  if (!options.shape.empty()) {
    // A single custom shape belongs to exactly one family; --mode picks it.
    if (options.mode == quant::QuantMode::kMxfp8Kv) {
      Mxfp8KvCase cfg;
      if (!parse_mxfp8_kv_shape(options.shape, cfg)) {
        std::cerr << "Invalid --shape string for mode=mxfp8_kv: " << options.shape << "\n";
        return -1;
      }
      mxfp8_kv_cases.push_back(cfg);
    } else if (options.mode == quant::QuantMode::kFp8PerTensor) {
      Fp8PerTensorCase cfg;
      if (!parse_fp8_per_tensor_shape(options.shape, cfg)) {
        std::cerr << "Invalid --shape string for mode=fp8_pertensor: " << options.shape << "\n";
        return -1;
      }
      fp8_cases.push_back(cfg);
    } else {
      Mxfp4Case cfg;
      cfg.name = "custom_mxfp4";
      if (!parse_mx_shape(options.shape, cfg)) {
        std::cerr << "Invalid --shape string: " << options.shape << "\n";
        return -1;
      }
      mxfp4_cases.push_back(cfg);
    }
  } else if (options.suite == "inkling_batched" || options.suite == "inkling-batched") {
    if (options.mode != quant::QuantMode::kAll && options.mode != quant::QuantMode::kMxfp4) {
      std::cerr << "Suite inkling_batched only exists for mode=mxfp4\n";
      return -1;
    }
    mxfp4_cases = inkling_batched_suite();
    run_batched = true;
  } else {
    if (quant::mode_selected(options.mode, quant::QuantMode::kMxfp4)) {
      mxfp4_cases = make_suite(options.suite);
    }
    if (quant::mode_selected(options.mode, quant::QuantMode::kMxfp8Kv)) {
      mxfp8_kv_cases = make_mxfp8_kv_suite(options.suite);
    }
    if (quant::mode_selected(options.mode, quant::QuantMode::kFp8PerTensor)) {
      fp8_cases = make_fp8_per_tensor_suite(options.suite);
    }
    if (mxfp4_cases.empty() && mxfp8_kv_cases.empty() && fp8_cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = quant::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "21_bmg_mxfp4_mapping: mxfp8_kv (Inkling KV cache), fp8_pertensor (static"
              << " per-tensor E4M3; divisor 2688 is Inkling's NVFP4 expert input_scale,"
              << " 448 is static_quant_fp8's), mxfp4 (reference only, not an Inkling op)\n";
    std::cout << "Suite=" << options.suite
              << " mode=" << quant::mode_text(options.mode)
              << " dtype=" << quant::dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << quant::bool_text(options.verify)
              << " benchmark=" << quant::bool_text(options.benchmark) << "\n";

    bool passed = true;
    if (run_batched) {
      passed &= run_batched_row_major_cases(queue, mxfp4_cases, options);
    } else if (!mxfp4_cases.empty()) {
      passed &= run_cases(queue, mxfp4_cases, options);
    }
    if (!mxfp8_kv_cases.empty()) {
      passed &= run_mxfp8_kv_cases(queue, mxfp8_kv_cases, options);
    }
    if (!fp8_cases.empty()) {
      passed &= run_fp8_per_tensor_cases(queue, fp8_cases, options);
    }
    return passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
