/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>

#include "16_bmg_moe_gate_topk_renorm.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace cutlass::examples::bmg_moe_gate {

/// The gate's contraction dim is InklingModelConfig.hidden_size (d_model).
/// Two configurations ship: 1536, the config default, and 6144, the production
/// checkpoint -- the latter attested by sglang's _INKLING_GATE_GEMV_HIDDEN and
/// by forward_fused's `x.shape[-1] == _INKLING_GATE_GEMV_HIDDEN` guard. The
/// width is a runtime argument (GateGemvParams::hidden); the production 6144
/// stays a compile-time specialization so its trip counts fold -- see
/// kGateHiddenSpecialized / launch_gate_gemv_hidden.
static constexpr int kGateHiddenSpecialized = 6144;
static constexpr int kGateRoutedExperts = 256;
static constexpr int kGateSharedExperts = 2;
static constexpr int kGateTotalExperts = kGateRoutedExperts + kGateSharedExperts;
static constexpr int kGateLogitsPad = 264;
static constexpr int kGateThreads = 256;
static constexpr int kGateFusedMaxTokens = 64;

struct GateGemvParams {
  cutlass::bfloat16_t const* x = nullptr;
  cutlass::bfloat16_t const* weight = nullptr;
  float* logits = nullptr;
  float const* bias = nullptr;
  float const* global_scale = nullptr;
  float* routed_weights = nullptr;
  float* shared_weights = nullptr;
  int32_t* indices = nullptr;
  int32_t* packed = nullptr;
  int32_t* ticket = nullptr;
  int64_t tokens = 0;
  int hidden = kGateHiddenSpecialized;
  float route_scale = 1.0f;
  GateActivation activation = GateActivation::kSigmoid;
  bool norm_after_topk = true;
};

CUTLASS_HOST_DEVICE
float gate_bf16_to_float(cutlass::bfloat16_t x) {
#if defined(__SYCL_DEVICE_ONLY__)
  uint32_t bits = static_cast<uint32_t>(x.raw()) << 16;
  return sycl::bit_cast<float>(bits);
#else
  return static_cast<float>(x);
#endif
}

/// HiddenStatic == 0 reads the width from GateGemvParams::hidden; a non-zero
/// value bakes it in (used for kGateHiddenSpecialized so the 6144 production
/// path keeps its compile-time trip counts).
template <int ExpertsPerWorkGroup,
          int SubGroupSize,
          bool Fused = false,
          bool Packed = false,
          int HiddenStatic = 0,
          GateActivation Act = GateActivation::kSigmoid>
class GateGemvKernel {
 public:
  static constexpr int kWarps = kGateThreads / SubGroupSize;

  GateGemvParams params;
  sycl::local_accessor<cutlass::bfloat16_t, 1> smem_weight;
  sycl::local_accessor<float, 1> smem_partials;
  sycl::local_accessor<int32_t, 1> smem_ticket;

  [[sycl::reqd_sub_group_size(SubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    static_assert(!Fused || SubGroupSize == kSubGroupSize);

    int const hidden = HiddenStatic != 0 ? HiddenStatic : params.hidden;

    sycl::sub_group sg = item.get_sub_group();
    int tid = static_cast<int>(item.get_local_id(0));
    int lane = static_cast<int>(sg.get_local_id());
    int warp = static_cast<int>(sg.get_group_id());
    int expert0 = static_cast<int>(item.get_group(0)) * ExpertsPerWorkGroup;
    int experts_this_group =
        expert0 + ExpertsPerWorkGroup <= kGateTotalExperts ? ExpertsPerWorkGroup : kGateTotalExperts - expert0;

    if (params.tokens > 0 && params.tokens <= 4) {
      float acc[4][ExpertsPerWorkGroup];
#pragma unroll
      for (int token = 0; token < 4; ++token) {
#pragma unroll
        for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
          acc[token][j] = 0.0f;
        }
      }

      for (int k = tid; k < hidden; k += kGateThreads) {
        float x_val[4];
#pragma unroll
        for (int token = 0; token < 4; ++token) {
          x_val[token] = 0.0f;
          if (token < params.tokens) {
            x_val[token] = gate_bf16_to_float(params.x[static_cast<int64_t>(token) * hidden + k]);
          }
        }

#pragma unroll
        for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
          if (j < experts_this_group) {
            cutlass::bfloat16_t const* w_row =
                params.weight + static_cast<int64_t>(expert0 + j) * hidden;
            float wv = gate_bf16_to_float(w_row[k]);
#pragma unroll
            for (int token = 0; token < 4; ++token) {
              if (token < params.tokens) {
                acc[token][j] = sycl::fma(x_val[token], wv, acc[token][j]);
              }
            }
          }
        }
      }

#pragma unroll
      for (int token = 0; token < 4; ++token) {
#pragma unroll
        for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
          float reduced = sycl::reduce_over_group(sg, acc[token][j], sycl::plus<float>());
          if (lane == 0) {
            smem_partials[warp * (4 * ExpertsPerWorkGroup) + token * ExpertsPerWorkGroup + j] = reduced;
          }
        }
      }

      item.barrier(sycl::access::fence_space::local_space);

      if (warp == 0) {
        int total_outputs = static_cast<int>(params.tokens) * ExpertsPerWorkGroup;
        if (lane < total_outputs) {
          int token_out = lane / ExpertsPerWorkGroup;
          int expert_j = lane - token_out * ExpertsPerWorkGroup;
          float sum = 0.0f;
#pragma unroll
          for (int w = 0; w < kWarps; ++w) {
            int offset = w * (4 * ExpertsPerWorkGroup) + token_out * ExpertsPerWorkGroup + expert_j;
            sum += smem_partials[offset];
          }
          if (expert_j < experts_this_group) {
            params.logits[static_cast<int64_t>(token_out) * kGateLogitsPad + expert0 + expert_j] = sum;
          }
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
        if (j < experts_this_group) {
          for (int k = tid; k < hidden; k += kGateThreads) {
            smem_weight[j * hidden + k] =
                params.weight[static_cast<int64_t>(expert0 + j) * hidden + k];
          }
        }
      }
      item.barrier(sycl::access::fence_space::local_space);

      int warps_per_token = 1;
      if (params.tokens <= 1) {
        warps_per_token = kWarps;
      } else if (params.tokens <= 2) {
        warps_per_token = kWarps / 2;
      } else if (params.tokens <= 4) {
        warps_per_token = kWarps / 4;
      } else if (params.tokens <= 8) {
        warps_per_token = kWarps / 8;
      }
      warps_per_token = warps_per_token < 1 ? 1 : warps_per_token;

      if (warps_per_token > 1) {
        int token = warp / warps_per_token;
        int slice = warp - token * warps_per_token;
        // ceil-div so a hidden size that is not a multiple of warps_per_token
        // still covers the whole row (the last slice is short or empty).
        int span = (hidden + warps_per_token - 1) / warps_per_token;
        int k_begin = slice * span;
        int k_end = k_begin + span < hidden ? k_begin + span : hidden;

        if (token < params.tokens) {
          cutlass::bfloat16_t const* x_row = params.x + static_cast<int64_t>(token) * hidden;
#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            float acc = 0.0f;
            if (j < experts_this_group) {
              for (int k = k_begin + lane; k < k_end; k += SubGroupSize) {
                float xv = gate_bf16_to_float(x_row[k]);
                float wv = gate_bf16_to_float(smem_weight[j * hidden + k]);
                acc = sycl::fma(xv, wv, acc);
              }
            }
            float reduced = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) {
              smem_partials[warp * ExpertsPerWorkGroup + j] = reduced;
            }
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (warp == 0) {
          int total_outputs = static_cast<int>(params.tokens) * ExpertsPerWorkGroup;
          if (lane < total_outputs) {
            int token_out = lane / ExpertsPerWorkGroup;
            int expert_j = lane - token_out * ExpertsPerWorkGroup;
            float sum = 0.0f;
            for (int s = 0; s < warps_per_token; ++s) {
              sum += smem_partials[(token_out * warps_per_token + s) * ExpertsPerWorkGroup + expert_j];
            }
            if (expert_j < experts_this_group) {
              params.logits[static_cast<int64_t>(token_out) * kGateLogitsPad + expert0 + expert_j] = sum;
            }
          }
        }
      } else {
        for (int64_t token = warp; token < params.tokens; token += kWarps) {
          cutlass::bfloat16_t const* x_row = params.x + token * hidden;
          float acc[ExpertsPerWorkGroup];
#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            acc[j] = 0.0f;
          }

          for (int k = lane; k < hidden; k += SubGroupSize) {
            float xv = gate_bf16_to_float(x_row[k]);
#pragma unroll
            for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
              if (j < experts_this_group) {
                float wv = gate_bf16_to_float(smem_weight[j * hidden + k]);
                acc[j] = sycl::fma(xv, wv, acc[j]);
              }
            }
          }

#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            float reduced = sycl::reduce_over_group(sg, acc[j], sycl::plus<float>());
            if (lane == 0 && j < experts_this_group) {
              params.logits[token * kGateLogitsPad + expert0 + j] = reduced;
            }
          }
        }
      }
    }

    if constexpr (Fused) {
      item.barrier(sycl::access::fence_space::global_and_local);
      sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);

      if (tid == 0) {
        sycl::atomic_ref<int32_t,
                         sycl::memory_order::acq_rel,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            counter(params.ticket[0]);
        smem_ticket[0] = counter.fetch_add(1);
      }

      item.barrier(sycl::access::fence_space::local_space);
      int32_t ticket_value = smem_ticket[0];
      int32_t last_ticket = static_cast<int32_t>(item.get_group_range(0)) - 1;
      if (ticket_value != last_ticket) {
        return;
      }

      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);

      GateParams gate_params;
      gate_params.logits = params.logits;
      gate_params.bias = params.bias;
      gate_params.global_scale = params.global_scale;
      gate_params.routed_weights = params.routed_weights;
      gate_params.shared_weights = params.shared_weights;
      gate_params.indices = params.indices;
      gate_params.packed = params.packed;
      gate_params.tokens = params.tokens;
      gate_params.logits_stride = kGateLogitsPad;
      gate_params.route_scale = params.route_scale;
      gate_params.activation = Act;
      gate_params.norm_after_topk = params.norm_after_topk;

      for (int64_t row = warp; row < params.tokens; row += kWarps) {
        detail::gate_topk_renorm_row<Packed, Act>(
            gate_params, sg, lane, row, row * static_cast<int64_t>(kGateLogitsPad));
      }

      item.barrier(sycl::access::fence_space::global_and_local);
      if (tid == 0) {
        sycl::atomic_ref<int32_t,
                         sycl::memory_order::acq_rel,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            counter(params.ticket[0]);
        counter.store(0, sycl::memory_order::release, sycl::memory_scope::device);
      }
    }
  }
};

template <int ExpertsPerWorkGroup,
          int SubGroupSize,
          bool Fused = false,
          bool Packed = false,
          int HiddenStatic = 0,
          GateActivation Act = GateActivation::kSigmoid>
sycl::event launch_gate_gemv_static(sycl::queue& queue, GateGemvParams const& params) {
  if (params.tokens == 0) {
    return {};
  }

  static_assert(ExpertsPerWorkGroup == 1 || ExpertsPerWorkGroup == 2 || ExpertsPerWorkGroup == 4);
  static_assert(SubGroupSize == 16 || SubGroupSize == 32);
  static_assert(!Fused || SubGroupSize == kSubGroupSize);
  if (params.hidden <= 0) {
    throw std::invalid_argument("gate GEMV requires hidden > 0");
  }
  if constexpr (HiddenStatic != 0) {
    if (params.hidden != HiddenStatic) {
      throw std::invalid_argument("gate GEMV hidden does not match the specialized width");
    }
  }
  if constexpr (Fused) {
    if (params.tokens > kGateFusedMaxTokens) {
      throw std::invalid_argument("fused gate GEMV supports at most 64 tokens");
    }
    if (params.ticket == nullptr) {
      throw std::invalid_argument("fused gate GEMV requires a non-null ticket pointer");
    }
  }
  // The tokens<=4 fast path never touches smem_weight; sizing it to 1 there
  // keeps a wide `hidden` from capping decode occupancy for nothing.
  bool const stages_weight = !(params.tokens > 0 && params.tokens <= 4);
  std::size_t smem_weight_elems =
      stages_weight ? static_cast<std::size_t>(ExpertsPerWorkGroup) * static_cast<std::size_t>(params.hidden)
                    : std::size_t{1};
  if (stages_weight) {
    std::size_t local_mem = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    std::size_t needed = smem_weight_elems * sizeof(cutlass::bfloat16_t) +
                         static_cast<std::size_t>((kGateThreads / SubGroupSize) * 4 * ExpertsPerWorkGroup) *
                             sizeof(float) +
                         sizeof(int32_t);
    if (needed > local_mem) {
      throw std::invalid_argument(
          "gate GEMV weight staging needs " + std::to_string(needed) + " bytes of SLM but the device has " +
          std::to_string(local_mem) + "; lower --hidden or --experts-per-wg");
    }
  }

  int64_t groups = (kGateTotalExperts + ExpertsPerWorkGroup - 1) / ExpertsPerWorkGroup;
  sycl::range<1> local(static_cast<std::size_t>(kGateThreads));
  sycl::range<1> global(static_cast<std::size_t>(groups * kGateThreads));

  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<cutlass::bfloat16_t, 1> smem_weight(sycl::range<1>(smem_weight_elems), cgh);
    sycl::local_accessor<float, 1> smem_partials(
        sycl::range<1>(
            static_cast<std::size_t>((kGateThreads / SubGroupSize) * 4 * ExpertsPerWorkGroup)),
        cgh);
    sycl::local_accessor<int32_t, 1> smem_ticket(sycl::range<1>(1), cgh);
    GateGemvKernel<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed, HiddenStatic, Act> kernel{
        params, smem_weight, smem_partials, smem_ticket};
    cgh.parallel_for<GateGemvKernel<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed, HiddenStatic, Act>>(
        sycl::nd_range<1>(global, local), kernel);
  });
}

inline int default_gate_gemv_experts_per_workgroup(int requested) {
  int value = requested == 0 ? 1 : requested;
  if (value != 1 && value != 2 && value != 4) {
    throw std::invalid_argument("experts_per_workgroup must be one of {0, 1, 2, 4}");
  }
  return value;
}

inline int default_gate_gemv_subgroup_size(int requested, int64_t tokens) {
  (void)tokens;
  int value = requested == 0 ? 32 : requested;
  if (value != 16 && value != 32) {
    throw std::invalid_argument("subgroup_size must be one of {0, 16, 32}");
  }
  return value;
}

/// Picks the compile-time-specialized instantiation when the runtime width is
/// the one sglang's GEMV shortcut is built for, otherwise the generic path.
template <int ExpertsPerWorkGroup,
          int SubGroupSize,
          bool Fused,
          bool Packed,
          GateActivation Act = GateActivation::kSigmoid>
sycl::event launch_gate_gemv_hidden(sycl::queue& queue, GateGemvParams const& params) {
  if (params.hidden == kGateHiddenSpecialized) {
    return launch_gate_gemv_static<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed,
                                   kGateHiddenSpecialized, Act>(queue, params);
  }
  return launch_gate_gemv_static<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed, 0, Act>(queue, params);
}

template <int ExpertsPerWorkGroup>
sycl::event launch_gate_gemv_experts(sycl::queue& queue, GateGemvParams const& params, int subgroup_size) {
  subgroup_size = default_gate_gemv_subgroup_size(subgroup_size, params.tokens);
  switch (subgroup_size) {
    case 16:
      return launch_gate_gemv_hidden<ExpertsPerWorkGroup, 16, false, false>(queue, params);
    case 32:
      return launch_gate_gemv_hidden<ExpertsPerWorkGroup, 32, false, false>(queue, params);
    default:
      throw std::invalid_argument("subgroup_size must be one of {0, 16, 32}");
  }
}

inline sycl::event launch_gate_gemv(
    sycl::queue& queue,
    GateGemvParams const& params,
    int experts_per_workgroup = 0,
    int subgroup_size = 0) {
  experts_per_workgroup = default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  switch (experts_per_workgroup) {
    case 1:
      return launch_gate_gemv_experts<1>(queue, params, subgroup_size);
    case 2:
      return launch_gate_gemv_experts<2>(queue, params, subgroup_size);
    case 4:
      return launch_gate_gemv_experts<4>(queue, params, subgroup_size);
    default:
      throw std::invalid_argument("experts_per_workgroup must be one of {0, 1, 2, 4}");
  }
}

template <bool Packed, GateActivation Act, int ExpertsPerWorkGroup>
sycl::event launch_gate_gemv_fused_experts(
    sycl::queue& queue,
    GateGemvParams const& params,
    int subgroup_size) {
  subgroup_size = default_gate_gemv_subgroup_size(subgroup_size, params.tokens);
  if (subgroup_size != kSubGroupSize) {
    throw std::invalid_argument("fused gate GEMV requires subgroup_size 32");
  }
  return launch_gate_gemv_hidden<ExpertsPerWorkGroup, kSubGroupSize, true, Packed, Act>(queue, params);
}

template <bool Packed, GateActivation Act>
sycl::event launch_gate_gemv_fused(
    sycl::queue& queue,
    GateGemvParams const& params,
    int experts_per_workgroup = 0,
    int subgroup_size = 0) {
  experts_per_workgroup = default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  switch (experts_per_workgroup) {
    case 1:
      return launch_gate_gemv_fused_experts<Packed, Act, 1>(queue, params, subgroup_size);
    case 2:
      return launch_gate_gemv_fused_experts<Packed, Act, 2>(queue, params, subgroup_size);
    case 4:
      return launch_gate_gemv_fused_experts<Packed, Act, 4>(queue, params, subgroup_size);
    default:
      throw std::invalid_argument("experts_per_workgroup must be one of {0, 1, 2, 4}");
  }
}

inline sycl::event launch_gate_gemv_fused(
    sycl::queue& queue,
    GateGemvParams const& params,
    bool packed,
    int experts_per_workgroup = 0,
    int subgroup_size = 0) {
  if (params.activation == GateActivation::kSoftmax) {
    return packed ? launch_gate_gemv_fused<true, GateActivation::kSoftmax>(
                        queue, params, experts_per_workgroup, subgroup_size)
                  : launch_gate_gemv_fused<false, GateActivation::kSoftmax>(
                        queue, params, experts_per_workgroup, subgroup_size);
  }
  return packed ? launch_gate_gemv_fused<true, GateActivation::kSigmoid>(
                      queue, params, experts_per_workgroup, subgroup_size)
                : launch_gate_gemv_fused<false, GateActivation::kSigmoid>(
                      queue, params, experts_per_workgroup, subgroup_size);
}

}  // namespace cutlass::examples::bmg_moe_gate
