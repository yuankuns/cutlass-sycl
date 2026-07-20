#pragma once

#include <sycl/sycl.hpp>

#include "16_bmg_moe_gate_topk_renorm.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"

#include <cstdint>
#include <stdexcept>

namespace cutlass::examples::bmg_moe_gate {

static constexpr int kGateHidden = 6144;
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
  float route_scale = 1.0f;
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

template <int ExpertsPerWorkGroup, int SubGroupSize, bool Fused = false, bool Packed = false>
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

      for (int k = tid; k < kGateHidden; k += kGateThreads) {
        float x_val[4];
#pragma unroll
        for (int token = 0; token < 4; ++token) {
          x_val[token] = 0.0f;
          if (token < params.tokens) {
            x_val[token] = gate_bf16_to_float(params.x[static_cast<int64_t>(token) * kGateHidden + k]);
          }
        }

#pragma unroll
        for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
          if (j < experts_this_group) {
            cutlass::bfloat16_t const* w_row =
                params.weight + static_cast<int64_t>(expert0 + j) * kGateHidden;
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
          for (int k = tid; k < kGateHidden; k += kGateThreads) {
            smem_weight[j * kGateHidden + k] =
                params.weight[static_cast<int64_t>(expert0 + j) * kGateHidden + k];
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
        int span = kGateHidden / warps_per_token;
        int k_begin = slice * span;
        int k_end = k_begin + span;

        if (token < params.tokens) {
          cutlass::bfloat16_t const* x_row = params.x + static_cast<int64_t>(token) * kGateHidden;
#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            float acc = 0.0f;
            if (j < experts_this_group) {
              for (int k = k_begin + lane; k < k_end; k += SubGroupSize) {
                float xv = gate_bf16_to_float(x_row[k]);
                float wv = gate_bf16_to_float(smem_weight[j * kGateHidden + k]);
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
          cutlass::bfloat16_t const* x_row = params.x + token * kGateHidden;
          float acc[ExpertsPerWorkGroup];
#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            acc[j] = 0.0f;
          }

          for (int k = lane; k < kGateHidden; k += SubGroupSize) {
            float xv = gate_bf16_to_float(x_row[k]);
#pragma unroll
            for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
              if (j < experts_this_group) {
                float wv = gate_bf16_to_float(smem_weight[j * kGateHidden + k]);
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

      for (int64_t row = warp; row < params.tokens; row += kWarps) {
        detail::gate_topk_renorm_row<Packed>(
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

template <int ExpertsPerWorkGroup, int SubGroupSize, bool Fused = false, bool Packed = false>
sycl::event launch_gate_gemv_static(sycl::queue& queue, GateGemvParams const& params) {
  if (params.tokens == 0) {
    return {};
  }

  static_assert(ExpertsPerWorkGroup == 1 || ExpertsPerWorkGroup == 2 || ExpertsPerWorkGroup == 4);
  static_assert(SubGroupSize == 16 || SubGroupSize == 32);
  static_assert(!Fused || SubGroupSize == kSubGroupSize);
  if constexpr (Fused) {
    if (params.tokens > kGateFusedMaxTokens) {
      throw std::invalid_argument("fused gate GEMV supports at most 64 tokens");
    }
    if (params.ticket == nullptr) {
      throw std::invalid_argument("fused gate GEMV requires a non-null ticket pointer");
    }
  }
  int64_t groups = (kGateTotalExperts + ExpertsPerWorkGroup - 1) / ExpertsPerWorkGroup;
  sycl::range<1> local(static_cast<std::size_t>(kGateThreads));
  sycl::range<1> global(static_cast<std::size_t>(groups * kGateThreads));

  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<cutlass::bfloat16_t, 1> smem_weight(
        sycl::range<1>(static_cast<std::size_t>(ExpertsPerWorkGroup * kGateHidden)), cgh);
    sycl::local_accessor<float, 1> smem_partials(
        sycl::range<1>(
            static_cast<std::size_t>((kGateThreads / SubGroupSize) * 4 * ExpertsPerWorkGroup)),
        cgh);
    sycl::local_accessor<int32_t, 1> smem_ticket(sycl::range<1>(1), cgh);
    GateGemvKernel<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed> kernel{
        params, smem_weight, smem_partials, smem_ticket};
    cgh.parallel_for<GateGemvKernel<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed>>(
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

template <int ExpertsPerWorkGroup>
sycl::event launch_gate_gemv_experts(sycl::queue& queue, GateGemvParams const& params, int subgroup_size) {
  subgroup_size = default_gate_gemv_subgroup_size(subgroup_size, params.tokens);
  switch (subgroup_size) {
    case 16:
      return launch_gate_gemv_static<ExpertsPerWorkGroup, 16, false, false>(queue, params);
    case 32:
      return launch_gate_gemv_static<ExpertsPerWorkGroup, 32, false, false>(queue, params);
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

template <bool Packed, int ExpertsPerWorkGroup>
sycl::event launch_gate_gemv_fused_experts(
    sycl::queue& queue,
    GateGemvParams const& params,
    int subgroup_size) {
  subgroup_size = default_gate_gemv_subgroup_size(subgroup_size, params.tokens);
  if (subgroup_size != kSubGroupSize) {
    throw std::invalid_argument("fused gate GEMV requires subgroup_size 32");
  }
  return launch_gate_gemv_static<ExpertsPerWorkGroup, kSubGroupSize, true, Packed>(queue, params);
}

template <bool Packed>
sycl::event launch_gate_gemv_fused(
    sycl::queue& queue,
    GateGemvParams const& params,
    int experts_per_workgroup = 0,
    int subgroup_size = 0) {
  experts_per_workgroup = default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  switch (experts_per_workgroup) {
    case 1:
      return launch_gate_gemv_fused_experts<Packed, 1>(queue, params, subgroup_size);
    case 2:
      return launch_gate_gemv_fused_experts<Packed, 2>(queue, params, subgroup_size);
    case 4:
      return launch_gate_gemv_fused_experts<Packed, 4>(queue, params, subgroup_size);
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
  return packed ? launch_gate_gemv_fused<true>(queue, params, experts_per_workgroup, subgroup_size)
                : launch_gate_gemv_fused<false>(queue, params, experts_per_workgroup, subgroup_size);
}

}  // namespace cutlass::examples::bmg_moe_gate
