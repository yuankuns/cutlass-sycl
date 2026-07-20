#pragma once

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cutlass::examples::bmg_moe_gate {

static constexpr int kRoutedExperts = 256;
static constexpr int kSharedExperts = 2;
static constexpr int kTotalExperts = kRoutedExperts + kSharedExperts;
static constexpr int kTopK = 6;
static constexpr int kTopAndShared = kTopK + kSharedExperts;
static constexpr int kSubGroupSize = 32;
static constexpr int kValuesPerLane = kRoutedExperts / kSubGroupSize;

struct GateParams {
  float const* logits = nullptr;
  float const* bias = nullptr;
  float const* global_scale = nullptr;
  float* routed_weights = nullptr;
  float* shared_weights = nullptr;
  int32_t* indices = nullptr;
  int32_t* packed = nullptr;
  int64_t tokens = 0;
  int64_t logits_stride = kTotalExperts;
  float route_scale = 1.0f;
};

inline uint16_t host_f32_to_bf16_rne(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

inline float host_bf16_to_f32(uint16_t x) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(x) << 16);
}

inline int32_t host_pack_routed(int32_t expert, float weight) {
  return static_cast<int32_t>((static_cast<uint32_t>(expert) << 16) |
                              static_cast<uint32_t>(host_f32_to_bf16_rne(weight)));
}

namespace detail {

inline float sigmoid_host(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

inline float sigmoid_device(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return 1.0f / (1.0f + sycl::native::exp(-x));
#else
  return sigmoid_host(x);
#endif
}

inline uint16_t f32_to_bf16_rne_device(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

inline int32_t pack_routed_device(int32_t expert, float weight) {
  return static_cast<int32_t>((static_cast<uint32_t>(expert) << 16) |
                              static_cast<uint32_t>(f32_to_bf16_rne_device(weight)));
}

inline bool score_better(float score, int idx, float best_score, int best_idx) {
  return score > best_score || (score == best_score && idx < best_idx);
}

}  // namespace detail

template <bool Packed, int RowsPerWorkGroup>
class GateTopKRenormKernel {
 public:
  explicit GateTopKRenormKernel(GateParams params) : params_(params) {}

  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();
    int lane = static_cast<int>(sg.get_local_id());
    int row_in_group = static_cast<int>(sg.get_group_id());
    int64_t row = static_cast<int64_t>(item.get_group(0)) * RowsPerWorkGroup + row_in_group;
    if (row >= params_.tokens) {
      return;
    }

    int64_t row_base = row * params_.logits_stride;
    float scores[kValuesPerLane];

#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      int expert = lane * kValuesPerLane + j;
      float s = detail::sigmoid_device(params_.logits[row_base + expert]);
      scores[j] = s + params_.bias[expert];
    }

    int selected_idx[kTopK];
    float selected_sigmoid[kTopK];

#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      float best_score = -FLT_MAX;
      int best_idx = INT32_MAX;

#pragma unroll
      for (int j = 0; j < kValuesPerLane; ++j) {
        int expert = lane * kValuesPerLane + j;
        float score = scores[j];
        if (detail::score_better(score, expert, best_score, best_idx)) {
          best_score = score;
          best_idx = expert;
        }
      }

#pragma unroll
      for (int offset = kSubGroupSize / 2; offset > 0; offset >>= 1) {
        float other_score = sycl::permute_group_by_xor(sg, best_score, offset);
        int other_idx = sycl::permute_group_by_xor(sg, best_idx, offset);
        if (detail::score_better(other_score, other_idx, best_score, best_idx)) {
          best_score = other_score;
          best_idx = other_idx;
        }
      }

      if (lane == 0) {
        selected_idx[k] = best_idx;
        selected_sigmoid[k] = best_score - params_.bias[best_idx];
      }

      int owner_lane = best_idx / kValuesPerLane;
      int owner_j = best_idx - owner_lane * kValuesPerLane;
      if (lane == owner_lane) {
        scores[owner_j] = -FLT_MAX;
      }
    }

    if (lane == 0) {
      float shared0 = detail::sigmoid_device(params_.logits[row_base + kRoutedExperts]);
      float shared1 = detail::sigmoid_device(params_.logits[row_base + kRoutedExperts + 1]);
      float sum = shared0 + shared1;

#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        sum += selected_sigmoid[k];
      }

      float scale = params_.route_scale * params_.global_scale[0] / sum;

#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        float weight = selected_sigmoid[k] * scale;
        if constexpr (Packed) {
          params_.packed[row * kTopK + k] = detail::pack_routed_device(selected_idx[k], weight);
        } else {
          params_.routed_weights[row * kTopK + k] = weight;
          params_.indices[row * kTopK + k] = selected_idx[k];
        }
      }
      params_.shared_weights[row * kSharedExperts] = shared0 * scale;
      params_.shared_weights[row * kSharedExperts + 1] = shared1 * scale;
    }
  }

 private:
  GateParams params_;
};

template <bool Packed, int RowsPerWorkGroup>
sycl::event launch_gate_topk_renorm_static(sycl::queue& queue, GateParams const& params) {
  if (params.tokens == 0) {
    return {};
  }

  int64_t groups = (params.tokens + RowsPerWorkGroup - 1) / RowsPerWorkGroup;
  sycl::range<1> local(static_cast<std::size_t>(RowsPerWorkGroup * kSubGroupSize));
  sycl::range<1> global(static_cast<std::size_t>(groups * RowsPerWorkGroup * kSubGroupSize));
  GateTopKRenormKernel<Packed, RowsPerWorkGroup> kernel(params);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(global, local), kernel);
  });
}

template <bool Packed>
sycl::event launch_gate_topk_renorm(sycl::queue& queue, GateParams const& params, int rows_per_workgroup = 0) {
  if (rows_per_workgroup == 0) {
    if constexpr (Packed) {
      rows_per_workgroup = params.tokens <= 8192 ? 2 : 1;
    } else {
      rows_per_workgroup = 1;
    }
  }

  switch (rows_per_workgroup) {
    case 1:
      return launch_gate_topk_renorm_static<Packed, 1>(queue, params);
    case 2:
      return launch_gate_topk_renorm_static<Packed, 2>(queue, params);
    case 4:
      return launch_gate_topk_renorm_static<Packed, 4>(queue, params);
    case 8:
      return launch_gate_topk_renorm_static<Packed, 8>(queue, params);
    default:
      throw std::invalid_argument("rows_per_workgroup must be one of {0, 1, 2, 4, 8}");
  }
}

inline sycl::event launch_gate_topk_renorm(
    sycl::queue& queue,
    GateParams const& params,
    bool packed,
    int rows_per_workgroup = 0) {
  return packed ? launch_gate_topk_renorm<true>(queue, params, rows_per_workgroup)
                : launch_gate_topk_renorm<false>(queue, params, rows_per_workgroup);
}

}  // namespace cutlass::examples::bmg_moe_gate
