/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
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

/// InklingGate's `gate_activation` config flag (sglang
/// srt/configs/inkling.py: Literal["sigmoid", "softmax"]).
enum class GateActivation : int { kSigmoid = 0, kSoftmax = 1 };

struct GateParams {
  float const* logits = nullptr;
  /// Per-routed-expert selection bias. nullptr models `use_gate_bias=false`.
  float const* bias = nullptr;
  /// Scalar learned gate scale. nullptr models `use_global_scale=false`.
  float const* global_scale = nullptr;
  float* routed_weights = nullptr;
  float* shared_weights = nullptr;
  int32_t* indices = nullptr;
  int32_t* packed = nullptr;
  int64_t tokens = 0;
  int64_t logits_stride = kTotalExperts;
  float route_scale = 1.0f;
  GateActivation activation = GateActivation::kSigmoid;
  /// `norm_after_topk=true` renormalizes the selected ++ shared logits;
  /// `false` takes the raw activated score at the selected indices.
  bool norm_after_topk = true;
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

/// Precise exp/log/log1p wrappers. The same code runs on the host (as the
/// verification reference) and on the device, so the two agree to ~1 ulp.
inline float gate_exp(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::exp(x);
#else
  return std::exp(x);
#endif
}

inline float gate_log(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::log(x);
#else
  return std::log(x);
#endif
}

inline float gate_log1p(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::log1p(x);
#else
  return std::log1p(x);
#endif
}

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

/// log(sigmoid(x)) in the numerically stable form sglang's triton gate uses:
/// `min(x, 0) - log1p(exp(-|x|))` (_inkling_compute_logsigmoid_norm,
/// sglang/srt/models/inkling_common/moe.py).
inline float logsigmoid(float x) {
  float min_x = x < 0.0f ? x : 0.0f;
  float abs_x = x < 0.0f ? -x : x;
  return min_x - gate_log1p(gate_exp(-abs_x));
}

/// Value fed to the post-top-k renormalization. `_renorm_topk_logits` applies
/// `exp(logsigmoid(l) - logsumexp(logsigmoid(l)))` for the sigmoid gate and
/// `topk_logits.softmax(-1)` for the softmax gate; both are
/// `exp(lp - logsumexp(lp))` with lp = logsigmoid(l) resp. lp = l.
inline float gate_pre_norm(GateActivation act, float logit) {
  return act == GateActivation::kSigmoid ? logsigmoid(logit) : logit;
}

/// Selection score before the bias: `sigmoid(logits)` or, for the softmax
/// gate, `softmax(logits)` over all `kTotalExperts` columns of the row.
/// `smax` / `sinv` carry the row's softmax max-shift and 1/sum.
inline float gate_activate(GateActivation act, float logit, float smax, float sinv) {
  return act == GateActivation::kSigmoid ? sigmoid_device(logit) : gate_exp(logit - smax) * sinv;
}

/// out[i] = exp(lp[i] - logsumexp(lp)), max-shifted exactly as
/// _inkling_compute_logsigmoid_norm does.
inline void gate_normalize_active(float const* lp, int count, float* out) {
  float max_lp = -FLT_MAX;
  for (int i = 0; i < count; ++i) {
    max_lp = lp[i] > max_lp ? lp[i] : max_lp;
  }
  float sum_exp = 0.0f;
  for (int i = 0; i < count; ++i) {
    sum_exp += gate_exp(lp[i] - max_lp);
  }
  float logsumexp = max_lp + gate_log(sum_exp);
  for (int i = 0; i < count; ++i) {
    out[i] = gate_exp(lp[i] - logsumexp);
  }
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

/// Ties break toward the lower expert id, matching gate_topk's
/// `indx_to_key(idx) = N_PAD - idx` packed into the sort key's low bits.
inline bool score_better(float score, int idx, float best_score, int best_idx) {
  return score > best_score || (score == best_score && idx < best_idx);
}

template <bool Packed, GateActivation Act>
inline void gate_topk_renorm_row(
    GateParams const& params,
    sycl::sub_group sg,
    int lane,
    int64_t row,
    int64_t row_base) {
  float raw[kValuesPerLane];
  float sel[kValuesPerLane];

  float shared_logit[kSharedExperts];
#pragma unroll
  for (int s = 0; s < kSharedExperts; ++s) {
    shared_logit[s] = params.logits[row_base + kRoutedExperts + s];
  }

#pragma unroll
  for (int j = 0; j < kValuesPerLane; ++j) {
    raw[j] = params.logits[row_base + lane * kValuesPerLane + j];
  }

  float softmax_max = 0.0f;
  float softmax_inv = 1.0f;
  if constexpr (Act == GateActivation::kSoftmax) {
    // scores = logits.softmax(dim=-1) over the full [tokens, 258] gate row,
    // i.e. the shared columns participate in the denominator.
    float row_max = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      row_max = raw[j] > row_max ? raw[j] : row_max;
    }
#pragma unroll
    for (int s = 0; s < kSharedExperts; ++s) {
      row_max = shared_logit[s] > row_max ? shared_logit[s] : row_max;
    }
    row_max = sycl::reduce_over_group(sg, row_max, sycl::maximum<float>());

    float sum_exp = 0.0f;
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      sum_exp += gate_exp(raw[j] - row_max);
    }
    sum_exp = sycl::reduce_over_group(sg, sum_exp, sycl::plus<float>());
#pragma unroll
    for (int s = 0; s < kSharedExperts; ++s) {
      sum_exp += gate_exp(shared_logit[s] - row_max);
    }
    softmax_max = row_max;
    softmax_inv = 1.0f / sum_exp;
  }

  // The use_gate_bias test is hoisted out of the per-lane loop: it is uniform
  // across the whole launch, so keeping it inside costs 8 predicated adds.
  float const* bias = params.bias;
  if (bias != nullptr) {
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      sel[j] = gate_activate(Act, raw[j], softmax_max, softmax_inv) + bias[lane * kValuesPerLane + j];
    }
  } else {
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      sel[j] = gate_activate(Act, raw[j], softmax_max, softmax_inv);
    }
  }

  int selected_idx[kTopK];

#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best_score = -FLT_MAX;
    int best_idx = INT32_MAX;

#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      int expert = lane * kValuesPerLane + j;
      if (score_better(sel[j], expert, best_score, best_idx)) {
        best_score = sel[j];
        best_idx = expert;
      }
    }

#pragma unroll
    for (int offset = kSubGroupSize / 2; offset > 0; offset >>= 1) {
      float other_score = sycl::permute_group_by_xor(sg, best_score, offset);
      int other_idx = sycl::permute_group_by_xor(sg, best_idx, offset);
      if (score_better(other_score, other_idx, best_score, best_idx)) {
        best_score = other_score;
        best_idx = other_idx;
      }
    }

    if (lane == 0) {
      selected_idx[k] = best_idx;
    }

    int owner_lane = best_idx / kValuesPerLane;
    int owner_j = best_idx - owner_lane * kValuesPerLane;
    if (lane == owner_lane) {
      sel[owner_j] = -FLT_MAX;
    }
  }

  if (lane == 0) {
    float scale = params.route_scale;
    if (params.global_scale != nullptr) {
      scale *= params.global_scale[0];
    }

    // Re-read the 6 selected raw logits instead of carrying them through the
    // butterfly. Six cache-resident scalar loads on one lane are much cheaper
    // than a third shuffle per reduction step plus keeping raw[] live: routing
    // the logit through the sort key measured ~1.7x slower on BMG.
    float selected_raw[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      selected_raw[k] = params.logits[row_base + selected_idx[k]];
    }

    float weights[kTopAndShared];
    if (params.norm_after_topk) {
      float lp[kTopAndShared];
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        lp[k] = gate_pre_norm(Act, selected_raw[k]);
      }
#pragma unroll
      for (int s = 0; s < kSharedExperts; ++s) {
        lp[kTopK + s] = gate_pre_norm(Act, shared_logit[s]);
      }
      gate_normalize_active(lp, kTopAndShared, weights);
#pragma unroll
      for (int i = 0; i < kTopAndShared; ++i) {
        weights[i] *= scale;
      }
    } else {
      // norm_after_topk=false: routed_scores.gather(topk_indices) * scale.
      // sglang returns shared_gammas=None here, so the shared slots have no
      // model-side counterpart; zero them so the contract stays explicit.
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        weights[k] = gate_activate(Act, selected_raw[k], softmax_max, softmax_inv) * scale;
      }
#pragma unroll
      for (int s = 0; s < kSharedExperts; ++s) {
        weights[kTopK + s] = 0.0f;
      }
    }

#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      if constexpr (Packed) {
        params.packed[row * kTopK + k] = pack_routed_device(selected_idx[k], weights[k]);
      } else {
        params.routed_weights[row * kTopK + k] = weights[k];
        params.indices[row * kTopK + k] = selected_idx[k];
      }
    }
#pragma unroll
    for (int s = 0; s < kSharedExperts; ++s) {
      params.shared_weights[row * kSharedExperts + s] = weights[kTopK + s];
    }
  }
}

}  // namespace detail

/// The InklingGate config flags this example covers. Defaults are the shipped
/// checkpoint's (gate_activation=sigmoid, use_gate_bias/use_global_scale=true,
/// norm_after_topk=true, route_scale=8.0).
struct GateConfig {
  GateActivation activation = GateActivation::kSigmoid;
  bool norm_after_topk = true;
  bool use_bias = true;
  bool use_global_scale = true;
  float route_scale = 8.0f;
  float global_scale = 1.25f;
};

/// CPU reference for one gate row, mirroring InklingGate.forward:
///   scores = sigmoid(logits) | logits.softmax(-1)            [258 columns]
///   idx    = topk(scores[:256] + bias, 6)                    ties -> lower id
///   norm_after_topk: exp(lp - logsumexp(lp)) over selected ++ shared, with
///                    lp = logsigmoid(logit) (sigmoid) or logit (softmax)
///   else:            scores.gather(idx)
///   weights *= route_scale [* global_scale]
/// `logits` points at the row's first column (kTotalExperts valid entries);
/// `bias` may be null when use_bias is false.
inline void gate_reference_row(
    GateConfig const& cfg,
    float const* logits,
    float const* bias,
    int32_t* out_indices,
    float* out_routed,
    float* out_shared) {
  float scores[kTotalExperts];
  if (cfg.activation == GateActivation::kSigmoid) {
    for (int e = 0; e < kTotalExperts; ++e) {
      scores[e] = detail::sigmoid_host(logits[e]);
    }
  } else {
    float row_max = -FLT_MAX;
    for (int e = 0; e < kTotalExperts; ++e) {
      row_max = logits[e] > row_max ? logits[e] : row_max;
    }
    float sum_exp = 0.0f;
    for (int e = 0; e < kTotalExperts; ++e) {
      sum_exp += detail::gate_exp(logits[e] - row_max);
    }
    float inv = 1.0f / sum_exp;
    for (int e = 0; e < kTotalExperts; ++e) {
      scores[e] = detail::gate_exp(logits[e] - row_max) * inv;
    }
  }

  float sel[kRoutedExperts];
  for (int e = 0; e < kRoutedExperts; ++e) {
    sel[e] = scores[e] + (bias != nullptr ? bias[e] : 0.0f);
  }

  int32_t selected[kTopK];
  for (int k = 0; k < kTopK; ++k) {
    float best_score = -FLT_MAX;
    int best_idx = INT32_MAX;
    for (int e = 0; e < kRoutedExperts; ++e) {
      if (detail::score_better(sel[e], e, best_score, best_idx)) {
        best_score = sel[e];
        best_idx = e;
      }
    }
    selected[k] = best_idx;
    sel[best_idx] = -FLT_MAX;
  }

  float scale = cfg.route_scale;
  if (cfg.use_global_scale) {
    scale *= cfg.global_scale;
  }

  float weights[kTopAndShared];
  if (cfg.norm_after_topk) {
    float lp[kTopAndShared];
    for (int k = 0; k < kTopK; ++k) {
      lp[k] = detail::gate_pre_norm(cfg.activation, logits[selected[k]]);
    }
    for (int s = 0; s < kSharedExperts; ++s) {
      lp[kTopK + s] = detail::gate_pre_norm(cfg.activation, logits[kRoutedExperts + s]);
    }
    detail::gate_normalize_active(lp, kTopAndShared, weights);
    for (int i = 0; i < kTopAndShared; ++i) {
      weights[i] *= scale;
    }
  } else {
    for (int k = 0; k < kTopK; ++k) {
      weights[k] = scores[selected[k]] * scale;
    }
    for (int s = 0; s < kSharedExperts; ++s) {
      weights[kTopK + s] = 0.0f;
    }
  }

  for (int k = 0; k < kTopK; ++k) {
    out_indices[k] = selected[k];
    out_routed[k] = weights[k];
  }
  for (int s = 0; s < kSharedExperts; ++s) {
    out_shared[s] = weights[kTopK + s];
  }
}

template <bool Packed, int RowsPerWorkGroup, GateActivation Act>
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
    detail::gate_topk_renorm_row<Packed, Act>(params_, sg, lane, row, row_base);
  }

 private:
  GateParams params_;
};

template <bool Packed, GateActivation Act, int RowsPerWorkGroup>
sycl::event launch_gate_topk_renorm_static(sycl::queue& queue, GateParams const& params) {
  if (params.tokens == 0) {
    return {};
  }

  int64_t groups = (params.tokens + RowsPerWorkGroup - 1) / RowsPerWorkGroup;
  sycl::range<1> local(static_cast<std::size_t>(RowsPerWorkGroup * kSubGroupSize));
  sycl::range<1> global(static_cast<std::size_t>(groups * RowsPerWorkGroup * kSubGroupSize));
  GateTopKRenormKernel<Packed, RowsPerWorkGroup, Act> kernel(params);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(global, local), kernel);
  });
}

template <bool Packed, GateActivation Act>
sycl::event launch_gate_topk_renorm(sycl::queue& queue, GateParams const& params, int rows_per_workgroup = 0) {
  if (rows_per_workgroup == 0) {
    rows_per_workgroup = params.tokens <= 16384 ? 2 : 1;
  }

  switch (rows_per_workgroup) {
    case 1:
      return launch_gate_topk_renorm_static<Packed, Act, 1>(queue, params);
    case 2:
      return launch_gate_topk_renorm_static<Packed, Act, 2>(queue, params);
    case 4:
      return launch_gate_topk_renorm_static<Packed, Act, 4>(queue, params);
    case 8:
      return launch_gate_topk_renorm_static<Packed, Act, 8>(queue, params);
    default:
      throw std::invalid_argument("rows_per_workgroup must be one of {0, 1, 2, 4, 8}");
  }
}

inline sycl::event launch_gate_topk_renorm(
    sycl::queue& queue,
    GateParams const& params,
    bool packed,
    int rows_per_workgroup = 0) {
  if (params.activation == GateActivation::kSoftmax) {
    return packed ? launch_gate_topk_renorm<true, GateActivation::kSoftmax>(queue, params, rows_per_workgroup)
                  : launch_gate_topk_renorm<false, GateActivation::kSoftmax>(queue, params, rows_per_workgroup);
  }
  return packed ? launch_gate_topk_renorm<true, GateActivation::kSigmoid>(queue, params, rows_per_workgroup)
                : launch_gate_topk_renorm<false, GateActivation::kSigmoid>(queue, params, rows_per_workgroup);
}

}  // namespace cutlass::examples::bmg_moe_gate
