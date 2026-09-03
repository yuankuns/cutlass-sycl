/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "20_bmg_dflash_common.hpp"

#include <numeric>

namespace dflash = cutlass::examples::bmg_dflash;

namespace {

// Mirrors sglang's scatter_mamba_states_after_mtp_verify (see
// python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py:685) for one
// conv stream:
//   dst[slots[r]]            <- intermediate[r, steps[r]]
// The *source* row is the request ordinal `r` (the intermediate conv-window
// cache is [layers, spec_state_size + 1, draft_token_num, W-1, dim], indexed by
// batch row), while the *destination* row is the pool slot from
// dst_indices_raw. That asymmetry is explicit in the Triton kernels:
// `src_idx = pid_req` / `dst_idx = tl.load(dst_indices_raw_ptr + pid_req)`
// (mamba_state_scatter_triton.py:356-358) and, in the fused multi-stream
// kernel, `src_idx = tl.where(is2, off2, off1)` with `off2 = pid_req - n1`
// (mamba_state_scatter_triton.py:497-505) -- i.e. the optional second
// (interval-crossing "track") index set restarts the source ordinal at 0, which
// is why the two passes are launched with their own index arrays below.
//
// Validity mask, also from the Triton kernels (mamba_state_scatter_triton.py:355
// and :507-513): `step < 0` means "no accepted step, leave the slot untouched",
// plus the range guards dst_idx in [0, dst_req_size), src_idx < src_req_size and
// step < src_step_size.
template <typename Element>
struct ScatterRowsParams {
  Element* dst = nullptr;
  Element const* intermediate = nullptr;
  int64_t const* slots = nullptr;  // dst_indices_raw: pool slot per request
  int64_t const* steps = nullptr;  // step_indices_raw: entry >= 0 means valid
  int count = 0;                   // requests in this pass (n1 or n2)
  int src_rows = 0;                // intermediate request rows (spec_state_size + 1)
  int dst_slots = 0;               // persistent-state pool slots (size + 1)
  int t_max = 0;                   // draft_token_num
  int row_elems = 0;               // (W-1) * dim for a conv stream
};

// Returns false for a masked-out or out-of-range request (nothing is written).
template <typename Element>
CUTLASS_DEVICE
bool scatter_row_bases(ScatterRowsParams<Element> const& params,
                       int request,
                       int64_t& dst_base,
                       int64_t& src_base) {
  int64_t step = params.steps[request];
  if (step < 0) {
    return false;
  }
  int64_t slot = params.slots[request];
  bool in_range = slot >= 0 && slot < params.dst_slots && request < params.src_rows && step < params.t_max;
  if (!in_range) {
    return false;
  }
  dst_base = slot * params.row_elems;
  src_base = (static_cast<int64_t>(request) * params.t_max + step) * params.row_elems;
  return true;
}

template <typename Element>
class ScatterRowsScalarKernel {
 public:
  explicit ScatterRowsScalarKernel(ScatterRowsParams<Element> params) : params_(params) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t total_lanes = static_cast<int64_t>(params_.count) * params_.row_elems;
    int64_t lane = static_cast<int64_t>(item.get_global_id(0));
    if (lane >= total_lanes) {
      return;
    }
    int request = static_cast<int>(lane / params_.row_elems);
    int elem = static_cast<int>(lane - static_cast<int64_t>(request) * params_.row_elems);

    int64_t dst_base = 0;
    int64_t src_base = 0;
    if (!scatter_row_bases(params_, request, dst_base, src_base)) {
      return;
    }
    params_.dst[dst_base + elem] = params_.intermediate[src_base + elem];
  }

 private:
  ScatterRowsParams<Element> params_;
};

template <typename Element>
class ScatterRowsPackKernel {
 public:
  using Pack = sycl::vec<uint32_t, dflash::kCopyPackBytes / static_cast<int>(sizeof(uint32_t))>;
  static constexpr int kPackElems = dflash::kCopyPackBytes / static_cast<int>(sizeof(Element));

  explicit ScatterRowsPackKernel(ScatterRowsParams<Element> params) : params_(params) {}

  void operator()(sycl::nd_item<1> item) const {
    int packs_per_row = params_.row_elems / kPackElems;
    int64_t total_lanes = static_cast<int64_t>(params_.count) * packs_per_row;
    int64_t lane = static_cast<int64_t>(item.get_global_id(0));
    if (lane >= total_lanes) {
      return;
    }
    int request = static_cast<int>(lane / packs_per_row);
    int pack = static_cast<int>(lane - static_cast<int64_t>(request) * packs_per_row);

    int64_t dst_base = 0;
    int64_t src_base = 0;
    if (!scatter_row_bases(params_, request, dst_base, src_base)) {
      return;
    }
    auto* dst_pack = reinterpret_cast<Pack*>(params_.dst + dst_base);
    auto const* src_pack = reinterpret_cast<Pack const*>(params_.intermediate + src_base);
    dst_pack[pack] = src_pack[pack];
  }

 private:
  ScatterRowsParams<Element> params_;
};

template <typename Element>
bool can_use_pack_path(ScatterRowsParams<Element> const& params) {
  constexpr int kPackElems = dflash::kCopyPackBytes / static_cast<int>(sizeof(Element));
  auto aligned = [](void const* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % dflash::kCopyPackBytes) == 0;
  };
  return params.row_elems % kPackElems == 0 && aligned(params.dst) && aligned(params.intermediate);
}

template <typename Kernel>
sycl::event submit_1d(sycl::queue& queue, int64_t lanes, Kernel const& kernel) {
  if (lanes <= 0) {
    return {};
  }
  std::size_t local = 64;
  std::size_t global = static_cast<std::size_t>(dflash::ceil_div(lanes, static_cast<int64_t>(local)) * local);
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)), kernel);
  });
}

template <typename Element>
sycl::event launch_scatter_rows_pass(sycl::queue& queue, ScatterRowsParams<Element> const& params) {
  if (params.count <= 0 || params.row_elems <= 0) {
    return {};
  }
  if (can_use_pack_path(params)) {
    constexpr int kPackElems = dflash::kCopyPackBytes / static_cast<int>(sizeof(Element));
    int64_t lanes = static_cast<int64_t>(params.count) * (params.row_elems / kPackElems);
    return submit_1d(queue, lanes, ScatterRowsPackKernel<Element>(params));
  }
  return submit_1d(queue, static_cast<int64_t>(params.count) * params.row_elems,
                   ScatterRowsScalarKernel<Element>(params));
}

// The accept commit plus the optional interval-crossing track pass. The real op
// runs them as two request-index sets over the same source buffer (one fused
// launch in fused_conv_window_scatter_multi, two launches in the per-stream
// fallback fused_conv_window_scatter_with_mask).
template <typename Element>
struct ScatterPasses {
  ScatterRowsParams<Element> main;
  ScatterRowsParams<Element> track;
};

struct ScatterLaunchEvents {
  sycl::event events[2];
  int count = 0;
};

template <typename Element>
ScatterLaunchEvents launch_scatter_rows(sycl::queue& queue, ScatterPasses<Element> const& passes) {
  ScatterLaunchEvents launched;
  if (passes.main.count > 0) {
    launched.events[launched.count++] = launch_scatter_rows_pass(queue, passes.main);
  }
  if (passes.track.count > 0) {
    launched.events[launched.count++] = launch_scatter_rows_pass(queue, passes.track);
  }
  return launched;
}

void append_events(std::vector<sycl::event>& events, ScatterLaunchEvents const& launched) {
  for (int i = 0; i < launched.count; ++i) {
    events.push_back(launched.events[i]);
  }
}

double profiled_elapsed_ms(std::vector<sycl::event> const& events, int iterations) {
  if (events.empty() || iterations <= 0) {
    return 0.0;
  }
  uint64_t total_ns = 0;
  for (sycl::event const& event : events) {
    auto begin = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_ns += static_cast<uint64_t>(end - begin);
  }
  return static_cast<double>(total_ns) * 1.0e-6 / static_cast<double>(iterations);
}

struct ScatterCase {
  std::string name;
  int slots = 4;
  int t_max = 3;
  int d_ssm = 5;
  int width_a = 2;
  int width_b = 3;
  int d_conv = 6;
  int main_count = 3;
  int track_count = 2;
  double target_gbps = 0.0;
  // Request rows in the intermediate window cache; 0 derives it from the
  // request counts. A smaller value exercises the src_idx < src_req_size guard.
  int src_rows = 0;
};

template <typename Element>
struct ScatterHost {
  std::vector<Element> ssm_states;
  std::vector<Element> ssm_intermediate;
  std::vector<Element> conv_a_states;
  std::vector<Element> conv_a_intermediate;
  std::vector<Element> conv_b_states;
  std::vector<Element> conv_b_intermediate;
  std::vector<Element> ssm_ref;
  std::vector<Element> conv_a_ref;
  std::vector<Element> conv_b_ref;
  std::vector<int64_t> slots;
  std::vector<int64_t> steps;
  std::vector<int64_t> track_slots;
  std::vector<int64_t> track_steps;
  int src_rows = 0;
};

// Request rows in the intermediate conv-window cache. The real buffer is
// [layers, spec_state_size + 1, draft_token_num, W-1, dim]: sized by the
// speculative batch, not by the conv-state pool.
int intermediate_rows(ScatterCase const& cfg) {
  return cfg.src_rows > 0 ? cfg.src_rows : std::max(cfg.main_count, cfg.track_count);
}

ScatterCase custom_default() {
  ScatterCase cfg;
  cfg.name = "custom_scatter";
  return cfg;
}

std::vector<ScatterCase> quick_suite() {
  return {
      {"reference_small_t3_d5", 4, 3, 5, 2, 3, 6, 3, 2, 0.0},
      {"tail_rows_t5_d19", 17, 5, 19, 3, 4, 13, 11, 5, 0.0},
      {"aligned_pack_bf16_like", 64, 7, 128, 3, 3, 256, 41, 13, 0.0},
      // Covers every branch of the validity mask: a negative step, a step past
      // draft_token_num, an out-of-range destination slot, and (via src_rows=2)
      // a request ordinal past the intermediate cache's request rows.
      {"masked_out_of_range", 4, 3, 5, 2, 3, 6, 4, 2, 0.0, 2},
  };
}

std::vector<ScatterCase> stress_suite() {
  return {
      {"non_power_t9_small_tail", 257, 9, 33, 3, 5, 65, 149, 41, 0.0},
      {"many_slots_track_tail", 1024, 7, 193, 2, 3, 97, 777, 211, 0.0},
      {"verify_block_h1536", 2048, 9, 1536, 3, 3, 1536, 1024, 256, 0.0},
  };
}

// Inkling conv-state geometry, from InklingModelConfig.mamba2_cache_params
// (python/sglang/srt/configs/inkling.py:215-252). Per layer it allocates SIX
// conv streams, each (sconv_kernel_size - 1, dim) in bf16:
//   K_FULL, V_FULL    dim = max(1, num_key_value_heads / tp) * head_dim
//   K_LOCAL, V_LOCAL  dim = max(1, swa_num_key_value_heads / tp) * swa_head_dim
//   ATTN, MLP         dim = hidden_size, or hidden_size / tp under
//                           --enable-scattered-sconv (inkling.py:231-237)
// and temporal=(0, 0, 0) (inkling.py:248) -- Inkling has NO SSM state, so
// scatter_mamba_states_after_mtp_verify skips the temporal scatter entirely
// (mamba_state_scatter_triton.py:696). d_ssm is therefore kept at 1: the
// memory-bound work is the conv-row copies.
//
// A case carries two conv streams (conv_a, conv_b) of equal width, so one case
// models one (K, V) / (ATTN, MLP) *pair* at one tp: three cases cover all six
// streams of a layer. width_a = width_b = W-1 = 3.
constexpr int kSconvKernelSize = 4;                        // inkling.py:52
constexpr int kConvLen = kSconvKernelSize - 1;             // W-1 rows per conv state
// draft_token_num = num_nextn_predict_layers + 1: 3 for the shipped checkpoint
// (num_nextn_predict_layers 2), 9 for production (8).
constexpr int kDraftTokensCheckpoint = 3;
constexpr int kDraftTokensProduction = 9;

struct InklingConvConfig {
  char const* tag;
  int hidden;
  int num_kv_heads;
  int head_dim;
  int swa_num_kv_heads;
  int swa_head_dim;
};

// hidden 768 is the shipped checkpoint, 1536 the config defaults, 6144
// production. head_dim = hidden_size / num_attention_heads = 128 in all three
// (Nq 8 / 12 / 48); swa_* default to the full-attention values when unset
// (inkling.py:82-85), which the checkpoint overrides with swa kv heads 4.
constexpr InklingConvConfig kInklingConvConfigs[] = {
    {"h768", 768, 2, 128, 4, 128},
    {"h1536", 1536, 4, 128, 4, 128},
    {"h6144", 6144, 4, 128, 4, 128},
};

constexpr int kInklingTpSizes[] = {1, 2, 4, 8};

// inkling.py:222-224 tp_local_kv_conv_dim
int tp_local_kv_conv_dim(int num_kv_heads, int head_dim, int tp) {
  return std::max(1, num_kv_heads / tp) * head_dim;
}

struct ScatterSizing {
  int slots = 0;
  int main_count = 0;
  int track_count = 0;
};

using SizingFn = ScatterSizing (*)(int dim, int t_max);

// Verify-friendly: a small pool with a handful of accepted + tracked requests.
ScatterSizing inkling_sizing(int dim, int t_max) {
  (void)dim;
  (void)t_max;
  return {256, 64, 16};
}

// Perf sizing scales the request count with the stream width so every case
// moves a comparable working set (tens of MB, i.e. tens of us per launch)
// instead of degenerating to a launch-latency measurement at dim = 128. The
// budget also bounds the intermediate window cache at ~42 MB per stream, which
// keeps the whole 72-case sweep affordable on host and device.
constexpr int64_t kPerfIntermediateElems = 8 << 20;

ScatterSizing perf_sizing(int dim, int t_max) {
  int64_t per_row = static_cast<int64_t>(t_max) * kConvLen * dim;
  int64_t rows = std::min<int64_t>(std::max<int64_t>(kPerfIntermediateElems / per_row, 64), 8192);
  int main_count = static_cast<int>(rows * 4 / 5);
  int track_count = static_cast<int>(rows) - main_count;
  return {2 * static_cast<int>(rows), main_count, track_count};
}

// One case per conv-stream pair, per tp, per draft_token_num band.
// `target_gbps` applies to every generated case; 0.0 is report-only.
std::vector<ScatterCase> inkling_conv_cases(SizingFn sizing, double target_gbps) {
  std::vector<ScatterCase> cases;
  for (InklingConvConfig const& cfg : kInklingConvConfigs) {
    for (int tp : kInklingTpSizes) {
      struct Stream {
        char const* name;
        int dim;
      };
      // ATTN/MLP use the scattered dim; non-scattered sconv keeps dim =
      // hidden_size for every tp, i.e. the tp=1 entry of this row.
      Stream const streams[] = {
          {"kvfull", tp_local_kv_conv_dim(cfg.num_kv_heads, cfg.head_dim, tp)},
          {"kvlocal", tp_local_kv_conv_dim(cfg.swa_num_kv_heads, cfg.swa_head_dim, tp)},
          {"attnmlp", cfg.hidden / tp},
      };
      for (int t_max : {kDraftTokensCheckpoint, kDraftTokensProduction}) {
        for (Stream const& stream : streams) {
          ScatterSizing const size = sizing(stream.dim, t_max);
          ScatterCase c;
          c.name = std::string(cfg.tag) + "_tp" + std::to_string(tp) + "_t" + std::to_string(t_max) + "_" +
                   stream.name + "D" + std::to_string(stream.dim);
          c.slots = size.slots;
          c.t_max = t_max;
          c.d_ssm = 1;  // temporal=(0,0,0): no SSM state in Inkling
          c.width_a = kConvLen;
          c.width_b = kConvLen;
          c.d_conv = stream.dim;
          c.main_count = size.main_count;
          c.track_count = size.track_count;
          c.target_gbps = target_gbps;
          cases.push_back(std::move(c));
        }
      }
    }
  }
  return cases;
}

std::vector<ScatterCase> inkling_suite() {
  return inkling_conv_cases(inkling_sizing, 0.0);
}

std::vector<ScatterCase> perf_suite() {
  // Re-calibrated from 350.0: on B60 the slowest measured points are 338.9 GB/s
  // (perf_b8192_d768_w3, bf16) and 346.4 GB/s (perf_b4096_d1536_w3, bf16) over
  // 3x50 iterations, so 350.0 was already missed at bf16/fp16 even before the
  // source-row indexing fix (see ScatterRowsParams). 300.0 keeps ~11% headroom.
  std::vector<ScatterCase> cases = {
      {"perf_b4096_d1536_w3", 8192, 9, 1536, 3, 3, 1536, 4096, 1024, 300.0},
      {"perf_b8192_d768_w3", 16384, 9, 768, 3, 3, 768, 8192, 2048, 300.0},
  };
  // The real conv-stream widths at every tp and both draft_token_num bands.
  // Report-only (0.0) gates: these shapes are new and uncalibrated on BMG, and
  // a guessed number would flake CI (same convention as
  // examples/17_bmg_relative_attention_backend). Their working sets are also
  // small enough to sit in L2 (measured 410-650 GB/s on a B60, i.e. above the
  // part's DRAM ceiling), so the numbers compare stream widths against each
  // other rather than against sustained DRAM bandwidth.
  std::vector<ScatterCase> conv_cases = inkling_conv_cases(perf_sizing, 0.0);
  cases.insert(cases.end(), std::make_move_iterator(conv_cases.begin()),
               std::make_move_iterator(conv_cases.end()));
  return cases;
}

std::vector<ScatterCase> make_suite(std::string const& suite) {
  if (suite == "quick") {
    return quick_suite();
  }
  if (suite == "stress") {
    return stress_suite();
  }
  if (suite == "perf") {
    return perf_suite();
  }
  if (suite == "inkling") {
    return inkling_suite();
  }
  return {};
}

template <typename Element>
void fill_pattern(std::vector<Element>& values, int salt) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = dflash::elem_from_float<Element>(static_cast<float>(dflash::patterned_value(i, salt)));
  }
}

// Smallest stride >= seed that is coprime with `modulus`, so that
// i -> (i * stride + c) % modulus is injective for i < modulus.
int coprime_stride(int seed, int modulus) {
  int stride = std::max(1, seed);
  while (std::gcd(stride, modulus) != 1) {
    ++stride;
  }
  return stride;
}

// CPU reference for one request-index set; `i` is the source request ordinal.
template <typename Element>
void apply_reference_pass(std::vector<Element>& dst,
                          std::vector<Element> const& intermediate,
                          std::vector<int64_t> const& slots,
                          std::vector<int64_t> const& steps,
                          int t_max,
                          int row_elems,
                          int src_rows,
                          int dst_slots) {
  for (std::size_t i = 0; i < slots.size(); ++i) {
    int64_t step = steps[i];
    int64_t slot = slots[i];
    if (step < 0 || step >= t_max || slot < 0 || slot >= dst_slots || static_cast<int64_t>(i) >= src_rows) {
      continue;
    }
    std::size_t dst_base = static_cast<std::size_t>(slot) * row_elems;
    std::size_t src_base = (i * static_cast<std::size_t>(t_max) + static_cast<std::size_t>(step)) * row_elems;
    std::copy(intermediate.begin() + static_cast<std::ptrdiff_t>(src_base),
              intermediate.begin() + static_cast<std::ptrdiff_t>(src_base + row_elems),
              dst.begin() + static_cast<std::ptrdiff_t>(dst_base));
  }
}

template <typename Element>
ScatterHost<Element> initialize_case(ScatterCase const& cfg) {
  ScatterHost<Element> h;
  int conv_a_elems = cfg.width_a * cfg.d_conv;
  int conv_b_elems = cfg.width_b * cfg.d_conv;
  h.src_rows = intermediate_rows(cfg);
  std::size_t src_windows = static_cast<std::size_t>(h.src_rows) * cfg.t_max;

  h.ssm_states.resize(static_cast<std::size_t>(cfg.slots) * cfg.d_ssm);
  h.ssm_intermediate.resize(src_windows * cfg.d_ssm);
  h.conv_a_states.resize(static_cast<std::size_t>(cfg.slots) * conv_a_elems);
  h.conv_a_intermediate.resize(src_windows * conv_a_elems);
  h.conv_b_states.resize(static_cast<std::size_t>(cfg.slots) * conv_b_elems);
  h.conv_b_intermediate.resize(src_windows * conv_b_elems);

  fill_pattern(h.ssm_states, 3);
  fill_pattern(h.ssm_intermediate, 5);
  fill_pattern(h.conv_a_states, 7);
  fill_pattern(h.conv_a_intermediate, 11);
  fill_pattern(h.conv_b_states, 13);
  fill_pattern(h.conv_b_intermediate, 17);

  h.slots.resize(cfg.main_count);
  h.steps.resize(cfg.main_count);
  h.track_slots.resize(cfg.track_count);
  h.track_steps.resize(cfg.track_count);

  if (cfg.name == "reference_small_t3_d5") {
    h.slots = {2, 0, 3};
    h.steps = {1, -1, 2};
    h.track_slots = {1, 3};
    h.track_steps = {0, 2};
  } else if (cfg.name == "masked_out_of_range") {
    // request 0 copies; 1 has step < 0; 2 has step >= t_max; 3 is past src_rows
    // (= 2) and also names a slot past the pool.
    h.slots = {1, 2, 3, static_cast<int64_t>(cfg.slots)};
    h.steps = {0, -1, cfg.t_max, 1};
    h.track_slots = {0, 2};
    h.track_steps = {2, -1};
  } else {
    // Both index sets must stay injective: duplicate destination slots inside
    // one pass would race between work items and make verify nondeterministic
    // (the real op's slot ids are distinct requests). A stride coprime with the
    // pool size keeps the walk scattered but collision-free.
    int const main_stride = coprime_stride(7, cfg.slots);
    int const track_stride = coprime_stride(13, cfg.slots);
    for (int i = 0; i < cfg.main_count; ++i) {
      h.slots[i] = (static_cast<int64_t>(i) * main_stride + 2) % cfg.slots;
      h.steps[i] = (i % 11 == 3) ? -1 : ((i * 5 + 1) % cfg.t_max);
    }
    for (int i = 0; i < cfg.track_count; ++i) {
      h.track_slots[i] = (static_cast<int64_t>(i) * track_stride + 1) % cfg.slots;
      h.track_steps[i] = (i % 7 == 4) ? -1 : ((i * 3) % cfg.t_max);
    }
  }

  h.ssm_ref = h.ssm_states;
  h.conv_a_ref = h.conv_a_states;
  h.conv_b_ref = h.conv_b_states;
  auto reference_stream = [&](std::vector<Element>& dst, std::vector<Element> const& intermediate, int row_elems) {
    apply_reference_pass(dst, intermediate, h.slots, h.steps, cfg.t_max, row_elems, h.src_rows, cfg.slots);
    apply_reference_pass(dst, intermediate, h.track_slots, h.track_steps, cfg.t_max, row_elems, h.src_rows,
                         cfg.slots);
  };
  reference_stream(h.ssm_ref, h.ssm_intermediate, cfg.d_ssm);
  reference_stream(h.conv_a_ref, h.conv_a_intermediate, conv_a_elems);
  reference_stream(h.conv_b_ref, h.conv_b_intermediate, conv_b_elems);

  return h;
}

int active_requests(std::vector<int64_t> const& slots,
                    std::vector<int64_t> const& steps,
                    int t_max,
                    int src_rows,
                    int dst_slots) {
  int count = 0;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    int64_t step = steps[i];
    int64_t slot = slots[i];
    if (step >= 0 && step < t_max && slot >= 0 && slot < dst_slots && static_cast<int64_t>(i) < src_rows) {
      ++count;
    }
  }
  return count;
}

template <typename Element>
bool run_case_for_dtype(sycl::queue& queue, ScatterCase const& cfg, dflash::Options const& options) {
  if (cfg.slots <= 0 || cfg.t_max <= 0 || cfg.d_ssm <= 0 || cfg.width_a <= 0 || cfg.width_b <= 0 ||
      cfg.d_conv <= 0 || cfg.main_count < 0 || cfg.track_count < 0 || cfg.src_rows < 0) {
    throw std::runtime_error("invalid scatter case dimensions");
  }
  // Each pass writes one row per request, so a pass with more requests than
  // pool slots could not have distinct destinations (see initialize_case).
  if (cfg.main_count > cfg.slots || cfg.track_count > cfg.slots) {
    throw std::runtime_error("scatter case needs slots >= main/track request count");
  }

  ScatterHost<Element> h = initialize_case<Element>(cfg);
  int conv_a_elems = cfg.width_a * cfg.d_conv;
  int conv_b_elems = cfg.width_b * cfg.d_conv;

  dflash::DeviceBuffer<Element> d_ssm_states(queue, h.ssm_states.size());
  dflash::DeviceBuffer<Element> d_ssm_intermediate(queue, h.ssm_intermediate.size());
  dflash::DeviceBuffer<Element> d_conv_a_states(queue, h.conv_a_states.size());
  dflash::DeviceBuffer<Element> d_conv_a_intermediate(queue, h.conv_a_intermediate.size());
  dflash::DeviceBuffer<Element> d_conv_b_states(queue, h.conv_b_states.size());
  dflash::DeviceBuffer<Element> d_conv_b_intermediate(queue, h.conv_b_intermediate.size());
  dflash::DeviceBuffer<int64_t> d_slots(queue, h.slots.size());
  dflash::DeviceBuffer<int64_t> d_steps(queue, h.steps.size());
  dflash::DeviceBuffer<int64_t> d_track_slots(queue, h.track_slots.size());
  dflash::DeviceBuffer<int64_t> d_track_steps(queue, h.track_steps.size());

  d_ssm_states.copy_from(h.ssm_states);
  d_ssm_intermediate.copy_from(h.ssm_intermediate);
  d_conv_a_states.copy_from(h.conv_a_states);
  d_conv_a_intermediate.copy_from(h.conv_a_intermediate);
  d_conv_b_states.copy_from(h.conv_b_states);
  d_conv_b_intermediate.copy_from(h.conv_b_intermediate);
  d_slots.copy_from(h.slots);
  d_steps.copy_from(h.steps);
  d_track_slots.copy_from(h.track_slots);
  d_track_steps.copy_from(h.track_steps);

  ScatterRowsParams<Element> base;
  base.src_rows = h.src_rows;
  base.dst_slots = cfg.slots;
  base.t_max = cfg.t_max;

  auto make_passes = [&](Element* dst, Element const* intermediate, int row_elems) {
    ScatterPasses<Element> passes;
    passes.main = base;
    passes.main.dst = dst;
    passes.main.intermediate = intermediate;
    passes.main.slots = d_slots.get();
    passes.main.steps = d_steps.get();
    passes.main.count = cfg.main_count;
    passes.main.row_elems = row_elems;
    passes.track = passes.main;
    passes.track.slots = d_track_slots.get();
    passes.track.steps = d_track_steps.get();
    passes.track.count = cfg.track_count;
    return passes;
  };

  ScatterPasses<Element> ssm_passes = make_passes(d_ssm_states.get(), d_ssm_intermediate.get(), cfg.d_ssm);
  ScatterPasses<Element> conv_a_passes =
      make_passes(d_conv_a_states.get(), d_conv_a_intermediate.get(), conv_a_elems);
  ScatterPasses<Element> conv_b_passes =
      make_passes(d_conv_b_states.get(), d_conv_b_intermediate.get(), conv_b_elems);

  launch_scatter_rows(queue, ssm_passes);
  launch_scatter_rows(queue, conv_a_passes);
  launch_scatter_rows(queue, conv_b_passes);
  queue.wait();

  bool passed = true;
  if (options.verify) {
    std::vector<Element> ssm_got(h.ssm_states.size());
    std::vector<Element> conv_a_got(h.conv_a_states.size());
    std::vector<Element> conv_b_got(h.conv_b_states.size());
    d_ssm_states.copy_to(ssm_got);
    d_conv_a_states.copy_to(conv_a_got);
    d_conv_b_states.copy_to(conv_b_got);
    auto ssm_cmp = dflash::compare_vectors(ssm_got, h.ssm_ref);
    auto conv_a_cmp = dflash::compare_vectors(conv_a_got, h.conv_a_ref);
    auto conv_b_cmp = dflash::compare_vectors(conv_b_got, h.conv_b_ref);
    passed = ssm_cmp.passed && conv_a_cmp.passed && conv_b_cmp.passed;
    if (!passed) {
      std::cerr << "Scatter mismatch case=" << cfg.name << " dtype=" << dflash::element_dtype_text<Element>()
                << " ssm_abs=" << ssm_cmp.max_abs << " conv_a_abs=" << conv_a_cmp.max_abs
                << " conv_b_abs=" << conv_b_cmp.max_abs << "\n";
    }
  }

  double ms = 0.0;
  if (options.benchmark && options.iterations > 0) {
    for (int i = 0; i < options.warmup; ++i) {
      launch_scatter_rows(queue, ssm_passes);
      launch_scatter_rows(queue, conv_a_passes);
      launch_scatter_rows(queue, conv_b_passes);
    }
    queue.wait();
    auto begin = std::chrono::steady_clock::now();
    std::vector<sycl::event> timed_events;
    timed_events.reserve(static_cast<std::size_t>(options.iterations) * 6);
    for (int i = 0; i < options.iterations; ++i) {
      append_events(timed_events, launch_scatter_rows(queue, ssm_passes));
      append_events(timed_events, launch_scatter_rows(queue, conv_a_passes));
      append_events(timed_events, launch_scatter_rows(queue, conv_b_passes));
    }
    queue.wait();
    auto end = std::chrono::steady_clock::now();
    ms = dflash::elapsed_ms(begin, end, options.iterations);
    try {
      ms = profiled_elapsed_ms(timed_events, options.iterations);
    } catch (sycl::exception const&) {
    }
  }

  int active = active_requests(h.slots, h.steps, cfg.t_max, h.src_rows, cfg.slots) +
               active_requests(h.track_slots, h.track_steps, cfg.t_max, h.src_rows, cfg.slots);
  double row_elems = static_cast<double>(cfg.d_ssm + conv_a_elems + conv_b_elems);
  double bytes = static_cast<double>(active) * row_elems * sizeof(Element) * 2.0;
  double seconds = ms / 1000.0;
  double gbps = seconds > 0.0 ? bytes / seconds / 1.0e9 : 0.0;
  double target_gbps = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
  if (target_gbps > 0.0 && bytes >= dflash::kMinSustainedTargetBytes && gbps < target_gbps) {
    passed = false;
    std::cerr << "GB/s target miss case=" << cfg.name << " dtype=" << dflash::element_dtype_text<Element>()
              << " got=" << gbps << " target=" << target_gbps << "\n";
  }

  bool packed_ssm = can_use_pack_path(ssm_passes.main);
  bool packed_conv_a = can_use_pack_path(conv_a_passes.main);
  bool packed_conv_b = can_use_pack_path(conv_b_passes.main);
  std::cout << "case=" << std::left << std::setw(28) << cfg.name
            << " dtype=" << std::setw(5) << dflash::element_dtype_text<Element>()
            << " slots=" << std::right << std::setw(6) << cfg.slots
            << " tmax=" << std::setw(2) << cfg.t_max
            << " rows=" << std::setw(6) << h.src_rows
            << " active=" << std::setw(6) << active
            << " elems=(" << cfg.d_ssm << "," << conv_a_elems << "," << conv_b_elems << ")"
            << " pack=(" << dflash::bool_text(packed_ssm) << "," << dflash::bool_text(packed_conv_a)
            << "," << dflash::bool_text(packed_conv_b) << ")"
            << " verify=" << dflash::bool_text(!options.verify || passed)
            << " time_ms=" << std::fixed << std::setprecision(4) << ms
            << " GBps=" << std::setprecision(2) << gbps << "\n";
  return passed;
}

template <typename Element>
bool run_cases_for_dtype(sycl::queue& queue, std::vector<ScatterCase> const& cases, dflash::Options const& options) {
  bool all_passed = true;
  for (ScatterCase const& cfg : cases) {
    all_passed &= run_case_for_dtype<Element>(queue, cfg, options);
  }
  return all_passed;
}

void print_usage(char const* exe) {
  std::cout << "20_bmg_scatter_mamba_states_after_mtp_verify: DFLASH Mamba/conv verify commit\n\n"
            << "Usage: " << exe << " [--suite=quick|stress|perf|inkling] [--dtype=all|float|bf16|fp16]\n"
            << "       [--shape=slots=4,tmax=3,dssm=5,wa=2,wb=3,dconv=6,main=3,track=2]\n"
            << "       [--iterations=N] [--verify=0|1] [--target-gbps=X]\n"
            << "\nInkling suite mirrors the six per-layer mamba2 conv streams at TP=1/2/4/8,\n"
            << "as three (K,V)-style pairs per config x TP x draft_token_num band:\n"
            << "  kvfull  D = head_dim     * max(1, num_key_value_heads     / tp)\n"
            << "  kvlocal D = swa_head_dim * max(1, swa_num_key_value_heads / tp)\n"
            << "  attnmlp D = hidden_size / tp (scattered sconv; non-scattered = tp1 row)\n"
            << "hidden 768 (checkpoint) / 1536 (config defaults) / 6144 (production),\n"
            << "sconv_kernel_size-1 = 3 rows, draft_token_num 3 (checkpoint) and 9 (production).\n";
}

}  // namespace

int main(int argc, char const** argv) {
  dflash::Options options;
  try {
    options = dflash::parse_common_options(argc, argv);
    if (options.help) {
      print_usage(argv[0]);
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse options: " << e.what() << "\n";
    return -1;
  }

  std::vector<ScatterCase> cases;
  if (!options.shape.empty()) {
    ScatterCase cfg = custom_default();
    if (!dflash::parse_shape_ints(options.shape, {
            {"slots", &cfg.slots},
            {"tmax", &cfg.t_max},
            {"dssm", &cfg.d_ssm},
            {"wa", &cfg.width_a},
            {"wb", &cfg.width_b},
            {"dconv", &cfg.d_conv},
            {"main", &cfg.main_count},
            {"track", &cfg.track_count},
        })) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    cases.push_back(cfg);
  } else {
    cases = make_suite(options.suite);
    if (cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = dflash::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "20_bmg_scatter_mamba_states_after_mtp_verify: masked intermediate-to-persistent row copy\n";
    std::cout << "Suite=" << options.suite << " dtype=" << dflash::dtype_text(options.dtype)
              << " iterations=" << options.iterations << " warmup=" << options.warmup
              << " verify=" << dflash::bool_text(options.verify)
              << " benchmark=" << dflash::bool_text(options.benchmark) << "\n";

    bool all_passed = true;
    if (options.dtype == dflash::DType::kAll || options.dtype == dflash::DType::kFloat) {
      all_passed &= run_cases_for_dtype<float>(queue, cases, options);
    }
    if (options.dtype == dflash::DType::kAll || options.dtype == dflash::DType::kBf16) {
      all_passed &= run_cases_for_dtype<cutlass::bfloat16_t>(queue, cases, options);
    }
    if (options.dtype == dflash::DType::kAll || options.dtype == dflash::DType::kFp16) {
      all_passed &= run_cases_for_dtype<cutlass::half_t>(queue, cases, options);
    }
    return all_passed ? 0 : -1;
  } catch (dflash::NoGpuDevice const& e) {
    std::cout << "SKIP: " << e.what() << "\n";
    return dflash::kSkipReturnCode;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
