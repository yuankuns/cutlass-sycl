/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "20_bmg_dflash_common.hpp"

namespace dflash = cutlass::examples::bmg_dflash;

namespace {

template <typename Element>
struct ScatterRowsParams {
  Element* dst = nullptr;
  Element const* intermediate = nullptr;
  int64_t const* slots = nullptr;
  int64_t const* steps = nullptr;
  int64_t const* track_slots = nullptr;
  int64_t const* track_steps = nullptr;
  int main_count = 0;
  int track_count = 0;
  int t_max = 0;
  int row_elems = 0;
};

template <typename Element>
CUTLASS_DEVICE
void scatter_request(ScatterRowsParams<Element> const& params,
                     int request,
                     int64_t& slot,
                     int64_t& step) {
  if (request < params.main_count) {
    slot = params.slots[request];
    step = params.steps[request];
  } else {
    int t = request - params.main_count;
    slot = params.track_slots[t];
    step = params.track_steps[t];
  }
}

template <typename Element>
class ScatterRowsScalarKernel {
 public:
  explicit ScatterRowsScalarKernel(ScatterRowsParams<Element> params) : params_(params) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t total_requests = static_cast<int64_t>(params_.main_count) + params_.track_count;
    int64_t total_lanes = total_requests * params_.row_elems;
    int64_t lane = static_cast<int64_t>(item.get_global_id(0));
    if (lane >= total_lanes) {
      return;
    }
    int request = static_cast<int>(lane / params_.row_elems);
    int elem = static_cast<int>(lane - static_cast<int64_t>(request) * params_.row_elems);

    int64_t slot = 0;
    int64_t step = -1;
    scatter_request(params_, request, slot, step);
    if (step < 0) {
      return;
    }

    int64_t dst_base = slot * params_.row_elems;
    int64_t src_base = (slot * params_.t_max + step) * params_.row_elems;
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
    int64_t total_requests = static_cast<int64_t>(params_.main_count) + params_.track_count;
    int64_t total_lanes = total_requests * packs_per_row;
    int64_t lane = static_cast<int64_t>(item.get_global_id(0));
    if (lane >= total_lanes) {
      return;
    }
    int request = static_cast<int>(lane / packs_per_row);
    int pack = static_cast<int>(lane - static_cast<int64_t>(request) * packs_per_row);

    int64_t slot = 0;
    int64_t step = -1;
    scatter_request(params_, request, slot, step);
    if (step < 0) {
      return;
    }

    int64_t dst_base = slot * params_.row_elems;
    int64_t src_base = (slot * params_.t_max + step) * params_.row_elems;
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
  int64_t requests = static_cast<int64_t>(params.main_count) + params.track_count;
  if (requests <= 0 || params.row_elems <= 0) {
    return {};
  }
  if (can_use_pack_path(params)) {
    constexpr int kPackElems = dflash::kCopyPackBytes / static_cast<int>(sizeof(Element));
    int64_t lanes = requests * (params.row_elems / kPackElems);
    return submit_1d(queue, lanes, ScatterRowsPackKernel<Element>(params));
  }
  return submit_1d(queue, requests * params.row_elems, ScatterRowsScalarKernel<Element>(params));
}

struct ScatterLaunchEvents {
  sycl::event events[2];
  int count = 0;
};

template <typename Element>
ScatterLaunchEvents launch_scatter_rows(sycl::queue& queue, ScatterRowsParams<Element> const& params) {
  ScatterLaunchEvents launched;
  ScatterRowsParams<Element> main_params = params;
  main_params.track_count = 0;
  main_params.track_slots = nullptr;
  main_params.track_steps = nullptr;
  if (main_params.main_count > 0) {
    launched.events[launched.count++] = launch_scatter_rows_pass(queue, main_params);
  }

  if (params.track_count > 0) {
    ScatterRowsParams<Element> track_params = params;
    track_params.slots = params.track_slots;
    track_params.steps = params.track_steps;
    track_params.main_count = params.track_count;
    track_params.track_count = 0;
    track_params.track_slots = nullptr;
    track_params.track_steps = nullptr;
    launched.events[launched.count++] = launch_scatter_rows_pass(queue, track_params);
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
};

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
  };
}

std::vector<ScatterCase> stress_suite() {
  return {
      {"non_power_t9_small_tail", 257, 9, 33, 3, 5, 65, 149, 41, 0.0},
      {"many_slots_track_tail", 1024, 7, 193, 2, 3, 97, 777, 211, 0.0},
      {"verify_block_h1536", 2048, 9, 1536, 3, 3, 1536, 1024, 256, 0.0},
  };
}

std::vector<ScatterCase> perf_suite() {
  return {
      {"perf_b4096_d1536_w3", 8192, 9, 1536, 3, 3, 1536, 4096, 1024, 350.0},
      {"perf_b8192_d768_w3", 16384, 9, 768, 3, 3, 768, 8192, 2048, 350.0},
  };
}

// Inkling suite: DFLASH target-verify commit copies rows of the per-layer
// mamba2 conv cache. sconv_kernel_size-1 is 3, so each conv row spans
// (K-1) * D bytes. Row-D follows InklingModelConfig.mamba2_cache_params:
//   kv_conv  = head_dim * max(1, num_kv_heads/tp)  → {512,256,128,128}
//              for TP=1/2/4/8 (head_dim=128, num_kv_heads=4).
//   stream   = hidden_size (non-scattered) or hidden_size/tp (scattered).
// Inkling has no temporal SSM cache (temporal=(0,0,0)), so d_ssm is kept
// small — the memory-bound path is the two conv-row copies. Slots reflect
// verify pool bands; draft_token_num=9 sets t_max; main/track counts
// approximate the target-verify + track-slot pass ratio.
//
// Each case models one conv-layer type at a given TP: both conv_a and
// conv_b share d_conv, matching the per-layer conv-state layout. Cases
// are named _<config>_tp<n>_<layer>D<row-dim>.
std::vector<ScatterCase> inkling_suite() {
  return {
      // hidden_size=1536 (config defaults) kv_conv layers.
      {"cfg_h1536_tp1_kvD512",   512, 9, 1, 3, 3,  512, 128, 16, 0.0},
      {"cfg_h1536_tp2_kvD256",   512, 9, 1, 3, 3,  256, 128, 16, 0.0},
      {"cfg_h1536_tp4_kvD128",   512, 9, 1, 3, 3,  128, 128, 16, 0.0},
      {"cfg_h1536_tp8_kvD128",   512, 9, 1, 3, 3,  128, 128, 16, 0.0},
      // hidden_size=1536 stream-conv layers (scattered sconv / non-scattered).
      {"cfg_h1536_tp1_streamD1536", 512, 9, 1, 3, 3, 1536, 128, 16, 0.0},
      {"cfg_h1536_tp2_streamD768",  512, 9, 1, 3, 3,  768, 128, 16, 0.0},
      {"cfg_h1536_tp4_streamD384",  512, 9, 1, 3, 3,  384, 128, 16, 0.0},
      {"cfg_h1536_tp8_streamD192",  512, 9, 1, 3, 3,  192, 128, 16, 0.0},
      // hidden_size=6144 (production) kv_conv layers.
      {"prod_h6144_tp1_kvD512",  256, 9, 1, 3, 3,  512,  96, 12, 0.0},
      {"prod_h6144_tp2_kvD256",  256, 9, 1, 3, 3,  256,  96, 12, 0.0},
      {"prod_h6144_tp4_kvD128",  256, 9, 1, 3, 3,  128,  96, 12, 0.0},
      {"prod_h6144_tp8_kvD128",  256, 9, 1, 3, 3,  128,  96, 12, 0.0},
      // hidden_size=6144 stream-conv layers.
      {"prod_h6144_tp1_streamD6144", 256, 9, 1, 3, 3, 6144, 96, 12, 0.0},
      {"prod_h6144_tp2_streamD3072", 256, 9, 1, 3, 3, 3072, 96, 12, 0.0},
      {"prod_h6144_tp4_streamD1536", 256, 9, 1, 3, 3, 1536, 96, 12, 0.0},
      {"prod_h6144_tp8_streamD768",  256, 9, 1, 3, 3,  768, 96, 12, 0.0},
  };
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

template <typename Element>
void apply_reference_pass(std::vector<Element>& dst,
                          std::vector<Element> const& intermediate,
                          std::vector<int64_t> const& slots,
                          std::vector<int64_t> const& steps,
                          int t_max,
                          int row_elems) {
  for (std::size_t i = 0; i < slots.size(); ++i) {
    int64_t step = steps[i];
    if (step < 0) {
      continue;
    }
    int64_t slot = slots[i];
    std::size_t dst_base = static_cast<std::size_t>(slot) * row_elems;
    std::size_t src_base = (static_cast<std::size_t>(slot) * t_max + static_cast<std::size_t>(step)) * row_elems;
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

  h.ssm_states.resize(static_cast<std::size_t>(cfg.slots) * cfg.d_ssm);
  h.ssm_intermediate.resize(static_cast<std::size_t>(cfg.slots) * cfg.t_max * cfg.d_ssm);
  h.conv_a_states.resize(static_cast<std::size_t>(cfg.slots) * conv_a_elems);
  h.conv_a_intermediate.resize(static_cast<std::size_t>(cfg.slots) * cfg.t_max * conv_a_elems);
  h.conv_b_states.resize(static_cast<std::size_t>(cfg.slots) * conv_b_elems);
  h.conv_b_intermediate.resize(static_cast<std::size_t>(cfg.slots) * cfg.t_max * conv_b_elems);

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
  } else {
    for (int i = 0; i < cfg.main_count; ++i) {
      h.slots[i] = (i * 7 + 2) % cfg.slots;
      h.steps[i] = (i % 11 == 3) ? -1 : ((i * 5 + 1) % cfg.t_max);
    }
    for (int i = 0; i < cfg.track_count; ++i) {
      h.track_slots[i] = (i * 13 + 1) % cfg.slots;
      h.track_steps[i] = (i % 7 == 4) ? -1 : ((i * 3) % cfg.t_max);
    }
  }

  h.ssm_ref = h.ssm_states;
  h.conv_a_ref = h.conv_a_states;
  h.conv_b_ref = h.conv_b_states;
  apply_reference_pass(h.ssm_ref, h.ssm_intermediate, h.slots, h.steps, cfg.t_max, cfg.d_ssm);
  apply_reference_pass(h.conv_a_ref, h.conv_a_intermediate, h.slots, h.steps, cfg.t_max, conv_a_elems);
  apply_reference_pass(h.conv_b_ref, h.conv_b_intermediate, h.slots, h.steps, cfg.t_max, conv_b_elems);
  apply_reference_pass(h.ssm_ref, h.ssm_intermediate, h.track_slots, h.track_steps, cfg.t_max, cfg.d_ssm);
  apply_reference_pass(h.conv_a_ref, h.conv_a_intermediate, h.track_slots, h.track_steps, cfg.t_max, conv_a_elems);
  apply_reference_pass(h.conv_b_ref, h.conv_b_intermediate, h.track_slots, h.track_steps, cfg.t_max, conv_b_elems);

  return h;
}

int active_requests(std::vector<int64_t> const& steps) {
  int count = 0;
  for (int64_t step : steps) {
    if (step >= 0) {
      ++count;
    }
  }
  return count;
}

template <typename Element>
bool run_case_for_dtype(sycl::queue& queue, ScatterCase const& cfg, dflash::Options const& options) {
  if (cfg.slots <= 0 || cfg.t_max <= 0 || cfg.d_ssm <= 0 || cfg.width_a <= 0 || cfg.width_b <= 0 ||
      cfg.d_conv <= 0 || cfg.main_count < 0 || cfg.track_count < 0) {
    throw std::runtime_error("invalid scatter case dimensions");
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

  ScatterRowsParams<Element> ssm_params;
  ssm_params.dst = d_ssm_states.get();
  ssm_params.intermediate = d_ssm_intermediate.get();
  ssm_params.slots = d_slots.get();
  ssm_params.steps = d_steps.get();
  ssm_params.track_slots = d_track_slots.get();
  ssm_params.track_steps = d_track_steps.get();
  ssm_params.main_count = cfg.main_count;
  ssm_params.track_count = cfg.track_count;
  ssm_params.t_max = cfg.t_max;
  ssm_params.row_elems = cfg.d_ssm;

  ScatterRowsParams<Element> conv_a_params = ssm_params;
  conv_a_params.dst = d_conv_a_states.get();
  conv_a_params.intermediate = d_conv_a_intermediate.get();
  conv_a_params.row_elems = conv_a_elems;

  ScatterRowsParams<Element> conv_b_params = ssm_params;
  conv_b_params.dst = d_conv_b_states.get();
  conv_b_params.intermediate = d_conv_b_intermediate.get();
  conv_b_params.row_elems = conv_b_elems;

  launch_scatter_rows(queue, ssm_params);
  launch_scatter_rows(queue, conv_a_params);
  launch_scatter_rows(queue, conv_b_params);
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
      launch_scatter_rows(queue, ssm_params);
      launch_scatter_rows(queue, conv_a_params);
      launch_scatter_rows(queue, conv_b_params);
    }
    queue.wait();
    auto begin = std::chrono::steady_clock::now();
    std::vector<sycl::event> timed_events;
    timed_events.reserve(static_cast<std::size_t>(options.iterations) * 6);
    for (int i = 0; i < options.iterations; ++i) {
      append_events(timed_events, launch_scatter_rows(queue, ssm_params));
      append_events(timed_events, launch_scatter_rows(queue, conv_a_params));
      append_events(timed_events, launch_scatter_rows(queue, conv_b_params));
    }
    queue.wait();
    auto end = std::chrono::steady_clock::now();
    ms = dflash::elapsed_ms(begin, end, options.iterations);
    try {
      ms = profiled_elapsed_ms(timed_events, options.iterations);
    } catch (sycl::exception const&) {
    }
  }

  int active = active_requests(h.steps) + active_requests(h.track_steps);
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

  bool packed_ssm = can_use_pack_path(ssm_params);
  bool packed_conv_a = can_use_pack_path(conv_a_params);
  bool packed_conv_b = can_use_pack_path(conv_b_params);
  std::cout << "case=" << std::left << std::setw(28) << cfg.name
            << " dtype=" << std::setw(5) << dflash::element_dtype_text<Element>()
            << " slots=" << std::right << std::setw(6) << cfg.slots
            << " tmax=" << std::setw(2) << cfg.t_max
            << " active=" << std::setw(6) << active
            << " rows=(" << cfg.d_ssm << "," << conv_a_elems << "," << conv_b_elems << ")"
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
            << "\nInkling suite mirrors the per-layer mamba2 conv-cache widths at TP=1/2/4/8:\n"
            << "kv_conv D = head_dim*max(1,num_kv_heads/tp), stream D = hidden_size/tp\n"
            << "(hidden 1536 cfg-defaults / 6144 production, sconv_kernel_size-1 = 3).\n";
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
