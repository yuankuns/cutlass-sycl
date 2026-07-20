#include "16_bmg_moe_gate_gemv.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe = cutlass::examples::bmg_moe_gate;

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
    ptr = sycl::malloc_device<T>(std::max<std::size_t>(n, 1), q);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept {
    queue = other.queue;
    ptr = other.ptr;
    count = other.count;
    other.queue = nullptr;
    other.ptr = nullptr;
    other.count = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      queue = other.queue;
      ptr = other.ptr;
      count = other.count;
      other.queue = nullptr;
      other.ptr = nullptr;
      other.count = 0;
    }
    return *this;
  }

  ~DeviceBuffer() {
    reset();
  }

  void reset() {
    if (ptr != nullptr) {
      sycl::free(ptr, *queue);
    }
    ptr = nullptr;
    queue = nullptr;
    count = 0;
  }

  T* get() const {
    return ptr;
  }

  void copy_from(std::vector<T> const& host) {
    if (!host.empty()) {
      queue->memcpy(ptr, host.data(), sizeof(T) * host.size()).wait();
    }
  }

  void copy_to(std::vector<T>& host) const {
    if (!host.empty()) {
      queue->memcpy(host.data(), ptr, sizeof(T) * host.size()).wait();
    }
  }
};

struct Options {
  std::string suite = "quick";
  int64_t tokens = -1;
  int iterations = 100;
  int warmup = 10;
  int experts_per_workgroup = 0;
  int subgroup_size = 0;
  bool verify = true;
  bool benchmark = true;
  bool help = false;
};

bool parse_bool(std::string const& value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

Options parse_options(int argc, char const** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto eq = arg.find('=');
    std::string key = eq == std::string::npos ? arg : arg.substr(0, eq);
    std::string value = eq == std::string::npos ? "" : arg.substr(eq + 1);
    if (key == "--help" || key == "-h") {
      options.help = true;
    } else if (key == "--suite") {
      options.suite = value;
    } else if (key == "--tokens") {
      options.tokens = std::stoll(value);
    } else if (key == "--iterations") {
      options.iterations = std::stoi(value);
    } else if (key == "--warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "--experts-per-wg") {
      options.experts_per_workgroup = std::stoi(value);
    } else if (key == "--subgroup") {
      options.subgroup_size = std::stoi(value);
    } else if (key == "--verify") {
      options.verify = parse_bool(value);
    } else if (key == "--benchmark") {
      options.benchmark = parse_bool(value);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return options;
}

void print_usage(char const* name) {
  std::cout << "Usage: " << name << " [options]\n\n"
            << "Options:\n"
            << "  --suite=quick|full|perf      Verification and benchmark suite (default quick)\n"
            << "  --tokens=<int>               Run one custom token count instead of suite cases\n"
            << "  --iterations=<int>           Benchmark iterations (default 100)\n"
            << "  --warmup=<int>               Benchmark warmup launches (default 10)\n"
            << "  --experts-per-wg=0|1|2|4     Experts computed by one workgroup, 0 selects default\n"
            << "  --subgroup=0|32              Fused epilogue requires a 32-lane subgroup\n"
            << "  --verify=0|1                 Run correctness checks (default 1)\n"
            << "  --benchmark=0|1              Run timing checks (default 1)\n";
}

struct CaseConfig {
  std::string name;
  int64_t tokens = 0;
  bool ties = false;
};

struct HostInputs {
  std::vector<cutlass::bfloat16_t> x;
  std::vector<cutlass::bfloat16_t> weight;
  std::vector<float> bias;
  float global_scale = 1.25f;
  float route_scale = 8.0f;
};

struct ReferenceOutput {
  std::vector<float> routed_weights;
  std::vector<float> shared_weights;
  std::vector<int32_t> indices;
};

struct DeviceOutputs {
  std::vector<float> workspace;
  std::vector<float> routed_weights;
  std::vector<float> shared_weights;
  std::vector<int32_t> indices;
  std::vector<int32_t> packed;
  int32_t ticket = -1;
};

HostInputs make_inputs(CaseConfig const& cfg, uint32_t seed) {
  HostInputs inputs;
  inputs.x.resize(static_cast<std::size_t>(cfg.tokens) * moe::kGateHidden);
  inputs.weight.resize(static_cast<std::size_t>(moe::kGateLogitsPad) * moe::kGateHidden);
  inputs.bias.resize(moe::kGateRoutedExperts);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> x_dist(-0.05f, 0.05f);
  std::uniform_real_distribution<float> w_dist(-0.02f, 0.02f);
  std::uniform_real_distribution<float> bias_dist(-0.50f, 0.50f);

  if (cfg.ties) {
    std::fill(inputs.x.begin(), inputs.x.end(), cutlass::bfloat16_t(0.0f));
    std::fill(inputs.weight.begin(), inputs.weight.end(), cutlass::bfloat16_t(0.0f));
    std::fill(inputs.bias.begin(), inputs.bias.end(), 0.0f);
    return inputs;
  }

  for (auto& v : inputs.x) {
    v = cutlass::bfloat16_t(x_dist(gen));
  }
  for (int e = 0; e < moe::kGateLogitsPad; ++e) {
    for (int k = 0; k < moe::kGateHidden; ++k) {
      float value = e < moe::kGateTotalExperts ? w_dist(gen) : 0.0f;
      inputs.weight[static_cast<std::size_t>(e) * moe::kGateHidden + k] = cutlass::bfloat16_t(value);
    }
  }
  for (auto& v : inputs.bias) {
    v = bias_dist(gen);
  }

  if (cfg.tokens > 0) {
    for (int k = 0; k < moe::kGateHidden; ++k) {
      inputs.x[static_cast<std::size_t>(cfg.tokens - 1) * moe::kGateHidden + k] =
          cutlass::bfloat16_t(((k % 23) - 11) * 0.0025f);
    }
  }

  return inputs;
}

std::vector<float> reference_gemv(CaseConfig const& cfg, HostInputs const& inputs) {
  std::vector<float> ref(static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad, -777.0f);
  for (int64_t token = 0; token < cfg.tokens; ++token) {
    for (int expert = 0; expert < moe::kGateTotalExperts; ++expert) {
      float acc = 0.0f;
      for (int k = 0; k < moe::kGateHidden; ++k) {
        float xv =
            moe::gate_bf16_to_float(inputs.x[static_cast<std::size_t>(token) * moe::kGateHidden + k]);
        float wv =
            moe::gate_bf16_to_float(inputs.weight[static_cast<std::size_t>(expert) * moe::kGateHidden + k]);
        acc = std::fma(xv, wv, acc);
      }
      ref[static_cast<std::size_t>(token) * moe::kGateLogitsPad + expert] = acc;
    }
  }
  return ref;
}

ReferenceOutput reference_gate_from_logits(
    CaseConfig const& cfg,
    HostInputs const& inputs,
    std::vector<float> const& logits) {
  ReferenceOutput ref;
  ref.routed_weights.resize(static_cast<std::size_t>(cfg.tokens) * moe::kTopK);
  ref.shared_weights.resize(static_cast<std::size_t>(cfg.tokens) * moe::kGateSharedExperts);
  ref.indices.resize(static_cast<std::size_t>(cfg.tokens) * moe::kTopK);

  std::vector<float> scores(moe::kGateRoutedExperts);
  for (int64_t row = 0; row < cfg.tokens; ++row) {
    int64_t row_base = row * moe::kGateLogitsPad;
    for (int expert = 0; expert < moe::kGateRoutedExperts; ++expert) {
      scores[expert] = moe::detail::sigmoid_host(logits[row_base + expert]) + inputs.bias[expert];
    }

    int32_t selected[moe::kTopK];
    float active[moe::kTopK + moe::kGateSharedExperts];
    for (int k = 0; k < moe::kTopK; ++k) {
      float best_score = -std::numeric_limits<float>::max();
      int best_idx = std::numeric_limits<int>::max();
      for (int expert = 0; expert < moe::kGateRoutedExperts; ++expert) {
        if (moe::detail::score_better(scores[expert], expert, best_score, best_idx)) {
          best_score = scores[expert];
          best_idx = expert;
        }
      }
      selected[k] = best_idx;
      scores[best_idx] = -std::numeric_limits<float>::max();
      active[k] = moe::detail::sigmoid_host(logits[row_base + best_idx]);
    }
    active[moe::kTopK] = moe::detail::sigmoid_host(logits[row_base + moe::kGateRoutedExperts]);
    active[moe::kTopK + 1] = moe::detail::sigmoid_host(logits[row_base + moe::kGateRoutedExperts + 1]);

    float sum = 0.0f;
    for (float value : active) {
      sum += value;
    }
    float scale = inputs.route_scale * inputs.global_scale / sum;

    for (int k = 0; k < moe::kTopK; ++k) {
      ref.indices[row * moe::kTopK + k] = selected[k];
      ref.routed_weights[row * moe::kTopK + k] = active[k] * scale;
    }
    for (int s = 0; s < moe::kGateSharedExperts; ++s) {
      ref.shared_weights[row * moe::kGateSharedExperts + s] = active[moe::kTopK + s] * scale;
    }
  }
  return ref;
}

DeviceOutputs run_fused_kernel(
    sycl::queue& queue,
    CaseConfig const& cfg,
    HostInputs const& inputs,
    bool packed,
    int experts_per_workgroup,
    int subgroup_size,
    int replays) {
  DeviceBuffer<cutlass::bfloat16_t> d_x(queue, inputs.x.size());
  DeviceBuffer<cutlass::bfloat16_t> d_weight(queue, inputs.weight.size());
  DeviceBuffer<float> d_bias(queue, inputs.bias.size());
  DeviceBuffer<float> d_global_scale(queue, 1);
  DeviceBuffer<float> d_workspace(queue, static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad);
  DeviceBuffer<float> d_routed(queue, static_cast<std::size_t>(cfg.tokens) * moe::kTopK);
  DeviceBuffer<float> d_shared(queue, static_cast<std::size_t>(cfg.tokens) * moe::kGateSharedExperts);
  DeviceBuffer<int32_t> d_indices(queue, static_cast<std::size_t>(cfg.tokens) * moe::kTopK);
  DeviceBuffer<int32_t> d_packed(queue, static_cast<std::size_t>(cfg.tokens) * moe::kTopK);
  DeviceBuffer<int32_t> d_ticket(queue, 1);

  d_x.copy_from(inputs.x);
  d_weight.copy_from(inputs.weight);
  d_bias.copy_from(inputs.bias);
  std::vector<float> global_scale = {inputs.global_scale};
  d_global_scale.copy_from(global_scale);
  std::vector<float> workspace(static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad, -777.0f);
  std::vector<float> routed(static_cast<std::size_t>(cfg.tokens) * moe::kTopK, -777.0f);
  std::vector<float> shared(static_cast<std::size_t>(cfg.tokens) * moe::kGateSharedExperts, -777.0f);
  std::vector<int32_t> indices(static_cast<std::size_t>(cfg.tokens) * moe::kTopK, -777);
  std::vector<int32_t> packed_out(static_cast<std::size_t>(cfg.tokens) * moe::kTopK, -777);
  std::vector<int32_t> ticket = {0};
  d_workspace.copy_from(workspace);
  d_routed.copy_from(routed);
  d_shared.copy_from(shared);
  d_indices.copy_from(indices);
  d_packed.copy_from(packed_out);
  d_ticket.copy_from(ticket);

  moe::GateGemvParams params;
  params.x = d_x.get();
  params.weight = d_weight.get();
  params.logits = d_workspace.get();
  params.bias = d_bias.get();
  params.global_scale = d_global_scale.get();
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.ticket = d_ticket.get();
  params.tokens = cfg.tokens;
  params.route_scale = inputs.route_scale;

  if (cfg.tokens > 0) {
    for (int replay = 0; replay < replays; ++replay) {
      moe::launch_gate_gemv_fused(queue, params, packed, experts_per_workgroup, subgroup_size);
    }
    queue.wait();
  }

  DeviceOutputs outputs;
  outputs.workspace.resize(workspace.size());
  outputs.routed_weights.resize(routed.size());
  outputs.shared_weights.resize(shared.size());
  outputs.indices.resize(indices.size());
  outputs.packed.resize(packed_out.size());
  d_workspace.copy_to(outputs.workspace);
  d_shared.copy_to(outputs.shared_weights);
  if (packed) {
    d_packed.copy_to(outputs.packed);
  } else {
    d_routed.copy_to(outputs.routed_weights);
    d_indices.copy_to(outputs.indices);
  }
  d_ticket.copy_to(ticket);
  outputs.ticket = ticket[0];
  return outputs;
}

bool close_enough(float got, float expected, float atol, float rtol) {
  float diff = std::abs(got - expected);
  return diff <= atol + rtol * std::abs(expected);
}

bool verify_case(
    sycl::queue& queue,
    CaseConfig const& cfg,
    bool packed,
    int experts_per_workgroup,
    int subgroup_size) {
  HostInputs inputs = make_inputs(cfg, 20260720u + static_cast<uint32_t>(cfg.tokens));
  std::vector<float> logits_ref = reference_gemv(cfg, inputs);
  DeviceOutputs got = run_fused_kernel(queue, cfg, inputs, packed, experts_per_workgroup, subgroup_size, 3);
  ReferenceOutput gate_ref = reference_gate_from_logits(cfg, inputs, got.workspace);

  int failures = 0;
  double max_logit_abs = 0.0;
  double max_logit_rel = 0.0;
  for (int64_t token = 0; token < cfg.tokens; ++token) {
    for (int expert = 0; expert < moe::kGateLogitsPad; ++expert) {
      std::size_t idx = static_cast<std::size_t>(token) * moe::kGateLogitsPad + expert;
      float actual = got.workspace[idx];
      float expected = logits_ref[idx];
      bool ok = expert < moe::kGateTotalExperts ? close_enough(actual, expected, 2.5e-4f, 2.0e-3f)
                                                : actual == -777.0f;
      double abs_err = std::abs(static_cast<double>(actual) - static_cast<double>(expected));
      double rel_err = abs_err / std::max(1.0e-12, std::abs(static_cast<double>(expected)));
      max_logit_abs = std::max(max_logit_abs, abs_err);
      max_logit_rel = std::max(max_logit_rel, rel_err);
      if (!ok && failures++ < 8) {
        std::cerr << "Logit mismatch case=" << cfg.name << " token=" << token << " expert=" << expert
                  << " got=" << actual << " expected=" << expected << " abs=" << abs_err << " rel=" << rel_err
                  << "\n";
      }
    }
  }

  float routed_atol = packed ? 2.0e-2f : 8.0e-5f;
  float routed_rtol = packed ? 2.0e-3f : 2.0e-4f;
  for (int64_t i = 0; i < cfg.tokens * moe::kTopK; ++i) {
    uint32_t packed_bits = packed ? static_cast<uint32_t>(got.packed[i]) : 0u;
    int32_t got_idx = packed ? static_cast<int32_t>(packed_bits >> 16) : got.indices[i];
    if (got_idx != gate_ref.indices[i] && failures++ < 8) {
      std::cerr << "Index mismatch case=" << cfg.name << (packed ? " packed" : " nonpacked")
                << " offset=" << i << " got=" << got_idx << " expected=" << gate_ref.indices[i] << "\n";
    }
    float got_w = packed ? moe::host_bf16_to_f32(static_cast<uint16_t>(packed_bits & 0xffffu))
                         : got.routed_weights[i];
    if (!close_enough(got_w, gate_ref.routed_weights[i], routed_atol, routed_rtol) && failures++ < 8) {
      std::cerr << "Routed weight mismatch case=" << cfg.name << (packed ? " packed" : " nonpacked")
                << " offset=" << i << " got=" << got_w << " expected=" << gate_ref.routed_weights[i] << "\n";
    }
  }

  for (int64_t i = 0; i < cfg.tokens * moe::kGateSharedExperts; ++i) {
    if (!close_enough(got.shared_weights[i], gate_ref.shared_weights[i], 8.0e-5f, 2.0e-4f) &&
        failures++ < 8) {
      std::cerr << "Shared weight mismatch case=" << cfg.name << (packed ? " packed" : " nonpacked")
                << " offset=" << i << " got=" << got.shared_weights[i]
                << " expected=" << gate_ref.shared_weights[i] << "\n";
    }
  }

  if (got.ticket != 0 && failures++ < 8) {
    std::cerr << "Ticket reset mismatch case=" << cfg.name << " got=" << got.ticket << " expected=0\n";
  }

  bool passed = failures == 0;
  std::cout << "verify " << std::setw(14) << cfg.name << " mode=" << (packed ? "packed   " : "nonpacked")
            << " tokens=" << std::setw(3) << cfg.tokens
            << " experts_per_wg=" << moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup)
            << " max_logit_abs=" << std::scientific << std::setprecision(3) << max_logit_abs
            << " max_logit_rel=" << max_logit_rel << " : " << (passed ? "PASS" : "FAIL")
            << std::defaultfloat << "\n";
  return passed;
}

double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

double estimated_fused_global_bytes(int64_t tokens, int experts_per_workgroup, bool packed) {
  int epw = moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  int64_t expert_groups = (moe::kGateTotalExperts + epw - 1) / epw;
  double weight_bytes =
      static_cast<double>(moe::kGateTotalExperts) * moe::kGateHidden * sizeof(cutlass::bfloat16_t);
  double x_bytes = static_cast<double>(expert_groups) * tokens * moe::kGateHidden * sizeof(cutlass::bfloat16_t);
  double workspace_bytes = static_cast<double>(tokens) * moe::kGateTotalExperts * sizeof(float) * 2.0;
  double output_bytes = packed ? static_cast<double>(tokens) *
                                     (moe::kTopK * sizeof(int32_t) + moe::kGateSharedExperts * sizeof(float))
                               : static_cast<double>(tokens) *
                                     (moe::kTopK * (sizeof(float) + sizeof(int32_t)) +
                                      moe::kGateSharedExperts * sizeof(float));
  return weight_bytes + x_bytes + workspace_bytes + output_bytes;
}

double fused_gemv_flops(int64_t tokens) {
  return 2.0 * static_cast<double>(tokens) * moe::kGateTotalExperts * moe::kGateHidden;
}

void benchmark_case(
    sycl::queue& queue,
    CaseConfig const& cfg,
    bool packed,
    int experts_per_workgroup,
    int subgroup_size,
    int warmup,
    int iterations) {
  if (cfg.tokens == 0) {
    return;
  }

  HostInputs inputs = make_inputs(cfg, 20260801u + static_cast<uint32_t>(cfg.tokens));
  DeviceBuffer<cutlass::bfloat16_t> d_x(queue, inputs.x.size());
  DeviceBuffer<cutlass::bfloat16_t> d_weight(queue, inputs.weight.size());
  DeviceBuffer<float> d_bias(queue, inputs.bias.size());
  DeviceBuffer<float> d_global_scale(queue, 1);
  DeviceBuffer<float> d_workspace(queue, static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad);
  DeviceBuffer<float> d_routed(queue, static_cast<std::size_t>(cfg.tokens) * moe::kTopK);
  DeviceBuffer<float> d_shared(queue, static_cast<std::size_t>(cfg.tokens) * moe::kGateSharedExperts);
  DeviceBuffer<int32_t> d_indices(queue, static_cast<std::size_t>(cfg.tokens) * moe::kTopK);
  DeviceBuffer<int32_t> d_packed(queue, static_cast<std::size_t>(cfg.tokens) * moe::kTopK);
  DeviceBuffer<int32_t> d_ticket(queue, 1);

  d_x.copy_from(inputs.x);
  d_weight.copy_from(inputs.weight);
  d_bias.copy_from(inputs.bias);
  std::vector<float> global_scale = {inputs.global_scale};
  d_global_scale.copy_from(global_scale);
  std::vector<int32_t> ticket = {0};
  d_ticket.copy_from(ticket);

  moe::GateGemvParams params;
  params.x = d_x.get();
  params.weight = d_weight.get();
  params.logits = d_workspace.get();
  params.bias = d_bias.get();
  params.global_scale = d_global_scale.get();
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.ticket = d_ticket.get();
  params.tokens = cfg.tokens;
  params.route_scale = inputs.route_scale;

  for (int i = 0; i < warmup; ++i) {
    moe::launch_gate_gemv_fused(queue, params, packed, experts_per_workgroup, subgroup_size).wait();
  }

  std::vector<sycl::event> events;
  events.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    events.push_back(moe::launch_gate_gemv_fused(queue, params, packed, experts_per_workgroup, subgroup_size));
  }
  queue.wait();

  double total_ms = 0.0;
  for (auto const& event : events) {
    total_ms += event_ms(event);
  }

  double avg_ms = total_ms / static_cast<double>(iterations);
  double bytes = estimated_fused_global_bytes(cfg.tokens, experts_per_workgroup, packed);
  double flops = fused_gemv_flops(cfg.tokens);
  double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
  double tops = flops / (avg_ms * 1.0e-3) / 1.0e12;
  double intensity = flops / bytes;

  std::cout << "bench  " << std::setw(14) << cfg.name << " mode=" << (packed ? "packed   " : "nonpacked")
            << " tokens=" << std::setw(3) << cfg.tokens
            << " experts_per_wg=" << moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup)
            << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms << " est_GB/s=" << std::setprecision(1)
            << gbps << " TOPS=" << std::setprecision(4) << tops << " flop_per_byte=" << std::setprecision(3)
            << intensity << std::defaultfloat << "\n";
}

std::vector<CaseConfig> make_suite(std::string const& suite, int64_t custom_tokens) {
  if (custom_tokens >= 0) {
    return {{"custom", custom_tokens, false}};
  }
  if (suite == "quick") {
    return {
        {"zero", 0, false},
        {"tie", 1, true},
        {"decode_1", 1, false},
        {"decode_2", 2, false},
        {"decode_3", 3, false},
        {"decode_4", 4, false},
        {"edge_5", 5, false},
        {"boundary_8", 8, false},
        {"fused_cap_64", 64, false},
    };
  }
  if (suite == "full") {
    return {
        {"zero", 0, false},
        {"tie", 1, true},
        {"decode_1", 1, false},
        {"decode_2", 2, false},
        {"decode_3", 3, false},
        {"decode_4", 4, false},
        {"edge_5", 5, false},
        {"edge_7", 7, false},
        {"boundary_8", 8, false},
        {"extend_17", 17, false},
        {"fused_cap_64", 64, false},
    };
  }
  if (suite == "perf") {
    return {
        {"decode_1", 1, false},
        {"decode_2", 2, false},
        {"decode_4", 4, false},
        {"boundary_8", 8, false},
        {"extend_17", 17, false},
        {"fused_cap_64", 64, false},
    };
  }
  throw std::invalid_argument("suite must be quick, full, or perf");
}

int main(int argc, char const** argv) {
  try {
    Options options = parse_options(argc, argv);
    if (options.help) {
      print_usage(argv[0]);
      return 0;
    }
    if (options.iterations <= 0 || options.warmup < 0) {
      throw std::invalid_argument("--iterations must be > 0 and --warmup must be >= 0");
    }
    (void)moe::default_gate_gemv_experts_per_workgroup(options.experts_per_workgroup);
    if (options.subgroup_size != 0 && options.subgroup_size != moe::kSubGroupSize) {
      throw std::invalid_argument("fused gate GEMV requires --subgroup=0 or --subgroup=32");
    }
    if (options.tokens > moe::kGateFusedMaxTokens) {
      throw std::invalid_argument("--tokens exceeds fused gate GEMV cap of 64");
    }

    sycl::queue queue(
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});

    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Inkling fused gate GEMV: bf16 GEMV -> top6 sigmoid gate -> "
                 "non-packed or packed outputs in one launch, max tokens=64\n";
    std::cout << "Roofline: GEMV dominates; at M<=64 the path is memory-bandwidth bound and avoids the "
                 "split GEMV/top-k launch boundary while preserving the padded [M,264] workspace contract.\n";

    std::vector<CaseConfig> cases = make_suite(options.suite, options.tokens);
    bool passed = true;
    if (options.verify && options.suite != "perf") {
      for (auto const& cfg : cases) {
        passed &= verify_case(queue, cfg, false, options.experts_per_workgroup, options.subgroup_size);
        passed &= verify_case(queue, cfg, true, options.experts_per_workgroup, options.subgroup_size);
      }
    }
    if (options.benchmark) {
      for (auto const& cfg : cases) {
        benchmark_case(
            queue,
            cfg,
            false,
            options.experts_per_workgroup,
            options.subgroup_size,
            options.warmup,
            options.iterations);
        benchmark_case(
            queue, cfg, true, options.experts_per_workgroup, options.subgroup_size, options.warmup, options.iterations);
      }
    }

    if (!passed) {
      return 1;
    }
    std::cout << "All requested checks passed.\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
