/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
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
  int hidden = moe::kGateHiddenSpecialized;
  std::string activation = "sigmoid";
  bool norm_after_topk = true;
  bool use_bias = true;
  bool use_global_scale = true;
  float route_scale = 8.0f;
  // --hidden and the five gate-config knobs only reach the kernel through the
  // --tokens custom case; tracked so passing one alone is an error rather than
  // a silent no-op that still prints PASS for the default config.
  bool config_set = false;
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
    } else if (key == "--hidden") {
      options.hidden = std::stoi(value);
      options.config_set = true;
    } else if (key == "--activation") {
      options.activation = value;
      options.config_set = true;
    } else if (key == "--norm-after-topk") {
      options.norm_after_topk = parse_bool(value);
      options.config_set = true;
    } else if (key == "--use-bias") {
      options.use_bias = parse_bool(value);
      options.config_set = true;
    } else if (key == "--use-global-scale") {
      options.use_global_scale = parse_bool(value);
      options.config_set = true;
    } else if (key == "--route-scale") {
      options.route_scale = std::stof(value);
      options.config_set = true;
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
            << "  --hidden=<int>               Gate contraction width for --tokens (default 6144)\n"
            << "  --activation=sigmoid|softmax InklingGate gate_activation for --tokens\n"
            << "  --norm-after-topk=0|1        InklingGate norm_after_topk for --tokens (default 1)\n"
            << "  --use-bias=0|1               InklingGate use_gate_bias for --tokens (default 1)\n"
            << "  --use-global-scale=0|1       InklingGate use_global_scale for --tokens (default 1)\n"
            << "  --route-scale=<float>        InklingGate route_scale for --tokens (default 8.0)\n"
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
  // InklingModelConfig.hidden_size (d_model): 1536 is the config default, 6144
  // the production checkpoint and the compile-time-specialized width.
  int hidden = moe::kGateHiddenSpecialized;
  moe::GateConfig gate{};
};

// InklingGate config variants. moe::GateConfig's defaults are the shipped
// checkpoint's (sigmoid, bias, global scale, norm_after_topk, route_scale 8.0).
moe::GateConfig gate_softmax() {
  moe::GateConfig cfg;
  cfg.activation = moe::GateActivation::kSoftmax;
  return cfg;
}

moe::GateConfig gate_no_norm() {
  moe::GateConfig cfg;
  cfg.norm_after_topk = false;
  return cfg;
}

moe::GateConfig gate_no_bias() {
  moe::GateConfig cfg;
  cfg.use_bias = false;
  return cfg;
}

moe::GateConfig gate_no_global_scale() {
  moe::GateConfig cfg;
  cfg.use_global_scale = false;
  return cfg;
}

moe::GateConfig gate_route_scale(float route_scale) {
  moe::GateConfig cfg;
  cfg.route_scale = route_scale;
  return cfg;
}

moe::GateConfig gate_from_options(Options const& options) {
  moe::GateConfig cfg;
  if (options.activation == "softmax") {
    cfg.activation = moe::GateActivation::kSoftmax;
  } else if (options.activation != "sigmoid") {
    throw std::invalid_argument("--activation must be sigmoid or softmax");
  }
  cfg.norm_after_topk = options.norm_after_topk;
  cfg.use_bias = options.use_bias;
  cfg.use_global_scale = options.use_global_scale;
  cfg.route_scale = options.route_scale;
  return cfg;
}

std::string gate_config_label(moe::GateConfig const& cfg) {
  std::string label = cfg.activation == moe::GateActivation::kSigmoid ? "sigmoid" : "softmax";
  label += cfg.norm_after_topk ? "/norm" : "/nonorm";
  label += cfg.use_bias ? "/bias" : "/nobias";
  label += cfg.use_global_scale ? "/gscale" : "/nogscale";
  return label;
}

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
  inputs.x.resize(static_cast<std::size_t>(cfg.tokens) * cfg.hidden);
  inputs.weight.resize(static_cast<std::size_t>(moe::kGateLogitsPad) * cfg.hidden);
  inputs.bias.resize(moe::kGateRoutedExperts);
  inputs.route_scale = cfg.gate.route_scale;
  inputs.global_scale = cfg.gate.global_scale;

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> x_dist(-0.05f, 0.05f);
  std::uniform_real_distribution<float> w_dist(-0.02f, 0.02f);
  // The GEMV logits land near zero, so softmax scores are ~1/258 and a
  // sigmoid-sized bias would decide the whole selection on its own.
  float bias_range = cfg.gate.activation == moe::GateActivation::kSigmoid ? 0.50f : 0.0005f;
  std::uniform_real_distribution<float> bias_dist(-bias_range, bias_range);

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
    for (int k = 0; k < cfg.hidden; ++k) {
      float value = e < moe::kGateTotalExperts ? w_dist(gen) : 0.0f;
      inputs.weight[static_cast<std::size_t>(e) * cfg.hidden + k] = cutlass::bfloat16_t(value);
    }
  }
  for (auto& v : inputs.bias) {
    v = bias_dist(gen);
  }

  if (cfg.tokens > 0) {
    for (int k = 0; k < cfg.hidden; ++k) {
      inputs.x[static_cast<std::size_t>(cfg.tokens - 1) * cfg.hidden + k] =
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
      for (int k = 0; k < cfg.hidden; ++k) {
        float xv = moe::gate_bf16_to_float(inputs.x[static_cast<std::size_t>(token) * cfg.hidden + k]);
        float wv =
            moe::gate_bf16_to_float(inputs.weight[static_cast<std::size_t>(expert) * cfg.hidden + k]);
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

  for (int64_t row = 0; row < cfg.tokens; ++row) {
    moe::gate_reference_row(
        cfg.gate,
        logits.data() + row * moe::kGateLogitsPad,
        cfg.gate.use_bias ? inputs.bias.data() : nullptr,
        ref.indices.data() + row * moe::kTopK,
        ref.routed_weights.data() + row * moe::kTopK,
        ref.shared_weights.data() + row * moe::kGateSharedExperts);
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
  // use_gate_bias=false / use_global_scale=false are modeled by null pointers,
  // exactly as InklingGate leaves self.bias / self.global_scale as None.
  params.bias = cfg.gate.use_bias ? d_bias.get() : nullptr;
  params.global_scale = cfg.gate.use_global_scale ? d_global_scale.get() : nullptr;
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.ticket = d_ticket.get();
  params.tokens = cfg.tokens;
  params.hidden = cfg.hidden;
  params.route_scale = inputs.route_scale;
  params.activation = cfg.gate.activation;
  params.norm_after_topk = cfg.gate.norm_after_topk;

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
  HostInputs inputs =
      make_inputs(cfg, 20260720u + static_cast<uint32_t>(cfg.tokens) + 7u * static_cast<uint32_t>(cfg.hidden));
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
            << " tokens=" << std::setw(3) << cfg.tokens << " hidden=" << std::setw(4) << cfg.hidden
            << " cfg=" << gate_config_label(cfg.gate)
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

double estimated_fused_global_bytes(int64_t tokens, int hidden, int experts_per_workgroup, bool packed) {
  int epw = moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  int64_t expert_groups = (moe::kGateTotalExperts + epw - 1) / epw;
  double weight_bytes =
      static_cast<double>(moe::kGateTotalExperts) * hidden * sizeof(cutlass::bfloat16_t);
  double x_bytes = static_cast<double>(expert_groups) * tokens * hidden * sizeof(cutlass::bfloat16_t);
  double workspace_bytes = static_cast<double>(tokens) * moe::kGateTotalExperts * sizeof(float) * 2.0;
  double output_bytes = packed ? static_cast<double>(tokens) *
                                     (moe::kTopK * sizeof(int32_t) + moe::kGateSharedExperts * sizeof(float))
                               : static_cast<double>(tokens) *
                                     (moe::kTopK * (sizeof(float) + sizeof(int32_t)) +
                                      moe::kGateSharedExperts * sizeof(float));
  return weight_bytes + x_bytes + workspace_bytes + output_bytes;
}

double fused_gemv_flops(int64_t tokens, int hidden) {
  return 2.0 * static_cast<double>(tokens) * moe::kGateTotalExperts * hidden;
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

  HostInputs inputs =
      make_inputs(cfg, 20260801u + static_cast<uint32_t>(cfg.tokens) + 7u * static_cast<uint32_t>(cfg.hidden));
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
  // use_gate_bias=false / use_global_scale=false are modeled by null pointers,
  // exactly as InklingGate leaves self.bias / self.global_scale as None.
  params.bias = cfg.gate.use_bias ? d_bias.get() : nullptr;
  params.global_scale = cfg.gate.use_global_scale ? d_global_scale.get() : nullptr;
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.ticket = d_ticket.get();
  params.tokens = cfg.tokens;
  params.hidden = cfg.hidden;
  params.route_scale = inputs.route_scale;
  params.activation = cfg.gate.activation;
  params.norm_after_topk = cfg.gate.norm_after_topk;

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
  double bytes = estimated_fused_global_bytes(cfg.tokens, cfg.hidden, experts_per_workgroup, packed);
  double flops = fused_gemv_flops(cfg.tokens, cfg.hidden);
  double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
  double tops = flops / (avg_ms * 1.0e-3) / 1.0e12;
  double intensity = flops / bytes;

  std::cout << "bench  " << std::setw(14) << cfg.name << " mode=" << (packed ? "packed   " : "nonpacked")
            << " tokens=" << std::setw(3) << cfg.tokens << " hidden=" << std::setw(4) << cfg.hidden
            << " cfg=" << gate_config_label(cfg.gate)
            << " experts_per_wg=" << moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup)
            << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms << " est_GB/s=" << std::setprecision(1)
            << gbps << " TOPS=" << std::setprecision(4) << tops << " flop_per_byte=" << std::setprecision(3)
            << intensity << std::defaultfloat << "\n";
}

// The two shipped gate widths, both InklingModelConfig.hidden_size values:
// 1536 (config default) and 6144 (production checkpoint, and the width
// sglang's _INKLING_GATE_GEMV_HIDDEN shortcut is built for). 1540 is not a model
// width; it only exercises the generic runtime-hidden path at an awkward size.
static constexpr int kHiddenDefault = 1536;
static constexpr int kHiddenProd = moe::kGateHiddenSpecialized;
static constexpr int kHiddenOdd = 1540;

std::vector<CaseConfig> make_suite(std::string const& suite, Options const& options) {
  if (options.tokens >= 0) {
    return {{"custom", options.tokens, false, options.hidden, gate_from_options(options)}};
  }
  if (suite == "quick") {
    return {
        {"zero", 0, false, kHiddenProd},
        {"tie", 1, true, kHiddenProd},
        {"decode_1", 1, false, kHiddenProd},
        {"decode_2", 2, false, kHiddenProd},
        {"decode_3", 3, false, kHiddenProd},
        {"decode_4", 4, false, kHiddenProd},
        {"edge_5", 5, false, kHiddenProd},
        {"boundary_8", 8, false, kHiddenProd},
        {"fused_cap_64", 64, false, kHiddenProd},
        {"h1536_decode_1", 1, false, kHiddenDefault},
        {"h1536_fused_cap_64", 64, false, kHiddenDefault},
        {"odd_decode_4", 4, false, kHiddenOdd},
        {"odd_fused_cap_64", 64, false, kHiddenOdd},
        {"softmax_4", 4, false, kHiddenProd, gate_softmax()},
        {"nonorm_4", 4, false, kHiddenProd, gate_no_norm()},
        {"nobias_4", 4, false, kHiddenProd, gate_no_bias()},
        {"nogscale_4", 4, false, kHiddenProd, gate_no_global_scale()},
    };
  }
  if (suite == "full") {
    std::vector<CaseConfig> cases;
    for (int hidden : {kHiddenProd, kHiddenDefault, kHiddenOdd}) {
      std::string tag = hidden == kHiddenProd ? "" : "h" + std::to_string(hidden) + "_";
      cases.push_back({tag + "zero", 0, false, hidden});
      cases.push_back({tag + "tie", 1, true, hidden});
      cases.push_back({tag + "decode_1", 1, false, hidden});
      cases.push_back({tag + "decode_2", 2, false, hidden});
      cases.push_back({tag + "decode_3", 3, false, hidden});
      cases.push_back({tag + "decode_4", 4, false, hidden});
      cases.push_back({tag + "edge_5", 5, false, hidden});
      cases.push_back({tag + "edge_7", 7, false, hidden});
      cases.push_back({tag + "boundary_8", 8, false, hidden});
      cases.push_back({tag + "extend_17", 17, false, hidden});
      cases.push_back({tag + "fused_cap_64", 64, false, hidden});
      // InklingGate config variants at each width.
      cases.push_back({tag + "softmax_17", 17, false, hidden, gate_softmax()});
      cases.push_back({tag + "nonorm_17", 17, false, hidden, gate_no_norm()});
      cases.push_back({tag + "nobias_17", 17, false, hidden, gate_no_bias()});
      cases.push_back({tag + "nogscale_17", 17, false, hidden, gate_no_global_scale()});
      // route_scale 8.0 is the checkpoint's (the default), 1.0 the config default.
      cases.push_back({tag + "rs1_17", 17, false, hidden, gate_route_scale(1.0f)});
    }
    return cases;
  }
  if (suite == "perf") {
    return {
        {"decode_1", 1, false, kHiddenProd},
        {"decode_2", 2, false, kHiddenProd},
        {"decode_4", 4, false, kHiddenProd},
        {"boundary_8", 8, false, kHiddenProd},
        {"extend_17", 17, false, kHiddenProd},
        {"fused_cap_64", 64, false, kHiddenProd},
        {"h1536_decode_1", 1, false, kHiddenDefault},
        {"h1536_decode_4", 4, false, kHiddenDefault},
        {"h1536_boundary_8", 8, false, kHiddenDefault},
        {"h1536_fused_cap_64", 64, false, kHiddenDefault},
        {"softmax_64", 64, false, kHiddenProd, gate_softmax()},
        {"nonorm_64", 64, false, kHiddenProd, gate_no_norm()},
        {"nobias_64", 64, false, kHiddenProd, gate_no_bias()},
        {"nogscale_64", 64, false, kHiddenProd, gate_no_global_scale()},
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
    if (options.hidden <= 0) {
      throw std::invalid_argument("--hidden must be > 0");
    }
    // Validate --activation even when no custom case runs, so a typo cannot be
    // mistaken for a passing softmax run.
    (void)gate_from_options(options);
    if (options.config_set && options.tokens < 0) {
      throw std::invalid_argument(
          "--hidden/--activation/--norm-after-topk/--use-bias/--use-global-scale/--route-scale only apply "
          "to the --tokens custom case; the suite tables carry their own configs, so pass --tokens too");
    }

    sycl::queue queue(
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});

    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Inkling fused gate GEMV: bf16 GEMV -> top6 sigmoid gate -> "
                 "non-packed or packed outputs in one launch, max tokens=64\n";
    std::cout << "Roofline: GEMV dominates; at M<=64 the path is memory-bandwidth bound and avoids the "
                 "split GEMV/top-k launch boundary while preserving the padded [M,264] workspace contract.\n";
    std::cout << "hidden is a runtime argument (6144 production/specialized, 1536 config default); the "
                 "gate epilogue covers gate_activation, norm_after_topk, use_gate_bias, use_global_scale and "
                 "route_scale.\n";

    std::vector<CaseConfig> cases = make_suite(options.suite, options);
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
