/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#include "16_bmg_moe_gate_topk_renorm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  int64_t stride = moe::kTotalExperts;
  int iterations = 100;
  int warmup = 10;
  int rows_per_workgroup = 0;
  std::string activation = "sigmoid";
  bool norm_after_topk = true;
  bool use_bias = true;
  bool use_global_scale = true;
  float route_scale = 8.0f;
  // --stride and the five gate knobs only reach the kernel through the --tokens
  // custom case; tracked so passing one alone is an error instead of a silent
  // no-op that prints PASS for the suite table's own config.
  bool config_set = false;
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
    } else if (key == "--stride") {
      options.stride = std::stoll(value);
      options.config_set = true;
    } else if (key == "--iterations") {
      options.iterations = std::stoi(value);
    } else if (key == "--warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "--rows-per-wg") {
      options.rows_per_workgroup = std::stoi(value);
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
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=quick|full       Verification and benchmark suite (default quick)\n"
      << "  --tokens=<int>           Run one custom token count instead of suite cases\n"
      << "  --stride=<int>           Custom logits row stride for --tokens (default 258)\n"
      << "  --iterations=<int>       Benchmark iterations (default 100)\n"
      << "  --warmup=<int>           Benchmark warmup launches (default 10)\n"
      << "  --rows-per-wg=0|1|2|4|8  Subgroups per workgroup, 0 selects mode-aware default\n"
      << "  --activation=sigmoid|softmax  InklingGate gate_activation for --tokens (default sigmoid)\n"
      << "  --norm-after-topk=0|1    InklingGate norm_after_topk for --tokens (default 1)\n"
      << "  --use-bias=0|1           InklingGate use_gate_bias for --tokens (default 1)\n"
      << "  --use-global-scale=0|1   InklingGate use_global_scale for --tokens (default 1)\n"
      << "  --route-scale=<float>    InklingGate route_scale for --tokens (default 8.0)\n"
      << "  --verify=0|1             Run correctness checks (default 1)\n"
      << "  --benchmark=0|1          Run timing checks (default 1)\n";
}

struct CaseConfig {
  std::string name;
  int64_t tokens = 0;
  int64_t stride = moe::kTotalExperts;
  bool ties = false;
  bool large_bias = false;
  moe::GateConfig gate{};
};

// InklingGate config variants. The defaults in moe::GateConfig are the shipped
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

moe::GateConfig gate_softmax_no_norm() {
  moe::GateConfig cfg;
  cfg.activation = moe::GateActivation::kSoftmax;
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
  std::vector<float> logits;
  std::vector<float> bias;
  float global_scale = 1.25f;
  float route_scale = 8.0f;
};

struct ReferenceOutput {
  std::vector<float> routed_weights;
  std::vector<float> shared_weights;
  std::vector<int32_t> indices;
};

HostInputs make_inputs(CaseConfig const& cfg, uint32_t seed) {
  if (cfg.stride < moe::kTotalExperts) {
    throw std::invalid_argument("stride must be >= 258");
  }

  HostInputs inputs;
  inputs.logits.resize(static_cast<std::size_t>(cfg.tokens * cfg.stride));
  inputs.bias.resize(moe::kRoutedExperts);
  inputs.route_scale = cfg.gate.route_scale;
  inputs.global_scale = cfg.gate.global_scale;

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> logit_dist(-5.0f, 5.0f);
  // The softmax gate's scores are ~1/258, so a sigmoid-sized bias would decide
  // the whole selection; scale it down to keep the top-k meaningful.
  float bias_range = cfg.gate.activation == moe::GateActivation::kSigmoid ? 0.05f : 0.0005f;
  std::uniform_real_distribution<float> bias_dist(-bias_range, bias_range);

  if (cfg.ties) {
    std::fill(inputs.logits.begin(), inputs.logits.end(), 0.0f);
    std::fill(inputs.bias.begin(), inputs.bias.end(), 0.0f);
    return inputs;
  }

  if (cfg.large_bias) {
    std::fill(inputs.logits.begin(), inputs.logits.end(), 0.0f);
    std::fill(inputs.bias.begin(), inputs.bias.end(), 1.0e8f);
    return inputs;
  }

  for (float& x : inputs.logits) {
    x = logit_dist(gen);
  }
  for (float& x : inputs.bias) {
    x = bias_dist(gen);
  }

  if (cfg.tokens > 0) {
    int64_t row = cfg.tokens - 1;
    int64_t base = row * cfg.stride;
    for (int i = 0; i < moe::kTopK + 2; ++i) {
      inputs.logits[base + i] = 1.0f;
      inputs.bias[i] = 0.0f;
    }
  }

  return inputs;
}

ReferenceOutput reference_gate(CaseConfig const& cfg, HostInputs const& inputs) {
  ReferenceOutput ref;
  ref.routed_weights.resize(static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  ref.shared_weights.resize(static_cast<std::size_t>(cfg.tokens * moe::kSharedExperts));
  ref.indices.resize(static_cast<std::size_t>(cfg.tokens * moe::kTopK));

  for (int64_t row = 0; row < cfg.tokens; ++row) {
    moe::gate_reference_row(
        cfg.gate,
        inputs.logits.data() + row * cfg.stride,
        cfg.gate.use_bias ? inputs.bias.data() : nullptr,
        ref.indices.data() + row * moe::kTopK,
        ref.routed_weights.data() + row * moe::kTopK,
        ref.shared_weights.data() + row * moe::kSharedExperts);
  }
  return ref;
}

struct DeviceOutputs {
  std::vector<float> routed_weights;
  std::vector<float> shared_weights;
  std::vector<int32_t> indices;
  std::vector<int32_t> packed;
};

DeviceOutputs run_kernel(
    sycl::queue& queue,
    CaseConfig const& cfg,
    HostInputs const& inputs,
    bool packed,
    int rows_per_workgroup) {
  DeviceBuffer<float> d_logits(queue, inputs.logits.size());
  DeviceBuffer<float> d_bias(queue, inputs.bias.size());
  DeviceBuffer<float> d_global_scale(queue, 1);
  DeviceBuffer<float> d_routed(queue, static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  DeviceBuffer<float> d_shared(queue, static_cast<std::size_t>(cfg.tokens * moe::kSharedExperts));
  DeviceBuffer<int32_t> d_indices(queue, static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  DeviceBuffer<int32_t> d_packed(queue, static_cast<std::size_t>(cfg.tokens * moe::kTopK));

  d_logits.copy_from(inputs.logits);
  d_bias.copy_from(inputs.bias);
  std::vector<float> global_scale = {inputs.global_scale};
  d_global_scale.copy_from(global_scale);

  moe::GateParams params;
  params.logits = d_logits.get();
  // use_gate_bias=false / use_global_scale=false are modeled by null pointers,
  // exactly as InklingGate leaves self.bias / self.global_scale as None.
  params.bias = cfg.gate.use_bias ? d_bias.get() : nullptr;
  params.global_scale = cfg.gate.use_global_scale ? d_global_scale.get() : nullptr;
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.tokens = cfg.tokens;
  params.logits_stride = cfg.stride;
  params.route_scale = inputs.route_scale;
  params.activation = cfg.gate.activation;
  params.norm_after_topk = cfg.gate.norm_after_topk;

  if (cfg.tokens > 0) {
    moe::launch_gate_topk_renorm(queue, params, packed, rows_per_workgroup).wait();
  }

  DeviceOutputs outputs;
  outputs.routed_weights.resize(static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  outputs.shared_weights.resize(static_cast<std::size_t>(cfg.tokens * moe::kSharedExperts));
  outputs.indices.resize(static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  outputs.packed.resize(static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  if (packed) {
    d_packed.copy_to(outputs.packed);
  } else {
    d_routed.copy_to(outputs.routed_weights);
    d_indices.copy_to(outputs.indices);
  }
  d_shared.copy_to(outputs.shared_weights);
  return outputs;
}

bool close_enough(float got, float expected, float atol, float rtol) {
  float diff = std::abs(got - expected);
  return diff <= atol + rtol * std::abs(expected);
}

void verify_case(
    sycl::queue& queue,
    CaseConfig const& cfg,
    bool packed,
    int rows_per_workgroup) {
  HostInputs inputs = make_inputs(cfg, 20260720u + static_cast<uint32_t>(cfg.tokens + cfg.stride));
  ReferenceOutput ref = reference_gate(cfg, inputs);
  DeviceOutputs got = run_kernel(queue, cfg, inputs, packed, rows_per_workgroup);

  float routed_atol = packed ? 2.0e-2f : 8.0e-5f;
  float routed_rtol = packed ? 2.0e-3f : 2.0e-4f;
  float shared_atol = 8.0e-5f;
  float shared_rtol = 2.0e-4f;

  for (int64_t i = 0; i < cfg.tokens * moe::kTopK; ++i) {
    int32_t got_idx = packed ? (got.packed[i] >> 16) : got.indices[i];
    if (got_idx != ref.indices[i]) {
      throw std::runtime_error(
          cfg.name + (packed ? " packed" : " non-packed") + " index mismatch at " + std::to_string(i) +
          ": got " + std::to_string(got_idx) + " expected " + std::to_string(ref.indices[i]));
    }

    float got_w = packed ? moe::host_bf16_to_f32(static_cast<uint16_t>(got.packed[i] & 0xffff))
                         : got.routed_weights[i];
    if (!close_enough(got_w, ref.routed_weights[i], routed_atol, routed_rtol)) {
      throw std::runtime_error(
          cfg.name + (packed ? " packed" : " non-packed") + " routed weight mismatch at " +
          std::to_string(i) + ": got " + std::to_string(got_w) + " expected " +
          std::to_string(ref.routed_weights[i]));
    }
  }

  for (int64_t i = 0; i < cfg.tokens * moe::kSharedExperts; ++i) {
    if (!close_enough(got.shared_weights[i], ref.shared_weights[i], shared_atol, shared_rtol)) {
      throw std::runtime_error(
          cfg.name + (packed ? " packed" : " non-packed") + " shared weight mismatch at " +
          std::to_string(i) + ": got " + std::to_string(got.shared_weights[i]) + " expected " +
          std::to_string(ref.shared_weights[i]));
    }
  }
}

double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

void benchmark_case(
    sycl::queue& queue,
    CaseConfig const& cfg,
    bool packed,
    int rows_per_workgroup,
    int warmup,
    int iterations) {
  if (cfg.tokens == 0) {
    return;
  }

  HostInputs inputs = make_inputs(cfg, 1337u + static_cast<uint32_t>(cfg.tokens));

  DeviceBuffer<float> d_logits(queue, inputs.logits.size());
  DeviceBuffer<float> d_bias(queue, inputs.bias.size());
  DeviceBuffer<float> d_global_scale(queue, 1);
  DeviceBuffer<float> d_routed(queue, static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  DeviceBuffer<float> d_shared(queue, static_cast<std::size_t>(cfg.tokens * moe::kSharedExperts));
  DeviceBuffer<int32_t> d_indices(queue, static_cast<std::size_t>(cfg.tokens * moe::kTopK));
  DeviceBuffer<int32_t> d_packed(queue, static_cast<std::size_t>(cfg.tokens * moe::kTopK));

  d_logits.copy_from(inputs.logits);
  d_bias.copy_from(inputs.bias);
  std::vector<float> global_scale = {inputs.global_scale};
  d_global_scale.copy_from(global_scale);

  moe::GateParams params;
  params.logits = d_logits.get();
  // use_gate_bias=false / use_global_scale=false are modeled by null pointers,
  // exactly as InklingGate leaves self.bias / self.global_scale as None.
  params.bias = cfg.gate.use_bias ? d_bias.get() : nullptr;
  params.global_scale = cfg.gate.use_global_scale ? d_global_scale.get() : nullptr;
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.tokens = cfg.tokens;
  params.logits_stride = cfg.stride;
  params.route_scale = inputs.route_scale;
  params.activation = cfg.gate.activation;
  params.norm_after_topk = cfg.gate.norm_after_topk;

  for (int i = 0; i < warmup; ++i) {
    moe::launch_gate_topk_renorm(queue, params, packed, rows_per_workgroup);
  }
  queue.wait();

  std::vector<sycl::event> events;
  events.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    events.push_back(moe::launch_gate_topk_renorm(queue, params, packed, rows_per_workgroup));
  }
  queue.wait();

  double total_ms = 0.0;
  for (auto const& event : events) {
    total_ms += event_ms(event);
  }
  double avg_ms = total_ms / static_cast<double>(iterations);
  double output_bytes = packed ? (moe::kTopK * sizeof(int32_t) + moe::kSharedExperts * sizeof(float))
                               : (moe::kTopK * sizeof(float) + moe::kTopK * sizeof(int32_t) +
                                  moe::kSharedExperts * sizeof(float));
  double bytes = static_cast<double>(cfg.tokens) * (moe::kTotalExperts * sizeof(float) + output_bytes);
  double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
  double rows_per_us = static_cast<double>(cfg.tokens) / (avg_ms * 1000.0);

  std::cout << "bench " << std::setw(14) << cfg.name << " mode=" << (packed ? "packed    " : "nonpacked ")
            << " tokens=" << std::setw(7) << cfg.tokens << " stride=" << std::setw(3) << cfg.stride
            << " avg_ms=" << std::fixed << std::setprecision(4) << std::setw(8) << avg_ms
            << " GB/s=" << std::setw(8) << std::setprecision(2) << gbps
            << " rows/us=" << std::setw(8) << std::setprecision(2) << rows_per_us << "\n";
}

std::vector<CaseConfig> verification_cases(std::string const& suite, Options const& options) {
  if (options.tokens >= 0) {
    return {{"custom", options.tokens, options.stride, false, false, gate_from_options(options)}};
  }
  std::vector<CaseConfig> cases = {
      {"zero", 0, moe::kTotalExperts, false},
      {"tie", 1, moe::kTotalExperts, true},
      {"decode1", 1, 264, false},
      {"largebias", 2, 264, false, true},
      {"small258", 3, moe::kTotalExperts, false},
      {"decode", 8, 264, false},
      {"draft9", 9, 264, false},
      {"tail", 17, 264, false},
      {"fusedcap64", 64, 264, false},
      {"oddstride", 65, 300, false},
      {"extend", 127, 264, false},
      {"medium", 512, 264, false},
      {"prod4096", 4096, 264, false},
      // InklingGate config variants the model supports.
      {"softmax1", 1, 264, false, false, gate_softmax()},
      {"softmax64", 64, 264, false, false, gate_softmax()},
      {"nonorm1", 1, 264, false, false, gate_no_norm()},
      {"nonorm64", 64, 264, false, false, gate_no_norm()},
      {"softmaxnonorm", 17, 264, false, false, gate_softmax_no_norm()},
      {"nobias1", 1, 264, false, false, gate_no_bias()},
      {"nobias64", 64, 264, false, false, gate_no_bias()},
      {"nogscale1", 1, 264, false, false, gate_no_global_scale()},
      {"nogscale64", 64, 264, false, false, gate_no_global_scale()},
      // route_scale: 8.0 is the checkpoint's (the default above), 1.0 the
      // InklingModelConfig default.
      {"rs1", 8, 264, false, false, gate_route_scale(1.0f)},
  };
  if (suite == "full") {
    cases.push_back({"prod8191", 8191, 264, false});
    cases.push_back({"softmax4096", 4096, 264, false, false, gate_softmax()});
    cases.push_back({"nonorm4096", 4096, 264, false, false, gate_no_norm()});
    cases.push_back({"nobias4096", 4096, 264, false, false, gate_no_bias()});
    cases.push_back({"nogscale4096", 4096, 264, false, false, gate_no_global_scale()});
  }
  return cases;
}

std::vector<CaseConfig> benchmark_cases(std::string const& suite, Options const& options) {
  if (options.tokens >= 0) {
    return {{"custom", options.tokens, options.stride, false, false, gate_from_options(options)}};
  }
  std::vector<CaseConfig> cases = {
      {"decode1", 1, 264, false},
      {"draft9", 9, 264, false},
      {"prod4096", 4096, 264, false},
      {"prefill16k", 16384, 264, false},
      {"large65536", 65536, 264, false},
      // The config variants change the per-row epilogue only; they are timed
      // at one large shape so a regression in the softmax row reduction or the
      // logsumexp renorm shows up.
      {"softmax16k", 16384, 264, false, false, gate_softmax()},
      {"nonorm16k", 16384, 264, false, false, gate_no_norm()},
      {"nobias16k", 16384, 264, false, false, gate_no_bias()},
      {"nogscale16k", 16384, 264, false, false, gate_no_global_scale()},
  };
  if (suite == "full") {
    cases.push_back({"large131k", 131072, 264, false});
  }
  return cases;
}

int main(int argc, char const** argv) {
  try {
    Options options = parse_options(argc, argv);
    if (options.help) {
      print_usage(argv[0]);
      return 0;
    }
    if (options.suite != "quick" && options.suite != "full") {
      throw std::invalid_argument("--suite must be quick or full");
    }
    if (options.iterations <= 0 || options.warmup < 0) {
      throw std::invalid_argument("--iterations must be > 0 and --warmup must be >= 0");
    }
    // Validate --activation even when no custom case runs, so a typo cannot be
    // mistaken for a passing softmax run.
    (void)gate_from_options(options);
    if (options.config_set && options.tokens < 0) {
      throw std::invalid_argument(
          "--stride/--activation/--norm-after-topk/--use-bias/--use-global-scale/--route-scale only apply "
          "to the --tokens custom case; the suite tables carry their own configs, so pass --tokens too");
    }

    sycl::queue queue(
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});

    std::cout << "device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    if (options.rows_per_workgroup == 0) {
      std::cout << "rows_per_workgroup: auto\n";
    } else {
      std::cout << "rows_per_workgroup: " << options.rows_per_workgroup << "\n";
    }

    if (options.verify) {
      for (auto const& cfg : verification_cases(options.suite, options)) {
        verify_case(queue, cfg, false, options.rows_per_workgroup);
        verify_case(queue, cfg, true, options.rows_per_workgroup);
        std::cout << "verify " << std::setw(14) << cfg.name << " tokens=" << std::setw(6) << cfg.tokens
                  << " stride=" << cfg.stride << " cfg=" << gate_config_label(cfg.gate)
                  << " route_scale=" << cfg.gate.route_scale << " ok\n";
      }
    }

    if (options.benchmark) {
      for (auto const& cfg : benchmark_cases(options.suite, options)) {
        benchmark_case(queue, cfg, false, options.rows_per_workgroup, options.warmup, options.iterations);
        benchmark_case(queue, cfg, true, options.rows_per_workgroup, options.warmup, options.iterations);
      }
    }

    return 0;
  } catch (std::exception const& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
