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
    } else if (key == "--iterations") {
      options.iterations = std::stoi(value);
    } else if (key == "--warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "--rows-per-wg") {
      options.rows_per_workgroup = std::stoi(value);
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
      << "  --verify=0|1             Run correctness checks (default 1)\n"
      << "  --benchmark=0|1          Run timing checks (default 1)\n";
}

struct CaseConfig {
  std::string name;
  int64_t tokens = 0;
  int64_t stride = moe::kTotalExperts;
  bool ties = false;
  bool large_bias = false;
};

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

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> logit_dist(-5.0f, 5.0f);
  std::uniform_real_distribution<float> bias_dist(-0.05f, 0.05f);

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

  std::vector<float> scores(moe::kRoutedExperts);
  for (int64_t row = 0; row < cfg.tokens; ++row) {
    int64_t row_base = row * cfg.stride;
    for (int e = 0; e < moe::kRoutedExperts; ++e) {
      scores[e] = moe::detail::sigmoid_host(inputs.logits[row_base + e]) + inputs.bias[e];
    }

    int32_t selected[moe::kTopK];
    float active[moe::kTopAndShared];
    for (int k = 0; k < moe::kTopK; ++k) {
      float best_score = -std::numeric_limits<float>::max();
      int best_idx = std::numeric_limits<int>::max();
      for (int e = 0; e < moe::kRoutedExperts; ++e) {
        if (moe::detail::score_better(scores[e], e, best_score, best_idx)) {
          best_score = scores[e];
          best_idx = e;
        }
      }
      selected[k] = best_idx;
      scores[best_idx] = -std::numeric_limits<float>::max();
      active[k] = moe::detail::sigmoid_host(inputs.logits[row_base + best_idx]);
    }

    active[moe::kTopK] = moe::detail::sigmoid_host(inputs.logits[row_base + moe::kRoutedExperts]);
    active[moe::kTopK + 1] = moe::detail::sigmoid_host(inputs.logits[row_base + moe::kRoutedExperts + 1]);

    float sum = 0.0f;
    for (float x : active) {
      sum += x;
    }
    float scale = inputs.route_scale * inputs.global_scale / sum;

    for (int k = 0; k < moe::kTopK; ++k) {
      ref.indices[row * moe::kTopK + k] = selected[k];
      ref.routed_weights[row * moe::kTopK + k] = active[k] * scale;
    }
    for (int s = 0; s < moe::kSharedExperts; ++s) {
      ref.shared_weights[row * moe::kSharedExperts + s] = active[moe::kTopK + s] * scale;
    }
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
  params.bias = d_bias.get();
  params.global_scale = d_global_scale.get();
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.tokens = cfg.tokens;
  params.logits_stride = cfg.stride;
  params.route_scale = inputs.route_scale;

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
  params.bias = d_bias.get();
  params.global_scale = d_global_scale.get();
  params.routed_weights = d_routed.get();
  params.shared_weights = d_shared.get();
  params.indices = d_indices.get();
  params.packed = d_packed.get();
  params.tokens = cfg.tokens;
  params.logits_stride = cfg.stride;
  params.route_scale = inputs.route_scale;

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

  std::cout << "bench " << std::setw(10) << cfg.name << " mode=" << (packed ? "packed    " : "nonpacked ")
            << " tokens=" << std::setw(7) << cfg.tokens << " stride=" << std::setw(3) << cfg.stride
            << " avg_ms=" << std::fixed << std::setprecision(4) << std::setw(8) << avg_ms
            << " GB/s=" << std::setw(8) << std::setprecision(2) << gbps
            << " rows/us=" << std::setw(8) << std::setprecision(2) << rows_per_us << "\n";
}

std::vector<CaseConfig> verification_cases(std::string const& suite, Options const& options) {
  if (options.tokens >= 0) {
    return {{"custom", options.tokens, options.stride, false}};
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
  };
  if (suite == "full") {
    cases.push_back({"prod8191", 8191, 264, false});
  }
  return cases;
}

std::vector<CaseConfig> benchmark_cases(std::string const& suite, Options const& options) {
  if (options.tokens >= 0) {
    return {{"custom", options.tokens, options.stride, false}};
  }
  std::vector<CaseConfig> cases = {
      {"decode1", 1, 264, false},
      {"draft9", 9, 264, false},
      {"prod4096", 4096, 264, false},
      {"prefill16k", 16384, 264, false},
      {"large65536", 65536, 264, false},
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
        std::cout << "verify " << std::setw(10) << cfg.name << " tokens=" << std::setw(6) << cfg.tokens
                  << " stride=" << cfg.stride << " ok\n";
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
