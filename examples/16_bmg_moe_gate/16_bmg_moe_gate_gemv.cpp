#include "16_bmg_moe_gate_gemv.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
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
            << "  --subgroup=0|16|32           Subgroup size, 0 selects default\n"
            << "  --verify=0|1                 Run correctness checks (default 1)\n"
            << "  --benchmark=0|1              Run timing checks (default 1)\n";
}

struct CaseConfig {
  std::string name;
  int64_t tokens = 0;
};

struct HostInputs {
  std::vector<cutlass::bfloat16_t> x;
  std::vector<cutlass::bfloat16_t> weight;
};

HostInputs make_inputs(CaseConfig const& cfg, uint32_t seed) {
  HostInputs inputs;
  inputs.x.resize(static_cast<std::size_t>(cfg.tokens) * moe::kGateHidden);
  inputs.weight.resize(static_cast<std::size_t>(moe::kGateLogitsPad) * moe::kGateHidden);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> x_dist(-0.05f, 0.05f);
  std::uniform_real_distribution<float> w_dist(-0.02f, 0.02f);

  for (auto& v : inputs.x) {
    v = cutlass::bfloat16_t(x_dist(gen));
  }
  for (int e = 0; e < moe::kGateLogitsPad; ++e) {
    for (int k = 0; k < moe::kGateHidden; ++k) {
      float value = e < moe::kGateTotalExperts ? w_dist(gen) : 0.0f;
      inputs.weight[static_cast<std::size_t>(e) * moe::kGateHidden + k] = cutlass::bfloat16_t(value);
    }
  }

  if (cfg.tokens > 0) {
    for (int k = 0; k < moe::kGateHidden; ++k) {
      inputs.x[static_cast<std::size_t>(cfg.tokens - 1) * moe::kGateHidden + k] =
          cutlass::bfloat16_t(((k % 17) - 8) * 0.003f);
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

std::vector<float> run_kernel(
    sycl::queue& queue,
    CaseConfig const& cfg,
    HostInputs const& inputs,
    int experts_per_workgroup,
    int subgroup_size) {
  DeviceBuffer<cutlass::bfloat16_t> d_x(queue, inputs.x.size());
  DeviceBuffer<cutlass::bfloat16_t> d_weight(queue, inputs.weight.size());
  DeviceBuffer<float> d_logits(queue, static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad);

  d_x.copy_from(inputs.x);
  d_weight.copy_from(inputs.weight);
  std::vector<float> logits(static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad, -777.0f);
  d_logits.copy_from(logits);

  moe::GateGemvParams params;
  params.x = d_x.get();
  params.weight = d_weight.get();
  params.logits = d_logits.get();
  params.tokens = cfg.tokens;

  if (cfg.tokens > 0) {
    moe::launch_gate_gemv(queue, params, experts_per_workgroup, subgroup_size).wait();
  }

  d_logits.copy_to(logits);
  return logits;
}

bool close_enough(float got, float expected, float atol, float rtol) {
  float diff = std::abs(got - expected);
  return diff <= atol + rtol * std::abs(expected);
}

bool verify_case(
    sycl::queue& queue,
    CaseConfig const& cfg,
    int experts_per_workgroup,
    int subgroup_size) {
  HostInputs inputs = make_inputs(cfg, 20260720u + static_cast<uint32_t>(cfg.tokens));
  std::vector<float> ref = reference_gemv(cfg, inputs);
  std::vector<float> got = run_kernel(queue, cfg, inputs, experts_per_workgroup, subgroup_size);

  int failures = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (int64_t token = 0; token < cfg.tokens; ++token) {
    for (int expert = 0; expert < moe::kGateLogitsPad; ++expert) {
      std::size_t idx = static_cast<std::size_t>(token) * moe::kGateLogitsPad + expert;
      float expected = ref[idx];
      float actual = got[idx];
      double abs_err = std::abs(static_cast<double>(actual) - static_cast<double>(expected));
      double rel_err = abs_err / std::max(1.0e-12, std::abs(static_cast<double>(expected)));
      max_abs = std::max(max_abs, abs_err);
      max_rel = std::max(max_rel, rel_err);
      bool ok = expert < moe::kGateTotalExperts ? close_enough(actual, expected, 2.5e-4f, 2.0e-3f)
                                                : actual == -777.0f;
      if (!ok && failures++ < 8) {
        std::cerr << "Mismatch case=" << cfg.name << " token=" << token << " expert=" << expert
                  << " got=" << actual << " expected=" << expected << " abs=" << abs_err << " rel=" << rel_err
                  << "\n";
      }
    }
  }

  bool passed = failures == 0;
  std::cout << "verify " << std::setw(16) << cfg.name << " tokens=" << std::setw(5) << cfg.tokens
            << " experts_per_wg=" << moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup)
            << " subgroup=" << moe::default_gate_gemv_subgroup_size(subgroup_size, cfg.tokens)
            << " max_abs=" << std::scientific << std::setprecision(3) << max_abs << " max_rel=" << max_rel
            << " : " << (passed ? "PASS" : "FAIL") << std::defaultfloat << "\n";
  return passed;
}

double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

double estimated_global_bytes(int64_t tokens, int experts_per_workgroup) {
  int epw = moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  int64_t expert_groups = (moe::kGateTotalExperts + epw - 1) / epw;
  double weight_bytes =
      static_cast<double>(moe::kGateTotalExperts) * moe::kGateHidden * sizeof(cutlass::bfloat16_t);
  double x_bytes = static_cast<double>(expert_groups) * tokens * moe::kGateHidden * sizeof(cutlass::bfloat16_t);
  double output_bytes = static_cast<double>(tokens) * moe::kGateTotalExperts * sizeof(float);
  return weight_bytes + x_bytes + output_bytes;
}

double gemv_flops(int64_t tokens) {
  return 2.0 * static_cast<double>(tokens) * moe::kGateTotalExperts * moe::kGateHidden;
}

void benchmark_case(
    sycl::queue& queue,
    CaseConfig const& cfg,
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
  DeviceBuffer<float> d_logits(queue, static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad);

  d_x.copy_from(inputs.x);
  d_weight.copy_from(inputs.weight);
  std::vector<float> logits(static_cast<std::size_t>(cfg.tokens) * moe::kGateLogitsPad, -777.0f);
  d_logits.copy_from(logits);

  moe::GateGemvParams params;
  params.x = d_x.get();
  params.weight = d_weight.get();
  params.logits = d_logits.get();
  params.tokens = cfg.tokens;

  for (int i = 0; i < warmup; ++i) {
    moe::launch_gate_gemv(queue, params, experts_per_workgroup, subgroup_size).wait();
  }

  std::vector<sycl::event> events;
  events.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    events.push_back(moe::launch_gate_gemv(queue, params, experts_per_workgroup, subgroup_size));
  }
  queue.wait();

  double total_ms = 0.0;
  for (auto const& event : events) {
    total_ms += event_ms(event);
  }

  double avg_ms = total_ms / static_cast<double>(iterations);
  double bytes = estimated_global_bytes(cfg.tokens, experts_per_workgroup);
  double flops = gemv_flops(cfg.tokens);
  double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
  double tops = flops / (avg_ms * 1.0e-3) / 1.0e12;
  double intensity = flops / bytes;

  std::cout << "bench  " << std::setw(16) << cfg.name << " tokens=" << std::setw(5) << cfg.tokens
            << " experts_per_wg=" << moe::default_gate_gemv_experts_per_workgroup(experts_per_workgroup)
            << " subgroup=" << moe::default_gate_gemv_subgroup_size(subgroup_size, cfg.tokens)
            << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms << " est_GB/s=" << std::setprecision(1)
            << gbps << " TOPS=" << std::setprecision(4) << tops << " flop_per_byte=" << std::setprecision(3)
            << intensity << std::defaultfloat << "\n";
}

std::vector<CaseConfig> make_suite(std::string const& suite, int64_t custom_tokens) {
  if (custom_tokens >= 0) {
    return {{"custom", custom_tokens}};
  }
  if (suite == "quick") {
    return {
        {"zero", 0},
        {"decode_1", 1},
        {"decode_2", 2},
        {"decode_3", 3},
        {"decode_4", 4},
        {"boundary_8", 8},
    };
  }
  if (suite == "full") {
    return {
        {"zero", 0},
        {"decode_1", 1},
        {"decode_2", 2},
        {"decode_3", 3},
        {"decode_4", 4},
        {"edge_5", 5},
        {"edge_7", 7},
        {"boundary_8", 8},
        {"extend_17", 17},
        {"fused_cap_64", 64},
    };
  }
  if (suite == "perf") {
    return {
        {"decode_1", 1},
        {"decode_2", 2},
        {"decode_3", 3},
        {"decode_4", 4},
        {"larger_64", 64},
        {"larger_512", 512},
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
    (void)moe::default_gate_gemv_subgroup_size(options.subgroup_size, 1);

    sycl::queue queue(
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});

    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Inkling gate GEMV: bf16 x bf16 -> fp32 logits [tokens, 264], writes columns [0, 258)\n";
    std::cout << "Roofline: production M<=4 has flop/byte ~= 0.50..0.80 at experts_per_wg=1, so this is "
                 "memory-bandwidth bound; 350 GB/s is the relevant target.\n";

    std::vector<CaseConfig> cases = make_suite(options.suite, options.tokens);
    bool passed = true;
    if (options.verify && options.suite != "perf") {
      for (auto const& cfg : cases) {
        passed &= verify_case(queue, cfg, options.experts_per_workgroup, options.subgroup_size);
      }
    }
    if (options.benchmark) {
      for (auto const& cfg : cases) {
        benchmark_case(
            queue, cfg, options.experts_per_workgroup, options.subgroup_size, options.warmup, options.iterations);
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
