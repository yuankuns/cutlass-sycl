/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "20_bmg_dflash_common.hpp"

namespace dflash = cutlass::examples::bmg_dflash;

namespace {

constexpr int kNameStride = 32;

struct DeviceGuardParams {
  char const* names = nullptr;
  uint8_t* legacy_cuda_guard = nullptr;
  uint8_t* dflash_supported_guard = nullptr;
  int count = 0;
  int stride = kNameStride;
};

class DeviceGuardKernel {
 public:
  explicit DeviceGuardKernel(DeviceGuardParams params) : params_(params) {}

  void operator()(sycl::id<1> id) const {
    int row = static_cast<int>(id[0]);
    if (row >= params_.count) {
      return;
    }
    char const* s = params_.names + row * params_.stride;
    bool is_cuda = s[0] == 'c' && s[1] == 'u' && s[2] == 'd' && s[3] == 'a';
    bool is_xpu = s[0] == 'x' && s[1] == 'p' && s[2] == 'u';
    bool is_level_zero = s[0] == 'l' && s[1] == 'e' && s[2] == 'v' && s[3] == 'e' &&
                         s[4] == 'l' && s[5] == '_' && s[6] == 'z' && s[7] == 'e' &&
                         s[8] == 'r' && s[9] == 'o';

    params_.legacy_cuda_guard[row] = static_cast<uint8_t>(is_cuda);
    params_.dflash_supported_guard[row] = static_cast<uint8_t>(is_cuda || is_xpu || is_level_zero);
  }

 private:
  DeviceGuardParams params_;
};

sycl::event launch_device_guard(sycl::queue& queue, DeviceGuardParams const& params) {
  if (params.count <= 0) {
    return {};
  }
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(params.count)), DeviceGuardKernel(params));
  });
}

struct GuardCase {
  std::string name;
  std::vector<std::string> devices;
};

std::vector<GuardCase> make_suite(std::string const& suite) {
  if (suite == "quick") {
    return {{
        "reference_cuda_vs_xpu",
        {"cuda:0", "xpu", "cuda", "cpu", "level_zero:gpu"},
    }};
  }
  if (suite == "stress") {
    return {{
        "prefix_variants",
        {"cuda:0", "cuda", "cuda_privateuseone", "xpu", "xpu:0", "level_zero:gpu", "level_zero",
         "cpu", "hip", "privateuseone"},
    }};
  }
  if (suite == "perf") {
    std::vector<std::string> devices;
    devices.reserve(1 << 16);
    char const* labels[] = {"cuda:0", "xpu", "level_zero:gpu", "cpu"};
    for (int i = 0; i < (1 << 16); ++i) {
      devices.emplace_back(labels[i & 3]);
    }
    return {{"many_device_strings", std::move(devices)}};
  }
  return {};
}

bool host_legacy_cuda_guard(std::string const& device) {
  return dflash::starts_with(device, "cuda");
}

bool host_dflash_supported_guard(std::string const& device) {
  return dflash::starts_with(device, "cuda") || dflash::starts_with(device, "xpu") ||
         dflash::starts_with(device, "level_zero");
}

bool run_case(sycl::queue& queue, GuardCase const& cfg, dflash::Options const& options) {
  std::vector<char> names(cfg.devices.size() * kNameStride, '\0');
  std::vector<uint8_t> legacy_ref(cfg.devices.size(), 0);
  std::vector<uint8_t> supported_ref(cfg.devices.size(), 0);

  for (std::size_t i = 0; i < cfg.devices.size(); ++i) {
    std::string const& text = cfg.devices[i];
    if (text.size() >= kNameStride) {
      throw std::runtime_error("device string too long for guard example");
    }
    std::copy(text.begin(), text.end(), names.begin() + static_cast<std::ptrdiff_t>(i * kNameStride));
    legacy_ref[i] = static_cast<uint8_t>(host_legacy_cuda_guard(text));
    supported_ref[i] = static_cast<uint8_t>(host_dflash_supported_guard(text));
  }

  dflash::DeviceBuffer<char> d_names(queue, names.size());
  dflash::DeviceBuffer<uint8_t> d_legacy(queue, cfg.devices.size());
  dflash::DeviceBuffer<uint8_t> d_supported(queue, cfg.devices.size());
  d_names.copy_from(names);

  DeviceGuardParams params;
  params.names = d_names.get();
  params.legacy_cuda_guard = d_legacy.get();
  params.dflash_supported_guard = d_supported.get();
  params.count = static_cast<int>(cfg.devices.size());
  params.stride = kNameStride;

  launch_device_guard(queue, params).wait();

  std::vector<uint8_t> legacy_got(cfg.devices.size(), 0);
  std::vector<uint8_t> supported_got(cfg.devices.size(), 0);
  d_legacy.copy_to(legacy_got);
  d_supported.copy_to(supported_got);

  bool passed = true;
  if (options.verify) {
    for (std::size_t i = 0; i < cfg.devices.size(); ++i) {
      bool ok = legacy_got[i] == legacy_ref[i] && supported_got[i] == supported_ref[i];
      if (!ok) {
        passed = false;
        std::cerr << "Mismatch case=" << cfg.name << " device=" << cfg.devices[i]
                  << " legacy got=" << int(legacy_got[i]) << " ref=" << int(legacy_ref[i])
                  << " supported got=" << int(supported_got[i]) << " ref=" << int(supported_ref[i]) << "\n";
      }
    }
  }

  double ms = 0.0;
  if (options.benchmark && options.iterations > 0) {
    for (int i = 0; i < options.warmup; ++i) {
      launch_device_guard(queue, params);
    }
    queue.wait();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iterations; ++i) {
      launch_device_guard(queue, params);
    }
    queue.wait();
    auto end = std::chrono::steady_clock::now();
    ms = dflash::elapsed_ms(begin, end, options.iterations);
  }

  std::cout << "case=" << std::left << std::setw(24) << cfg.name
            << " strings=" << std::right << std::setw(7) << cfg.devices.size()
            << " verify=" << dflash::bool_text(!options.verify || passed)
            << " time_ms=" << std::fixed << std::setprecision(4) << ms << "\n";
  return passed;
}

void print_usage(char const* exe) {
  std::cout << "20_bmg_dflash_device_guard: DFLASH CUDA guard replacement check\n\n"
            << "Usage: " << exe << " [--suite=quick|stress|perf] [--iterations=N] [--verify=0|1]\n";
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

  std::vector<GuardCase> cases = make_suite(options.suite);
  if (cases.empty()) {
    std::cerr << "Unknown suite: " << options.suite << "\n";
    return -1;
  }

  try {
    sycl::queue queue = dflash::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "20_bmg_dflash_device_guard: legacy startswith(cuda), replacement accepts cuda/xpu/level_zero\n";

    bool all_passed = true;
    for (GuardCase const& cfg : cases) {
      all_passed &= run_case(queue, cfg, options);
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
