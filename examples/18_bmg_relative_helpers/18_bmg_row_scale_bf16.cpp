/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "18_bmg_relative_helpers_common.hpp"

namespace relh = cutlass::examples::relative_helpers;

namespace {

relh::RowCase custom_default() {
  relh::RowCase cfg;
  cfg.name = "custom_row_scale";
  cfg.rows = 4;
  cfg.inner = 5;
  cfg.stride = 7;
  return cfg;
}

std::vector<relh::RowCase> quick_suite() {
  return {
      {"tiny_reference_05_2", 4, 5, 7, 0.0},
      {"tail_inner_17", 37, 17, 25, 0.0},
      {"aligned_decode_r", 32, 256, 256, 0.0},
      {"strided_prefill_r", 1024, 256, 320, 0.0},
  };
}

std::vector<relh::RowCase> inkling_suite() {
  return {
      {"inkling_t1_h16_d16", 1, 256, 256, 0.0},
      {"inkling_t32_h16_d16", 32, 256, 256, 0.0},
      {"inkling_t2048_strided", 2048, 256, 320, 0.0},
      {"inkling_tail_inner", 4096, 257, 320, 0.0},
  };
}

std::vector<relh::RowCase> perf_suite() {
  return {
      {"perf_64k_x256", 65536, 256, 256, 350.0},
      {"perf_64k_x256_strided", 65536, 256, 320, 340.0},
      {"perf_32k_x1024", 32768, 1024, 1024, 350.0},
  };
}

std::vector<relh::RowCase> make_suite(std::string const& suite) {
  if (suite == "quick") {
    return quick_suite();
  }
  if (suite == "inkling") {
    return inkling_suite();
  }
  if (suite == "perf") {
    return perf_suite();
  }
  return {};
}

}  // namespace

int main(int argc, char const** argv) {
  relh::Options options;
  try {
    options = relh::parse_options(argc, argv);
    if (options.help) {
      std::cout << "18_bmg_row_scale_bf16: vectorized strided-row scale helper\n\n";
      relh::print_common_usage(argv[0], "quick|inkling|perf", "rows=<int>,inner=<int>,stride=<int>|pad=<int>");
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<relh::RowCase> cases;
  if (!options.shape.empty()) {
    relh::RowCase cfg = custom_default();
    if (!relh::parse_row_shape(options.shape, cfg)) {
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
    sycl::queue queue = relh::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "18_bmg_row_scale_bf16: out[row,:] = dtype(fp32(x[row,:]) * tau[row])\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << relh::dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << relh::bool_text(options.verify)
              << " benchmark=" << relh::bool_text(options.benchmark) << "\n";

    bool all_passed = true;
    if (options.dtype == relh::DType::kAll || options.dtype == relh::DType::kBf16) {
      all_passed &= relh::run_row_cases_for_dtype<cutlass::bfloat16_t, true>(queue, cases, options, "scale");
    }
    if (options.dtype == relh::DType::kAll || options.dtype == relh::DType::kFp16) {
      all_passed &= relh::run_row_cases_for_dtype<cutlass::half_t, true>(queue, cases, options, "scale");
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
