/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "18_bmg_relative_helpers_common.hpp"

namespace relh = cutlass::examples::relative_helpers;

namespace {

relh::RelProjCase custom_default() {
  relh::RelProjCase cfg;
  cfg.name = "custom_rel_proj";
  cfg.t = 3;
  cfg.h = 2;
  cfg.d = 4;
  cfg.e = 5;
  cfg.r_stride_t = cfg.h * cfg.d;
  cfg.proj_per_head = true;
  cfg.tau_mode = relh::kTauPostRow;
  return cfg;
}

std::vector<relh::RelProjCase> quick_suite() {
  return {
      {"tiny_reference_05_1", 3, 2, 4, 5, 8, true, relh::kTauPostRow, 0.0},
      {"production_t1_tau", 1, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
      {"production_t32_tau", 32, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
      {"tail_dims_pre_row", 7, 3, 13, 29, 44, false, relh::kTauPreRow, 0.0},
      {"tail_e_no_tau", 8, 5, 16, 257, 80, false, relh::kTauNone, 0.0},
  };
}

std::vector<relh::RelProjCase> inkling_suite() {
  return {
      {"inkling_decode_t1", 1, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
      {"inkling_decode_t16", 16, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
      {"inkling_small_t_limit", 32, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
      {"inkling_strided_rows", 32, 16, 16, 1024, 320, false, relh::kTauPreToken, 0.0},
      {"inkling_no_tau", 32, 16, 16, 1024, 256, false, relh::kTauNone, 0.0},
  };
}

std::vector<relh::RelProjCase> perf_suite() {
  return {
      {"perf_t1_h16_d16_e1024", 1, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
      {"perf_t16_h16_d16_e1024", 16, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
      {"perf_t32_h16_d16_e1024", 32, 16, 16, 1024, 256, false, relh::kTauPreToken, 0.0},
  };
}

std::vector<relh::RelProjCase> make_suite(std::string const& suite) {
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
      std::cout << "18_bmg_rel_proj_small_t: Inkling small-token relative projection\n\n";
      relh::print_common_usage(
          argv[0],
          "quick|inkling|perf",
          "t=<int>,h=<int>,d=<int>,e=<int>,rpad=<int>,proj=shared|head,tau=none|pre_token|pre_row|post_token|post_row");
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<relh::RelProjCase> cases;
  if (!options.shape.empty()) {
    relh::RelProjCase cfg = custom_default();
    if (!relh::parse_rel_shape(options.shape, cfg)) {
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
    std::cout << "18_bmg_rel_proj_small_t: Inkling small-token relative projection\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << relh::dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << relh::bool_text(options.verify)
              << " benchmark=" << relh::bool_text(options.benchmark) << "\n";

    bool all_passed = true;
    if (options.dtype == relh::DType::kAll || options.dtype == relh::DType::kBf16) {
      all_passed &= relh::run_rel_cases_for_dtype<cutlass::bfloat16_t>(queue, cases, options);
    }
    if (options.dtype == relh::DType::kAll || options.dtype == relh::DType::kFp16) {
      all_passed &= relh::run_rel_cases_for_dtype<cutlass::half_t>(queue, cases, options);
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
