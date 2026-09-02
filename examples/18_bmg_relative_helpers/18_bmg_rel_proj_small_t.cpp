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
      {"local_e512_tau", 9, 12, 16, 512, 1984, false, relh::kTauPreToken, 0.0},
      {"segmented_e1022_tau", 9, 5, 16, 1022, 80, false, relh::kTauPreToken, 0.0},
      {"odd_e1023_tau", 9, 5, 16, 1023, 80, false, relh::kTauPreToken, 0.0},
      {"tail_dims_pre_row", 7, 3, 13, 29, 44, false, relh::kTauPreRow, 0.0},
      {"tail_e_no_tau", 8, 5, 16, 257, 80, false, relh::kTauNone, 0.0},
  };
}

// Inkling rel_proj_small_t is only reached inside the tau-fused small-t band
// (T <= _REL_PROJ_TAU_KERNEL_MAX_T = 32); real invocations are:
//   * decode: T = 1
//   * target-verify: T = draft_token_num = 9
//   * extend head/tail into the small band: T = 32
// Every call passes d_rel = 16. Global attention uses rel_extent = 1024, while
// local attention uses local_extent = sliding_window_size = 512. Per-rank head
// geometry:
//   config defaults (hidden=1536): H = 12/6/3 for TP=1/2/4 (TP=8 skipped, 12%8)
//   production      (hidden=6144): H = 48/24/12/6 for TP=1/2/4/8
// r is a strided view into the packed [q||k||v||r] qkvr row, so r_stride_t is
// the whole qkvr row width per token: h*head_dim + 2*kv*head_dim + h*d_rel.
// Perf gates below are the B60 bf16 measurements with roughly 10% headroom.
// The harness only enforces target_gbps once a case moves at least
// kMinSustainedTargetBytes (32 MiB), so for this suite -- whose largest case
// moves 30 MiB -- they document the measured level rather than fail a build.
//
// The four T=1 cases that sit near 200 GB/s and below are pinned by the launch
// floor, not by the kernel: the smallest launch this harness can measure on B60
// is 0.731 us (a one-row row_compact) and the smallest rel_proj launch is
// 1.041 us. Reaching 370 GB/s would need 0.631 us for 6 rows at E=1024
// (233472 B) or for 12 rows at E=512, and 0.316 us for 3 rows -- all below the
// empty-launch floor, so no kernel change can get there. The 12-row E=1024
// decode cases do clear it, at 385-387 GB/s.
std::vector<relh::RelProjCase> inkling_suite() {
  return {
      // Config defaults, decode T=1.
      {"cfg_tp1_decode_t1",     1, 12, 16, 1024, 12*128 + 2*4*128 + 12*16, false, relh::kTauPreToken, 350.0},
      {"cfg_tp2_decode_t1",     1,  6, 16, 1024,  6*128 + 2*2*128 +  6*16, false, relh::kTauPreToken, 190.0},
      {"cfg_tp4_decode_t1",     1,  3, 16, 1024,  3*128 + 2*1*128 +  3*16, false, relh::kTauPreToken, 96.0},
      // Config defaults, target-verify T=9 (draft_token_num).
      {"cfg_tp2_verify_t9",     9,  6, 16, 1024,  6*128 + 2*2*128 +  6*16, false, relh::kTauPreToken, 1200.0},
      {"cfg_tp4_verify_t9",     9,  3, 16, 1024,  3*128 + 2*1*128 +  3*16, false, relh::kTauPreToken, 690.0},
      // Config defaults, small-t band max T=32.
      {"cfg_tp2_extend_t32",   32,  6, 16, 1024,  6*128 + 2*2*128 +  6*16, false, relh::kTauPreToken, 2550.0},
      {"cfg_tp4_extend_t32",   32,  3, 16, 1024,  3*128 + 2*1*128 +  3*16, false, relh::kTauPreToken, 1900.0},

      // Production, decode T=1.
      {"prod_tp1_decode_t1",    1, 48, 16, 1024, 48*128 + 2*4*128 + 48*16, false, relh::kTauPreToken, 1100.0},
      {"prod_tp2_decode_t1",    1, 24, 16, 1024, 24*128 + 2*2*128 + 24*16, false, relh::kTauPreToken, 610.0},
      {"prod_tp4_decode_t1",    1, 12, 16, 1024, 12*128 + 2*1*128 + 12*16, false, relh::kTauPreToken, 350.0},
      {"prod_tp8_decode_t1",    1,  6, 16, 1024,  6*128 + 2*1*128 +  6*16, false, relh::kTauPreToken, 190.0},
      // Production, target-verify T=9.
      {"prod_tp2_verify_t9",    9, 24, 16, 1024, 24*128 + 2*2*128 + 24*16, false, relh::kTauPreToken, 2700.0},
      {"prod_tp4_verify_t9",    9, 12, 16, 1024, 12*128 + 2*1*128 + 12*16, false, relh::kTauPreToken, 2050.0},
      {"prod_tp8_verify_t9",    9,  6, 16, 1024,  6*128 + 2*1*128 +  6*16, false, relh::kTauPreToken, 1200.0},
      // Production, small-t band max T=32.
      {"prod_tp2_extend_t32",  32, 24, 16, 1024, 24*128 + 2*2*128 + 24*16, false, relh::kTauPreToken, 3150.0},
      {"prod_tp4_extend_t32",  32, 12, 16, 1024, 12*128 + 2*1*128 + 12*16, false, relh::kTauPreToken, 2700.0},
      {"prod_tp8_extend_t32",  32,  6, 16, 1024,  6*128 + 2*1*128 +  6*16, false, relh::kTauPreToken, 2550.0},

      // Production local attention: local_extent = sliding_window_size = 512.
      {"prod_tp4_local_decode_t1",  1, 12, 16, 512, 12*128 + 2*1*128 + 12*16, false, relh::kTauPreToken, 185.0},
      {"prod_tp4_local_verify_t9",  9, 12, 16, 512, 12*128 + 2*1*128 + 12*16, false, relh::kTauPreToken, 1250.0},
      {"prod_tp4_local_extend_t32", 32, 12, 16, 512, 12*128 + 2*1*128 + 12*16, false, relh::kTauPreToken, 2200.0},

      // Tau OFF path still lands in the kernel when the caller opts out of
      // fused-tau (SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU=0); cover it at prod TP=4.
      {"prod_tp4_no_tau_t9",    9, 12, 16, 1024, 12*128 + 2*1*128 + 12*16, false, relh::kTauNone,     2350.0},
  };
}

std::vector<relh::RelProjCase> perf_suite() {
  return {
      {"perf_t1_h16_d16_e1024", 1, 16, 16, 1024, 256, false, relh::kTauPreToken, 440.0},
      {"perf_t16_h16_d16_e1024", 16, 16, 16, 1024, 256, false, relh::kTauPreToken, 2950.0},
      {"perf_t32_h16_d16_e1024", 32, 16, 16, 1024, 256, false, relh::kTauPreToken, 2900.0},

      // Production per-rank shapes at T=32 (the tau-fused band's max T).
      {"perf_prod_tp2_t32", 32, 24, 16, 1024, 24*128 + 2*2*128 + 24*16, false, relh::kTauPreToken, 3150.0},
      {"perf_prod_tp4_t32", 32, 12, 16, 1024, 12*128 + 2*1*128 + 12*16, false, relh::kTauPreToken, 2700.0},
      {"perf_prod_tp8_t32", 32,  6, 16, 1024,  6*128 + 2*1*128 +  6*16, false, relh::kTauPreToken, 2500.0},
      {"perf_prod_tp1_t32", 32, 48, 16, 1024, 48*128 + 2*4*128 + 48*16, false, relh::kTauPreToken, 3400.0},
      {"perf_prod_tp1_local_t32", 32, 48, 16, 512, 48*128 + 2*4*128 + 48*16, false, relh::kTauPreToken, 3150.0},
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
