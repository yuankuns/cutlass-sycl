/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "18_bmg_relative_helpers_common.hpp"

namespace relh = cutlass::examples::relative_helpers;

namespace {

relh::RowCase custom_default() {
  relh::RowCase cfg;
  cfg.name = "custom_row_compact";
  cfg.rows = 5;
  cfg.inner = 4;
  cfg.stride = 7;
  return cfg;
}

std::vector<relh::RowCase> quick_suite() {
  return {
      {"tiny_reference_05_3_copy", 5, 4, 7, 0.0},
      {"tail_inner_19", 41, 19, 31, 0.0},
      {"aligned_decode_r", 32, 256, 256, 0.0},
      {"strided_prefill_r", 1024, 256, 320, 0.0},
  };
}

// row_compact_bf16 is Inkling's strided-to-contiguous helper on the r operand
// of RelLogitsProj._project (attn.py:158-161). It fires when T > 48 (past the
// zero-copy strided GEMM band) so the following einsum runs on a contiguous r.
// Shape: rows=T, inner = h_tp * d_rel, stride = qkvr row width
//   = h_tp*head_dim + 2*kv_tp*head_dim + h_tp*d_rel.
// Real T floors: T > 48 → cover 64 (band entry), 2048 (mid-prefill),
// 16384 (max_prefill_tokens). Per-rank shapes at both shipped configs.
std::vector<relh::RowCase> inkling_suite() {
  return {
      // Config defaults (hidden=1536, h=12, kv=4).
      {"cfg_tp2_t64",     64,  6*16,  6*128 + 2*2*128 +  6*16, 0.0},
      {"cfg_tp4_t64",     64,  3*16,  3*128 + 2*1*128 +  3*16, 0.0},
      {"cfg_tp2_t2k",   2048,  6*16,  6*128 + 2*2*128 +  6*16, 0.0},
      {"cfg_tp4_t2k",   2048,  3*16,  3*128 + 2*1*128 +  3*16, 0.0},
      {"cfg_tp2_t16k", 16384,  6*16,  6*128 + 2*2*128 +  6*16, 0.0},
      {"cfg_tp4_t16k", 16384,  3*16,  3*128 + 2*1*128 +  3*16, 0.0},

      // Production (hidden=6144, h=48, kv=4).
      {"prod_tp2_t64",    64, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"prod_tp4_t64",    64, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"prod_tp8_t64",    64,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},
      {"prod_tp2_t2k",  2048, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"prod_tp4_t2k",  2048, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"prod_tp8_t2k",  2048,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},
      {"prod_tp2_t16k",16384, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"prod_tp4_t16k",16384, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"prod_tp8_t16k",16384,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},
  };
}

std::vector<relh::RowCase> perf_suite() {
  return {
      {"perf_64k_x256", 65536, 256, 256, 350.0},
      {"perf_64k_x256_strided", 65536, 256, 320, 340.0},
      {"perf_32k_x1024", 32768, 1024, 1024, 350.0},

      // Per-rank strided-to-contiguous compact at production T = 16384.
      {"perf_prod_tp2_t16k", 16384, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"perf_prod_tp4_t16k", 16384, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"perf_prod_tp8_t16k", 16384,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},
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
      std::cout << "18_bmg_row_compact_bf16: strided-row to contiguous-row compact helper\n\n";
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
    std::cout << "18_bmg_row_compact_bf16: contiguous copy of strided rows\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << relh::dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << relh::bool_text(options.verify)
              << " benchmark=" << relh::bool_text(options.benchmark) << "\n";

    bool all_passed = true;
    if (options.dtype == relh::DType::kAll || options.dtype == relh::DType::kBf16) {
      all_passed &= relh::run_row_cases_for_dtype<cutlass::bfloat16_t, false>(queue, cases, options, "compact");
    }
    if (options.dtype == relh::DType::kAll || options.dtype == relh::DType::kFp16) {
      all_passed &= relh::run_row_cases_for_dtype<cutlass::half_t, false>(queue, cases, options, "compact");
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
