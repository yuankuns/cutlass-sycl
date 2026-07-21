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

// row_scale_bf16 is Inkling's per-token log-scaling helper (log_scaling_tau.py).
// It runs in two positions:
//   (a) prescale the r operand of RelLogitsProj outside the small-t kernel
//       band (T > 32). Shape: rows=T, inner = h_tp*d_rel, stride = qkvr row
//       width (h_tp*head_dim + 2*kv_tp*head_dim + h_tp*d_rel).
//   (b) post-scale the projected rel logits when the fused-tau flag is off.
//       Shape: rows=T, inner = h_tp*rel_extent, contiguous.
// Configs (see [[inkling_model_shapes]]):
//   config defaults hidden=1536 -> h=12, kv=4  |  h_tp = 12/6/3 (TP=1/2/4)
//   production       hidden=6144 -> h=48, kv=4  |  h_tp = 48/24/12/6 (TP=1/2/4/8)
// A prefill chunk in Inkling caps at max_prefill_tokens=16384, so pick T at the
// upper end of the "t > 32" band (T = 16384) plus a mid-band T = 2048.
std::vector<relh::RowCase> inkling_suite() {
  return {
      // (a) prescale r-operand path (strided from qkvr), T just above small-t band.
      {"cfg_tp2_rop_t64",    64,  6*16,  6*128 + 2*2*128 +  6*16, 0.0},
      {"cfg_tp4_rop_t64",    64,  3*16,  3*128 + 2*1*128 +  3*16, 0.0},
      {"prod_tp2_rop_t64",   64, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"prod_tp4_rop_t64",   64, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"prod_tp8_rop_t64",   64,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},

      // (a) prescale r-operand at prefill mid-chunk T = 2048.
      {"cfg_tp2_rop_t2k",   2048,  6*16,  6*128 + 2*2*128 +  6*16, 0.0},
      {"prod_tp2_rop_t2k",  2048, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"prod_tp4_rop_t2k",  2048, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"prod_tp8_rop_t2k",  2048,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},

      // (a) prescale r-operand at the max_prefill_tokens=16384 chunk cap.
      {"prod_tp2_rop_t16k",16384, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"prod_tp4_rop_t16k",16384, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"prod_tp8_rop_t16k",16384,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},

      // (b) post-scale rel-logits path, contiguous inner = h_tp * rel_extent.
      // Cover TP=2/4/8 at decode T=1, target-verify T=9, and prefill T=2048.
      {"cfg_tp2_post_t1",       1,  6*1024,  6*1024, 0.0},
      {"cfg_tp4_post_t9",       9,  3*1024,  3*1024, 0.0},
      {"prod_tp2_post_t1",      1, 24*1024, 24*1024, 0.0},
      {"prod_tp4_post_t9",      9, 12*1024, 12*1024, 0.0},
      {"prod_tp8_post_t9",      9,  6*1024,  6*1024, 0.0},
      {"prod_tp2_post_t2k",  2048, 24*1024, 24*1024, 0.0},
      {"prod_tp4_post_t2k",  2048, 12*1024, 12*1024, 0.0},
      {"prod_tp8_post_t2k",  2048,  6*1024,  6*1024, 0.0},

      // Local-layer variant: rel_extent = local_extent = sliding_window_size = 512.
      {"prod_tp4_local_post_t2k", 2048, 12*512, 12*512, 0.0},
  };
}

std::vector<relh::RowCase> perf_suite() {
  return {
      {"perf_64k_x256", 65536, 256, 256, 350.0},
      {"perf_64k_x256_strided", 65536, 256, 320, 340.0},
      {"perf_32k_x1024", 32768, 1024, 1024, 350.0},

      // Post-scale rel-logits at production T = 16384 across TP=2/4/8.
      {"perf_prod_tp2_post_t16k", 16384, 24*1024, 24*1024, 0.0},
      {"perf_prod_tp4_post_t16k", 16384, 12*1024, 12*1024, 0.0},
      {"perf_prod_tp8_post_t16k", 16384,  6*1024,  6*1024, 0.0},

      // Prescale r-operand at production T = 16384 across TP=2/4/8 (strided).
      {"perf_prod_tp2_rop_t16k", 16384, 24*16, 24*128 + 2*2*128 + 24*16, 0.0},
      {"perf_prod_tp4_rop_t16k", 16384, 12*16, 12*128 + 2*1*128 + 12*16, 0.0},
      {"perf_prod_tp8_rop_t16k", 16384,  6*16,  6*128 + 2*1*128 +  6*16, 0.0},
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
