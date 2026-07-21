/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "17_bmg_relative_attention_common.hpp"

namespace rel = cutlass::examples::relative_attention;

namespace {

rel::AttentionCase base_case() {
  rel::AttentionCase cfg;
  cfg.name = "custom_sliding_window";
  cfg.batch = 2;
  cfg.max_seq_len = 64;
  cfg.heads = 8;
  cfg.kv_heads = 4;
  cfg.d = 64;
  cfg.dv = 64;
  cfg.window_left = 32;
  cfg.window_right = 0;
  cfg.use_window = true;
  cfg.causal = true;
  return cfg;
}

std::vector<rel::AttentionCase> quick_suite() {
  return {
      {"tiny_reference_04_2", 1, 5, 2, 2, 3, 4, 0, 0, 0, 0, 0, 2, 0, 0.0f,
       false, true, true, false, false, false, 0.0},
      {"irregular_window_tail", 5, 29, 8, 2, 64, 33, 0, 2, 1, 4, 3, 7, 0, 0.0f,
       false, true, true, true, false, true, 0.0},
      {"decode_long_window", 16, 257, 16, 4, 96, 64, 0, 0, 0, 0, 0, 128, 0, 0.0f,
       false, true, true, true, true, true, 0.0},
      {"bidirectional_local", 2, 31, 4, 4, 40, 24, 0, 0, 0, 0, 0, 3, 2, 6.0f,
       false, true, false, true, false, false, 0.0},
  };
}

// Local/SWA layer (InklingAttention with is_local=true) at real per-rank
// shapes. sliding_window_size defaults to 512 -> window_left = 511
// (RadixAttention passes ``sliding_window_size - 1`` per pr31358_config).
// GQA head split follows the full-attention layer above:
//   config defaults: heads=12, kv=4  |  production: heads=48, kv=4
// TP=8 skipped for config defaults (12 % 8 != 0).
std::vector<rel::AttentionCase> inkling_suite() {
  return {
      // Legacy scale-only cases (kept for parity with earlier CI).
      {"inkling_window_128", 8, 128, 16, 4, 128, 128, 0, 0, 0, 0, 0, 64, 0, 0.0f,
       false, true, true, true, false, false, 0.0},
      {"inkling_window_decode", 64, 1024, 32, 4, 128, 128, 0, 0, 0, 0, 0, 256, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"inkling_window_tail_dims", 6, 97, 12, 4, 80, 96, 0, 1, 2, 3, 4, 31, 0, 0.0f,
       false, true, true, true, false, true, 0.0},

      // Config-defaults (h=12, kv=4) extend/prefill across TP=1/2/4 with the
      // real 512-token sliding window (backend passes size-1 = 511).
      {"inkling_cfg_tp1_win_extend", 2, 640, 12, 4, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, false, false, 0.0},
      {"inkling_cfg_tp2_win_extend", 2, 640, 6, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, false, false, 0.0},
      {"inkling_cfg_tp4_win_extend", 2, 640, 3, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, false, false, 0.0},

      // Production (h=48, kv=4) extend/prefill across TP=2/4/8.
      {"inkling_prod_tp2_win_extend", 2, 640, 24, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, false, false, 0.0},
      {"inkling_prod_tp4_win_extend", 2, 640, 12, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, false, false, 0.0},
      {"inkling_prod_tp8_win_extend", 2, 640, 6, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, false, false, 0.0},

      // Config-defaults decode across TP=2/4 (kv-cache seqlen>window so the
      // window-clipping path fires).
      {"inkling_cfg_tp2_win_decode", 24, 1024, 6, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"inkling_cfg_tp4_win_decode", 24, 1024, 3, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},

      // Production decode across TP=2/4/8.
      {"inkling_prod_tp2_win_decode", 24, 1024, 24, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"inkling_prod_tp4_win_decode", 24, 1024, 12, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"inkling_prod_tp8_win_decode", 24, 1024, 6, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
  };
}

std::vector<rel::AttentionCase> perf_suite() {
  // Perf gates left at 0 (report-only) for the newly added TP shapes until they
  // are calibrated on BMG; the two legacy 350 GB/s gates stay in place.
  return {
      {"perf_window_4k", 8, 512, 16, 4, 128, 128, 0, 0, 0, 0, 0, 256, 0, 0.0f,
       false, true, true, false, false, false, 350.0},
      {"perf_decode_window_8k", 128, 2048, 32, 4, 128, 128, 0, 0, 0, 0, 0, 512, 0, 0.0f,
       false, true, true, true, true, false, 350.0},

      // Chunked-prefill sized extend for the local layer.
      {"perf_cfg_tp2_win_extend_4k", 1, 4096, 6, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, false, false, false, 0.0},
      {"perf_cfg_tp4_win_extend_4k", 1, 4096, 3, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, false, false, false, 0.0},
      {"perf_prod_tp2_win_extend_4k", 1, 4096, 24, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, false, false, false, 0.0},
      {"perf_prod_tp4_win_extend_4k", 1, 4096, 12, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, false, false, false, 0.0},
      {"perf_prod_tp8_win_extend_4k", 1, 4096, 6, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, false, false, false, 0.0},

      // Decode at production batch/context.
      {"perf_cfg_tp2_win_decode_b128", 128, 2048, 6, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"perf_cfg_tp4_win_decode_b128", 128, 2048, 3, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"perf_prod_tp2_win_decode_b128", 128, 2048, 24, 2, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"perf_prod_tp4_win_decode_b128", 128, 2048, 12, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"perf_prod_tp8_win_decode_b128", 128, 2048, 6, 1, 128, 128, 0, 0, 0, 0, 0, 511, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
  };
}

std::vector<rel::AttentionCase> make_suite(std::string const& suite) {
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
  return rel::run_suite(
      argc,
      argv,
      "quick|inkling|perf",
      base_case(),
      make_suite,
      "17_bmg_xpu_flash_attention_window: causal/local-window attention backend");
}
