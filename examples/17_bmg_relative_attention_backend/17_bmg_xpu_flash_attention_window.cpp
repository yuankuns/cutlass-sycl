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

std::vector<rel::AttentionCase> inkling_suite() {
  return {
      {"inkling_window_128", 8, 128, 16, 4, 128, 128, 0, 0, 0, 0, 0, 64, 0, 0.0f,
       false, true, true, true, false, false, 0.0},
      {"inkling_window_decode", 64, 1024, 32, 4, 128, 128, 0, 0, 0, 0, 0, 256, 0, 0.0f,
       false, true, true, true, true, false, 0.0},
      {"inkling_window_tail_dims", 6, 97, 12, 4, 80, 96, 0, 1, 2, 3, 4, 31, 0, 0.0f,
       false, true, true, true, false, true, 0.0},
  };
}

std::vector<rel::AttentionCase> perf_suite() {
  return {
      {"perf_window_4k", 8, 512, 16, 4, 128, 128, 0, 0, 0, 0, 0, 256, 0, 0.0f,
       false, true, true, false, false, false, 350.0},
      {"perf_decode_window_8k", 128, 2048, 32, 4, 128, 128, 0, 0, 0, 0, 0, 512, 0, 0.0f,
       false, true, true, true, true, false, 350.0},
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
