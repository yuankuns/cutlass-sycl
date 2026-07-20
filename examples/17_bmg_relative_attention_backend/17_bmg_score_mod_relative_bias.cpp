/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "17_bmg_relative_attention_common.hpp"

namespace rel = cutlass::examples::relative_attention;

namespace {

rel::AttentionCase base_case() {
  rel::AttentionCase cfg;
  cfg.name = "custom_relative_bias";
  cfg.batch = 2;
  cfg.max_seq_len = 32;
  cfg.heads = 8;
  cfg.kv_heads = 4;
  cfg.d = 64;
  cfg.dv = 64;
  cfg.rel_len = 64;
  cfg.use_relative_bias = true;
  cfg.causal = true;
  return cfg;
}

std::vector<rel::AttentionCase> quick_suite() {
  return {
      {"tiny_reference_04_1", 1, 4, 2, 2, 3, 5, 4, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 0.0},
      {"irregular_gqa_tail", 4, 17, 6, 2, 63, 19, 9, 1, 2, 3, 4, -1, -1, 0.0f,
       true, false, true, true, false, true, 0.0},
      {"decode_tail_rel", 8, 65, 8, 2, 64, 32, 32, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, true, 0.0},
      {"softcap_noncausal", 2, 23, 4, 4, 48, 24, 12, 0, 1, 2, 0, -1, -1, 8.0f,
       true, false, false, true, false, false, 0.0},
  };
}

std::vector<rel::AttentionCase> inkling_suite() {
  return {
      {"inkling_extend_h16", 8, 96, 16, 4, 128, 128, 128, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},
      {"inkling_decode_h32", 32, 512, 32, 4, 128, 128, 256, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"inkling_tail_dims", 6, 79, 12, 4, 96, 80, 64, 3, 5, 7, 9, -1, -1, 0.0f,
       true, false, true, true, false, true, 0.0},
  };
}

std::vector<rel::AttentionCase> perf_suite() {
  return {
      {"perf_extend_4k", 8, 512, 16, 4, 128, 128, 256, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 350.0},
      {"perf_decode_b256", 256, 1024, 32, 4, 128, 128, 512, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 350.0},
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
      "17_bmg_score_mod_relative_bias: relative-bias score_mod attention");
}
