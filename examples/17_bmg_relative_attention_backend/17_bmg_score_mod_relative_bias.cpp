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

// Inkling full-attention layer (InklingAttention with is_local=false) at real
// per-rank shapes. Config-defaults hidden_size=1536 -> num_heads=12; production
// hidden_size=6144 -> num_heads=48. Both use num_kv_heads=4, head_dim=128,
// d_rel=16, rel_extent=1024 (upstream_model_attn.py:180 + config defaults).
// After TP split: num_tp_heads = num_heads / tp,
// num_tp_kv_heads = max(1, num_kv_heads / tp).
// (Config defaults + TP=8 skipped: num_heads=12 not divisible by 8.)
std::vector<rel::AttentionCase> inkling_suite() {
  return {
      // Legacy scale-only cases (kept for parity with earlier CI).
      {"inkling_extend_h16", 8, 96, 16, 4, 128, 128, 128, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},
      {"inkling_decode_h32", 32, 512, 32, 4, 128, 128, 256, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"inkling_tail_dims", 6, 79, 12, 4, 96, 80, 64, 3, 5, 7, 9, -1, -1, 0.0f,
       true, false, true, true, false, true, 0.0},

      // Config-defaults (h=12, kv=4) extend/prefill across TP=1/2/4.
      {"inkling_cfg_tp1_extend", 2, 320, 12, 4, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},
      {"inkling_cfg_tp2_extend", 2, 320, 6, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},
      {"inkling_cfg_tp4_extend", 2, 320, 3, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},

      // Production (h=48, kv=4) extend/prefill across TP=2/4/8.
      {"inkling_prod_tp2_extend", 2, 256, 24, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},
      {"inkling_prod_tp4_extend", 2, 256, 12, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},
      {"inkling_prod_tp8_extend", 2, 256, 6, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, false, false, 0.0},

      // Config-defaults decode across TP=2/4 (q_len=1 per seq).
      {"inkling_cfg_tp2_decode", 24, 640, 6, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"inkling_cfg_tp4_decode", 24, 640, 3, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},

      // Production decode across TP=4/8. (TP=2 h=24 covered above; kv broadcast
      // dominates decode when kv=1, so TP=4/8 is where GQA scheduling matters.)
      {"inkling_prod_tp2_decode", 24, 640, 24, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"inkling_prod_tp4_decode", 24, 640, 12, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"inkling_prod_tp8_decode", 24, 640, 6, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
  };
}

std::vector<rel::AttentionCase> perf_suite() {
  // Perf gates left at 0 (report-only) for the newly added TP shapes until they
  // are calibrated on BMG; the two legacy gates (350 GB/s) stay in place.
  return {
      {"perf_extend_4k", 8, 512, 16, 4, 128, 128, 256, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 350.0},
      {"perf_decode_b256", 256, 1024, 32, 4, 128, 128, 512, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 350.0},

      // Chunked-prefill sized extend (max_prefill_tokens=16384 -> per-chunk
      // 4k tokens is a common working point) at real per-rank shapes.
      {"perf_cfg_tp2_extend_4k", 1, 4096, 6, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 0.0},
      {"perf_cfg_tp4_extend_4k", 1, 4096, 3, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 0.0},
      {"perf_prod_tp2_extend_4k", 1, 4096, 24, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 0.0},
      {"perf_prod_tp4_extend_4k", 1, 4096, 12, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 0.0},
      {"perf_prod_tp8_extend_4k", 1, 4096, 6, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, false, false, false, 0.0},

      // Decode at production batch/context (batch=128, kv-cache seqlen=2k).
      {"perf_cfg_tp2_decode_b128", 128, 2048, 6, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"perf_cfg_tp4_decode_b128", 128, 2048, 3, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"perf_prod_tp2_decode_b128", 128, 2048, 24, 2, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"perf_prod_tp4_decode_b128", 128, 2048, 12, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
      {"perf_prod_tp8_decode_b128", 128, 2048, 6, 1, 128, 128, 1024, 0, 0, 0, 0, -1, -1, 0.0f,
       true, false, true, true, true, false, 0.0},
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
