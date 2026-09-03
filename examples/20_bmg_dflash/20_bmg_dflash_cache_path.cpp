/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "20_bmg_dflash_common.hpp"

#include <oneapi/mkl/blas.hpp>

namespace dflash = cutlass::examples::bmg_dflash;

namespace {

// dflash::patterned_value() only mixes the low 16 bits of the index, so weight
// rows built from it repeat exactly every 256 vocab entries. That is harmless
// for the legacy cases (their +v*0.1 ramp forces argmax == V-1) but it makes an
// honest argmax reference ambiguous, so the production-vocab cases use this
// splitmix64-style hash instead. Range is [-0.75, 0.75), matching
// patterned_value() so magnitudes stay comparable.
inline float hash_uniform(uint64_t index, uint64_t salt) {
  uint64_t x = index * 0x9E3779B97F4A7C15ull + salt * 0xBF58476D1CE4E5B9ull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return (static_cast<float>(x >> 40) / 8388608.0f - 1.0f) * 0.75f;
}

// logits_mup_width_multiplier: InklingCausalLLM.forward() divides the trunk
// hidden state by config.logits_mup_width_multiplier (3.0 in the shipped
// thinkingmachines/Inkling checkpoint) before the LM head, and the MTP chain
// does the same in InklingMTP. Greedy argmax is invariant under a positive
// scalar, so this cannot change the sampled token, but the model materializes
// the divided [tokens, hidden] tensor, so the example models it as its own
// elementwise pass (rather than folding 1/mup into the GEMM alpha) to keep the
// launch sequence and the memory traffic one-for-one with the real op.
// The same tail also applies final_logit_softcapping when it is set; it is
// null in the shipped checkpoint, so no softcap pass is modeled here.
struct MupDivideParams {
  float const* in = nullptr;
  float* out = nullptr;
  int64_t count = 0;
  float divisor = 1.0f;
};

class MupDivideKernel {
 public:
  explicit MupDivideKernel(MupDivideParams params) : params_(params) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t i = static_cast<int64_t>(item.get_global_id(0));
    if (i >= params_.count) {
      return;
    }
    params_.out[i] = params_.in[i] / params_.divisor;
  }

 private:
  MupDivideParams params_;
};

sycl::event launch_mup_divide(sycl::queue& queue, MupDivideParams const& params) {
  if (params.count <= 0 || params.out == nullptr) {
    return {};
  }
  std::size_t rounded =
      static_cast<std::size_t>(dflash::ceil_div(params.count, dflash::kThreads)) * dflash::kThreads;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(rounded), sycl::range<1>(dflash::kThreads)),
        MupDivideKernel(params));
  });
}

struct MaskedGatherParams {
  int64_t const* req_to_token = nullptr;
  int64_t const* req_pool_indices = nullptr;
  int64_t const* pos2d = nullptr;
  uint8_t const* mask = nullptr;
  int32_t const* out_offsets = nullptr;
  int64_t* out = nullptr;
  int batch = 0;
  int draft_tokens = 0;
  int table_width = 0;
};

class MaskedGatherKernel {
 public:
  explicit MaskedGatherKernel(MaskedGatherParams params) : params_(params) {}

  void operator()(sycl::id<1> id) const {
    int lane = static_cast<int>(id[0]);
    int total = params_.batch * params_.draft_tokens;
    if (lane >= total || params_.mask[lane] == 0) {
      return;
    }
    int b = lane / params_.draft_tokens;
    int64_t req_slot = params_.req_pool_indices[b];
    int64_t pos = params_.pos2d[lane];
    int32_t out_idx = params_.out_offsets[lane];
    params_.out[out_idx] = params_.req_to_token[req_slot * params_.table_width + pos];
  }

 private:
  MaskedGatherParams params_;
};

sycl::event launch_masked_gather(sycl::queue& queue, MaskedGatherParams const& params) {
  int total = params.batch * params.draft_tokens;
  if (total <= 0) {
    return {};
  }
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), MaskedGatherKernel(params));
  });
}

struct GreedyArgmaxParams {
  float const* hidden = nullptr;
  float const* weight = nullptr;
  float* partial_values = nullptr;
  int32_t* partial_indices = nullptr;
  int64_t* out_tokens = nullptr;
  int tokens = 0;
  int hidden_dim = 0;
  int vocab = 0;
  int vocab_tiles = 0;
};

template <int RowsPerGroup, int SubGroupSize>
class GreedyArgmaxTileKernel {
 public:
  GreedyArgmaxTileKernel(GreedyArgmaxParams params,
                         sycl::local_accessor<float, 1> tile_values,
                         sycl::local_accessor<int32_t, 1> tile_indices,
                         sycl::local_accessor<float, 1> hidden_cache)
      : params_(params), tile_values_(tile_values), tile_indices_(tile_indices), hidden_cache_(hidden_cache) {}

  [[sycl::reqd_sub_group_size(SubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();
    int lane = static_cast<int>(sg.get_local_id());
    int row_in_tile = static_cast<int>(sg.get_group_id());
    int group = static_cast<int>(item.get_group(0));
    int token = group / params_.vocab_tiles;
    int tile = group - token * params_.vocab_tiles;
    int vocab_row = tile * RowsPerGroup + row_in_tile;
    bool valid = token < params_.tokens && vocab_row < params_.vocab;
    int local_id = static_cast<int>(item.get_local_id(0));
    int local_size = static_cast<int>(item.get_local_range(0));

    if (token < params_.tokens) {
      float const* hidden_row = params_.hidden + static_cast<int64_t>(token) * params_.hidden_dim;
      for (int h = local_id; h < params_.hidden_dim; h += local_size) {
        hidden_cache_[h] = hidden_row[h];
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    float acc = 0.0f;
    if (valid) {
      float const* weight_row = params_.weight + static_cast<int64_t>(vocab_row) * params_.hidden_dim;
      for (int h = lane; h < params_.hidden_dim; h += SubGroupSize) {
        acc = sycl::fma(hidden_cache_[h], weight_row[h], acc);
      }
    }

    float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) {
      tile_values_[row_in_tile] = valid ? sum : -3.4028234663852886e38f;
      tile_indices_[row_in_tile] = valid ? vocab_row : INT32_MAX;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (item.get_local_id(0) == 0) {
      float best = -3.4028234663852886e38f;
      int32_t best_idx = INT32_MAX;
#pragma unroll
      for (int i = 0; i < RowsPerGroup; ++i) {
        float value = tile_values_[i];
        int32_t idx = tile_indices_[i];
        if (idx < params_.vocab && (value > best || (value == best && idx < best_idx))) {
          best = value;
          best_idx = idx;
        }
      }
      int partial_offset = token * params_.vocab_tiles + tile;
      params_.partial_values[partial_offset] = best;
      params_.partial_indices[partial_offset] = best_idx;
    }
  }

 private:
  GreedyArgmaxParams params_;
  sycl::local_accessor<float, 1> tile_values_;
  sycl::local_accessor<int32_t, 1> tile_indices_;
  sycl::local_accessor<float, 1> hidden_cache_;
};

class GreedyArgmaxFinalizeKernel {
 public:
  explicit GreedyArgmaxFinalizeKernel(GreedyArgmaxParams params) : params_(params) {}

  void operator()(sycl::id<1> id) const {
    int token = static_cast<int>(id[0]);
    if (token >= params_.tokens) {
      return;
    }
    float best = -3.4028234663852886e38f;
    int32_t best_idx = 0;
    int base = token * params_.vocab_tiles;
    for (int tile = 0; tile < params_.vocab_tiles; ++tile) {
      float value = params_.partial_values[base + tile];
      int32_t idx = params_.partial_indices[base + tile];
      if (idx < params_.vocab && (value > best || (value == best && idx < best_idx))) {
        best = value;
        best_idx = idx;
      }
    }
    params_.out_tokens[token] = static_cast<int64_t>(best_idx);
  }

 private:
  GreedyArgmaxParams params_;
};

sycl::event launch_greedy_argmax(sycl::queue& queue, GreedyArgmaxParams params) {
  if (params.tokens <= 0 || params.vocab <= 0 || params.hidden_dim <= 0) {
    return {};
  }
  params.vocab_tiles = static_cast<int>(dflash::ceil_div(params.vocab, dflash::kGreedyRowsPerGroup));
  sycl::event tile_event = queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> tile_values(dflash::kGreedyRowsPerGroup, cgh);
    sycl::local_accessor<int32_t, 1> tile_indices(dflash::kGreedyRowsPerGroup, cgh);
    sycl::local_accessor<float, 1> hidden_cache(static_cast<std::size_t>(params.hidden_dim), cgh);
    sycl::range<1> local(static_cast<std::size_t>(dflash::kGreedyRowsPerGroup * dflash::kSubGroup));
    sycl::range<1> global(static_cast<std::size_t>(params.tokens * params.vocab_tiles) * local[0]);
    GreedyArgmaxTileKernel<dflash::kGreedyRowsPerGroup, dflash::kSubGroup> kernel(
        params, tile_values, tile_indices, hidden_cache);
    cgh.parallel_for(sycl::nd_range<1>(global, local), kernel);
  });
  (void)tile_event;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(params.tokens)), GreedyArgmaxFinalizeKernel(params));
  });
}

struct LogitsArgmaxParams {
  float const* logits = nullptr;
  int64_t* out_tokens = nullptr;
  int tokens = 0;
  int vocab = 0;
};

class LogitsArgmaxKernel {
 public:
  LogitsArgmaxKernel(LogitsArgmaxParams params,
                     sycl::local_accessor<float, 1> values,
                     sycl::local_accessor<int32_t, 1> indices)
      : params_(params), values_(values), indices_(indices) {}

  void operator()(sycl::nd_item<1> item) const {
    int token = static_cast<int>(item.get_group(0));
    int tid = static_cast<int>(item.get_local_id(0));
    int local = static_cast<int>(item.get_local_range(0));
    float best = -3.4028234663852886e38f;
    int32_t best_idx = 0;

    float const* row = params_.logits + static_cast<int64_t>(token) * params_.vocab;
    for (int v = tid; v < params_.vocab; v += local) {
      float value = row[v];
      if (value > best || (value == best && v < best_idx)) {
        best = value;
        best_idx = v;
      }
    }
    values_[tid] = best;
    indices_[tid] = best_idx;
    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = local / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        float other = values_[tid + stride];
        int32_t other_idx = indices_[tid + stride];
        if (other > values_[tid] || (other == values_[tid] && other_idx < indices_[tid])) {
          values_[tid] = other;
          indices_[tid] = other_idx;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (tid == 0) {
      params_.out_tokens[token] = static_cast<int64_t>(indices_[0]);
    }
  }

 private:
  LogitsArgmaxParams params_;
  sycl::local_accessor<float, 1> values_;
  sycl::local_accessor<int32_t, 1> indices_;
};

sycl::event launch_logits_argmax(sycl::queue& queue, LogitsArgmaxParams const& params) {
  if (params.tokens <= 0 || params.vocab <= 0) {
    return {};
  }
  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> values(dflash::kThreads, cgh);
    sycl::local_accessor<int32_t, 1> indices(dflash::kThreads, cgh);
    sycl::range<1> local(dflash::kThreads);
    sycl::range<1> global(static_cast<std::size_t>(params.tokens) * dflash::kThreads);
    cgh.parallel_for(sycl::nd_range<1>(global, local), LogitsArgmaxKernel(params, values, indices));
  });
}

sycl::event launch_gemm_greedy_argmax(
    sycl::queue& queue,
    float const* hidden,
    float const* weight,
    float* logits,
    int64_t* out_tokens,
    int tokens,
    int hidden_dim,
    int vocab) {
  if (tokens <= 0 || hidden_dim <= 0 || vocab <= 0) {
    return {};
  }
  oneapi::mkl::blas::row_major::gemm(
      queue,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans,
      tokens,
      vocab,
      hidden_dim,
      1.0f,
      hidden,
      hidden_dim,
      weight,
      hidden_dim,
      0.0f,
      logits,
      vocab);

  LogitsArgmaxParams params;
  params.logits = logits;
  params.out_tokens = out_tokens;
  params.tokens = tokens;
  params.vocab = vocab;
  return launch_logits_argmax(queue, params);
}

struct CachePathCase {
  std::string name;
  int batch = 2;
  int draft_tokens = 3;
  int req_rows = 3;
  int table_width = 10;
  int tokens = 4;
  int hidden_dim = 5;
  int vocab = 7;
  double target_tops = 0.0;
  // logits_mup_width_multiplier; <= 0.0 models the null (unset) config value.
  double mup = 0.0;
  // Greedy-argmax reference budget: -1 = full CPU reference over every row,
  // 0 = benchmark only (gather is still verified), N > 0 = CPU reference over N
  // deterministically sampled rows. The CPU reference is O(rows*V*H), so the
  // production vocab shards (V up to 201024) cannot afford the full form.
  int verify_rows = -1;
  // true: hash_uniform() weights with a planted per-row maximum (see
  // initialize_case) so the argmax is well separated and row-dependent.
  // false: the legacy +v*0.1 vocab ramp, whose argmax is always V-1.
  bool planted = false;
};

struct CachePathHost {
  std::vector<int64_t> req_to_token;
  std::vector<int64_t> req_pool_indices;
  std::vector<int64_t> pos2d;
  std::vector<uint8_t> mask;
  std::vector<int32_t> out_offsets;
  std::vector<int64_t> gathered_ref;
  std::vector<float> hidden;
  std::vector<float> weight;
  // Parallel arrays: greedy_ref[i] is the reference token for row
  // verify_row_ids[i]. Empty when the case is benchmark-only.
  std::vector<int> verify_row_ids;
  std::vector<int64_t> greedy_ref;
};

CachePathCase custom_default() {
  CachePathCase cfg;
  cfg.name = "custom_cache_path";
  return cfg;
}

std::vector<CachePathCase> quick_suite() {
  return {
      {"reference_small_b2_t3_h5_v7", 2, 3, 3, 10, 4, 5, 7, 0.0},
      {"tail_vocab_b5_t9_h33_v79", 5, 9, 8, 31, 17, 33, 79, 0.0},
      {"nondivisible_h97_v257", 7, 9, 11, 73, 23, 97, 257, 0.0},
  };
}

std::vector<CachePathCase> stress_suite() {
  return {
      {"emptyish_masks_h17_v31", 13, 9, 17, 101, 29, 17, 31, 0.0},
      {"prime_dims_h193_v1009", 16, 9, 23, 257, 64, 193, 1009, 0.0},
      {"draft_verify_h384_v2048", 32, 9, 40, 2048, 128, 384, 2048, 0.0},
  };
}

std::vector<CachePathCase> perf_suite() {
  return {
      {"perf_draft_h1536_v8192", 128, 9, 160, 8192, 512, 1536, 8192, 1.0},
      {"perf_prod_shard_h768_v16384", 256, 9, 320, 8192, 1024, 768, 16384, 1.0},
      // Real ParallelLMHead geometry: padded_vocab_size=201024 unsharded, and
      // the TP=8 shard at the production trunk width. Benchmark only (the CPU
      // reference is intractable at this V). target_tops stays 0.0 =
      // report-only: these shapes have no calibrated BMG number yet, and a
      // guessed gate would flake CI (see examples/17 for the same convention).
      {"perf_prodvocab_h768_tp1_v201024", 256, 3, 320, 16384, 1024, 768, 201024, 0.0, 3.0, 0, true},
      {"perf_prodvocab_h6144_tp8_v25128", 256, 9, 320, 16384, 1024, 6144, 25128, 0.0, 3.0, 0, true},
  };
}

// Inkling suite: draft-worker cache_path fuses (a) masked req_to_token gather
// with draft_token_num verify positions and (b) LM-head greedy argmax over the
// full hidden state. Cases sweep the three shipped hidden_size targets (768 in
// the thinkingmachines/Inkling checkpoint, 1536 config default, 6144
// production) and TP=1/2/4/8. The gather side scales with
// bs*draft_token_num (draft_token_num = num_nextn_predict_layers+1, i.e. 3 for
// the checkpoint's 2 MTP layers and 9 for production's 8); the greedy side
// scales with the LM-head vocab shard V/tp. max_prefill_tokens=16384 sets the
// req_to_token column bound.
//
// Vocab coverage comes in two tiers:
//   * the 8192-anchored shards keep the full O(N*V*H) CPU reference tractable;
//   * the production shards use the real ParallelLMHead vocab,
//     InklingModelConfig.padded_vocab_size = 201024 (unpadded 200058), sharded
//     over TP as {201024, 100512, 50256, 25128}. Reference strategy for these:
//     a *sampled* CPU reference over a few deterministically chosen rows
//     (verify_rows > 0; verify_rows == 0 means benchmark-only and is used by
//     the perf suite, where --verify=0 anyway). A sampled row is a full honest
//     argmax over all V, and row 0's planted maximum always sits at V-1, so a
//     wrong-vocab / wrong-stride / dropped-tail bug is still caught; the
//     sampling only gives up coverage of the *other* rows. It was chosen over a
//     GPU-side second implementation because two fp32 reductions on the device
//     share the same failure modes, and over --verify=0 alone because that
//     would leave the largest V with no numeric coverage at all. The 201024
//     cases also need ~617 MB of fp32 weights per case, which is why they use
//     the narrow 768 trunk.
std::vector<CachePathCase> inkling_suite() {
  return {
      // config defaults hidden_size=1536, TP sweep over verify-band bs and vocab shard.
      {"cfg_h1536_tp1_bs4_vshard8192",  4, 9,   8, 16384,  36, 1536, 8192, 0.0},
      {"cfg_h1536_tp2_bs4_vshard4096",  4, 9,   8, 16384,  36, 1536, 4096, 0.0},
      {"cfg_h1536_tp4_bs4_vshard2048",  4, 9,   8, 16384,  36, 1536, 2048, 0.0},
      {"cfg_h1536_tp8_bs4_vshard1024",  4, 9,   8, 16384,  36, 1536, 1024, 0.0},
      // production hidden_size=6144 TP sweep.
      {"prod_h6144_tp1_bs4_vshard8192", 4, 9,   8, 16384,  36, 6144, 8192, 0.0},
      {"prod_h6144_tp2_bs4_vshard4096", 4, 9,   8, 16384,  36, 6144, 4096, 0.0},
      {"prod_h6144_tp4_bs4_vshard2048", 4, 9,   8, 16384,  36, 6144, 2048, 0.0},
      {"prod_h6144_tp8_bs4_vshard1024", 4, 9,   8, 16384,  36, 6144, 1024, 0.0},
      // Wider batch to cover the target-verify Q band (bs*draft_token_num=144).
      {"cfg_h1536_tp4_bs16_vshard2048", 16, 9, 24, 16384, 144, 1536, 2048, 0.0},
      {"prod_h6144_tp4_bs16_vshard2048", 16, 9, 24, 16384, 144, 6144, 2048, 0.0},
      // Checkpoint trunk geometry hidden_size=768 with draft_token_num=3 and
      // the checkpoint's logits_mup_width_multiplier=3.0 pre-head divide.
      {"ckpt_h768_tp1_bs4_vshard8192", 4, 3, 8, 16384, 12, 768, 8192, 0.0, 3.0, -1, true},
      {"ckpt_h768_tp4_bs4_vshard2048", 4, 3, 8, 16384, 12, 768, 2048, 0.0, 3.0, -1, true},
      // logits_mup_width_multiplier = null (the config default): no divide.
      // Same shape as the case above, so the two together show the argmax is
      // invariant under the positive scalar while the launch sequence differs.
      {"ckpt_h768_tp4_bs4_nomup", 4, 3, 8, 16384, 12, 768, 2048, 0.0, 0.0, -1, true},
      // Production padded_vocab_size=201024 sharded over TP={1,2,4,8}, sampled
      // reference (see the suite comment for why).
      {"prodvocab_h768_tp1_v201024", 4, 3, 8, 16384, 12, 768, 201024, 0.0, 3.0, 2, true},
      {"prodvocab_h768_tp2_v100512", 4, 3, 8, 16384, 12, 768, 100512, 0.0, 3.0, 2, true},
      {"prodvocab_h768_tp4_v50256", 4, 3, 8, 16384, 12, 768, 50256, 0.0, 3.0, 4, true},
      {"prodvocab_h768_tp8_v25128", 4, 3, 8, 16384, 12, 768, 25128, 0.0, 3.0, 4, true},
      // Production vocab shards at the wider trunks, draft_token_num=9.
      {"prodvocab_h1536_tp8_v25128", 4, 9, 8, 16384, 36, 1536, 25128, 0.0, 3.0, 2, true},
      {"prodvocab_h6144_tp8_v25128", 4, 9, 8, 16384, 36, 6144, 25128, 0.0, 3.0, 1, true},
      {"prodvocab_h6144_tp4_v50256", 4, 9, 8, 16384, 36, 6144, 50256, 0.0, 3.0, 1, true},
  };
}

std::vector<CachePathCase> make_suite(std::string const& suite) {
  if (suite == "quick") {
    return quick_suite();
  }
  if (suite == "stress") {
    return stress_suite();
  }
  if (suite == "perf") {
    return perf_suite();
  }
  if (suite == "inkling") {
    return inkling_suite();
  }
  return {};
}

// Planted argmax target for a row of the hidden state (planted cases only).
// Row 0 targets the very last vocab entry so the sampled reference always
// covers the final vocab tile -- a kernel that dropped the V tail would be
// caught even at V = 201024 -- row 1 targets entry 0, and the rest are hashed
// across the whole shard rather than walking a short linear stride.
int planted_target(int token, int vocab) {
  if (token == 0) {
    return vocab - 1;
  }
  if (token == 1) {
    return 0;
  }
  uint64_t bits = static_cast<uint64_t>(token) * 0x9E3779B97F4A7C15ull;
  bits ^= bits >> 29;
  bits *= 0xBF58476D1CE4E5B9ull;
  bits ^= bits >> 32;
  return static_cast<int>(bits % static_cast<uint64_t>(vocab));
}

// Deterministic subset of [0, tokens) of size min(count, tokens); count < 0
// selects every row. Rows 0 and tokens-1 are always picked first so the
// tail-of-vocab and index-0 planted targets are never sampled away.
std::vector<int> select_verify_rows(int tokens, int count) {
  std::vector<int> rows;
  if (tokens <= 0 || count == 0) {
    return rows;
  }
  if (count < 0 || count >= tokens) {
    rows.resize(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
      rows[static_cast<std::size_t>(i)] = i;
    }
    return rows;
  }
  rows.push_back(0);
  if (count > 1 && tokens > 1) {
    rows.push_back(tokens - 1);
  }
  std::vector<int> rest;
  for (int i = 1; i + 1 < tokens; ++i) {
    rest.push_back(i);
  }
  std::mt19937 rng(0x5eed5eedu);
  std::shuffle(rest.begin(), rest.end(), rng);
  for (int row : rest) {
    if (static_cast<int>(rows.size()) >= count) {
      break;
    }
    rows.push_back(row);
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

CachePathHost initialize_case(CachePathCase const& cfg, bool want_greedy_ref) {
  CachePathHost h;
  h.req_to_token.resize(static_cast<std::size_t>(cfg.req_rows) * cfg.table_width);
  for (std::size_t i = 0; i < h.req_to_token.size(); ++i) {
    h.req_to_token[i] = static_cast<int64_t>(i);
  }

  h.req_pool_indices.resize(cfg.batch);
  h.pos2d.resize(static_cast<std::size_t>(cfg.batch) * cfg.draft_tokens);
  h.mask.resize(static_cast<std::size_t>(cfg.batch) * cfg.draft_tokens);
  h.out_offsets.resize(static_cast<std::size_t>(cfg.batch) * cfg.draft_tokens, -1);

  if (cfg.name == "reference_small_b2_t3_h5_v7") {
    h.req_pool_indices = {2, 0};
    h.pos2d = {1, 3, 0, 2, 4, 6};
    h.mask = {1, 0, 1, 1, 1, 0};
  } else {
    for (int b = 0; b < cfg.batch; ++b) {
      h.req_pool_indices[b] = (b * 7 + 2) % cfg.req_rows;
      for (int t = 0; t < cfg.draft_tokens; ++t) {
        int idx = b * cfg.draft_tokens + t;
        h.pos2d[idx] = (b * 13 + t * 17 + 1) % cfg.table_width;
        h.mask[idx] = static_cast<uint8_t>(((b + 2 * t) % 5) != 1);
      }
    }
  }

  int32_t out_count = 0;
  for (int i = 0; i < cfg.batch * cfg.draft_tokens; ++i) {
    if (h.mask[i] != 0) {
      h.out_offsets[i] = out_count++;
      int b = i / cfg.draft_tokens;
      int64_t req = h.req_pool_indices[b];
      int64_t pos = h.pos2d[i];
      h.gathered_ref.push_back(h.req_to_token[static_cast<std::size_t>(req) * cfg.table_width + pos]);
    }
  }

  h.hidden.resize(static_cast<std::size_t>(cfg.tokens) * cfg.hidden_dim);
  h.weight.resize(static_cast<std::size_t>(cfg.vocab) * cfg.hidden_dim);
  if (cfg.planted) {
    // Well-conditioned argmax at any V: weights are hashed noise, and row t of
    // the hidden state is a copy of the weight row it should select. The winning
    // logit is then ||w_target||^2 ~= 0.1875*H while every other logit is a
    // near-orthogonal inner product with standard deviation ~0.1875*sqrt(H), so
    // the top-1 margin grows as sqrt(H) and stays far outside fp32 accumulation
    // noise even at V = 201024.
    for (std::size_t i = 0; i < h.weight.size(); ++i) {
      h.weight[i] = hash_uniform(i, 23);
    }
    for (int token = 0; token < cfg.tokens; ++token) {
      int target = planted_target(token, cfg.vocab);
      for (int k = 0; k < cfg.hidden_dim; ++k) {
        std::size_t idx = static_cast<std::size_t>(token) * cfg.hidden_dim + k;
        std::size_t w_idx = static_cast<std::size_t>(target) * cfg.hidden_dim + k;
        h.hidden[idx] = h.weight[w_idx] + 0.1f * hash_uniform(idx, 11);
      }
    }
  } else {
    for (std::size_t i = 0; i < h.hidden.size(); ++i) {
      h.hidden[i] = 0.5f + 0.25f * static_cast<float>(dflash::patterned_value(i, 11));
    }
    for (int v = 0; v < cfg.vocab; ++v) {
      for (int k = 0; k < cfg.hidden_dim; ++k) {
        std::size_t idx = static_cast<std::size_t>(v) * cfg.hidden_dim + k;
        h.weight[idx] = 0.05f * static_cast<float>(dflash::patterned_value(idx, 23)) +
                        static_cast<float>(v) * 0.1f;
      }
    }
  }

  if (!want_greedy_ref) {
    return h;
  }

  h.verify_row_ids = select_verify_rows(cfg.tokens, cfg.verify_rows);
  h.greedy_ref.resize(h.verify_row_ids.size());
  float divisor = static_cast<float>(cfg.mup);
  std::vector<float> head_row(static_cast<std::size_t>(cfg.hidden_dim));
  for (std::size_t r = 0; r < h.verify_row_ids.size(); ++r) {
    int token = h.verify_row_ids[r];
    // Same fp32 divide the device pass performs, hoisted out of the V loop.
    for (int k = 0; k < cfg.hidden_dim; ++k) {
      float x = h.hidden[static_cast<std::size_t>(token) * cfg.hidden_dim + k];
      head_row[static_cast<std::size_t>(k)] = cfg.mup > 0.0 ? x / divisor : x;
    }
    float best = -std::numeric_limits<float>::infinity();
    int64_t best_idx = 0;
    for (int v = 0; v < cfg.vocab; ++v) {
      float acc = 0.0f;
      for (int k = 0; k < cfg.hidden_dim; ++k) {
        acc = std::fma(
            head_row[static_cast<std::size_t>(k)],
            h.weight[static_cast<std::size_t>(v) * cfg.hidden_dim + k],
            acc);
      }
      if (acc > best || (acc == best && v < best_idx)) {
        best = acc;
        best_idx = v;
      }
    }
    h.greedy_ref[r] = best_idx;
  }

  return h;
}

bool run_case(sycl::queue& queue, CachePathCase const& cfg, dflash::Options const& options) {
  if (cfg.batch < 0 || cfg.draft_tokens < 0 || cfg.req_rows <= 0 || cfg.table_width <= 0 ||
      cfg.tokens < 0 || cfg.hidden_dim <= 0 || cfg.vocab <= 0) {
    throw std::runtime_error("invalid cache path case dimensions");
  }

  bool want_greedy_ref = options.verify && cfg.verify_rows != 0;
  CachePathHost h = initialize_case(cfg, want_greedy_ref);

  dflash::DeviceBuffer<int64_t> d_req_to_token(queue, h.req_to_token.size());
  dflash::DeviceBuffer<int64_t> d_req_pool(queue, h.req_pool_indices.size());
  dflash::DeviceBuffer<int64_t> d_pos2d(queue, h.pos2d.size());
  dflash::DeviceBuffer<uint8_t> d_mask(queue, h.mask.size());
  dflash::DeviceBuffer<int32_t> d_offsets(queue, h.out_offsets.size());
  dflash::DeviceBuffer<int64_t> d_gathered(queue, h.gathered_ref.size());
  dflash::DeviceBuffer<float> d_hidden(queue, h.hidden.size());
  dflash::DeviceBuffer<float> d_weight(queue, h.weight.size());
  dflash::DeviceBuffer<float> d_logits(queue, static_cast<std::size_t>(cfg.tokens) * cfg.vocab);
  dflash::DeviceBuffer<int64_t> d_greedy(queue, static_cast<std::size_t>(cfg.tokens));
  // Only allocated when the config sets logits_mup_width_multiplier.
  dflash::DeviceBuffer<float> d_hidden_mup;
  if (cfg.mup > 0.0) {
    d_hidden_mup = dflash::DeviceBuffer<float>(queue, h.hidden.size());
  }

  d_req_to_token.copy_from(h.req_to_token);
  d_req_pool.copy_from(h.req_pool_indices);
  d_pos2d.copy_from(h.pos2d);
  d_mask.copy_from(h.mask);
  d_offsets.copy_from(h.out_offsets);
  d_hidden.copy_from(h.hidden);
  d_weight.copy_from(h.weight);

  MaskedGatherParams gather_params;
  gather_params.req_to_token = d_req_to_token.get();
  gather_params.req_pool_indices = d_req_pool.get();
  gather_params.pos2d = d_pos2d.get();
  gather_params.mask = d_mask.get();
  gather_params.out_offsets = d_offsets.get();
  gather_params.out = d_gathered.get();
  gather_params.batch = cfg.batch;
  gather_params.draft_tokens = cfg.draft_tokens;
  gather_params.table_width = cfg.table_width;

  MupDivideParams mup_params;
  mup_params.in = d_hidden.get();
  mup_params.out = d_hidden_mup.get();
  mup_params.count = static_cast<int64_t>(h.hidden.size());
  mup_params.divisor = static_cast<float>(cfg.mup);

  // The head consumes the mup-divided hidden state when the config sets the
  // multiplier, and the raw hidden state when it is null.
  float const* head_input = cfg.mup > 0.0 ? d_hidden_mup.get() : d_hidden.get();

  auto launch_iteration = [&]() {
    launch_masked_gather(queue, gather_params);
    if (cfg.mup > 0.0) {
      launch_mup_divide(queue, mup_params);
    }
    launch_gemm_greedy_argmax(
        queue, head_input, d_weight.get(), d_logits.get(), d_greedy.get(), cfg.tokens, cfg.hidden_dim, cfg.vocab);
  };

  launch_iteration();
  queue.wait();

  bool passed = true;
  if (options.verify) {
    std::vector<int64_t> gathered_got(h.gathered_ref.size(), 0);
    d_gathered.copy_to(gathered_got);
    if (gathered_got != h.gathered_ref) {
      passed = false;
      std::cerr << "Gather mismatch case=" << cfg.name << "\n";
    }
    // The greedy argmax is invariant under the positive mup scalar, so it can
    // never observe a wrong divisor. Check the divided tensor itself on a small
    // strided sample; the device does the same fp32 divide, so it must match
    // bit-for-bit.
    if (cfg.mup > 0.0 && !h.hidden.empty()) {
      std::size_t probe = std::min<std::size_t>(h.hidden.size(), 64);
      std::size_t stride = std::max<std::size_t>(h.hidden.size() / probe, 1);
      std::vector<float> mup_got(h.hidden.size(), 0.0f);
      d_hidden_mup.copy_to(mup_got);
      for (std::size_t i = 0; i < h.hidden.size(); i += stride) {
        float want = h.hidden[i] / static_cast<float>(cfg.mup);
        if (mup_got[i] != want) {
          passed = false;
          std::cerr << "mup divide mismatch case=" << cfg.name << " index=" << i
                    << " got=" << mup_got[i] << " ref=" << want << "\n";
          break;
        }
      }
    }
    if (!h.verify_row_ids.empty()) {
      std::vector<int64_t> greedy_got(static_cast<std::size_t>(cfg.tokens), 0);
      d_greedy.copy_to(greedy_got);
      for (std::size_t r = 0; r < h.verify_row_ids.size(); ++r) {
        int token = h.verify_row_ids[r];
        if (greedy_got[static_cast<std::size_t>(token)] != h.greedy_ref[r]) {
          passed = false;
          std::cerr << "Greedy mismatch case=" << cfg.name << " token=" << token
                    << " got=" << greedy_got[static_cast<std::size_t>(token)]
                    << " ref=" << h.greedy_ref[r] << "\n";
          break;
        }
      }
    }
  }

  double ms = 0.0;
  if (options.benchmark && options.iterations > 0) {
    for (int i = 0; i < options.warmup; ++i) {
      launch_iteration();
    }
    queue.wait();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iterations; ++i) {
      launch_iteration();
    }
    queue.wait();
    auto end = std::chrono::steady_clock::now();
    ms = dflash::elapsed_ms(begin, end, options.iterations);
  }

  // ms covers the whole launch sequence, including the mup divide when the case
  // has one, while flops/gather_bytes only count the head GEMM and the gather.
  // A mup case therefore reports slightly lower greedy_TOPS than the identical
  // shape without the divide (measured ~6% at H=768,V=2048, where the extra
  // launch is a large fraction of a very short iteration). That is why every
  // mup case keeps target_tops = 0.0; a gate must be calibrated per case, not
  // carried over from the non-mup twin.
  double seconds = ms / 1000.0;
  double gather_bytes = static_cast<double>(h.gathered_ref.size()) * (sizeof(int64_t) * 2.0) +
                        static_cast<double>(cfg.batch) * cfg.draft_tokens *
                            (sizeof(uint8_t) + sizeof(int64_t) + sizeof(int32_t));
  double flops = 2.0 * static_cast<double>(cfg.tokens) * cfg.vocab * cfg.hidden_dim;
  double gbps = seconds > 0.0 ? gather_bytes / seconds / 1.0e9 : 0.0;
  double tops = seconds > 0.0 ? flops / seconds / 1.0e12 : 0.0;
  double target_tops = options.target_tops_set ? options.target_tops : cfg.target_tops;
  if (target_tops > 0.0 && tops < target_tops) {
    passed = false;
    std::cerr << "TOPS target miss case=" << cfg.name << " got=" << tops
              << " target=" << target_tops << "\n";
  }

  std::string mup_text = "null";
  if (cfg.mup > 0.0) {
    std::ostringstream mup_stream;
    mup_stream << std::fixed << std::setprecision(1) << cfg.mup;
    mup_text = mup_stream.str();
  }

  bool full_ref = want_greedy_ref && h.verify_row_ids.size() == static_cast<std::size_t>(cfg.tokens);
  std::string ref_text = "none";
  if (want_greedy_ref) {
    ref_text = full_ref ? "full" : ("rows:" + std::to_string(h.verify_row_ids.size()));
  }

  // Distinguish "every row checked" from "some rows checked" and "greedy output
  // never compared", so a log grep cannot mistake a benchmark-only case for a
  // numerically verified one.
  std::string verify_text = "off";
  if (options.verify) {
    if (!passed) {
      verify_text = "false";
    } else if (full_ref) {
      verify_text = "true";
    } else if (want_greedy_ref) {
      verify_text = "sampled";
    } else {
      verify_text = "gather-only";
    }
  }

  std::cout << "case=" << std::left << std::setw(32) << cfg.name
            << " B=" << std::right << std::setw(4) << cfg.batch
            << " T=" << std::setw(3) << cfg.draft_tokens
            << " H=" << std::setw(5) << cfg.hidden_dim
            << " V=" << std::setw(6) << cfg.vocab
            << " mup=" << std::left << std::setw(4) << mup_text << std::right
            << " valid=" << std::setw(6) << h.gathered_ref.size()
            << " ref=" << std::left << std::setw(7) << ref_text
            << " verify=" << std::setw(11) << verify_text << std::right
            << " time_ms=" << std::fixed << std::setprecision(4) << ms
            << " gather_GBps=" << std::setprecision(2) << gbps
            << " greedy_TOPS=" << std::setprecision(3) << tops << "\n";
  return passed;
}

void print_usage(char const* exe) {
  std::cout << "20_bmg_dflash_cache_path: DFLASH masked req_to_token gather + greedy LM-head argmax\n\n"
            << "Usage: " << exe << " [--suite=quick|stress|perf|inkling]\n"
            << "       [--shape=B=2,T=3,rows=3,table=10,tokens=4,H=5,V=7,mup=3,vrows=-1,planted=1]\n"
            << "       [--iterations=N] [--verify=0|1] [--target-tops=X]\n"
            << "\nInkling suite sweeps hidden 768/1536/6144, draft_token_num 3 (checkpoint)\n"
            << "and 9 (production), table=16384 (max_prefill_tokens), and the LM-head\n"
            << "vocab shard for TP=1/2/4/8 both at an 8192 anchor and at the real\n"
            << "padded_vocab_size=201024. mup is logits_mup_width_multiplier (0 = null);\n"
            << "vrows caps the greedy CPU reference (-1 = all rows, 0 = benchmark only);\n"
            << "planted=1 selects the well-conditioned argmax data required at large V\n"
            << "(planted=0 keeps the legacy +v*0.1 ramp, whose argmax is always V-1).\n";
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

  std::vector<CachePathCase> cases;
  if (!options.shape.empty()) {
    CachePathCase cfg = custom_default();
    // parse_shape_ints() only binds ints, so mup is given as an integer
    // multiplier (0 = null, the shipped checkpoint uses 3) and planted as 0|1.
    int mup_int = 0;
    int planted_int = 0;
    if (!dflash::parse_shape_ints(options.shape, {
            {"B", &cfg.batch},
            {"T", &cfg.draft_tokens},
            {"rows", &cfg.req_rows},
            {"table", &cfg.table_width},
            {"tokens", &cfg.tokens},
            {"H", &cfg.hidden_dim},
            {"V", &cfg.vocab},
            {"mup", &mup_int},
            {"vrows", &cfg.verify_rows},
            {"planted", &planted_int},
        })) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    cfg.mup = static_cast<double>(mup_int);
    cfg.planted = planted_int != 0;
    cases.push_back(cfg);
  } else {
    cases = make_suite(options.suite);
    if (cases.empty()) {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = dflash::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "20_bmg_dflash_cache_path: masked gather is memory-bound; greedy argmax reports TOPS\n";
    std::cout << "Suite=" << options.suite << " iterations=" << options.iterations
              << " warmup=" << options.warmup << " verify=" << dflash::bool_text(options.verify)
              << " benchmark=" << dflash::bool_text(options.benchmark) << "\n";

    bool all_passed = true;
    for (CachePathCase const& cfg : cases) {
      all_passed &= run_case(queue, cfg, options);
    }
    return all_passed ? 0 : -1;
  } catch (dflash::NoGpuDevice const& e) {
    std::cout << "SKIP: " << e.what() << "\n";
    return dflash::kSkipReturnCode;
  } catch (std::bad_alloc const&) {
    // The production vocab cases need ~1.5 GB of device memory
    // (V=201024 weights plus tokens x V logits). Report that as a skip rather
    // than a failure so a smaller or busier card does not look like a bug.
    std::cout << "SKIP: out of memory allocating the LM-head working set; "
                 "use a smaller --shape or --suite\n";
    return dflash::kSkipReturnCode;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
