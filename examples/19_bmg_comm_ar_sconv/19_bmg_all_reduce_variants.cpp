/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling 06.1 all-reduce variants for CUTLASS SYCL.
 *
 * This standalone BMG example models the CUDA symmetric-memory variants with
 * world-size rank buffers resident on one XPU. CUDA multimem and in-kernel
 * cross-GPU barriers do not exist in this examples environment, so the kernels
 * preserve the dataflow and rounding seams while SYCL event dependencies stand
 * in for the remote barriers:
 *
 *   two-shot     : reduce-scatter-style full reduction, then all-gather copy
 *   full one-shot: every rank independently reduces the full range
 *   push one-shot: fold each rank into a staging slot, then reduce the stage
 *   direct       : XPU wrapper fallback, one output element per work-item
 *
 * Roofline: for bf16/fp16 and TP4, one output element performs three useful
 * adds and streams roughly four input elements plus one output element
 * (10 bytes without shared partials), so the arithmetic intensity is about
 * 0.3 FLOP/B. These paths are memory/latency bound; perf output reports
 * effective GB/s instead of TOPS.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include "cutlass/util/command_line.h"
#include "19_bmg_comm_ar_sconv_common.hpp"

#include <functional>
#include <map>

namespace cutlass::examples::comm_ar_sconv {

constexpr int kThreads = 256;

enum class Variant {
  kAll,
  kTwoShot,
  kFullOneShot,
  kPushOneShot,
  kDirect
};

struct VariantPerfThresholds {
  double two_shot = 0.0;
  double full_oneshot = 0.0;
  double push_oneshot = 0.0;
  double direct = 0.0;
};

struct Options {
  std::string suite = "quick";
  DType dtype = DType::kAll;
  Variant variant = Variant::kAll;
  int iterations = 5;
  bool verify = true;
  double perf_threshold_scale = 1.0;
};

struct CaseConfig {
  std::string name;
  int world = 4;
  int n = 0;
  bool use_shared = false;
  VariantPerfThresholds min_gbps;
};

template <typename Element_>
struct AllReduceParams {
  using Element = Element_;

  Element const* __restrict__ in;
  Element const* __restrict__ shared;
  Element* __restrict__ scratch;
  Element* __restrict__ stage;
  Element* __restrict__ out;
  int world;
  int n;
  int use_shared;
};

template <typename Element>
class DirectAllReduceKernel;

template <typename Element>
class DirectAllReducePack4Kernel;

template <typename Element>
class FullOneShotAllReduceKernel;

template <typename Element>
class FullOneShotAllReducePack4Kernel;

template <typename Element>
class TwoShotReduceKernel;

template <typename Element>
class TwoShotReducePack4Kernel;

template <typename Element>
class TwoShotGatherKernel;

template <typename Element>
class TwoShotGatherPack4Kernel;

template <typename Element>
class PushStageKernel;

template <typename Element>
class PushStagePack4Kernel;

template <typename Element>
class PushReduceKernel;

template <typename Element>
class PushReducePack4Kernel;

std::string variant_text(Variant variant) {
  switch (variant) {
    case Variant::kAll:
      return "all";
    case Variant::kTwoShot:
      return "two_shot";
    case Variant::kFullOneShot:
      return "full_oneshot";
    case Variant::kPushOneShot:
      return "push_oneshot";
    case Variant::kDirect:
      return "direct";
  }
  return "unknown";
}

bool parse_variant(std::string const& text, Variant& variant) {
  if (text == "all") {
    variant = Variant::kAll;
    return true;
  }
  if (text == "two_shot") {
    variant = Variant::kTwoShot;
    return true;
  }
  if (text == "full_oneshot" || text == "v4") {
    variant = Variant::kFullOneShot;
    return true;
  }
  if (text == "push_oneshot" || text == "v5") {
    variant = Variant::kPushOneShot;
    return true;
  }
  if (text == "direct" || text == "fallback") {
    variant = Variant::kDirect;
    return true;
  }
  return false;
}

template <typename Element>
CUTLASS_DEVICE
float folded_rank_value(AllReduceParams<Element> const& params, int rank, int idx) {
  std::size_t off = static_cast<std::size_t>(rank) * params.n + idx;
  float value = element_to_float(params.in[off]);
  if (params.use_shared) {
    value += element_to_float(params.shared[off]);
    return element_to_float(round_to_element<Element>(value));
  }
  return value;
}

template <typename Element>
CUTLASS_DEVICE
Element element_from_pack4(uint64_t raw, int lane) {
  return Element::bitcast(static_cast<uint16_t>(raw >> (16 * lane)));
}

template <typename Element>
CUTLASS_DEVICE
uint64_t load_pack4(Element const* ptr) {
  return *reinterpret_cast<uint64_t const*>(ptr);
}

template <typename Element>
CUTLASS_DEVICE
void store_pack4(Element* ptr, uint64_t raw) {
  *reinterpret_cast<uint64_t*>(ptr) = raw;
}

template <typename Element>
CUTLASS_DEVICE
uint64_t pack4_from_floats(float const (&values)[4]) {
  uint64_t raw = 0;
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    raw |= static_cast<uint64_t>(Element(values[v]).raw()) << (16 * v);
  }
  return raw;
}

template <typename Element>
CUTLASS_DEVICE
float folded_pack_lane(AllReduceParams<Element> const& params, int rank, int pack, int lane) {
  std::size_t off = static_cast<std::size_t>(rank) * params.n + pack * 4;
  uint64_t in_raw = load_pack4(params.in + off);
  float value = element_to_float(element_from_pack4<Element>(in_raw, lane));
  if (params.use_shared) {
    uint64_t shared_raw = load_pack4(params.shared + off);
    value += element_to_float(element_from_pack4<Element>(shared_raw, lane));
    return element_to_float(Element(value));
  }
  return value;
}

template <typename Element>
sycl::event launch_direct(sycl::queue& q, AllReduceParams<Element> const& params) {
  if ((params.n % 4) == 0) {
    int pack_n = params.n / 4;
    int total_packs = params.world * pack_n;
    int global_packs = ceil_div(total_packs, kThreads) * kThreads;
    return q.parallel_for<DirectAllReducePack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global_packs), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= total_packs) {
            return;
          }
          int pack = linear % pack_n;
          int out_rank = linear / pack_n;
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int r = 0; r < params.world; ++r) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[v] += folded_pack_lane(params, r, pack, v);
            }
          }
          store_pack4(params.out + static_cast<std::size_t>(out_rank) * params.n + pack * 4, pack4_from_floats<Element>(acc));
        });
  }
  int total = params.world * params.n;
  int global = ceil_div(total, kThreads) * kThreads;
  return q.parallel_for<DirectAllReduceKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int idx = linear % params.n;
        float acc = 0.0f;
        for (int r = 0; r < params.world; ++r) {
          acc += folded_rank_value(params, r, idx);
        }
        params.out[linear] = Element(acc);
      });
}

template <typename Element>
sycl::event launch_full_oneshot(sycl::queue& q, AllReduceParams<Element> const& params) {
  if ((params.n % 4) == 0) {
    int pack_n = params.n / 4;
    int total_packs = params.world * pack_n;
    int global_packs = ceil_div(total_packs, kThreads) * kThreads;
    return q.parallel_for<FullOneShotAllReducePack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global_packs), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= total_packs) {
            return;
          }
          int pack = linear % pack_n;
          int out_rank = linear / pack_n;
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int r = 0; r < params.world; ++r) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[v] += folded_pack_lane(params, r, pack, v);
            }
          }
          store_pack4(params.out + static_cast<std::size_t>(out_rank) * params.n + pack * 4, pack4_from_floats<Element>(acc));
        });
  }
  int total = params.world * params.n;
  int global = ceil_div(total, kThreads) * kThreads;
  return q.parallel_for<FullOneShotAllReduceKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int idx = linear % params.n;
        float acc = 0.0f;
        for (int r = 0; r < params.world; ++r) {
          acc += folded_rank_value(params, r, idx);
        }
        params.out[linear] = Element(acc);
      });
}

template <typename Element>
sycl::event launch_two_shot(sycl::queue& q, AllReduceParams<Element> const& params) {
  if ((params.n % 4) == 0) {
    int pack_n = params.n / 4;
    int global_reduce = ceil_div(pack_n, kThreads) * kThreads;
    auto reduce = q.parallel_for<TwoShotReducePack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global_reduce), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int pack = static_cast<int>(item.get_global_linear_id());
          if (pack >= pack_n) {
            return;
          }
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int r = 0; r < params.world; ++r) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[v] += folded_pack_lane(params, r, pack, v);
            }
          }
          store_pack4(params.scratch + pack * 4, pack4_from_floats<Element>(acc));
        });
    int total_packs = params.world * pack_n;
    int global_gather = ceil_div(total_packs, kThreads) * kThreads;
    return q.parallel_for<TwoShotGatherPack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global_gather), sycl::range<1>(kThreads)),
        reduce,
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= total_packs) {
            return;
          }
          int pack = linear % pack_n;
          int rank = linear / pack_n;
          uint64_t raw = load_pack4(params.scratch + pack * 4);
          store_pack4(params.out + static_cast<std::size_t>(rank) * params.n + pack * 4, raw);
        });
  }
  int global_reduce = ceil_div(params.n, kThreads) * kThreads;
  auto reduce = q.parallel_for<TwoShotReduceKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global_reduce), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        int idx = static_cast<int>(item.get_global_linear_id());
        if (idx >= params.n) {
          return;
        }
        float acc = 0.0f;
        for (int r = 0; r < params.world; ++r) {
          acc += folded_rank_value(params, r, idx);
        }
        params.scratch[idx] = Element(acc);
      });
  int total = params.world * params.n;
  int global_gather = ceil_div(total, kThreads) * kThreads;
  return q.parallel_for<TwoShotGatherKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global_gather), sycl::range<1>(kThreads)),
      reduce,
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int idx = linear % params.n;
        params.out[linear] = params.scratch[idx];
      });
}

template <typename Element>
sycl::event launch_push_oneshot(sycl::queue& q, AllReduceParams<Element> const& params) {
  if ((params.n % 4) == 0) {
    int pack_n = params.n / 4;
    int total_packs = params.world * pack_n;
    int global_stage = ceil_div(total_packs, kThreads) * kThreads;
    auto push = q.parallel_for<PushStagePack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global_stage), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= total_packs) {
            return;
          }
          int rank = linear / pack_n;
          int pack = linear - rank * pack_n;
          float values[4];
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            values[v] = folded_pack_lane(params, rank, pack, v);
          }
          store_pack4(params.stage + static_cast<std::size_t>(rank) * params.n + pack * 4, pack4_from_floats<Element>(values));
        });
    return q.parallel_for<PushReducePack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global_stage), sycl::range<1>(kThreads)),
        push,
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= total_packs) {
            return;
          }
          int out_rank = linear / pack_n;
          int pack = linear - out_rank * pack_n;
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int r = 0; r < params.world; ++r) {
            uint64_t raw = load_pack4(params.stage + static_cast<std::size_t>(r) * params.n + pack * 4);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[v] += element_to_float(element_from_pack4<Element>(raw, v));
            }
          }
          store_pack4(params.out + static_cast<std::size_t>(out_rank) * params.n + pack * 4, pack4_from_floats<Element>(acc));
        });
  }
  int total = params.world * params.n;
  int global_stage = ceil_div(total, kThreads) * kThreads;
  auto push = q.parallel_for<PushStageKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global_stage), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int rank = linear / params.n;
        int idx = linear - rank * params.n;
        params.stage[linear] = Element(folded_rank_value(params, rank, idx));
      });
  return q.parallel_for<PushReduceKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global_stage), sycl::range<1>(kThreads)),
      push,
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int idx = linear % params.n;
        float acc = 0.0f;
        for (int r = 0; r < params.world; ++r) {
          acc += element_to_float(params.stage[static_cast<std::size_t>(r) * params.n + idx]);
        }
        params.out[linear] = Element(acc);
      });
}

template <typename Element>
void reference_allreduce(
    CaseConfig const& cfg,
    std::vector<Element> const& in,
    std::vector<Element> const& shared,
    std::vector<Element>& ref) {
  ref.assign(static_cast<std::size_t>(cfg.world) * cfg.n, Element(0.0f));
  for (int idx = 0; idx < cfg.n; ++idx) {
    float acc = 0.0f;
    for (int r = 0; r < cfg.world; ++r) {
      std::size_t off = static_cast<std::size_t>(r) * cfg.n + idx;
      float value = element_to_float(in[off]);
      if (cfg.use_shared) {
        value += element_to_float(shared[off]);
        value = element_to_float(Element(value));
      }
      acc += value;
    }
    Element rounded(acc);
    for (int r = 0; r < cfg.world; ++r) {
      ref[static_cast<std::size_t>(r) * cfg.n + idx] = rounded;
    }
  }
}

std::vector<CaseConfig> quick_suite() {
  return {
      {"tp2_tail_n7", 2, 7, false},
      {"tp4_decode_shared_n1536", 4, 1536, true},
      {"tp8_tail_n257", 8, 257, true},
      {"tp4_prod_n6144", 4, 6144, false},
  };
}

std::vector<CaseConfig> stress_suite() {
  return {
      {"stress_tp1_n1", 1, 1, false},
      {"stress_tp2_n3_shared", 2, 3, true},
      {"stress_tp4_n8191", 4, 8191, false},
      {"stress_tp8_n12289_shared", 8, 12289, true},
  };
}

std::vector<CaseConfig> perf_suite() {
  // Inkling AR feeds hidden-sized tensors. Perf points cover TP=2/4/8 at
  // sizes that model per-token decode residency (n = hidden) up through
  // per-chunk prefill (n = 4096 * hidden, i.e. one 4k-token chunk of a
  // 16384-cap prefill). Larger sizes are avoided because the example holds
  // (world+1)*n resident on ONE XPU (simulating world ranks).
  return {
      {"perf_tp2_n1048576", 2, 1024 * 1024, false, {150.0, 150.0, 150.0, 150.0}},
      {"perf_tp4_n1048576", 4, 1024 * 1024, false, {250.0, 250.0, 250.0, 250.0}},
      {"perf_tp4_n1048576_shared", 4, 1024 * 1024, true, {250.0, 250.0, 250.0, 250.0}},
      {"perf_tp8_n1048576", 8, 1024 * 1024, false, {250.0, 250.0, 250.0, 250.0}},
      {"perf_tp8_n2097152", 8, 2 * 1024 * 1024, false, {250.0, 250.0, 150.0, 250.0}},

      // Per-4k-token-chunk residual reduce shapes.
      // cfg hidden=1536 → 4096*1536 = 6291456 elements.
      {"perf_tp2_chunk_cfg",  2, 4096 * 1536, false, {200.0, 200.0, 200.0, 200.0}},
      {"perf_tp4_chunk_cfg",  4, 4096 * 1536, false, {250.0, 250.0, 150.0, 250.0}},
      {"perf_tp8_chunk_cfg",  8, 4096 * 1536, false, {250.0, 150.0, 150.0, 150.0}},
      // prod hidden=6144 → 4096*6144 = 25165824 elements.
      {"perf_tp2_chunk_prod", 2, 4096 * 6144, false, {120.0, 120.0, 150.0, 120.0}},
      {"perf_tp4_chunk_prod", 4, 4096 * 6144, false, {180.0, 160.0, 160.0, 160.0}},

      // Target-verify decode band residency (batch=16 * draft_token_num=9 = 144
      // rows; residual is one row per token). Reduce n = 144 * hidden.
      {"perf_tp2_verify_cfg",  2, 144 * 1536, false, {90.0, 90.0, 90.0, 90.0}},
      {"perf_tp4_verify_cfg",  4, 144 * 1536, false, {150.0, 150.0, 150.0, 150.0}},
      {"perf_tp8_verify_cfg",  8, 144 * 1536, false, {200.0, 200.0, 200.0, 200.0}},
      {"perf_tp2_verify_prod", 2, 144 * 6144, false, {180.0, 180.0, 180.0, 180.0}},
      {"perf_tp4_verify_prod", 4, 144 * 6144, false, {250.0, 250.0, 250.0, 250.0}},
      {"perf_tp8_verify_prod", 8, 144 * 6144, false, {250.0, 250.0, 250.0, 250.0}},
  };
}

double variant_min_gbps(Variant variant, VariantPerfThresholds const& thresholds) {
  switch (variant) {
    case Variant::kTwoShot:
      return thresholds.two_shot;
    case Variant::kFullOneShot:
      return thresholds.full_oneshot;
    case Variant::kPushOneShot:
      return thresholds.push_oneshot;
    case Variant::kDirect:
      return thresholds.direct;
    case Variant::kAll:
      break;
  }
  return 0.0;
}

double variant_effective_bytes(Variant variant, CaseConfig const& cfg, std::size_t element_bytes) {
  double n = static_cast<double>(cfg.n);
  double w = static_cast<double>(cfg.world);
  double shared = cfg.use_shared ? 1.0 : 0.0;
  switch (variant) {
    case Variant::kTwoShot:
      return element_bytes * (n * w * (1.0 + shared) + n + w * n + w * n);
    case Variant::kPushOneShot:
      return element_bytes * (w * n * (1.0 + shared) + w * n + w * w * n + w * n);
    case Variant::kFullOneShot:
    case Variant::kDirect:
      return element_bytes * (w * n * (w * (1.0 + shared) + 1.0));
    case Variant::kAll:
      break;
  }
  return 0.0;
}

template <typename Element>
bool run_variant(
    sycl::queue& q,
    CaseConfig const& cfg,
    Variant variant,
    Options const& options,
    AllReduceParams<Element> const& params,
    std::vector<Element> const& ref) {
  auto launch = [&]() {
    switch (variant) {
      case Variant::kTwoShot:
        return launch_two_shot(q, params);
      case Variant::kFullOneShot:
        return launch_full_oneshot(q, params);
      case Variant::kPushOneShot:
        return launch_push_oneshot(q, params);
      case Variant::kDirect:
      case Variant::kAll:
        return launch_direct(q, params);
    }
    return launch_direct(q, params);
  };

  launch().wait();
  bool passed = true;
  if (options.verify) {
    std::vector<Element> got(ref.size());
    q.memcpy(got.data(), params.out, sizeof(Element) * got.size()).wait();
    std::ostringstream label;
    label << cfg.name << "/" << variant_text(variant) << "/" << element_dtype_text<Element>();
    passed = compare_vectors(label.str(), got, ref, default_atol<Element>(), default_rtol<Element>());
  }

  double ms = time_ms(q, options.iterations, launch);
  double bytes = variant_effective_bytes(variant, cfg, sizeof(Element));
  double gbps = bytes / (ms * 1.0e6);
  std::ostringstream perf_label;
  perf_label << cfg.name << "/" << variant_text(variant) << "/" << element_dtype_text<Element>();
  double min_gbps = variant_min_gbps(variant, cfg.min_gbps);
  passed &= check_min_gbps(perf_label.str(), gbps, min_gbps, options.perf_threshold_scale);
  min_gbps = scaled_min_gbps(min_gbps, options.perf_threshold_scale);
  std::cout << "[allreduce] " << std::left << std::setw(30) << cfg.name << " variant=" << std::setw(13)
            << variant_text(variant) << " dtype=" << std::setw(4) << element_dtype_text<Element>()
            << " world=" << cfg.world << " n=" << cfg.n << " shared=" << bool_text(cfg.use_shared)
            << " time_ms=" << std::fixed << std::setprecision(4) << ms << " eff_GBps=" << std::setprecision(2)
            << gbps << " min_GBps=" << min_gbps << " " << (passed ? "PASSED" : "FAILED") << "\n";
  return passed;
}

template <typename Element>
bool run_case(sycl::queue& q, CaseConfig const& cfg, Options const& options) {
  std::size_t elems = static_cast<std::size_t>(cfg.world) * cfg.n;
  std::vector<Element> h_in(elems);
  std::vector<Element> h_shared(elems);
  std::vector<Element> h_ref;
  fill_random(h_in, 20260720u + static_cast<uint32_t>(cfg.world * 17 + cfg.n), -0.75f, 0.75f);
  fill_random(h_shared, 20260721u + static_cast<uint32_t>(cfg.world * 31 + cfg.n), -0.25f, 0.25f);
  reference_allreduce(cfg, h_in, h_shared, h_ref);

  DeviceBuffer<Element> d_in(q, elems);
  DeviceBuffer<Element> d_shared(q, elems);
  DeviceBuffer<Element> d_scratch(q, cfg.n);
  DeviceBuffer<Element> d_stage(q, elems);
  DeviceBuffer<Element> d_out(q, elems);
  d_in.copy_from(h_in);
  d_shared.copy_from(h_shared);

  AllReduceParams<Element> params{
      d_in.get(),
      cfg.use_shared ? d_shared.get() : nullptr,
      d_scratch.get(),
      d_stage.get(),
      d_out.get(),
      cfg.world,
      cfg.n,
      cfg.use_shared ? 1 : 0};

  std::vector<Variant> variants;
  if (options.variant == Variant::kAll) {
    variants = {Variant::kTwoShot, Variant::kFullOneShot, Variant::kPushOneShot, Variant::kDirect};
  } else {
    variants = {options.variant};
  }

  bool passed = true;
  for (Variant variant : variants) {
    passed &= run_variant(q, cfg, variant, options, params, h_ref);
  }
  return passed;
}

template <typename Element>
bool run_typed(sycl::queue& q, std::vector<CaseConfig> const& cases, Options const& options) {
  bool passed = true;
  for (auto const& cfg : cases) {
    passed &= run_case<Element>(q, cfg, options);
  }
  return passed;
}

void print_usage(char const* name) {
  std::cout << "Usage: " << name << " [options]\n"
            << "  --suite=<quick|stress|perf>\n"
            << "  --dtype=<all|bf16|fp16>\n"
            << "  --variant=<all|two_shot|full_oneshot|push_oneshot|direct>\n"
            << "  --iterations=<int>\n"
            << "  --verify=<0|1>\n"
            << "  --perf-threshold-scale=<float> (0 disables perf thresholds)\n";
}

}  // namespace cutlass::examples::comm_ar_sconv

int main(int argc, char const** argv) {
  using namespace cutlass::examples::comm_ar_sconv;

  cutlass::CommandLine cmd(argc, argv);
  Options options;
  cmd.get_cmd_line_argument("suite", options.suite, options.suite);
  cmd.get_cmd_line_argument("iterations", options.iterations, options.iterations);
  cmd.get_cmd_line_argument("perf-threshold-scale", options.perf_threshold_scale, options.perf_threshold_scale);
  int verify = options.verify ? 1 : 0;
  cmd.get_cmd_line_argument("verify", verify, verify);
  options.verify = verify != 0;

  std::string dtype_arg = dtype_text(options.dtype);
  cmd.get_cmd_line_argument("dtype", dtype_arg, dtype_arg);
  if (!parse_dtype(dtype_arg, options.dtype)) {
    std::cerr << "Unknown dtype: " << dtype_arg << "\n";
    print_usage(argv[0]);
    return -1;
  }
  std::string variant_arg = variant_text(options.variant);
  cmd.get_cmd_line_argument("variant", variant_arg, variant_arg);
  if (!parse_variant(variant_arg, options.variant)) {
    std::cerr << "Unknown variant: " << variant_arg << "\n";
    print_usage(argv[0]);
    return -1;
  }
  if (cmd.check_cmd_line_flag("help")) {
    print_usage(argv[0]);
    return 0;
  }

  std::vector<CaseConfig> cases;
  if (options.suite == "quick") {
    cases = quick_suite();
  } else if (options.suite == "stress") {
    cases = stress_suite();
  } else if (options.suite == "perf") {
    cases = perf_suite();
  } else {
    std::cerr << "Unknown suite: " << options.suite << "\n";
    print_usage(argv[0]);
    return -1;
  }

  sycl::queue q{sycl::gpu_selector_v};
  print_device(q);

  bool passed = true;
  if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
    passed &= run_typed<cutlass::bfloat16_t>(q, cases, options);
  }
  if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
    passed &= run_typed<cutlass::half_t>(q, cases, options);
  }
  return passed ? 0 : -1;
}
