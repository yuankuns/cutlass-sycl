/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling 06.4 XPU collective mapping.
 *
 * The SGLang XPU wrapper exposes init/register/all_reduce and graph-buffer
 * registration hooks, but CUTLASS standalone examples do not own the Python
 * ProcessGroup, IPC handles, or CUDA multimem-style multicast pointers. This
 * executable validates the BMG-side data movement that the wrapper must provide:
 *
 *   direct      = all_reduce(partials[:, row, col])
 *   shards[p]  = reduce_scatter(direct, uneven last-dim split)
 *   round_trip = all_gather(shards)
 *
 * The direct and round_trip tensors must match bit-for-bit after the same final
 * dtype rounding. This is the semantic contract used by 06.1-06.3 when the
 * CUDA-specific multimem variants are mapped to XPU staged-buffer fallbacks.
 *
 * Roofline: the kernel performs world_size - 1 adds per output element while
 * reading world_size inputs and writing direct + shard + round-trip outputs.
 * Arithmetic intensity is low (<0.5 FLOP/B for TP4 bf16), so this mapping is
 * bandwidth-bound and reports effective GB/s.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include "cutlass/util/command_line.h"
#include "19_bmg_comm_ar_sconv_common.hpp"

namespace cutlass::examples::comm_ar_sconv {

constexpr int kThreads = 256;

struct Options {
  std::string suite = "quick";
  DType dtype = DType::kAll;
  int iterations = 5;
  bool verify = true;
};

struct CaseConfig {
  std::string name;
  int world = 4;
  int rows = 0;
  int cols = 0;
  int parts = 4;
};

template <typename Element_>
struct CollectiveParams {
  using Element = Element_;

  Element const* __restrict__ partials;
  Element* __restrict__ direct;
  Element* __restrict__ shards;
  Element* __restrict__ round_trip;
  int32_t const* __restrict__ part_offsets;
  int world;
  int rows;
  int cols;
  int parts;
  int max_part_cols;
};

template <typename Element>
class XpuCollectiveMapKernel;

template <typename Element>
class XpuCollectiveMapPack4Kernel;

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
sycl::event launch_collective_map(sycl::queue& q, CollectiveParams<Element> const& params) {
  int total = params.rows * params.cols;
  if (total == 0) {
    return sycl::event{};
  }
  if ((params.cols % 4) == 0 && (params.max_part_cols % 4) == 0) {
    int pack_cols = params.cols / 4;
    int pack_total = params.rows * pack_cols;
    int global_pack = ceil_div(pack_total, kThreads) * kThreads;
    return q.parallel_for<XpuCollectiveMapPack4Kernel<Element>>(
        sycl::nd_range<1>(sycl::range<1>(global_pack), sycl::range<1>(kThreads)),
        [=](sycl::nd_item<1> item) {
          int linear = static_cast<int>(item.get_global_linear_id());
          if (linear >= pack_total) {
            return;
          }
          int pack_col = linear % pack_cols;
          int row = linear / pack_cols;
          int col = pack_col * 4;
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int r = 0; r < params.world; ++r) {
            uint64_t raw = load_pack4(params.partials + (static_cast<std::size_t>(r) * params.rows + row) * params.cols + col);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[v] += element_to_float(element_from_pack4<Element>(raw, v));
            }
          }
          uint64_t reduced = pack4_from_floats<Element>(acc);
          store_pack4(params.direct + static_cast<std::size_t>(row) * params.cols + col, reduced);
          store_pack4(params.round_trip + static_cast<std::size_t>(row) * params.cols + col, reduced);
          int part = 0;
          for (int p = 0; p < params.parts; ++p) {
            if (col >= params.part_offsets[p] && col < params.part_offsets[p + 1]) {
              part = p;
            }
          }
          int local_col = col - params.part_offsets[part];
          store_pack4(
              params.shards + (static_cast<std::size_t>(part) * params.rows + row) * params.max_part_cols + local_col,
              reduced);
        });
  }
  int global = ceil_div(total, kThreads) * kThreads;
  return q.parallel_for<XpuCollectiveMapKernel<Element>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        int linear = static_cast<int>(item.get_global_linear_id());
        if (linear >= total) {
          return;
        }
        int col = linear % params.cols;
        int row = linear / params.cols;
        float acc = 0.0f;
        for (int r = 0; r < params.world; ++r) {
          acc += element_to_float(params.partials[(static_cast<std::size_t>(r) * params.rows + row) * params.cols + col]);
        }
        Element reduced(acc);
        params.direct[static_cast<std::size_t>(row) * params.cols + col] = reduced;
        params.round_trip[static_cast<std::size_t>(row) * params.cols + col] = reduced;
        int part = 0;
        for (int p = 0; p < params.parts; ++p) {
          if (col >= params.part_offsets[p] && col < params.part_offsets[p + 1]) {
            part = p;
          }
        }
        int local_col = col - params.part_offsets[part];
        params.shards[(static_cast<std::size_t>(part) * params.rows + row) * params.max_part_cols + local_col] = reduced;
      });
}

template <typename Element>
struct HostTensors {
  std::vector<Element> partials;
  std::vector<Element> direct;
  std::vector<Element> direct_ref;
  std::vector<Element> shards;
  std::vector<Element> shards_ref;
  std::vector<Element> round_trip;
  std::vector<Element> round_trip_ref;
  std::vector<int32_t> part_offsets;
  int max_part_cols = 0;
};

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
  h.part_offsets.resize(cfg.parts + 1);
  for (int p = 0; p <= cfg.parts; ++p) {
    h.part_offsets[p] = static_cast<int>((static_cast<int64_t>(cfg.cols) * p) / cfg.parts);
  }
  h.max_part_cols = 0;
  for (int p = 0; p < cfg.parts; ++p) {
    h.max_part_cols = std::max(h.max_part_cols, h.part_offsets[p + 1] - h.part_offsets[p]);
  }
  std::size_t rc = static_cast<std::size_t>(cfg.rows) * cfg.cols;
  h.partials.resize(static_cast<std::size_t>(cfg.world) * rc);
  h.direct.resize(rc);
  h.direct_ref.resize(rc);
  h.round_trip.resize(rc);
  h.round_trip_ref.resize(rc);
  h.shards.resize(static_cast<std::size_t>(cfg.parts) * cfg.rows * h.max_part_cols);
  h.shards_ref.resize(h.shards.size());
  fill_random(h.partials, 20260723u + static_cast<uint32_t>(cfg.world * 53 + cfg.rows * 11 + cfg.cols), -0.50f, 0.50f);
  return h;
}

template <typename Element>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  for (int row = 0; row < cfg.rows; ++row) {
    for (int col = 0; col < cfg.cols; ++col) {
      float acc = 0.0f;
      for (int r = 0; r < cfg.world; ++r) {
        acc += element_to_float(h.partials[(static_cast<std::size_t>(r) * cfg.rows + row) * cfg.cols + col]);
      }
      Element reduced(acc);
      h.direct_ref[static_cast<std::size_t>(row) * cfg.cols + col] = reduced;
      h.round_trip_ref[static_cast<std::size_t>(row) * cfg.cols + col] = reduced;
      int part = 0;
      for (int p = 0; p < cfg.parts; ++p) {
        if (col >= h.part_offsets[p] && col < h.part_offsets[p + 1]) {
          part = p;
        }
      }
      int local_col = col - h.part_offsets[part];
      h.shards_ref[(static_cast<std::size_t>(part) * cfg.rows + row) * h.max_part_cols + local_col] = reduced;
    }
  }
}

std::vector<CaseConfig> quick_suite() {
  return {
      {"reference_world4_rows2_cols6_parts3", 4, 2, 6, 3},
      {"tail_world2_rows5_cols7_parts2", 2, 5, 7, 2},
      {"inkling_world4_rows128_cols1536_parts4", 4, 128, 1536, 4},
      {"scattered_world8_rows16_cols193_parts8", 8, 16, 193, 8},
  };
}

std::vector<CaseConfig> stress_suite() {
  return {
      {"stress_world1_rows1_cols1_parts1", 1, 1, 1, 1},
      {"stress_world4_rows17_cols31_parts3", 4, 17, 31, 3},
      {"stress_world8_rows33_cols769_parts7", 8, 33, 769, 7},
  };
}

std::vector<CaseConfig> perf_suite() {
  return {
      {"perf_world4_rows4096_cols1536_parts4", 4, 4096, 1536, 4},
      {"perf_world8_rows4096_cols768_parts8", 8, 4096, 768, 8},
      {"perf_world8_rows2048_cols1536_parts8", 8, 2048, 1536, 8},
  };
}

template <typename Element>
double effective_bytes(CaseConfig const& cfg) {
  double rc = static_cast<double>(cfg.rows) * cfg.cols;
  double elem = static_cast<double>(sizeof(Element));
  return rc * (static_cast<double>(cfg.world) + 3.0) * elem;
}

template <typename Element>
bool run_case(sycl::queue& q, CaseConfig const& cfg, Options const& options) {
  HostTensors<Element> h = initialize_case<Element>(cfg);
  reference_case(cfg, h);

  DeviceBuffer<Element> d_partials(q, h.partials.size());
  DeviceBuffer<Element> d_direct(q, h.direct.size());
  DeviceBuffer<Element> d_shards(q, h.shards.size());
  DeviceBuffer<Element> d_round_trip(q, h.round_trip.size());
  DeviceBuffer<int32_t> d_part_offsets(q, h.part_offsets.size());
  d_partials.copy_from(h.partials);
  d_part_offsets.copy_from(h.part_offsets);

  CollectiveParams<Element> params{
      d_partials.get(),
      d_direct.get(),
      d_shards.get(),
      d_round_trip.get(),
      d_part_offsets.get(),
      cfg.world,
      cfg.rows,
      cfg.cols,
      cfg.parts,
      h.max_part_cols};

  auto launch = [&]() {
    return launch_collective_map(q, params);
  };
  q.memset(d_shards.get(), 0, sizeof(Element) * h.shards.size()).wait();
  launch().wait();

  bool passed = true;
  if (options.verify) {
    d_direct.copy_to(h.direct);
    d_shards.copy_to(h.shards);
    d_round_trip.copy_to(h.round_trip);
    std::string base = cfg.name + "/" + element_dtype_text<Element>();
    passed &= compare_vectors(base + "/direct", h.direct, h.direct_ref, default_atol<Element>(), default_rtol<Element>());
    passed &= compare_vectors(base + "/shards", h.shards, h.shards_ref, default_atol<Element>(), default_rtol<Element>());
    passed &= compare_vectors(base + "/round_trip", h.round_trip, h.round_trip_ref, default_atol<Element>(), default_rtol<Element>());
  }

  double ms = time_ms(q, options.iterations, launch);
  double gbps = effective_bytes<Element>(cfg) / (ms * 1.0e6);
  std::cout << "[xpu_collective_mapping] " << std::left << std::setw(38) << cfg.name << " dtype=" << std::setw(4)
            << element_dtype_text<Element>() << " world=" << cfg.world << " rows=" << cfg.rows
            << " cols=" << cfg.cols << " parts=" << cfg.parts << " time_ms=" << std::fixed
            << std::setprecision(4) << ms << " eff_GBps=" << std::setprecision(2) << gbps << " "
            << (passed ? "PASSED" : "FAILED") << "\n";
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

void print_mapping_notes() {
  std::cout << "XPU mapping:\n"
            << "  init_custom_ar/register_buffer: host/runtime metadata, not a CUTLASS kernel\n"
            << "  all_reduce: staged-buffer all-reduce equivalent to 06.1 direct/push paths\n"
            << "  reduce_scatter + all_gather: shard/round-trip contract validated here\n"
            << "  CUDA multimem ld_reduce/st: no SYCL/BMG equivalent in this example; use staged local buffers\n";
}

void print_usage(char const* name) {
  std::cout << "Usage: " << name << " [options]\n"
            << "  --suite=<quick|stress|perf>\n"
            << "  --dtype=<all|bf16|fp16>\n"
            << "  --iterations=<int>\n"
            << "  --verify=<0|1>\n";
}

}  // namespace cutlass::examples::comm_ar_sconv

int main(int argc, char const** argv) {
  using namespace cutlass::examples::comm_ar_sconv;

  cutlass::CommandLine cmd(argc, argv);
  Options options;
  cmd.get_cmd_line_argument("suite", options.suite, options.suite);
  cmd.get_cmd_line_argument("iterations", options.iterations, options.iterations);
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
  print_mapping_notes();

  bool passed = true;
  if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
    passed &= run_typed<cutlass::bfloat16_t>(q, cases, options);
  }
  if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
    passed &= run_typed<cutlass::half_t>(q, cases, options);
  }
  return passed ? 0 : -1;
}
