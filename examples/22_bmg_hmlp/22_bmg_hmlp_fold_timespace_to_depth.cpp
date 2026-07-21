/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "22_bmg_hmlp_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hmlp = cutlass::examples::bmg_hmlp;

namespace {

enum class DType {
  kAll,
  kF32,
  kBf16,
  kFp16
};

struct Options {
  std::string suite = "quick";
  std::string shape;
  DType dtype = DType::kAll;
  int iterations = 20;
  int warmup = 5;
  bool verify = true;
  bool benchmark = true;
  bool target_gbps_set = false;
  double target_gbps = 0.0;
  bool help = false;
};

struct FoldCase {
  std::string name;
  int B = 1;
  int T = 1;
  int H = 1;
  int W = 1;
  int C = 1;
  int t_fold = 1;
  int hw_fold = 1;
  double target_gbps = 0.0;
};

bool parse_bool(std::string const& value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool parse_dtype(std::string const& text, DType& dtype) {
  if (text == "all") {
    dtype = DType::kAll;
    return true;
  }
  if (text == "f32" || text == "float") {
    dtype = DType::kF32;
    return true;
  }
  if (text == "bf16") {
    dtype = DType::kBf16;
    return true;
  }
  if (text == "fp16" || text == "f16") {
    dtype = DType::kFp16;
    return true;
  }
  return false;
}

std::string dtype_text(DType dtype) {
  switch (dtype) {
    case DType::kAll: return "all";
    case DType::kF32: return "f32";
    case DType::kBf16: return "bf16";
    case DType::kFp16: return "fp16";
  }
  return "unknown";
}

Options parse_options(int argc, char const** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto eq = arg.find('=');
    std::string key = eq == std::string::npos ? arg : arg.substr(0, eq);
    std::string value = eq == std::string::npos ? "" : arg.substr(eq + 1);
    if (key == "--help" || key == "-h") {
      options.help = true;
    } else if (key == "--suite") {
      options.suite = value;
    } else if (key == "--shape") {
      options.shape = value;
    } else if (key == "--dtype") {
      if (!parse_dtype(value, options.dtype)) {
        throw std::invalid_argument("unknown dtype: " + value);
      }
    } else if (key == "--iterations") {
      options.iterations = std::stoi(value);
    } else if (key == "--warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "--verify") {
      options.verify = parse_bool(value);
    } else if (key == "--benchmark") {
      options.benchmark = parse_bool(value);
    } else if (key == "--target-gbps") {
      options.target_gbps = std::stod(value);
      options.target_gbps_set = true;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return options;
}

void print_usage(char const* name) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=quick|inkling|perf\n"
      << "  --shape=name=<s>,b=<int>,t=<int>,h=<int>,w=<int>,c=<int>,tf=<int>,hwf=<int>\n"
      << "  --dtype=all|f32|bf16|fp16\n"
      << "  --iterations=<int>       Timed kernel iterations; 0 skips timing\n"
      << "  --warmup=<int>           Warmup launches before timing\n"
      << "  --verify=0|1             Run exact CPU reference comparison\n"
      << "  --benchmark=0|1          Run profiling-event timing\n"
      << "  --target-gbps=<float>    Optional sustained effective GB/s gate\n";
}

bool parse_shape(std::string const& text, FoldCase& cfg) {
  if (text.empty()) {
    return true;
  }
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto eq = item.find('=');
    if (eq == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, eq);
    std::string value = item.substr(eq + 1);
    if (key == "name") {
      cfg.name = value;
    } else if (key == "b") {
      cfg.B = std::stoi(value);
    } else if (key == "t") {
      cfg.T = std::stoi(value);
    } else if (key == "h") {
      cfg.H = std::stoi(value);
    } else if (key == "w") {
      cfg.W = std::stoi(value);
    } else if (key == "c") {
      cfg.C = std::stoi(value);
    } else if (key == "tf") {
      cfg.t_fold = std::stoi(value);
    } else if (key == "hwf") {
      cfg.hw_fold = std::stoi(value);
    } else if (key == "target_gbps") {
      cfg.target_gbps = std::stod(value);
    } else {
      return false;
    }
  }
  return true;
}

void validate_case(FoldCase& cfg) {
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
  if (cfg.B <= 0 || cfg.T <= 0 || cfg.H <= 0 || cfg.W <= 0 || cfg.C <= 0) {
    throw std::invalid_argument(cfg.name + " has a non-positive shape dimension");
  }
  if (cfg.t_fold <= 0 || cfg.hw_fold <= 0) {
    throw std::invalid_argument(cfg.name + " has a non-positive fold factor");
  }
  if (cfg.T % cfg.t_fold != 0 || cfg.H % cfg.hw_fold != 0 || cfg.W % cfg.hw_fold != 0) {
    throw std::invalid_argument(cfg.name + " fold factors do not divide T/H/W");
  }
}

int64_t element_count(FoldCase const& cfg) {
  return static_cast<int64_t>(cfg.B) * cfg.T * cfg.H * cfg.W * cfg.C;
}

std::vector<FoldCase> quick_suite() {
  return {
      {"oracle_hw3_tf2_c3", 2, 4, 6, 6, 3, 2, 3, 0.0},
      {"oracle_t3_hw2_c2", 1, 6, 4, 8, 2, 3, 2, 0.0},
      {"identity_tail_c5", 2, 2, 2, 2, 5, 1, 1, 0.0},
      {"nonpow3_tail_c17", 3, 3, 9, 6, 17, 3, 3, 0.0},
      {"patch16_rgb_batch", 512, 1, 16, 16, 3, 1, 16, 0.0},
      {"layer_hw2_c64", 128, 1, 8, 8, 64, 1, 2, 0.0},
  };
}

std::vector<FoldCase> inkling_suite() {
  // The Inkling vision tower (HMLPPatchEncoder) is a plain nn.Linear stack with
  // no TP shard/all-gather -- every rank runs the full replicated tower -- so
  // TP=1/2/4/8 all consume identical fold shapes. The cases below track the
  // real per-layer fold sequence emitted by plan_out_scales() at the shipped
  // vision configs, and the patch=14 branch adds the non-pow2 hwf=7 fold.
  return {
      // Shipped n_layers=1 case: single 16x16 spatial fold on the RGB input.
      {"patch16_n1_rgb_8k", 8192, 1, 16, 16, 3, 1, 16, 0.0},
      // Shipped n_layers=4 case (patch_size=16, tps=1) per-layer fold sequence.
      // Each layer folds hw=2; the last spatial stage collapses to 1x1 before
      // the final linear projects to decoder_dmodel (1536 cfg / 6144 prod).
      {"patch16_n4_L0_rgb_hw2", 4096, 1, 16, 16, 3, 1, 2, 0.0},
      {"patch16_n4_L1_hw2_c64", 4096, 1, 8, 8, 64, 1, 2, 0.0},
      {"patch16_n4_L2_hw2_c64", 4096, 1, 4, 4, 64, 1, 2, 0.0},
      {"patch16_n4_L3_hw2_c192", 4096, 1, 2, 2, 192, 1, 2, 0.0},
      // Alt shipped n_layers=2 stage L1 (fold hw=4 aggregates 4x4 patches).
      {"patch16_n2_L1_hw4_c64", 2048, 1, 4, 4, 64, 1, 4, 0.0},
      // patch_size=14 branch: primes are {7,2} so the first layer folds hw=7,
      // exercising a non-pow2 hw stride. Keep temporal_nonpow_tf3 for tf-side
      // coverage; this row is the hw-side twin.
      {"patch14_n2_L0_hw7_c3", 1024, 1, 14, 14, 3, 1, 7, 0.0},
      // temporal_patch_size=2 tail fold: purely temporal fold with hw=1 (a fold
      // that reshapes but never permutes spatial dims), which used to hit the
      // identity_tail path -- covered explicitly at the shipped 16x16 grid.
      {"tps2_L3_temporal_c768", 256, 2, 16, 16, 768, 2, 1, 0.0},
      // Legacy inkling coverage retained.
      {"spatial_stage_hw2_c128", 2048, 1, 4, 4, 128, 1, 2, 0.0},
      {"temporal_stage_tf2_c128", 1024, 4, 4, 4, 128, 2, 2, 0.0},
      {"temporal_nonpow_tf3", 512, 6, 6, 6, 96, 3, 3, 0.0},
      {"full_spacetime_fold", 4096, 2, 4, 4, 64, 2, 4, 0.0},
  };
}

std::vector<FoldCase> perf_suite() {
  // Vision tower runs on every rank (no TP shard), so the perf sweep exercises
  // the same per-layer shapes at a larger num_patches to hit the vectorized
  // paths. Sizes scale num_patches proportional to the C x hw_fold_area growth
  // so total bytes stay within a comparable band.
  return {
      {"perf_patch16_n1_rgb_64k", 65536, 1, 16, 16, 3, 1, 16, 170.0},
      // Shipped n_layers=4 per-layer perf shapes.
      {"perf_patch16_n4_L0_rgb", 32768, 1, 16, 16, 3, 1, 2, 140.0},
      {"perf_patch16_n4_L1_c64", 16384, 1, 8, 8, 64, 1, 2, 170.0},
      {"perf_patch16_n4_L2_c64", 16384, 1, 4, 4, 64, 1, 2, 160.0},
      {"perf_patch16_n4_L3_c192", 16384, 1, 2, 2, 192, 1, 2, 170.0},
      // patch=14 non-pow2 first-layer fold at scale.
      {"perf_patch14_n2_L0_hw7", 8192, 1, 14, 14, 3, 1, 7, 80.0},
      // Legacy perf entries kept for regression continuity.
      {"perf_spatial_hw2_c64", 16384, 1, 8, 8, 64, 1, 2, 170.0},
      {"perf_spatial_hw2_c256", 8192, 1, 8, 8, 256, 1, 2, 170.0},
      {"perf_temporal_tf2_c128", 4096, 4, 4, 4, 128, 2, 2, 170.0},
  };
}

std::vector<FoldCase> make_suite(std::string const& suite) {
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

template <typename Element>
std::string element_name() {
  if constexpr (std::is_same_v<Element, float>) {
    return "f32";
  } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  }
  return "fp16";
}

template <typename Element>
Element make_input_value(int64_t index) {
  uint32_t bits = static_cast<uint32_t>((index * 1103515245ull + 12345ull) >> 8);
  float value = static_cast<float>(static_cast<int>(bits % 4096) - 2048) / 257.0f;
  return static_cast<Element>(value);
}

template <typename Element>
std::vector<Element> make_input(FoldCase const& cfg) {
  std::vector<Element> input(static_cast<std::size_t>(element_count(cfg)));
  for (int64_t i = 0; i < element_count(cfg); ++i) {
    input[static_cast<std::size_t>(i)] = make_input_value<Element>(i);
  }
  return input;
}

template <typename Element>
std::vector<Element> reference_fold(FoldCase const& cfg, std::vector<Element> const& input) {
  int t_new = cfg.T / cfg.t_fold;
  int h_new = cfg.H / cfg.hw_fold;
  int w_new = cfg.W / cfg.hw_fold;
  int fold_count = cfg.t_fold * cfg.hw_fold * cfg.hw_fold;
  std::vector<Element> expected(input.size());

  for (int b = 0; b < cfg.B; ++b) {
    for (int t_out = 0; t_out < t_new; ++t_out) {
      for (int h_out = 0; h_out < h_new; ++h_out) {
        for (int w_out = 0; w_out < w_new; ++w_out) {
          int64_t outer = (((static_cast<int64_t>(b) * t_new + t_out) * h_new + h_out) * w_new + w_out);
          for (int fold = 0; fold < fold_count; ++fold) {
            int wf = fold % cfg.hw_fold;
            int fold_tmp = fold / cfg.hw_fold;
            int hf = fold_tmp % cfg.hw_fold;
            int tf = fold_tmp / cfg.hw_fold;
            int64_t dst = (outer * fold_count + fold) * cfg.C;
            int64_t src = (((static_cast<int64_t>(b) * cfg.T + t_out * cfg.t_fold + tf) * cfg.H +
                            h_out * cfg.hw_fold + hf) *
                               cfg.W +
                           w_out * cfg.hw_fold + wf) *
                cfg.C;
            for (int c = 0; c < cfg.C; ++c) {
              expected[static_cast<std::size_t>(dst + c)] = input[static_cast<std::size_t>(src + c)];
            }
          }
        }
      }
    }
  }
  return expected;
}

template <typename Element>
bool raw_equal(Element a, Element b) {
  if constexpr (std::is_same_v<Element, float>) {
    return a == b;
  } else {
    return a.raw() == b.raw();
  }
}

template <typename Element>
void verify_case(sycl::queue& queue, FoldCase const& cfg) {
  std::vector<Element> input = make_input<Element>(cfg);
  std::vector<Element> expected = reference_fold(cfg, input);
  std::vector<Element> got(expected.size());

  hmlp::DeviceBuffer<Element> d_input(queue, input.size());
  hmlp::DeviceBuffer<Element> d_output(queue, got.size());
  d_input.copy_from(input);
  hmlp::launch_fold_timespace_to_depth(
      queue,
      d_input.get(),
      d_output.get(),
      cfg.B,
      cfg.T,
      cfg.H,
      cfg.W,
      cfg.C,
      cfg.t_fold,
      cfg.hw_fold)
      .wait();
  d_output.copy_to(got);

  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!raw_equal(got[i], expected[i])) {
      throw std::runtime_error(
          cfg.name + " " + element_name<Element>() + " mismatch at " + std::to_string(i));
    }
  }
}

double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

template <typename Element>
void benchmark_case(sycl::queue& queue, FoldCase const& cfg, Options const& options) {
  if (options.iterations == 0) {
    return;
  }

  std::vector<Element> input = make_input<Element>(cfg);
  hmlp::DeviceBuffer<Element> d_input(queue, input.size());
  hmlp::DeviceBuffer<Element> d_output(queue, input.size());
  d_input.copy_from(input);

  for (int i = 0; i < options.warmup; ++i) {
    hmlp::launch_fold_timespace_to_depth(
        queue,
        d_input.get(),
        d_output.get(),
        cfg.B,
        cfg.T,
        cfg.H,
        cfg.W,
        cfg.C,
        cfg.t_fold,
        cfg.hw_fold);
  }
  queue.wait();

  std::vector<sycl::event> events;
  events.reserve(static_cast<std::size_t>(options.iterations));
  for (int i = 0; i < options.iterations; ++i) {
    events.push_back(hmlp::launch_fold_timespace_to_depth(
        queue,
        d_input.get(),
        d_output.get(),
        cfg.B,
        cfg.T,
        cfg.H,
        cfg.W,
        cfg.C,
        cfg.t_fold,
        cfg.hw_fold));
  }
  queue.wait();

  double total_ms = 0.0;
  for (auto const& event : events) {
    total_ms += event_ms(event);
  }
  double avg_ms = total_ms / static_cast<double>(options.iterations);
  double bytes = 2.0 * static_cast<double>(element_count(cfg)) * static_cast<double>(sizeof(Element));
  double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
  double target = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;

  int t_new = cfg.T / cfg.t_fold;
  int h_new = cfg.H / cfg.hw_fold;
  int w_new = cfg.W / cfg.hw_fold;
  bool contiguous = hmlp::is_contiguous_reinterpret(
      hmlp::make_fold_params<Element>(
          d_input.get(), d_output.get(), cfg.B, cfg.T, cfg.H, cfg.W, cfg.C, cfg.t_fold, cfg.hw_fold));

  std::cout << "bench " << std::setw(24) << cfg.name
            << " dtype=" << std::setw(4) << element_name<Element>()
            << " in=[" << cfg.B << "," << cfg.T << "," << cfg.H << "," << cfg.W << "," << cfg.C << "]"
            << " out=[" << cfg.B << "," << t_new << "," << h_new << "," << w_new << ","
            << cfg.t_fold * cfg.hw_fold * cfg.hw_fold * cfg.C << "]"
            << " fold=(" << cfg.t_fold << "," << cfg.hw_fold << ")"
            << " path=" << (contiguous ? "memcpy" : "shuffle")
            << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms
            << " GB/s=" << std::setprecision(2) << gbps << "\n";

  if (target > 0.0 && gbps < target) {
    throw std::runtime_error(
        cfg.name + " " + element_name<Element>() + " bandwidth " + std::to_string(gbps) +
        " GB/s below target " + std::to_string(target));
  }
}

template <typename Element>
bool run_cases_for_dtype(sycl::queue& queue, std::vector<FoldCase> const& cases, Options const& options) {
  for (FoldCase const& cfg : cases) {
    if (options.verify) {
      verify_case<Element>(queue, cfg);
      std::cout << "verify " << std::setw(24) << cfg.name
                << " dtype=" << std::setw(4) << element_name<Element>()
                << " elems=" << element_count(cfg) << " ok\n";
    }
    if (options.benchmark) {
      benchmark_case<Element>(queue, cfg, options);
    }
  }
  return true;
}

}  // namespace

int main(int argc, char const** argv) {
  try {
    Options options = parse_options(argc, argv);
    if (options.help) {
      print_usage(argv[0]);
      return 0;
    }
    if (options.iterations < 0 || options.warmup < 0) {
      throw std::invalid_argument("--iterations and --warmup must be non-negative");
    }

    std::vector<FoldCase> cases;
    if (!options.shape.empty()) {
      FoldCase cfg;
      if (!parse_shape(options.shape, cfg)) {
        throw std::invalid_argument("invalid --shape string: " + options.shape);
      }
      validate_case(cfg);
      cases.push_back(cfg);
    } else {
      cases = make_suite(options.suite);
      if (cases.empty()) {
        throw std::invalid_argument("--suite must be quick, inkling, or perf");
      }
      for (FoldCase& cfg : cases) {
        validate_case(cfg);
      }
    }

    sycl::queue queue(
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});

    std::cout << "device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "22_bmg_hmlp_fold_timespace_to_depth"
              << " suite=" << options.suite
              << " dtype=" << dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " verify=" << (options.verify ? "true" : "false")
              << " benchmark=" << (options.benchmark ? "true" : "false") << "\n";

    bool all_passed = true;
    if (options.dtype == DType::kAll || options.dtype == DType::kF32) {
      all_passed &= run_cases_for_dtype<float>(queue, cases, options);
    }
    if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
      all_passed &= run_cases_for_dtype<cutlass::bfloat16_t>(queue, cases, options);
    }
    if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
      all_passed &= run_cases_for_dtype<cutlass::half_t>(queue, cases, options);
    }
    return all_passed ? 0 : 1;
  } catch (std::exception const& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
