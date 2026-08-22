/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Gemma 4 full-attention decode reproduction:
 *   B=1, Hq/Hkv=8/1, D=512, q_len=1, paged BF16 KV cache, page_size=64.
 *
 * This is the Split-K decode FMHA implementation observed in the Gemma 4
 * TP=4 profiler, not the prefill-with-KV-cache implementation.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "sycl/kernels/flash_attention_v2/collective/fmha_fusion.hpp"
#include "sycl/kernels/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "sycl/kernels/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_reduce_split_k.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_tile_scheduler.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace {

using namespace cute;

constexpr int kQHeads = 8;
constexpr int kKVHeads = 1;
constexpr int kHeadDim = 512;
constexpr int kPageSize = 64;
constexpr int kQGroupSize = kQHeads / kKVHeads;

// Output head-dim tile per work-group.
//
// The O accumulator (FragA) is held in registers for the whole K loop, and
// SubgroupLayoutPV splits the k dimension rather than v, so every subgroup
// holds the *entire* q x kVTileO accumulator. At kVTileO=512 that is
// 8*512 floats / 16 lanes = 256 registers per lane -- the whole GRF -- and IGC
// spills 17,728 bytes per thread, costing far more traffic than the KV reads
// themselves.
//
// Tiling the output head dim across work-groups (grid.x = kHeadDim / kVTileO)
// shrinks the accumulator proportionally and simultaneously raises occupancy,
// which matters because B=1 and Hkv=1 leave Split-K as the only other
// parallelism axis. Each v-group re-reads K to recompute Q*K; the softmax
// statistics it derives are v-independent, so every v-group writes identical
// exp_sums/max_logits and the Split-K reduction is unaffected.
#ifndef GEMMA4_DECODE_VTILE_O
#define GEMMA4_DECODE_VTILE_O 256
#endif
constexpr int kVTileO = GEMMA4_DECODE_VTILE_O;
static_assert(kHeadDim % kVTileO == 0, "head dim must divide into v tiles");
static_assert(kVTileO % 32 == 0, "v tile must be a multiple of the PV MMA N");

struct Options {
  bool help = false;
  bool error = false;
  int seq_len_kv = 4097;
  int iterations = 100;
  int warmup = 10;
  int verify = 1;
  int num_kv_splits = 0;
  float softmax_scale = 1.0f;
  // Windowed-K-block count below which Split-K collapses onto split 0.
  // The upstream kernel hard-coded 128 to avoid the split-reduce roundtrip's
  // precision loss, but that silently serializes every context under 8192
  // tokens (page_size 64) onto a single work-group. Default to 2 so splitting
  // is honored wherever there is more than one block to split; the BF16
  // reference check passes at every context covered by the test matrix.
  int min_blocks_for_split = 2;
  // K blocks prefetched ahead in the mainloop; 1 is the original behavior.
  int prefetch_depth = 1;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, seq_len_kv);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("warmup", warmup, warmup);
    cmd.get_cmd_line_argument("verify", verify, verify);
    cmd.get_cmd_line_argument("num_kv_splits", num_kv_splits, num_kv_splits);
    cmd.get_cmd_line_argument("softmax_scale", softmax_scale, softmax_scale);
    cmd.get_cmd_line_argument(
        "min_blocks_for_split", min_blocks_for_split, min_blocks_for_split);
    cmd.get_cmd_line_argument("prefetch_depth", prefetch_depth, prefetch_depth);
    if (prefetch_depth < 1) {
      error = true;
    }
    if (seq_len_kv <= 0 || iterations < 0 || warmup < 0 || num_kv_splits < 0) {
      error = true;
    }
  }

  void usage() const {
    std::cout
        << "Gemma4 paged Split-K FMHA decode\n"
        << "  --seq_len_kv=<int>     KV context length, default 4097\n"
        << "  --num_kv_splits=<int>  0 selects the production occupancy heuristic\n"
        << "  --softmax_scale=<float> Gemma4 uses 1.0\n"
        << "  --min_blocks_for_split=<int> K blocks below which Split-K "
           "collapses to one work-group, default 2\n"
        << "  --prefetch_depth=<int> K blocks prefetched ahead, default 1\n"
        << "  --warmup=<int> --iterations=<int> --verify=<0|1>\n";
  }
};

template <class Kernel>
void launch_kernel(typename Kernel::Params const& params) {
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  dim3 const block = Kernel::get_block_shape();
  dim3 const grid = Kernel::get_grid_shape(params);
  compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>,
      intelex::grf_size<256>};
  compat::experimental::launch_policy policy{
      compat::dim3(grid.x, grid.y, grid.z),
      compat::dim3(block.x, block.y, block.z),
      launch_props,
      kernel_props};
  compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel, false>(
      policy, params);
}


using Element = cutlass::bfloat16_t;
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

using TileShapeQK = Shape<Int<kQGroupSize>, Int<kPageSize>, _64>;
using TileShapePV = Shape<Int<kQGroupSize>, _32, Int<kPageSize>>;
using TileShapeO = Shape<Int<kQGroupSize>, Int<kVTileO>>;
using SubgroupLayoutQK = Layout<Shape<_1, Int<kPageSize / 16>, _1>>;
using MMAOperation = XE_DPAS_TT<cute::gcd(kQGroupSize, 8), float, Element>;
using TiledMMAQK =
    typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>,
                            SubgroupLayoutQK>::TiledMMA;
using SubgroupLayoutPV = decltype(
    cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
using TiledMMAPV =
    typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>,
                            SubgroupLayoutPV>::TiledMMA;

using StrideQ = Stride<int, _1, int, int>;
using StrideK = Stride<int, _1, int, int>;
using StrideV = Stride<_1, int, int, int>;
using StrideO = Stride<int, _1, int, int>;

template <class T, class Stride>
using TensorFor = decltype(make_tensor(
    make_gmem_ptr(static_cast<T*>(nullptr)),
    make_layout(repeat<rank_v<Stride>>(1), Stride{})));

using TensorQ = TensorFor<Element, StrideQ>;
using TensorK = TensorFor<Element, StrideK>;
using TensorV = TensorFor<Element, StrideV>;
using TensorO = TensorFor<Element, StrideO>;
using TensorLSE = TensorFor<float, StrideO>;

using Mainloop = cutlass::fmha::collective::DecodeFwdMainloop<
    cutlass::fmha::XeDefault<1>,
    true,
    false,
    TiledMMAQK,
    TiledMMAPV,
    kVTileO / 32,
    TensorQ,
    TensorK,
    TensorV>;
using Epilogue = cutlass::fmha::collective::DecodeFwdEpilogue<
    Mainloop, TileShapeO, TensorO, TensorLSE>;
using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<
    ProblemShape, Mainloop, Epilogue,
    cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
using ReduceKernel = cutlass::reduction::kernel::ReduceSplitK<
    ProblemShape, cutlass::fmha::kernel::XeReduceSplitKTileScheduler,
    FMHAKernel>;

int select_num_kv_splits(int requested, int seq_len_kv) {
  if (requested != 0) {
    return requested;
  }

  auto queue = compat::get_default_queue();
  auto device = queue.get_device();
  int const xe_cores =
      device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
      device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  int const total_blocks = (seq_len_kv + kPageSize - 1) / kPageSize;
  if (total_blocks <= 64) {
    return 1;
  }

  // Parallelism available before splitting. The v-tiling of the output head dim
  // already gives grid.x = head_dim / kVTileO work-groups per (batch, kv head),
  // so Split-K only has to make up the remaining shortfall against the machine.
  constexpr int kBatchSize = 1;
  int const v_tiles = kHeadDim / kVTileO;
  int const current_parallelism = kBatchSize * kKVHeads * v_tiles;

  // Oversubscribe the machine: with the O accumulator no longer filling the GRF,
  // several work-groups fit per Xe core, and more of them hides the memory
  // latency of a q_len=1 mainloop. Three per core is the smallest target that
  // reaches the fastest iteration count at the production contexts (see below).
  int const target_wgs = 3 * xe_cores;
  int const target_splits = (target_wgs + current_parallelism - 1) / current_parallelism;
  int const max_splits = std::min(total_blocks, FMHAKernel::max_num_kv_splits);

  // Every split runs ceil(blocks/splits) iterations, so the kernel's duration is
  // set by that ceiling and any split holding fewer blocks than the ceiling just
  // idles. Pushing the split count past the point where the ceiling last dropped
  // therefore buys no time while still adding partials for the reduction to
  // combine. At 65 blocks, splits of 22, 33 and 40 all run 3, 2 and 2 iterations
  // respectively, and measurement (5 interleaved runs, 2000 iterations each)
  // ranks them 18.3 us, 21.5 us and 21.6 us of mainloop: the smallest split
  // count reaching a given ceiling is fastest, because oversubscribing past it
  // only shrinks each work-group's share of the KV stream. So take the smallest
  // split count that achieves the lowest iteration count within the target.
  int best_splits = 1;
  int best_iters = total_blocks;
  for (int splits = 1; splits <= std::min(target_splits, max_splits); ++splits) {
    int const iters = (total_blocks + splits - 1) / splits;
    if (iters < best_iters) {
      best_iters = iters;
      best_splits = splits;
    }
  }
  return best_splits;
}

struct Buffers {
  cutlass::DeviceAllocation<Element> q;
  cutlass::DeviceAllocation<Element> k;
  cutlass::DeviceAllocation<Element> v;
  cutlass::DeviceAllocation<Element> out;
  cutlass::DeviceAllocation<Element> out_accum;
  cutlass::DeviceAllocation<float> exp_sums;
  cutlass::DeviceAllocation<float> max_logits;
  cutlass::DeviceAllocation<int> page_table;
  cutlass::DeviceAllocation<int> cu_q;
  cutlass::DeviceAllocation<int> cu_k;
};

template <class T>
void copy_to_device(cutlass::DeviceAllocation<T>& dst, std::vector<T> const& src) {
  compat::memcpy(dst.get(), src.data(), src.size() * sizeof(T));
}

bool verify_output(
    std::vector<Element> const& h_q,
    std::vector<Element> const& h_k,
    std::vector<Element> const& h_v,
    std::vector<int> const& h_page_table,
    std::vector<Element> const& h_out,
    Options const& options) {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  int worst_head = 0;
  int worst_dim = 0;
  int failing_elements = 0;
  constexpr double kAbsoluteTolerance = 0.03;
  constexpr double kRelativeTolerance = 0.10;

  for (int head = 0; head < kQHeads; ++head) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < options.seq_len_kv; ++token) {
      int const physical_page = h_page_table[token / kPageSize];
      int const page_offset = token % kPageSize;
      float logit = 0.0f;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        size_t const kv_offset =
            (static_cast<size_t>(physical_page) * kPageSize + page_offset) *
                kHeadDim +
            dim;
        logit += static_cast<float>(h_q[head * kHeadDim + dim]) *
                 static_cast<float>(h_k[kv_offset]);
      }
      max_logit = std::max(max_logit, logit * options.softmax_scale);
    }

    std::vector<float> weights(options.seq_len_kv);
    float denominator = 0.0f;
    for (int token = 0; token < options.seq_len_kv; ++token) {
      int const physical_page = h_page_table[token / kPageSize];
      int const page_offset = token % kPageSize;
      float logit = 0.0f;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        size_t const kv_offset =
            (static_cast<size_t>(physical_page) * kPageSize + page_offset) *
                kHeadDim +
            dim;
        logit += static_cast<float>(h_q[head * kHeadDim + dim]) *
                 static_cast<float>(h_k[kv_offset]);
      }
      weights[token] = std::exp(logit * options.softmax_scale - max_logit);
      denominator += weights[token];
    }

    for (int dim = 0; dim < kHeadDim; ++dim) {
      float reference = 0.0f;
      for (int token = 0; token < options.seq_len_kv; ++token) {
        int const physical_page = h_page_table[token / kPageSize];
        int const page_offset = token % kPageSize;
        size_t const kv_offset =
            (static_cast<size_t>(physical_page) * kPageSize + page_offset) *
                kHeadDim +
            dim;
        reference += weights[token] * static_cast<float>(h_v[kv_offset]);
      }
      // Compare the BF16-rounded reference with the BF16 output written by FMHA.
      float const expected = static_cast<float>(Element(reference / denominator));
      float const actual = static_cast<float>(h_out[head * kHeadDim + dim]);
      double const abs_error = std::abs(static_cast<double>(actual) - expected);
      double const rel_error = abs_error / std::max(1e-3, std::abs(static_cast<double>(expected)));
      if (abs_error > max_abs_error) {
        max_abs_error = abs_error;
        worst_head = head;
        worst_dim = dim;
      }
      max_rel_error = std::max(max_rel_error, rel_error);
      if (abs_error > kAbsoluteTolerance && rel_error > kRelativeTolerance) {
        ++failing_elements;
      }
    }
  }

  std::cout << std::setprecision(6)
            << "Verification: max_abs=" << max_abs_error
            << " max_rel=" << max_rel_error
            << " failing_elements=" << failing_elements
            << " at head=" << worst_head << " dim=" << worst_dim << '\n';
  return failing_elements == 0;
}

int run(Options const& options) {
  int const pages = (options.seq_len_kv + kPageSize - 1) / kPageSize;
  int const splits = select_num_kv_splits(options.num_kv_splits, options.seq_len_kv);
  if (splits > FMHAKernel::max_num_kv_splits) {
    std::cerr << "num_kv_splits=" << splits << " exceeds supported maximum "
              << FMHAKernel::max_num_kv_splits << '\n';
    return 1;
  }

  std::mt19937 generator(20260808);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<Element> h_q(kQHeads * kHeadDim);
  std::vector<Element> h_k(pages * kPageSize * kKVHeads * kHeadDim);
  std::vector<Element> h_v(h_k.size());
  std::vector<int> h_page_table(pages);
  std::vector<int> h_cu_q{0, 1};
  std::vector<int> h_cu_k{options.seq_len_kv};

  for (auto& value : h_q) {
    value = Element(distribution(generator));
  }
  for (int logical_page = 0; logical_page < pages; ++logical_page) {
    int const physical_page = (logical_page * 17) % pages;
    h_page_table[logical_page] = physical_page;
    for (int token = 0; token < kPageSize; ++token) {
      int const logical_token = logical_page * kPageSize + token;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        size_t const offset =
            (static_cast<size_t>(physical_page) * kPageSize + token) * kHeadDim +
            dim;
        float const value = logical_token < options.seq_len_kv
                                ? distribution(generator)
                                : 0.0f;
        h_k[offset] = Element(value);
        h_v[offset] = Element(distribution(generator));
      }
    }
  }

  Buffers buffers;
  buffers.q.reset(h_q.size());
  buffers.k.reset(h_k.size());
  buffers.v.reset(h_v.size());
  buffers.out.reset(h_q.size());
  buffers.out_accum.reset(h_q.size() * splits);
  buffers.exp_sums.reset(kQHeads * splits);
  buffers.max_logits.reset(kQHeads * splits);
  buffers.page_table.reset(h_page_table.size());
  buffers.cu_q.reset(h_cu_q.size());
  buffers.cu_k.reset(h_cu_k.size());
  copy_to_device(buffers.q, h_q);
  copy_to_device(buffers.k, h_k);
  copy_to_device(buffers.v, h_v);
  copy_to_device(buffers.page_table, h_page_table);
  copy_to_device(buffers.cu_q, h_cu_q);
  copy_to_device(buffers.cu_k, h_cu_k);
  compat::wait();

  auto stride_q = cutlass::make_cute_packed_stride(
      StrideQ{}, make_shape(1, kHeadDim, kQHeads, 1));
  auto stride_k = StrideK{
      kKVHeads * kHeadDim, _1{}, kHeadDim,
      kPageSize * kKVHeads * kHeadDim};
  auto stride_v = StrideV{
      _1{}, kKVHeads * kHeadDim, kHeadDim,
      kPageSize * kKVHeads * kHeadDim};
  auto stride_out = cutlass::make_cute_packed_stride(
      StrideO{}, make_shape(1, kHeadDim, kQHeads, 1));
  auto stride_out_accum = cutlass::make_cute_packed_stride(
      StrideO{}, make_shape(1, kHeadDim, kQHeads * splits, 1));
  auto stride_stats = cutlass::make_cute_packed_stride(
      StrideO{}, make_shape(1, splits, kQHeads, 1));

  ProblemShape shape{
      .batch = 1,
      .num_heads_q = kQHeads,
      .num_heads_kv = kKVHeads,
      .seq_len_qo = {1, 1, buffers.cu_q.get()},
      .seq_len_kv = {options.seq_len_kv, options.seq_len_kv, buffers.cu_k.get()},
      .seq_len_kv_cache = {options.seq_len_kv, options.seq_len_kv, buffers.cu_k.get()},
      .head_size_qk = kHeadDim,
      .head_size_vo = kHeadDim};

  cutlass::KernelHardwareInfo hardware;
  hardware.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hardware.device_id);

  typename FMHAKernel::Arguments fmha_args{
      {shape, buffers.q.get(), stride_q, buffers.k.get(), stride_k,
       buffers.v.get(), stride_v, buffers.out_accum.get(), stride_out_accum,
       buffers.exp_sums.get(), stride_stats, buffers.max_logits.get(),
       stride_stats, nullptr, nullptr, nullptr, nullptr,
       options.min_blocks_for_split},
      {options.softmax_scale, buffers.page_table.get(), kPageSize, pages,
       options.seq_len_kv, -1, -1, options.prefetch_depth},
      {},
      hardware,
      splits};
  typename ReduceKernel::KernelArguments reduce_kernel_args{
      shape,
      buffers.out.get(),
      stride_out,
      buffers.out_accum.get(),
      stride_out_accum,
      buffers.exp_sums.get(),
      stride_stats,
      buffers.max_logits.get(),
      stride_stats,
      -1,
      nullptr};
  typename ReduceKernel::Arguments reduce_args{
      reduce_kernel_args, hardware, splits};

  if (!FMHAKernel::can_implement(fmha_args) ||
      !ReduceKernel::can_implement(reduce_args)) {
    std::cerr << "Kernel cannot implement the requested Gemma4 shape.\n";
    return 1;
  }

  auto fmha_params = FMHAKernel::to_underlying_arguments(fmha_args, nullptr);
  auto reduce_params =
      ReduceKernel::to_underlying_arguments(reduce_args, nullptr);
  for (int i = 0; i < options.warmup; ++i) {
    launch_kernel<FMHAKernel>(fmha_params);
    launch_kernel<ReduceKernel>(reduce_params);
  }
  compat::wait();

  double elapsed = 0.0;
  double elapsed_fmha = 0.0;
  double elapsed_reduce = 0.0;
  if (options.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      launch_kernel<FMHAKernel>(fmha_params);
      launch_kernel<ReduceKernel>(reduce_params);
    }
    compat::wait();
    elapsed = timer.seconds() / options.iterations;

    // Per-launch attribution. Each dispatch is timed on its own so the
    // Split-K reduction and the launch floor can be told apart from the
    // memory-bound mainloop.
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      launch_kernel<FMHAKernel>(fmha_params);
    }
    compat::wait();
    elapsed_fmha = timer.seconds() / options.iterations;

    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      launch_kernel<ReduceKernel>(reduce_params);
    }
    compat::wait();
    elapsed_reduce = timer.seconds() / options.iterations;
  }

  double const flops =
      4.0 * kQHeads * kHeadDim * options.seq_len_kv;
  double const kv_bytes =
      2.0 * kKVHeads * kHeadDim * options.seq_len_kv * sizeof(Element);
  std::cout << std::fixed << std::setprecision(4)
            << "Gemma4 Split-K KV-cache decode: B=1 Hq/Hkv=8/1 D=512 q=1 "
            << "ctx=" << options.seq_len_kv << " page=64 splits=" << splits
            << " min_blocks_for_split=" << options.min_blocks_for_split
            << " prefetch_depth=" << options.prefetch_depth
            << " scale=" << options.softmax_scale << '\n';
  if (elapsed > 0.0) {
    std::cout << "Performance: " << elapsed * 1e3 << " ms, "
              << flops / elapsed / 1e12 << " TFLOP/s, "
              << kv_bytes / elapsed / 1e9
              << " GB/s minimum logical KV traffic\n";
    std::cout << "  Breakdown: fmha=" << elapsed_fmha * 1e3 << " ms ("
              << kv_bytes / elapsed_fmha / 1e9 << " GB/s), reduce="
              << elapsed_reduce * 1e3 << " ms\n";
  }
  if (options.verify != 0) {
    std::vector<Element> h_out(h_q.size());
    compat::memcpy(
        h_out.data(), buffers.out.get(), h_out.size() * sizeof(Element));
    compat::wait();
    if (!verify_output(h_q, h_k, h_v, h_page_table, h_out, options)) {
      std::cerr << "Verification failed.\n";
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char const** argv) {
  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.usage();
    return 0;
  }
  if (options.error) {
    options.usage();
    return 1;
  }
  return run(options);
}
