/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "fmha_fusion.hpp"
#include "xe_fmha_cache_hint_copy.hpp"

#ifndef FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
#define FMHA_PREFILL_ENABLE_SCORE_BLOCK2D 0
#endif

// Number of consecutive K blocks whose Q*K GEMM shares one pass of Q loads.
// Q re-read traffic in GEMM1 is heads*seq_q*head_dim*2*(seq_kv/TILED_KV) bytes --
// independent of TILED_Q, so the only ways to shrink it are a wider K step (which
// spills at head_dim=512) or amortizing the Q loads over a group of K blocks.
// Grouping costs one extra S accumulator per additional block and leaves the K/P
// fragment widths alone, which is why it fits where TILED_KV=128 does not.
#ifndef FMHA_PREFILL_QK_GROUP
#define FMHA_PREFILL_QK_GROUP 1
#endif

// Give GEMM1 a launch of its own in the ScoreBlock2D path: mode 0 computes Q*K and
// stores the scores but runs no GEMM2 and no epilogue, and each later mode owns one
// output tile. Mode 0 then needs no O accumulator, which is the single largest
// register consumer (rows_per_SG * TILED_OUT floats), so the freed budget is what
// lets FMHA_PREFILL_QK_GROUP > 1 fit -- at group 1 it spills 3.2KB/thread. The cost
// is one extra pass over the score scratch, since no launch now fuses store with a
// tile. Only worth it when GEMM1 dominates, which it does at head_dim=512.
#ifndef FMHA_PREFILL_SPLIT_STORE
#define FMHA_PREFILL_SPLIT_STORE 0
#endif

// Alternate the direction of GEMM1's head-dim walk between consecutive K blocks.
#ifndef FMHA_PREFILL_ZIGZAG_D
#define FMHA_PREFILL_ZIGZAG_D 1
#endif

// Cap how many head-dim chunks the *initial* Q/K prefetch issues (0 = all). The WG's Q
// tile is 256KB at head_dim=512, already L1-sized, so prefetching all 16 chunks plus K
// up front evicts the front of Q before GEMM1 reaches it. Addresses the "limit D
// prefetch for large head size" TODO below.
#ifndef FMHA_PREFILL_INIT_PF_DEPTH
#define FMHA_PREFILL_INIT_PF_DEPTH 0
#endif

// Issue the next K block's head-dim prefetches in the order that block will actually
// consume them. With ZigzagD the walk direction flips every block, so a fixed 0..nD-1
// prefetch hands the reverse-walking block its first-needed chunk last -- and at
// head_dim=512 there are 16 chunks in flight, so that chunk can be evicted before use.
// Addresses the "reorder K prefetches" TODO below. Costs no registers.
#ifndef FMHA_PREFILL_PF_ZIGZAG
#define FMHA_PREFILL_PF_ZIGZAG 0
#endif

// Stage GEMM1's K through SLM, a head-dim *group* at a time. SubgroupLayoutQK splits Q
// only, so all NUM_SG subgroups load the entire K block out of L1: K's request volume is
// NUM_SG x its footprint, 8x Q's at head_dim=512 (see the K-request-volume analysis).
// Staging cooperatively means each subgroup loads 1/NUM_SG of the group from global once
// and everyone then reads it from SLM.
//
// A group, rather than a single reduction step, for the two reasons that killed the
// per-step attempt:
//   1. make_coop_block_2d_copy_B cannot split a tile more ways than it has DPAS atom
//      blocks, and one step's (TILED_KV, KSTEP) K tile has only (TILED_KV/16)*(KSTEP/16)
//      = 8 -> an 8-way split, not 32. A group of K_SLM steps has K_SLM times as many.
//   2. Per-step staging costs 2 workgroup barriers per step = 2 * head_dim/KSTEP = 32 per
//      K block; a group of 4 costs 8.
//
// K_SLM is that group width in units of KSTEP, so it must satisfy
// (TILED_KV/16) * (KSTEP*K_SLM/16) >= NUM_SG: at TILED_KV=64, KSTEP=32, NUM_SG=32 the
// minimum is 4, giving a 128-wide head-dim group and 16KB of SLM. 0 disables staging.
//
// MEASURED AND REJECTED, kept default-off as a recorded refutation. At
// b1/hq32/sq4096/sk4096/dv512: 13.7 TFLOPS staged vs 47.7 direct, correct either way
// (bad=0 on all six verify shapes). FMHA_PREFILL_K_SLM_STAGES=2 changes nothing (13.6 ->
// 13.7), so it is not exposed load latency, and FMHA_PREFILL_K_SLM_NO_S2R splits the cost:
// keeping the cooperative load + SLM write but feeding the DPAS from global still only
// reaches 34.2. So the write side costs ~13 TFLOPS and the SLM->register read another ~21.
// The redundant L1 requests this was meant to remove are worth only 3.3 TFLOPS (the pin-K
// probe), and the read side alone cannot be removed -- the DPAS needs its B operand in
// registers -- so no variant of this approach can pay for itself. The SLM round trip is
// simply more expensive than re-requesting K from L1.
#ifndef FMHA_PREFILL_K_SLM
#define FMHA_PREFILL_K_SLM 0
#endif

// Number of SLM buffers K staging rotates through. With one buffer the global load for
// group g+1 cannot start until every subgroup has finished reading group g, so the load
// latency is fully exposed and GEMM1 stalls on it; measured 13.6 TFLOPS vs 47.7 direct.
// Two or more let the next group's load overlap this group's DPAS, at K_SLM * TILED_KV *
// KSTEP * sizeof(ElementK) = 16KB of SLM per stage.
#ifndef FMHA_PREFILL_K_SLM_STAGES
#define FMHA_PREFILL_K_SLM_STAGES 2
#endif

// Diagnostic: keep the whole staging path (cooperative global load, SLM write, barriers)
// but feed the DPAS from the direct global K load instead of from SLM. Results stay
// correct -- the staged copy simply becomes dead weight -- so this splits the staged
// path's cost into "write side" (which this keeps) and "read side" (which it drops)
// without a skip-the-load probe's instruction-stream distortion.
#ifndef FMHA_PREFILL_K_SLM_NO_S2R
#define FMHA_PREFILL_K_SLM_NO_S2R 0
#endif

// Drop GEMM1/GEMM2's workgroup split barrier. This mainloop shares nothing through
// SLM (SharedStorage is empty), so the barrier is only a heuristic that keeps the
// subgroups marching in step to improve locality on the shared K/V loads.
// 0 = keep everywhere, 1 = drop everywhere, 2 = drop only in the ScoreBlock2D load
// kernel. Measured: the barrier earns its keep in the GEMM1 kernel (it groups the
// shared K loads) but costs time in the load kernel, which has no GEMM1 to group.
#ifndef FMHA_PREFILL_NO_SPLIT_BARRIER
#define FMHA_PREFILL_NO_SPLIT_BARRIER 2
#endif

// Hold the subgroup's *entire* Q slice in registers for the whole K loop, so GEMM1 loads
// Q once per workgroup instead of once per K block.
//
// Q re-read traffic is heads*seq_q*head_dim*2*(seq_kv/TILED_KV) bytes and is the measured
// bottleneck at head_dim=512 (~8.6 GB at b1/hq32/sq4096, all L1/L2 hits). QK_GROUP
// attacks the same term but only divides it by the group width, and grouping costs one S
// accumulator per extra block. Full residency divides it by seq_kv/TILED_KV outright --
// i.e. removes the re-read entirely -- and costs no accumulators.
//
// The register arithmetic is what makes this worth trying: with TILED_Q=256 and 32
// subgroups a subgroup owns 8 Q rows, so its whole Q slice is 8*512*2 = 8 KB, which is
// 512 B/lane across 16 lanes. That is large but not obviously fatal against a 256-GRF
// (8 KB/lane) budget, and unlike TILED_KV/TILED_Q changes it leaves every fragment width
// and the O accumulator untouched. Independent of Q residency the K fragment is still
// re-loaded per block, so this trades registers for Q traffic only.
//
// The value is the *number of head-dim chunks* to hold, up to head_dim / KSTEP (16 at
// head_dim=512, KSTEP=32); 0 disables the feature, and any smaller value makes residency
// partial (leading chunks resident, the rest loaded per block as before). It cannot be
// derived from the tensor, because size<3>(tQgQ) is a dynamic int -- and a compile-time
// count is required anyway: the fragment array has to be indexed by a statically-unrolled
// loop variable or the compiler puts it in scratch instead of registers.
//
// MEASURED AND REJECTED; kept default-off as a recorded refutation. Every setting is
// correct (bad=0, six verify shapes) and every setting loses, monotonically in the amount
// held, because the register arithmetic above is wrong in practice -- spill grows ~600
// B/thread per resident chunk, roughly 20x the 32 B/lane a chunk's data occupies, so the
// A fragments do not stay in registers at all:
//
//   chunks | spill B/thread | TFLOPS @ b1/hq32/sq2048 | @ sq4096
//   0      | 640/704        | 44.2                    | 47.77  (keeper)
//   4      | 1856-2816      | 41.4                    | 42.95
//   8      | 4096-4928      | 42.7                    | 43.10
//   16     | 9600-10688     | 19.4                    | timeout
//
// This is the same register wall as TILED_KV=128, TILED_Q=512 and N_SPLIT=2: the Q
// re-read is real and is the bottleneck, but registers are not where the re-read can be
// removed. Note the K fragment is reloaded per block regardless, so residency never
// removed more than the Q term.
#ifndef FMHA_PREFILL_Q_RESIDENT
#define FMHA_PREFILL_Q_RESIDENT 0
#endif

// Give the ScoreBlock2D workspace accesses explicit LSC cache hints: store `.uc.wb` (do
// not allocate in L1, but keep it in L3 for the reader) and load `.st.ca` (streaming, so
// the single-use score lines do not evict Q/K).
//
// The score round-trip's *bandwidth* is nearly free (see the workspace measurement), but
// its L1 *footprint* is not: it shares the cache with the Q/K re-reads that bound this
// kernel. This is the one lever that cuts the re-read cost without moving any operand,
// which is why it survives after the tile sweep, N-split, K SLM staging, KSTEP, grouping
// and Q register residency were all refuted. It needs inline asm, hence its own header
// and its own branch.
#ifndef FMHA_PREFILL_SCORE_CACHE_HINT
#define FMHA_PREFILL_SCORE_CACHE_HINT 0
#endif

// Same mechanism aimed at the actual bottleneck: hint the K-cache loads instead of the score
// surface. K is the operand whose L1 requests are NUM_SG-redundant, so a streaming hint on K
// is the only known way to reduce its interference with the Q re-reads without moving data.
#ifndef FMHA_PREFILL_K_HINT
#define FMHA_PREFILL_K_HINT 0
#endif

namespace cutlass::fmha {

template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.

};  // namespace cutlass::fmha

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// ---- K SLM staging (FMHA_PREFILL_K_SLM) ----
// Types for staging GEMM1's K through SLM one head-dim group at a time. They live in a
// template specialized on Enable so that the cooperative-copy machinery -- and in
// particular its "block size should not be less than sg size" assertion, which a single
// reduction step cannot satisfy -- is only instantiated when staging is switched on.
template <
    bool Enable,
    class TiledMMAQK,
    class SubgroupLayoutQK,
    class TensorK2D,
    class TiledCopyKStep,
    int Steps,
    int Stages>
struct KSlmTraits {};

template <class TiledMMAQK, class SubgroupLayoutQK, class TensorK2D, class TiledCopyKStep, int Steps, int Stages>
struct KSlmTraits<true, TiledMMAQK, SubgroupLayoutQK, TensorK2D, TiledCopyKStep, Steps, Stages> {
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  // A GEMM1-shaped MMA whose reduction mode spans a whole head-dim group instead of a
  // single step. Widening it is what supplies the DPAS atom blocks the cooperative copy
  // needs in order to split the tile NUM_SG ways.
  using TileShapeGrp =
      decltype(make_shape(get<0>(TileShapeQK{}), get<1>(TileShapeQK{}), get<2>(TileShapeQK{}) * Int<Steps>{}));
  using TiledMMAGrp =
      typename cute::TiledMMAHelper<typename TiledMMAQK::Atom, Layout<TileShapeGrp>, SubgroupLayoutQK>::TiledMMA;

  // Global -> registers, one 1/NUM_SG slice of the group per subgroup, and the matching
  // registers -> SLM write.
  using CoopCopy = decltype(make_coop_block_2d_copy_B(TiledMMAGrp{}, TensorK2D{}));
  using GrpCopies = decltype(make_B_slm_copies(TiledMMAGrp{}, CoopCopy{}));
  using R2S = remove_cvref_t<std::tuple_element_t<0, GrpCopies>>;

  // SLM -> registers for one reduction step. Built from the *unwidened* MMA so the K
  // fragment feeding the DPAS keeps its original width: staging must not cost registers,
  // which is the whole reason the group only exists in SLM and not in the MMA.
  using StepCopies = decltype(make_B_slm_copies(TiledMMAQK{}, TiledCopyKStep{}));
  using S2R = remove_cvref_t<std::tuple_element_t<1, StepCopies>>;

  using Element = typename TiledMMAGrp::ValTypeB;
  // Three views of the same Stages-deep buffer. The r2s tiler is the group tile
  // (TILED_KV, KSTEP*Steps), laid out compactly; the per-step view splits its second mode
  // so GEMM1 can read one reduction step at a time. Both are compact in the same order, so
  // they address SLM identically, and the trailing mode selects the stage.
  using GrpLayout = decltype(make_layout(append<3>(typename R2S::Tiler_MN{}, Int<Stages>{})));
  using StepLayout = decltype(make_layout(
      make_shape(get<1>(TileShapeQK{}), get<2>(TileShapeQK{}), Int<Steps>{}, Int<Stages>{})));
  static_assert(cosize_v<GrpLayout> == cosize_v<StepLayout>, "group and per-step SLM views must agree");
};

// Storage for the above. Empty when staging is off, so the kernel's mainloop/epilogue
// SLM union -- and therefore the launch's work-group scratch request -- stays at zero.
template <class Traits, bool Enable>
struct KSlmStorage {};

template <class Traits>
struct KSlmStorage<Traits, true> {
  cute::array_aligned<typename Traits::Element, cute::cosize_v<typename Traits::GrpLayout>> smem_k;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool CachedKV_,
    bool PagedKV_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TensorK_cache_,
    class TensorV_cache_,
    class TiledCopyQ_ = void,        // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,        // Optional TiledCopy for loading K
    class TiledCopyV_ = void,        // Optional TiledCopy for loading V
    class TiledCopyK_cache_ = void,  // Optional TiledCopy for loading K_cache
    class TiledCopyV_cache_ = void,  // Optional TiledCopy for loading V_cache
    bool LocalMask_ = false,
    // PackGQA: the M tile holds the head_group_q query heads of one GQA group
    // (decode only, seq_len_qo == 1). All packed rows share the single decode
    // KV position, so per-row masking must use a fixed decode row. Default
    // false keeps prefill (and non-packed decode) unaffected.
    bool PackGQA_ = false>
struct FMHAFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    bool CachedKV_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TensorK_cache_,
    class TensorV_cache_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    class TiledCopyK_cache_,
    class TiledCopyV_cache_,
    bool LocalMask_,
    bool PackGQA_>
struct FMHAFwdMainloop<
    XeDefault<Stages>,
    CausalMask_,
    CachedKV_,
    PagedKV_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TensorK_cache_,
    TensorV_cache_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    TiledCopyK_cache_,
    TiledCopyV_cache_,
    LocalMask_,
    PackGQA_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ =
      conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK =
      conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV =
      conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;
  using TensorK_cache = TensorK_cache_;
  using TensorV_cache = TensorV_cache_;
  using TensorK_cache2D = decltype(TensorK_cache_{}(append<rank_v<TensorK_cache_>>(make_coord(_, _), 0)));
  using TensorV_cache2D = decltype(TensorV_cache_{}(append<rank_v<TensorV_cache_>>(make_coord(_, _), 0)));
  // K's L1 request volume is NUM_SG x its footprint (every subgroup loads the whole block),
  // and those requests evict the Q lines that must survive the block. FMHA_PREFILL_K_HINT
  // marks the K loads streaming so the cache stops retaining them; it changes no data
  // movement at all, unlike SLM staging or an N split (both measured and refuted).
  using TiledCopyK_cache = conditional_t<
      is_void_v<TiledCopyK_cache_>,
      conditional_t<
          FMHA_PREFILL_K_HINT != 0,
          decltype(make_block_2d_copy_B_hinted(TiledMMAQK{}, TensorK_cache2D{})),
          decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK_cache2D{}))>,
      TiledCopyK_cache_>;
  using TiledCopyV_cache = conditional_t<
      is_void_v<TiledCopyV_cache_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV_cache2D{})),
      TiledCopyV_cache_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  // ScoreBlock2D scratch element type. The values round-tripped through the
  // workspace are pre-softmax logits consumed immediately by exp2, so half
  // precision suffices: the store folds in params.scale and the load path's
  // running-max subtraction keeps the exponent in range. Halving the element
  // width halves both the workspace footprint and the workspace traffic, and at
  // head_dim=512 that fp32 round-trip was ~half of all DRAM traffic on large
  // shapes -- which is what actually bounds this kernel.
  using ElementScoreStore = half_t;

  using SingleFragA = FragC<TiledMMAPV>;                       // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool LocalMask = LocalMask_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool ScoreBlock2D = FMHA_PREFILL_ENABLE_SCORE_BLOCK2D;
  static constexpr int QKGroup = FMHA_PREFILL_QK_GROUP;
  static_assert(QKGroup >= 1, "FMHA_PREFILL_QK_GROUP must be at least 1");
  // Grouping only exists to amortize GEMM1's Q loads, so it is pointless in the
  // ScoreBlock2D load launches -- and there it is actively harmful, since they would
  // still pay for QKGroup S accumulators they never fill.
  template <int Mode>
  static constexpr int group_for_mode() {
    return (ScoreBlock2D && Mode >= 1) ? 1 : QKGroup;
  }
  // Keep the subgroup's whole Q slice live across the K loop. Only meaningful in launches
  // that actually run GEMM1: the ScoreBlock2D load launches reload S from the scratch, so
  // there they would hold Q fragments nothing reads.
  static constexpr bool ScoreCacheHint = ScoreBlock2D && FMHA_PREFILL_SCORE_CACHE_HINT;
  static constexpr int QResidentChunks = FMHA_PREFILL_Q_RESIDENT;
  static constexpr bool QResident = QResidentChunks > 0;
  // Residency replaces the per-block Q load outright, so it subsumes grouping; allowing
  // both would keep KGrp S accumulators live for no remaining benefit.
  static_assert(!QResident || QKGroup == 1, "FMHA_PREFILL_Q_RESIDENT and QK_GROUP > 1 are exclusive");
  static_assert(!QResident || FMHA_PREFILL_K_SLM == 0, "FMHA_PREFILL_Q_RESIDENT and K_SLM are exclusive");
  static constexpr bool ZigzagD = FMHA_PREFILL_ZIGZAG_D;
  static constexpr bool PfZigzag = FMHA_PREFILL_PF_ZIGZAG;
  static constexpr int InitPfDepth = FMHA_PREFILL_INIT_PF_DEPTH;
  static constexpr int NoSplitBarrierMode = FMHA_PREFILL_NO_SPLIT_BARRIER;
  // When set, ScoreBlock2D mode 0 is store-only: it skips GEMM2 and the epilogue,
  // so the number of launches is one more than the number of output tiles.
  static constexpr bool SplitStore = ScoreBlock2D && FMHA_PREFILL_SPLIT_STORE;
  // Head-dim group width for K SLM staging, in units of the GEMM1 reduction step.
  static constexpr int KSlmSteps = FMHA_PREFILL_K_SLM;
  // Staging publishes K through SLM, so its barriers must be reached by every subgroup
  // the same number of times. Under CausalMask they are not: the kernel derives the K
  // block count from seq_coord, which carries a per-subgroup Q offset, so each subgroup
  // runs its own trip count. Causal therefore keeps the direct per-subgroup loads.
  static constexpr bool KSlm = KSlmSteps > 0 && !CausalMask_;
  static constexpr int KSlmStages = FMHA_PREFILL_K_SLM_STAGES;
  // The staged loop writes the next group's buffer while reading this group's, which is
  // what lets one barrier per group suffice; with a single buffer those are the same
  // memory and the loop would need a second barrier (and would expose the load latency
  // anyway -- the reason this knob exists).
  static_assert(!KSlm || KSlmStages >= 2, "FMHA_PREFILL_K_SLM_STAGES must be at least 2");
  using KSlmT = KSlmTraits<
      KSlm,
      TiledMMAQK,
      SubgroupLayoutQK,
      TensorK_cache2D,
      TiledCopyK_cache,
      (KSlm ? KSlmSteps : 1),
      (KSlm ? KSlmStages : 1)>;
  // The staged path publishes one K block's group at a time, so it has no place to put a
  // second block's. Grouping exists to amortize Q loads and staging to cut K requests;
  // combining them would need one SLM buffer per block in the group.
  static_assert(!KSlm || QKGroup == 1, "FMHA_PREFILL_K_SLM and FMHA_PREFILL_QK_GROUP > 1 are exclusive");

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int max_num_pages_per_seq = 0;
    int window_size_left = -1;
    int window_size_right = -1;
    ElementScoreStore* ptr_score = nullptr;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data: a staging buffer for one head-dim group of the K block, empty unless
  // FMHA_PREFILL_K_SLM is set.
  struct SharedStorage : KSlmStorage<KSlmT, KSlm> {};

  Params params;
  SharedStorage& shared;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage& shared_) : params(params_), shared(shared_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* workspace) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.ptr_page_table,
        args.page_size,
        args.max_num_pages_per_seq,
        args.window_size_left,
        args.window_size_right,
        ScoreBlock2D ? reinterpret_cast<ElementScoreStore*>(workspace) : nullptr};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE
  int get_physical_k_tile(int K, int l_coord, int seq_len_kv_cache) {
    int next_page_logical_idx = K * get<1>(TileShapeQK{}) / params.page_size;
    // get<1>(TileShapeQK{}) usually smaller than page_size.
    // assuming page_size is multiple of get<1>(TileShapeQK{})
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    // int batch_offset =
    //     params.num_pages_per_seq ? params.num_pages_per_seq[l_coord] : l_coord * (seq_len_kv_cache /
    //     params.page_size);
    int batch_offset = l_coord * params.max_num_pages_per_seq;

    return params.ptr_page_table[batch_offset + next_page_logical_idx] * tiles_per_page + K % tiles_per_page;
  }

  template <int StaticScoreMode = -1, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int seq_len_kv_cache,
      int l_coord,
      int full_tile_offset,
      int discard_seq_coord,
      TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
      TensorV_cache2D const& V_cache_2D = TensorV_cache2D{},
      ElementScoreStore* score_head_ptr = nullptr) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape());   // (k,d)
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape());   // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)
#if FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
    // score_head_ptr already points at this workgroup's own block, so the score
    // surface is exactly one Q tile tall (not the whole seq_len_qo) and rows are
    // addressed block-locally. Element type is ElementScoreStore, narrower than
    // ElementS, so the block-2D atoms below are selected for that width and move
    // half the bytes.
    auto score_shape = make_shape(get<0>(TileShapeQK{}), seq_len_kv_cache);
    auto score_layout = make_layout(score_shape, make_stride(seq_len_kv_cache, Int<1>{}));
    Tensor Score = make_tensor(make_gmem_ptr(score_head_ptr), score_layout);
    Tensor cScore = make_identity_tensor(score_shape);  // (q,k)
#endif

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});          // (q,d,D)
    Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    Tensor gK_cache = local_tile(cK_cache, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV_cache = local_tile(cV_cache, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_cache_split = local_tile(gV_cache, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)
#if FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
    // Q coord is 0: this block holds only this workgroup's Q tile.
    Tensor gScore = local_tile(cScore, take<0, 2>(TileShapeQK{}), make_coord(_0{}, _));  // (q,k,K)
#endif

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};
    TiledCopyK_cache copy_k_cache{K_cache_2D};
    TiledCopyV_cache copy_v_cache{V_cache_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
#if FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
    // The score scratch is written once and read once, so it has no reason to occupy L1 --
    // where it only evicts the Q/K lines GEMM1 re-reads. FMHA_PREFILL_SCORE_CACHE_HINT
    // swaps in block-2D ops that carry an explicit LSC hint saying so; see
    // xe_fmha_cache_hint_copy.hpp for why this is reachable from kernel code at all.
    auto copy_score_store = [&] {
      if constexpr (ScoreCacheHint) {
        return make_block_2d_copy_D_hinted(mma_qk, Score);
      } else {
        return make_block_2d_copy_D(mma_qk, Score);
      }
    }();
    auto copy_score_load = [&] {
      if constexpr (ScoreCacheHint) {
        return make_block_2d_copy_C_hinted(mma_qk, Score);
      } else {
        return make_block_2d_copy_C(mma_qk, Score);
      }
    }();
#endif

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_copy_k_cache = copy_k_cache.get_slice(thr_id);
    auto thr_copy_v_cache = copy_v_cache.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);
#if FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
    auto thr_copy_score_store = copy_score_store.get_slice(thr_id);
    auto thr_copy_score_load = copy_score_load.get_slice(thr_id);
#endif

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    // One S accumulator per K block in the group; KGrp == 1 reduces to the old
    // single accumulator, so the extra register cost is opt-in. The load launches
    // run no GEMM1, so they are pinned to 1 rather than paying for accumulators
    // they never fill (see group_for_mode).
    constexpr int KGrp = group_for_mode<StaticScoreMode>();
    FragS tSrS_grp[KGrp];

    // Whole-Q-slice residency: one A fragment per head-dim chunk, filled once before the K
    // loop. Sized 1 (and left unfilled) in the launches that run no GEMM1, so they pay
    // nothing.
    constexpr bool QRes = QResident && !(ScoreBlock2D && StaticScoreMode >= 1);
    constexpr int NDStatic = QRes ? QResidentChunks : 1;
    using FragQ = decltype(tSrQ);
    FragQ tSrQ_all[NDStatic];
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
#if FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
    auto tScoreStoreR = thr_copy_score_store.partition_sg_fragment_S(gScore(_, _, 0));
    auto tScoreStoreG = thr_copy_score_store.partition_D(gScore);
    auto tScoreLoadG = thr_copy_score_load.partition_S(gScore);
    auto tScoreLoadR = thr_copy_score_load.partition_sg_fragment_D(gScore(_, _, 0));
#endif

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto prefetch_k_cache = make_block_2d_prefetch(copy_k_cache);
    auto prefetch_v_cache = make_block_2d_prefetch(copy_v_cache);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pKgK_cache = prefetch_k_cache.get_slice(thr_id).partition_S(gK_cache);
    auto pVgV_cache = prefetch_v_cache.get_slice(thr_id).partition_S(gV_cache_split);

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    int kblocks_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));
    int page_idx = blk_k0;
    int next_page_idx = blk_k0;
    if constexpr (PagedKV) {
      next_page_idx = get_physical_k_tile(blk_k0, l_coord, seq_len_kv_cache);
    }
    if constexpr (!(ScoreBlock2D && StaticScoreMode >= 1)) {
      // Only the leading chunks: the rest arrive via the in-loop prefetch anyway, and
      // issuing all of them here just evicts the ones GEMM1 needs first.
      const int nQpf = InitPfDepth > 0 ? cute::min(int(size<3>(pQgQ)), InitPfDepth) : int(size<3>(pQgQ));
      const int nKpf = InitPfDepth > 0 ? cute::min(int(size<4>(pKgK)), InitPfDepth) : int(size<4>(pKgK));
      for (int D = 0; D < nQpf; D++) {
        prefetch(prefetch_q, pQgQ(_, _, _, D));
      }
      for (int D = 0; D < nKpf; D++) {
        prefetch(prefetch_k_cache, pKgK_cache(_, _, _, next_page_idx, D));
      }
    }
    // Always initialize the per-WG accumulators: the caller (kernel) may pass
    // blk_k0 > 0 when sliding-window pruning skips leading K blocks, so we can
    // no longer key initialization off of (blk_k0 == 0).
    // The store-only launch never touches these, and skipping the init is what lets
    // the compiler drop the accumulator instead of keeping it live.
    if constexpr (!(SplitStore && StaticScoreMode == 0)) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    constexpr bool SkipSplitBarrier =
        (NoSplitBarrierMode == 1) || (NoSplitBarrierMode == 2 && ScoreBlock2D && StaticScoreMode >= 1);

    /* Load the subgroup's whole Q slice once, ahead of the K loop, when it stays resident. */
    if constexpr (QRes) {
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < NDStatic; D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        reorder(tQrQ, tSrQ_all[D]);
      }
    }

    /* Main loop, blocked in k -- outer loop steps whole groups of KGrp blocks. */
    const int k_end = cute::min(blk_k1, kblocks_cache);
    for (int K_grp = blk_k0; K_grp < k_end; K_grp += KGrp) {
      /* GEMM 1 for the whole group: one pass of Q loads feeds KGrp K blocks, so
         Q is read seq_kv/(TILED_KV*KGrp) times instead of seq_kv/TILED_KV. */
      if constexpr (!(ScoreBlock2D && StaticScoreMode >= 1)) {
        int grp_page_idx[KGrp];
        CUTLASS_PRAGMA_UNROLL
        for (int g = 0; g < KGrp; g++) {
          // Clamp the tail so a short final group re-reads a valid page instead of
          // running off the end; those lanes' results are simply never consumed.
          int Kg = cute::min(K_grp + g, k_end - 1);
          grp_page_idx[g] = PagedKV ? get_physical_k_tile(Kg, l_coord, seq_len_kv_cache) : Kg;
          clear(tSrS_grp[g]);
        }
        if constexpr (KSlm) {
          // Stage K through SLM one head-dim group at a time. All NUM_SG subgroups need
          // the whole K block, so loading it directly (below) costs NUM_SG identical L1
          // request streams; here each subgroup instead fetches 1/NUM_SG of the group and
          // publishes it, and everyone reads the block back out of SLM.
          //
          // Requires head_dim % (KSTEP * KSlmSteps) == 0, otherwise the trailing group
          // would stage out-of-bounds columns and feed them to the DPAS.
          typename KSlmT::CoopCopy coop_copy_k{K_cache_2D};
          typename KSlmT::R2S r2s_k{};
          typename KSlmT::S2R s2r_k{};

          // The group tile is (TILED_KV, KSTEP * KSlmSteps), so the head dim is walked in
          // nDg groups rather than nD single steps.
          Tensor gK_cache_grp =
              local_tile(cK_cache, typename KSlmT::TileShapeGrp{}, make_coord(_, _, _), Step<X, _1, _1>{});

          // Two views of the same KSlmStages-deep buffer: the group shape the cooperative
          // copy writes, and the (TILED_KV, KSTEP, KSlmSteps) shape GEMM1 reads a step at a
          // time. Both are compact in the same order, so they address SLM identically.
          Tensor sK_grp = make_tensor(make_smem_ptr(shared.smem_k.data()), typename KSlmT::GrpLayout{});
          Tensor sK_step = make_tensor(make_smem_ptr(shared.smem_k.data()), typename KSlmT::StepLayout{});

          auto thr_coop_k = coop_copy_k.get_slice(thr_id);
          auto thr_r2s_k = r2s_k.get_slice(thr_id);
          auto thr_s2r_k = s2r_k.get_slice(thr_id);

          Tensor tKgK_coop = thr_coop_k.partition_S(gK_cache_grp);  // (atom_val,k',d',K,Dg)
          auto tKrK_coop = thr_coop_k.partition_sg_fragment_D(gK_cache_grp(_, _, 0, 0));
          auto tKrK_stage = thr_r2s_k.partition_sg_fragment_S(gK_cache_grp(_, _, 0, 0));
          auto tKrK_out = thr_r2s_k.retile_S(tKrK_stage);
          auto tKsK_out = thr_r2s_k.partition_D(sK_grp);   // (atom_val,k',d',stage)
          auto tKsK_in = thr_s2r_k.partition_S(sK_step);   // (atom_val,k',d',KSlmSteps,stage)
          auto tSrK_in = thr_s2r_k.retile_D(tSrK);

          const int nDg = size<4>(tKgK_coop);
          // Same serpentine motivation as the direct path below, at group granularity.
          auto dg_of = [&](int i) { return (ZigzagD && (K_grp & 1)) ? (nDg - 1 - i) : i; };
          auto load_grp = [&](int i, int stage) {
            copy(coop_copy_k, tKgK_coop(_, _, _, grp_page_idx[0], dg_of(i)), tKrK_coop);
            reorder(tKrK_coop, tKrK_stage);
            copy(r2s_k, tKrK_out, tKsK_out(_, _, _, stage));
          };
          auto publish = [] {
            barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
            barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
          };

          // Prologue: fill all but one stage, so the steady-state loop always has the group
          // it is about to consume already resident.
          CUTLASS_PRAGMA_UNROLL
          for (int p = 0; p < KSlmStages - 1; p++) {
            if (p < nDg) {
              load_grp(p, p);
            }
          }
          publish();

          for (int Dgi = 0; Dgi < nDg; Dgi++) {
            const int rd_stage = Dgi % KSlmStages;
            const int wr = Dgi + KSlmStages - 1;
            // Issue the next group's global load *before* consuming this one, so its
            // latency hides behind the DPAS below rather than stalling the whole
            // workgroup at the barrier. This is the entire reason for multiple stages.
            if (wr < nDg) {
              load_grp(wr, wr % KSlmStages);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < KSlmSteps; s++) {
              const int D = dg_of(Dgi) * KSlmSteps + s;
              copy(copy_q, tQgQ(_, _, _, D), tQrQ);
              reorder(tQrQ, tSrQ);
              if constexpr (FMHA_PREFILL_K_SLM_NO_S2R) {
                copy(copy_k_cache, tKgK_cache(_, _, _, grp_page_idx[0], D), tKrK);
                reorder(tKrK, tSrK);
              } else {
                copy(s2r_k, tKsK_in(_, _, _, s, rd_stage), tSrK_in);
              }
              cute::gemm(mma_qk, tSrQ, tSrK, tSrS_grp[0]);
            }

            // One barrier per group, doing double duty: it publishes the group just
            // written and, because the writer is always at least one stage ahead of every
            // reader, it also guarantees the reads of the stage about to be overwritten
            // have all retired.
            publish();
          }
        } else if constexpr (QRes) {
          // Q is already in registers, so this walks only K. The bound must be the
          // compile-time chunk count -- not size<4>(tKgK), which is a dynamic int -- so
          // that Di, and hence the tSrQ_all index, is a constant in each unrolled body;
          // a runtime index would turn the fragment array into indirect scratch. The
          // caller must set FMHA_PREFILL_Q_RESIDENT <= head_dim / KSTEP; setting it equal
          // holds all of Q and removes the re-read outright, while a smaller value makes
          // residency partial -- the leading NDStatic chunks come from registers and the
          // rest are loaded per block as before. Partial residency is what grades the
          // register cost against the traffic saved.
          //
          // Serpentine order is dropped here: it exists to keep the previously-loaded Q
          // chunk resident in L1, and the resident chunks have no Q load left to help.
          CUTLASS_PRAGMA_UNROLL
          for (int D = 0; D < NDStatic; D++) {
            copy(copy_k_cache, tKgK_cache(_, _, _, grp_page_idx[0], D), tKrK);
            reorder(tKrK, tSrK);
            cute::gemm(mma_qk, tSrQ_all[D], tSrK, tSrS_grp[0]);
          }
          const int nD_tail = size<4>(tKgK);
          for (int D = NDStatic; D < nD_tail; D++) {
            copy(copy_q, tQgQ(_, _, _, D), tQrQ);
            reorder(tQrQ, tSrQ);
            copy(copy_k_cache, tKgK_cache(_, _, _, grp_page_idx[0], D), tKrK);
            reorder(tKrK, tSrK);
            cute::gemm(mma_qk, tSrQ, tSrK, tSrS_grp[0]);
          }
        } else {
          const int nD = size<4>(tKgK);
          CUTLASS_PRAGMA_UNROLL
          for (int Di = 0; Di < nD; Di++) {
            // Serpentine walk over the head-dim chunks: consecutive K blocks traverse D
            // in opposite directions, so the Q chunk loaded last by one block is the
            // first one the next block needs and is still resident. The WG's Q tile is
            // 256KB at head_dim=512 -- right at L1 capacity -- so a straight 0..nD-1
            // walk evicts the front of Q before it comes back around. Costs no extra
            // registers, unlike widening TILED_KV or grouping K blocks.
            const int D = (ZigzagD && ((K_grp / KGrp) & 1)) ? (nD - 1 - Di) : Di;
            copy(copy_q, tQgQ(_, _, _, D), tQrQ);
            reorder(tQrQ, tSrQ);
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < KGrp; g++) {
              copy(copy_k_cache, tKgK_cache(_, _, _, grp_page_idx[g], D), tKrK);
              reorder(tKrK, tSrK);
              cute::gemm(mma_qk, tSrQ, tSrK, tSrS_grp[g]);
            }
          }
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int g = 0; g < KGrp; g++) {
        const int K = K_grp + g;
        if (K >= k_end) {
          break;
        }
        auto& tSrS = tSrS_grp[g];

        /* Split barrier to keep threads together */
        if constexpr (!SkipSplitBarrier) {
          barrier_arrive(ScopeWorkgroup);
        }

        bool need_causal = false;
        if constexpr (CausalMask) {
          need_causal = K >= blk_k1_causal;
        }

        page_idx = next_page_idx;
        next_page_idx = K + 1;
        if constexpr (PagedKV) {
          next_page_idx = get_physical_k_tile(next_page_idx, l_coord, seq_len_kv_cache);
        }

        if constexpr (ScoreBlock2D && StaticScoreMode >= 1) {
#if FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
          copy(copy_score_load, tScoreLoadG(_, _, _, K), tScoreLoadR);
          reorder(tScoreLoadR, tSrS);
#endif
        }

        /* V prefetch for GEMM 2 -- the store-only launch runs no GEMM2, so V never
           enters its cache footprint. */
        if constexpr (!(SplitStore && StaticScoreMode == 0)) {
          CUTLASS_PRAGMA_UNROLL
          for (int VV = 0; VV < VTiles; VV++) {
            prefetch(prefetch_v_cache, pVgV_cache(_, _, _, VV, page_idx));
          }
        }

        /* Causal masking */
        if constexpr (CausalMask && !(ScoreBlock2D && StaticScoreMode >= 1)) {
          if (need_causal) {
            int lane_id = thr_id % intel::sg_size;
            constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
            int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) + (thr_id / intel::sg_size) * sg_tile_q;

            constexpr int k_tile = get<1>(TileShapeQK{});
            constexpr int n_reps = k_tile / intel::sg_size;
            // Size off the type, not the variable: tSrS is a reference into the
            // group array and so is not itself a constant expression.
            constexpr int elems_per_n = FragS{}.size() / n_reps;
            int k_base = K * k_tile;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < n_reps; n++) {
              int col = k_base + n * intel::sg_size + lane_id;
              int causal_bound = col - full_tile_offset - row_base;
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < elems_per_n; j++) {
                if (j < causal_bound) {
                  tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
                }
              }
            }
          }
        }

        /* Local/sliding window masking */
        if constexpr (LocalMask && !(ScoreBlock2D && StaticScoreMode >= 1)) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            // PackGQA decode: every packed M row is the same decode token, so the
            // KV position is full_tile_offset regardless of the per-row (head)
            // index. Non-packed keeps the per-row sequence position.
            int row_kv_idx = (PackGQA_ ? 0 : row_idx) + full_tile_offset;
            bool left_mask = col_idx < row_kv_idx - params.window_size_left;
            bool right_mask = col_idx > row_kv_idx + params.window_size_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }

        /* k masking for remainder tiles */
        if (check_remainder_k && K == total_blk - 1) {
          FragSCol k_rem_mask;
          int k_val = get<0>(tKgK_cache(0, 0, 0, K, 0));
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
        }

#if FMHA_PREFILL_ENABLE_SCORE_BLOCK2D
        if constexpr (ScoreBlock2D && StaticScoreMode == 0) {
          // Store the raw (unscaled) logits; the reorder narrows fp32 -> ElementScoreStore.
          // params.scale is applied on the load side instead, so the narrowing error is
          // multiplied down by scale before it reaches exp2 rather than landing directly
          // in the exponent. -INFINITY from the causal/remainder masks is representable
          // in half, so masked lanes still exponentiate to zero.
          reorder(tSrS, tScoreStoreR);
          copy(copy_score_store, tScoreStoreR, tScoreStoreG(_, _, _, K));
        }
#endif

        // Store-only launch: the scores are on their way to the scratch and no output
        // tile belongs to this launch, so skip softmax and GEMM2 entirely. Leaving the
        // O accumulator untouched is the point -- it is what frees the registers.
        if constexpr (!(SplitStore && StaticScoreMode == 0)) {
          /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
          ElementS softmax_scale = params.scale;
          auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum, softmax_scale);
          reorder(tSrS, tArP);

          /* GEMM 2: A += P * V, split in v dimension. */
          CUTLASS_PRAGMA_UNROLL
          for (int VV = 0; VV < VTiles; VV++) {
            copy(copy_v_cache, tVgV_cache(_, _, _, VV, page_idx), tVrV);
            reorder(tVrV, tArV);
            if (K != blk_k0) {
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < tArA.size() / VTiles; i++) {
                tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
              }
            }
            cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
          }
        }

        /* K prefetch */
        if constexpr (!(ScoreBlock2D && StaticScoreMode >= 1)) {
          const int nPf = size<4>(pKgK);
          // Match the head-dim order the *next* group will walk in, so its first
          // chunk is the first one prefetched rather than the last.
          const bool rev = PfZigzag && ZigzagD && (((K_grp / KGrp) + 1) & 1);
          for (int Di = 0; Di < nPf; Di++) {
            const int D = rev ? (nPf - 1 - Di) : Di;
            prefetch(prefetch_k_cache, pKgK_cache(_, _, _, next_page_idx, D));
          }
        }

        if constexpr (!SkipSplitBarrier) {
          barrier_wait(ScopeWorkgroup);
        }
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow softmax(
      bool first_block,     // First softmax block?
      FragS& tS,            // Softmax src/dst block
      FragSRow& tS_max,     // Softmax row-wise max accumulator
      FragSRow& tS_sum,     // Softmax row-wise sum accumulator
      ElementS qk_scale) {  // Q*K scale

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima and compute rescale factor */
    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), qk_scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(qk_scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);

    return rescale;
  }
};

template <
    class DispatchPolicy_,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void,  // Optional TiledCopy for loading V
    bool LocalMask_ = false>
struct DecodeFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <
    int Stages,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    bool LocalMask_>
struct DecodeFwdMainloop<
    XeDefault<Stages>,
    PagedKV_,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    LocalMask_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ =
      conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK =
      conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV =
      conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                       // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  // static_assert(is_same_v<decltype(FragSRow{}.shape()), float>, "dtype
  // mismatched");
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool Fp8KV = is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool LocalMask = LocalMask_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;
    // Paged KV Cache
    int const* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int window_size_left;
    int window_size_right;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  DecodeFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.window_size_left,
        args.window_size_right};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});          // (q,d,D)
    Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto copyQ = make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{});

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    // PagedKV
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int tile_idx = blk_k0;
    int b_offset = idx_b * params.max_pages_per_seq;
    if constexpr (PagedKV) {
      int page_local_idx = tile_idx * get<1>(TileShapeQK{}) / params.page_size;
      tile_idx = params.ptr_page_table[b_offset + page_local_idx] * tiles_per_page + tile_idx % tiles_per_page;
    }

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    for (int D = 0; D < size<4>(pKgK); D++) {
      prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // FP8 KV Scale: Currently we only support per-tensor scale for KV
    float scale_k = 1.f, scale_v = 1.f;
    if constexpr (Fp8KV) {
      scale_k = *static_cast<const float*>(params.scale_k);
      scale_v = *static_cast<const float*>(params.scale_v);
    }

    /* Main loop, blocked in k. */
    int next_tile_idx;
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Split barrier to keep threads together */
      // barrier_arrive(ScopeWorkgroup);

      auto tKgK_cache = PagedKV ? tKgK(_, _, _, tile_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache = PagedKV ? tVgV(_, _, _, _, tile_idx) : tVgV(_, _, _, _, K);

      /* GEMM 1: S = K * Q */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);

        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        if constexpr (Fp8KV) {
          for (int i = 0; i < tSrK.size(); ++i) {
            tSrK(i) = static_cast<ElementQ>(scale_k * static_cast<float>(tSrK(i)));
          }
        }

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }
      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, tile_idx));

      /* Causal masking */
      // No Causal masking in decoding
      // if constexpr (CausalMask) {
      //   if (K == blk_k1 - 1) {
      //     // Need to get global col and row indices to mask the elements
      //     Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
      //     Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
      //     make_coord(get<0>(blk_qv), K)); auto cS_thread =
      //     thr_mma_qk.partition_C(gP); CUTLASS_PRAGMA_UNROLL for (int i = 0; i
      //     < tSrS.size(); ++i) {
      //       int row_idx = get<0>(cS_thread(i));
      //       int col_idx = get<1>(cS_thread(i));
      //       if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
      //         tSrS(i) = ElementS(-INFINITY);
      //       }
      //     }
      //   }
      // }

      /* Local/sliding window masking */
      if constexpr (LocalMask) {
        // For decode, all packed GQA heads share the same KV position
        // (seq_len_kv - 1). Use a fixed decode row for all elements.
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Apply softmax and scaling */
      softmax(K == 0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        reorder(tVrV, tArV);
        if constexpr (Fp8KV) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArV.size(); ++i) {
            tArV(i) = static_cast<ElementQ>(scale_v * static_cast<float>(tArV(i)));
          }
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      barrier();

      // next tile_idx
      next_tile_idx = K + 1;
      if constexpr (PagedKV) {
        int next_page_local_idx = next_tile_idx * get<1>(TileShapeQK{}) / params.page_size;
        if (next_page_local_idx < params.max_pages_per_seq) {
          next_tile_idx =
              params.ptr_page_table[b_offset + next_page_local_idx] * tiles_per_page + next_tile_idx % tiles_per_page;
        } else {
          // set to last page
          next_tile_idx = params.max_pages_per_seq * tiles_per_page - 1;
        }
      }
      tile_idx = next_tile_idx;

      /* K prefetch */
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
      }

      // barrier_wait(ScopeWorkgroup);
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void softmax(
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <typename MMAOp, typename WGTileQK, typename SGLayoutQK, typename TileV>
CUTLASS_HOST_DEVICE constexpr auto
get_tiled_mma_pv(MMAOp const&, WGTileQK const& wg_tile_qk, SGLayoutQK const& sg_layout_qk, TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(size(SGLayoutPV{}) == size(SGLayoutQK{}), "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

}  // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
