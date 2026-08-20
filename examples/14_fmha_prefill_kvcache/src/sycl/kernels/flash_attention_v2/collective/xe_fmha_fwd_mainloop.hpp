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
// Now default 8. Measured neutral (47.30 vs 47.7) before D_SKEW existed and left off; once
// the walk is skewed it helps, because the initial burst then spans the whole head-dim
// spread instead of 32 subgroups piling onto the same leading chunks. With PF_ZIGZAG=1 at
// b1/hq32/sq4096: uncapped 49.55, depth 4 -> 49.57, depth 8 -> 49.62, depth 12 -> see the
// D-skew note. Re-measure alongside a same-batch reference before changing it.
#define FMHA_PREFILL_INIT_PF_DEPTH 8
#endif

// Issue the next K block's head-dim prefetches in the order that block will actually
// consume them. With ZigzagD the walk direction flips every block, so a fixed 0..nD-1
// prefetch hands the reverse-walking block its first-needed chunk last -- and at
// head_dim=512 there are 16 chunks in flight, so that chunk can be evicted before use.
// Addresses the "reorder K prefetches" TODO below. Costs no registers.
#ifndef FMHA_PREFILL_PF_ZIGZAG
// Now default 1. Measured neutral (47.58 vs 47.7) before D_SKEW existed; with the skewed
// walk it is a small consistent win (49.50-49.55 vs 49.32), since matching the consume
// order matters more once that order differs per subgroup.
#define FMHA_PREFILL_PF_ZIGZAG 1
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

// Split GEMM1's head-dim reduction across several partial S accumulators, summed once at the
// end of the K block. Purely an instruction-level-parallelism knob: it moves no data, issues
// the same loads, and leaves every cache footprint unchanged.
//
// Motivated by the per-launch device-time breakdown (FMHA_PROFILE_PER_LAUNCH), which shows
// the two launches doing the same amount of GEMM2 but taking 18.4 ms and 4.6 ms at
// b1/hq32/sq4096. Subtracting gives GEMM1 ~13.7 ms for the same FLOP count GEMM2 does in
// ~4.6, i.e. ~40 vs ~59 TFLOPS. The structural difference is dependency depth, not
// bandwidth: GEMM2 splits V into VTiles independent accumulators, so it runs VTiles
// concurrent DPAS chains, while GEMM1 chains every head-dim chunk into the single S
// accumulator. At head_dim=512 with KSTEP=32 that is a 16-deep serial chain per n-position,
// so each DPAS waits on its predecessor's writeback.
//
// This is NOT FMHA_PREFILL_QK_GROUP: grouping adds one accumulator per K *block*, and each
// still carries the full-depth chain. This splits the chain itself, so it is the only knob
// here that touches GEMM1's latency rather than its traffic.
//
// Costs SPLIT-1 extra S accumulators (FragS is small next to the O accumulator, unlike the
// Q residency fragments) plus SPLIT-1 fragment adds per K block. Must be a power of two so
// the accumulator index stays a compile-time constant inside the unrolled loop; there is no
// divisibility requirement, since slot 0 is the existing group accumulator and a partial
// final round is summed in like any other.
//
// MEASURED AND REJECTED; kept default-off as a recorded refutation. Correct at every setting
// (bad=0, six verify shapes) and slower at every setting, and the per-launch numbers show
// exactly where it goes wrong -- k1 (the load launch, which runs no GEMM1) is unchanged to
// three digits, so the change is properly isolated, while k0 balloons:
//
//   split | spill B/thread | k0 ms | k1 ms | TFLOPS @ b1/hq32/sq4096
//   1     | 640/704        | 18.41 | 4.65  | 47.69  (keeper)
//   2     | 6464-7360      | 34.97 | 4.66  | 27.74
//   4     | 1984-2688      | 60.38 | 4.63  | 16.91
//
// So GEMM1's ~40 TFLOPS is not a dependency-depth problem. FragS is small -- a few hundred
// bytes per lane -- yet asking for one more copy of it across the D loop costs thousands of
// bytes of spill, the same disproportion Q residency hit (~600 B/thread per 32 B/lane
// chunk). The generalization worth keeping: inside GEMM1's head-dim loop there is no
// register slack at all, so *any* variant whose mechanism is "hold one more thing across
// that loop" is refuted in advance -- Q residency, KSTEP widening, TILED_KV widening,
// N-split and this. Note spill is not even monotone in the split (4 spills less than 2 and
// is slower still), which means the scheduler is thrashing rather than simply overflowing.
#ifndef FMHA_PREFILL_QK_ACC_SPLIT
#define FMHA_PREFILL_QK_ACC_SPLIT 1
#endif

// Share the softmax statistics between ScoreBlock2D launches instead of recomputing them.
//
// Every launch currently re-derives the whole online softmax from the same stored logits:
// a row-max reduction, an exp2 per element, a row-sum reduction, and -- the expensive part --
// a rescale of the *entire* O accumulator on every K block, because the running max keeps
// moving. Mode 0 already computes the final per-row max and sum by the time it finishes, and
// those values are identical for every output tile, since all tiles share the same S.
//
// So mode 0 writes them once (one max + one sum per Q row, i.e. TILED_Q floats each -- 2 KB
// per workgroup at TILED_Q=256, next to the score block's megabytes) and the load launches
// read them back and compute P = exp2(scale*S - final_max) directly. Two things then
// disappear from the load launches:
//   1. both row reductions, replaced by a broadcast of a value read from memory;
//   2. the per-K-block `tArA *= rescale` over the whole O accumulator -- the max is already
//      final, so there is nothing to rescale. That loop touches rows_per_SG * TILED_OUT
//      floats per K block, so it is the real target here.
// The epilogue's division by the sum is unaffected: tA_sum is filled from the stored value.
//
// Motivated by the per-launch breakdown: GEMM1 dominates (18.4 ms vs 4.6), but 47.8 -> 50
// TFLOPS needs only ~1.1 ms of the 23.0, and this removes work from the load launch without
// touching GEMM1's register-starved head-dim loop at all -- the constraint that refuted
// Q residency, KSTEP, TILED_KV, N-split and QK_ACC_SPLIT.
//
// MEASURED: exactly neutral, so it ships default-off. b1/hq32/dv512/sq4096, same-batch
// reference, spill unchanged at 640/704:
//     off: k0=18.400 k1=4.688 total=23.088 -> 47.62 TFLOPS
//     on:  k0=18.409 k1=4.678 total=23.087 -> 47.62 TFLOPS
// Accuracy improves slightly (max_abs 0.00129 vs 0.00153), as expected from using a final
// rather than a moving max. The point is *where* it is neutral: k1 shed both row reductions
// and the whole-accumulator rescale and moved 0.2%. So the load launch is not bound by the
// softmax arithmetic at all -- it is bound by its V and score loads plus GEMM2's DPAS, and
// removing ALU work from it buys nothing. Kept because it is the clean disproof of "the load
// launches redo work launch 0 already did", and because the seeded path is a building block
// for anything that wants the final max up front.
#ifndef FMHA_PREFILL_SHARE_SOFTMAX_STATS
#define FMHA_PREFILL_SHARE_SOFTMAX_STATS 0
#endif

// Stagger where each subgroup starts GEMM1's head-dim walk, by this many chunks per
// subgroup (0 = off, all subgroups start at chunk 0 as before).
//
// Motivated by reconciling three measurements that look contradictory. Pinning K's address
// is worth +3.3 TFLOPS (51.14 vs 47.82) and pinning Q's ~+2.5 -- yet halving GEMM1's K
// *request count* (KSTEP=64, once TILED_KV=32 makes it spill-free) is worth exactly 0%, and
// its reorders 0.03%. If neither request volume nor data movement costs anything but
// address identity does, then the cost is request *simultaneity*: SubgroupLayoutQK splits Q
// only, so all 32 subgroups walk D in lockstep and issue the same K address in the same
// cycle, serializing on a single cache line. Both pin probes remove exactly that collision,
// which is why they beat every volume-reduction variant.
//
// GEMM1's D loop is a reduction into tSrS, so each subgroup may traverse it in any order --
// only the fp summation order changes, which is why this is free where everything else
// costs registers. Skewing by 1 gives 32 subgroups 16 distinct chunks at head_dim=512
// (nD=16), so pressure per chunk drops ~16x. This is the one lever the pin probes' 51.14
// suggests is real and that no other variant reaches.
//
// MEASURED: this works, and it is the first hd=512 variant to beat the baseline.
// b1/hq32/dv512/sq4096, same-batch reference, bad=0 on all six verify shapes:
//     off:    k0=18.384 k1=4.649 total=23.033 -> 47.74 TFLOPS
//     skew=1: k0=17.904 k1=4.643 total=22.547 -> **48.77** TFLOPS
//     skew=2: k0=17.973 k1=4.660 total=22.633 -> 48.58 TFLOPS
// The gain is entirely in k0 with k1 unchanged to three digits, which is exactly what the
// simultaneity diagnosis predicts -- only GEMM1 has the lockstep D walk. Spill rises just
// 640/704 -> 704/768, i.e. this does not fight the register wall that refuted the rest.
// skew=1 is the default-worthy value; larger strides spread addresses further but give
// fewer distinct chunks, and measure slightly worse.
//
// **Default ON at 1**, because the gain holds across every shape measured and never
// regresses (b1/hq32/sq2048 42.78 -> 43.33, b1/hq32/sq4096 47.68 -> 48.67, b2/hq32/sq2048
// 45.68 -> 46.24, b1/hq16/sq4096 46.73 -> 47.29, b4/hq32/sq1024 39.18 -> 40.14).
//
// It composes with nothing that costs registers, which follows from *why* it is free: it
// changes only traversal order. Combining it with QK_GROUP=2 spills 7.6KB and gives 23.8
// TFLOPS; with TILED_KV=96 it spills 4.9KB and gives 24.4. Skew does not buy headroom for
// the variants the register wall already refuted -- do not retry them with it enabled.
#ifndef FMHA_PREFILL_D_SKEW
#define FMHA_PREFILL_D_SKEW 1
#endif

// Apply the same per-subgroup rotation to GEMM2's V-tile prefetch. Separate from D_SKEW so
// the two can be attributed independently: D_SKEW only touches GEMM1, and the score-load
// launch (k1, 4.65 of the 22.6 ms) contains no GEMM1 at all, so if simultaneity also costs
// there it has to be reachable from the V side. Prefetch order is free to permute -- it
// feeds no DPAS -- unlike GEMM2's VTiles *consume* loop, whose accumulator index must stay
// a compile-time constant.
#ifndef FMHA_PREFILL_V_PF_SKEW
#define FMHA_PREFILL_V_PF_SKEW 0
#endif

// Advance the D_SKEW rotation by this many chunks per K block as well (0 = fixed rotation).
// The idea was that D_SKEW spreads the 32 subgroups in space but leaves the *same*
// arrangement every K block, so any residual collision recurs identically 64 times.
// Free for the same reason D_SKEW is -- it only permutes a reduction's traversal order --
// and it is spill-neutral (640/704) and correct (bad=0 on all six shapes).
//
// Measured 49.43 (skew 1) and 49.33 (skew 2) against an in-batch reference holding at
// 49.66/49.57, i.e. a small consistent regression, so this stays off. The reason is that
// there is no residual collision left to move: nD = 16 at head_dim=512 and there are 32
// subgroups, so `sg * 1 % 16` already lands exactly 2 subgroups on each chunk, which is
// the pigeonhole optimum for any function of sg alone. A per-K-block phase rotates both
// members of every colliding pair by the same amount, so the pairs are unchanged; all it
// does is break the serpentine Q reuse ZIGZAG_D relies on. Kept for the record: the
// spatial-arrangement axis is closed, not merely untried.
#ifndef FMHA_PREFILL_K_SKEW
#define FMHA_PREFILL_K_SKEW 0
#endif

// Issue the *next* K block's prefetch before softmax/GEMM2 instead of after them.
// By default those 16 prefetches sit at the very end of the loop body, immediately
// before the split barrier, so they have almost no time to land before the next
// block's GEMM1 issues the matching loads. Hoisting them above softmax+GEMM2 gives
// them that entire window -- the prefetch distance goes from ~0 to one GEMM2.
// Free in the same sense PF_ZIGZAG is: prefetch feeds no DPAS and writes no fragment,
// so only the issue point moves. Implemented as two guarded copies of the same loop
// rather than a hoist, so the default path's statements stay textually identical
// (this mainloop's register allocation is sensitive to far less than that).
//
// Measured neutral -- 49.64/49.70 against an interleaved reference at 49.69/49.63,
// spill unchanged -- so K's prefetches already land in time even from the loop tail
// and this stays off. It is kept because the probe is what identified the real gap:
// V had no such one-block distance at all, and giving it one is V_PF_NEXT below.
#ifndef FMHA_PREFILL_PF_EARLY
#define FMHA_PREFILL_PF_EARLY 0
#endif

// Prefetch the *next* K block's V instead of the current block's.
// K is prefetched a whole block ahead, but V is not: the V prefetch sits immediately
// before the GEMM2 consume loop that loads the very same page, so its prefetch
// distance is a softmax rather than a K block. Retargeting it at next_page_idx gives
// V the same one-block distance K already has, at the cost of the first block's V
// arriving cold. Register-free (prefetch only), and unlike V_PF_SKEW it changes which
// block is fetched rather than the order within one. The next index must be *clamped*
// to the last block: the first version reused the unclamped next_page_idx (as the K
// prefetch does) and hung the device, RC=124 at sq2048 and sq4096, though the six small
// verify shapes all passed -- so a prefetch address one past the page table is not
// harmless here even though it is for K.
//
// 0 = off, 1 = every launch, 2 = only the ScoreBlock2D load launches.
// Mode 1 splits sharply by launch, which is what makes 2 the useful setting: k1 (score
// load + GEMM2, no GEMM1) went 4.68 -> 4.43 ms, a 5.3% win, while k0 (GEMM1 + store +
// GEMM2) went 17.48 -> 18.75, a 7% loss -- net 47.42 vs 49.60. The asymmetry is the
// point: in k0 the extra block of V in flight competes with GEMM1's Q and K for L1,
// and GEMM1 is what the whole kernel is bound on; in k1 there is no GEMM1 to compete
// with, so V's latency is exposed and covering it is free. Same launch-conditional
// shape as NO_SPLIT_BARRIER=2, for the same reason.
#ifndef FMHA_PREFILL_V_PF_NEXT
#define FMHA_PREFILL_V_PF_NEXT 2
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

// The relative-bias surface is the caller's tensor as-is -- dense [q, h, k], no padding on
// either axis -- so the block 2D atom is only legal when that shape happens to satisfy the
// hardware rules (see cute/atom/copy_traits_xe_2d.hpp): 64B-aligned base, width and pitch a
// multiple of 4B and at least 64B, and an x offset that is a multiple of 4 elements. The
// batch and head offsets are carried in the surface coordinates rather than the base pointer
// (see FMHAFwdMainloop::add_rel_bias_tile), so only the strides matter, which makes this a
// host-side question: the answer is baked into the kernel as RelBiasBlock2D and the other
// load never gets compiled. head_stride is both the surface width of head 0 (so >= 64B, i.e.
// >= 32 elements) and the x-offset granularity of the later heads (so a multiple of 4
// elements). An odd seqlen_k fails the pitch rule; those runs read the bias element by
// element instead.
CUTLASS_HOST_DEVICE inline bool rel_bias_can_block_2d(
    void const* ptr_rel_bias, int64_t rel_bias_token_stride, int64_t rel_bias_head_stride) {
  constexpr int64_t kElemsPer64B = 64 / int64_t(sizeof(cutlass::bfloat16_t));
  return (rel_bias_head_stride % 4 == 0) && (rel_bias_head_stride >= kElemsPer64B) &&
      (rel_bias_token_stride % 2 == 0) && (rel_bias_token_stride >= kElemsPer64B) &&
      (reinterpret_cast<uintptr_t>(ptr_rel_bias) % 64 == 0);
}

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
    bool PackGQA_ = false,
    bool HasRelBias_ = false,
    // Whether the relative-bias tensor can be read with the block 2D atom. Decided on
    // the host from the caller's pointer and strides (rel_bias_can_block_2d) and baked
    // in here, because the two loads are different enough that having both in one
    // kernel costs the fast one register pressure it cannot afford.
    bool RelBiasBlock2D_ = true>
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
    bool PackGQA_,
    bool HasRelBias_,
    bool RelBiasBlock2D_>
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
    PackGQA_,
    HasRelBias_,
    RelBiasBlock2D_> {
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
  using TiledCopyK_cache = conditional_t<
      is_void_v<TiledCopyK_cache_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK_cache2D{})),
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
  static constexpr bool HasRelBias = HasRelBias_;
  static constexpr bool RelBiasBlock2D = RelBiasBlock2D_;
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
  static constexpr int QResidentChunks = FMHA_PREFILL_Q_RESIDENT;
  static constexpr bool QResident = QResidentChunks > 0;
  // Residency replaces the per-block Q load outright, so it subsumes grouping; allowing
  // both would keep KGrp S accumulators live for no remaining benefit.
  static_assert(!QResident || QKGroup == 1, "FMHA_PREFILL_Q_RESIDENT and QK_GROUP > 1 are exclusive");
  static_assert(!QResident || FMHA_PREFILL_K_SLM == 0, "FMHA_PREFILL_Q_RESIDENT and K_SLM are exclusive");
  // GEMM1 dependency-chain split. Confined to the plain (non-grouped, non-staged,
  // non-resident) GEMM1 path: each of the other paths already restructures the same loop,
  // and combining them would multiply accumulator counts for no separable measurement.
  // Share the final softmax row statistics across launches rather than recomputing them.
  // Meaningless without the score round-trip, since there is only one launch then.
  static constexpr bool ShareSoftmaxStats = ScoreBlock2D && FMHA_PREFILL_SHARE_SOFTMAX_STATS;
  // Floats per workgroup for one statistic. Sized by the thread-flat layout the mainloop
  // writes (thr_id * FragARow::size()), not by TILED_Q: every thread in the workgroup gets
  // its own slot, so the maxima occupy [0, kStatsPerWG) and the sums [kStatsPerWG, 2*...).
  static constexpr int kStatsPerWG =
      int(SGPerWG::value) * int(intel::sg_size) * int(decltype(reduce<1>(FragA{}, sycl::plus<void>{})){}.size());
  // Under SplitStore mode 0 skips softmax entirely (it runs no GEMM2), so it has no
  // statistics to publish -- the two features are mutually exclusive as written.
  static_assert(
      !(ScoreBlock2D && FMHA_PREFILL_SHARE_SOFTMAX_STATS && FMHA_PREFILL_SPLIT_STORE),
      "FMHA_PREFILL_SHARE_SOFTMAX_STATS and FMHA_PREFILL_SPLIT_STORE are exclusive");
  // Mode 0 is the only launch that runs GEMM1, so it is the only one that can produce the
  // statistics; with SplitStore it also runs no GEMM2, which is fine -- it still completes
  // the full online softmax over every K block, which is all the statistics require.
  static constexpr int QKAccSplit = FMHA_PREFILL_QK_ACC_SPLIT;
  static_assert(QKAccSplit >= 1, "FMHA_PREFILL_QK_ACC_SPLIT must be at least 1");
  // Power of two: the loop indexes the accumulator array with `Di & (QKAccSplit-1)`, which
  // must fold to a constant in each unrolled body or the array lands in scratch.
  static_assert((QKAccSplit & (QKAccSplit - 1)) == 0, "FMHA_PREFILL_QK_ACC_SPLIT must be a power of two");
  static_assert(QKAccSplit == 1 || QKGroup == 1, "FMHA_PREFILL_QK_ACC_SPLIT and QK_GROUP > 1 are exclusive");
  static_assert(QKAccSplit == 1 || FMHA_PREFILL_Q_RESIDENT == 0, "QK_ACC_SPLIT and Q_RESIDENT are exclusive");
  static_assert(QKAccSplit == 1 || FMHA_PREFILL_K_SLM == 0, "QK_ACC_SPLIT and K_SLM are exclusive");
  static constexpr bool ZigzagD = FMHA_PREFILL_ZIGZAG_D;
  // Per-subgroup rotation of GEMM1's head-dim walk; 0 = off. Exclusive with Q residency,
  // which pins specific chunks in registers and so requires a fixed per-subgroup order.
  static constexpr int DSkew = FMHA_PREFILL_D_SKEW;
  static_assert(DSkew == 0 || FMHA_PREFILL_Q_RESIDENT == 0, "D_SKEW and Q_RESIDENT are exclusive");
  // Extra per-K-block phase on top of D_SKEW; 0 = the rotation is fixed for the whole K loop.
  // Only meaningful with D_SKEW > 0, which is where it is applied.
  static constexpr int KSkew = FMHA_PREFILL_K_SKEW;
  // Move the next block's K prefetch ahead of softmax/GEMM2 instead of after them.
  static constexpr bool PfEarly = FMHA_PREFILL_PF_EARLY;
  // Aim the V prefetch one K block ahead, matching the distance K already gets.
  // Mode 2 restricts it to the load launches; resolved in the mainloop body, since
  // StaticScoreMode is a parameter of that function rather than of this class.
  static constexpr int VPfNextMode = FMHA_PREFILL_V_PF_NEXT;
  // Per-subgroup rotation of GEMM2's V-tile *prefetch* order; 0 = off.
  static constexpr int VPfSkew = FMHA_PREFILL_V_PF_SKEW;
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
    cutlass::bfloat16_t const* ptr_rel_bias = nullptr;
    int64_t rel_bias_token_stride = 0;
    int64_t rel_bias_head_stride = 0;
    int rel_bias_extent = 0;
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
    Params params{
        val,
        args.ptr_page_table,
        args.page_size,
        args.max_num_pages_per_seq,
        args.window_size_left,
        args.window_size_right,
        ScoreBlock2D ? reinterpret_cast<ElementScoreStore*>(workspace) : nullptr,
        args.ptr_rel_bias,
        args.rel_bias_token_stride,
        args.rel_bias_head_stride,
        args.rel_bias_extent};
    return params;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Adds one K block of the relative bias into the scores, folding the still-pending
  // softmax scale into the same multiply-add. RelBiasBlock2D picks the fast path: one 2D
  // block load covers the whole tile. The other instantiation loads element by element and
  // serves the surfaces the block 2D atom cannot address at all. The choice is made on the
  // host, so only one of the two is ever compiled into a given kernel.
  template <typename QVCoord>
  CUTLASS_DEVICE void add_rel_bias_tile(
      FragS& scores,
      ElementS& score_scale,
      TiledMMAQK const& mma_qk,
      int bias_col,
      int bias_rows,
      int q_head,
      int q_token_offset,
      QVCoord const& blk_qv,
      int thr_id) const {
    constexpr ElementS kLog2e = ElementS(1.4426950408889634074);
    constexpr int k_tile = get<1>(TileShapeQK{});
    int const bias_cols = static_cast<int>(params.rel_bias_head_stride);
    if constexpr (RelBiasBlock2D) {
      // The surface is the [q, h*k] tensor based at the (64B-aligned) allocation: the row
      // offset of this batch and the column offset of this head go into the tile
      // coordinates instead, where they carry no alignment requirement beyond the x offset
      // checked in rel_bias_can_block_2d. Height stops at the end of this batch's rows and
      // width at the end of this head's columns, so both a Q tile overhanging the sequence
      // and a K tile overhanging seqlen_k read zeros there -- rather than the next batch,
      // the next head, or past the tensor. That is what lets the K tail go through the same
      // load as every other block; the k-remainder mask has already put -inf in those
      // scores, so the bound is what keeps the value that is added to it well defined
      // rather than another head's logits.
      //
      // Width is a whole number of heads, so it inherits head_stride's alignment, and the
      // pitch stays the full row stride. Both extents fit an int -- the width is at most
      // one row -- while the strides stay 64-bit, which is what addresses the whole tensor.
      auto surface_shape = make_shape(q_token_offset + bias_rows, (q_head + 1) * bias_cols);
      auto surface_layout = make_layout(surface_shape, make_stride(params.rel_bias_token_stride, Int<1>{}));
      Tensor Bias = make_tensor(make_gmem_ptr(params.ptr_rel_bias), surface_layout);
      Tensor cBias = domain_offset(
          make_coord(q_token_offset, q_head * bias_cols), make_identity_tensor(make_shape(bias_rows, bias_cols)));
      Tensor gBias = local_tile(cBias, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), _));
      auto copy_bias_load = make_block_2d_copy_C(mma_qk, Bias);
      auto thr_copy_bias_load = copy_bias_load.get_slice(thr_id);
      auto tBiasLoadG = thr_copy_bias_load.partition_S(gBias);
      auto tBiasLoadR = thr_copy_bias_load.partition_sg_fragment_D(gBias(_, _, 0));
      auto bias = make_subgroup_tensor(make_fragment_like<cutlass::bfloat16_t>(scores.layout()), scores.tv_layout());
      copy(copy_bias_load, tBiasLoadG(_, _, _, bias_col / k_tile), tBiasLoadR);
      reorder(tBiasLoadR, bias);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < scores.size(); ++i) {
        ElementS const scaled_bias = kLog2e * static_cast<ElementS>(bias(i));
        scores(i) = sycl::mad(score_scale, scores(i), scaled_bias);
      }
    } else {
      // The block 2D load bounds-checks against the surface and substitutes zero outside
      // it; the scalar path has to do that itself, which is what the guards below are for.
      // Row/column come from the same lane arithmetic the causal mask uses -- fragment
      // element n*elems_per_n + j is row j of the subgroup's rows and column n*sg_size +
      // lane -- rather than a partitioned coordinate tensor, which costs registers in a
      // loop this hot. Lane l therefore reads column l of a 16-wide run, so each element
      // is still one contiguous 32B gather across the subgroup.
      auto const bias_head_ptr = params.ptr_rel_bias +
          int64_t(q_token_offset) * params.rel_bias_token_stride + int64_t(q_head) * params.rel_bias_head_stride;
      constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
      constexpr int n_reps = k_tile / intel::sg_size;
      constexpr int elems_per_n = FragS{}.size() / n_reps;
      int const lane_id = thr_id % intel::sg_size;
      int const row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) + (thr_id / intel::sg_size) * sg_tile_q;
      int const rows = cute::min(elems_per_n, cute::max(0, bias_rows - row_base));
      auto const row_ptr = bias_head_ptr + int64_t(row_base) * params.rel_bias_token_stride;
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < n_reps; ++n) {
        int const col = bias_col + n * intel::sg_size + lane_id;
        auto const col_ptr = row_ptr + col;
        bool const col_ok = col < bias_cols;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < elems_per_n; ++j) {
          ElementS bias = ElementS(0);
          if (col_ok && j < rows) {
            bias = static_cast<ElementS>(col_ptr[int64_t(j) * params.rel_bias_token_stride]);
          }
          scores(n * elems_per_n + j) = sycl::mad(score_scale, scores(n * elems_per_n + j), kLog2e * bias);
        }
      }
    }
    score_scale = ElementS(1);
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void apply_relative_bias(
      FragS& scores,
      int K,
      ElementS& score_scale,
      TiledMMAQK const& mma_qk,
      int seq_len_kv_cache,
      int q_head,
      int q_token_offset,
      QVCoord const& blk_qv,
      int thr_id,
      int full_tile_offset) const {
    if constexpr (HasRelBias) {
      constexpr int k_tile = get<1>(TileShapeQK{});
      constexpr int q_tile = get<0>(TileShapeQK{});
      int const bias_col = K * k_tile;
      // Columns past seqlen_k carry no bias at all (and no memory): the k-remainder mask
      // has already driven those scores to -inf.
      if (bias_col >= params.rel_bias_head_stride) return;
      // The relative bias b(i,j) = R[i, i-j] is exactly zero unless the distance
      // rel = row_kv - col falls in [0, extent). The host zero-fills the rest, so a
      // K block whose whole [bias_col, bias_col+k_tile) column span lies outside the
      // band for every row_kv in this Q tile contributes only zeros -- skip its load
      // and add entirely. row_kv for this tile spans [R0, R1]; a straight-through walk
      // reads the full row, which at seq >> extent is mostly zeros. This prune is exact:
      // no accuracy change, only traffic removed.
      int const R0 = get<0>(blk_qv) * q_tile + full_tile_offset;  // min row_kv in tile
      int const R1 = R0 + q_tile - 1;                             // max row_kv in tile
      bool const in_band = (bias_col <= R1) && (bias_col + k_tile - 1 > R0 - params.rel_bias_extent);
      if (!in_band) return;
      // Rows index the query, so the surface is exactly this batch's query length tall --
      // the kernel builds full_tile_offset as seq_len_kv_cache - seq_len_qo. Rows past it
      // belong to the next batch (or to nothing at all, at the end of the tensor) and read
      // as zero, which is what a Q tile overhanging the sequence needs.
      int const bias_rows = seq_len_kv_cache - full_tile_offset;
      add_rel_bias_tile(scores, score_scale, mma_qk, bias_col, bias_rows, q_head, q_token_offset, blk_qv, thr_id);
    }
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
      int q_head,
      int q_token_offset,
      int full_tile_offset,
      int discard_seq_coord,
      TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
      TensorV_cache2D const& V_cache_2D = TensorV_cache2D{},
      ElementScoreStore* score_head_ptr = nullptr,
      // Base of this workgroup's softmax-statistics slot (2 * TILED_Q floats: maxima then
      // sums). Null unless ShareSoftmaxStats.
      ElementS* stats_wg_ptr = nullptr) {
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
    auto copy_score_store = make_block_2d_copy_D(mma_qk, Score);
    auto copy_score_load = make_block_2d_copy_C(mma_qk, Score);
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
      // Same skew as the in-loop prefetch and the loads: start each subgroup at the chunk
      // it will consume first, and stop all 32 from issuing one address at one instant.
      const int sg = int(thr_id / intel::sg_size);
      for (int Di = 0; Di < nQpf; Di++) {
        int D = Di;
        if constexpr (DSkew > 0) {
          D += (sg * DSkew) % nQpf;
          if (D >= nQpf) {
            D -= nQpf;
          }
        }
        prefetch(prefetch_q, pQgQ(_, _, _, D));
      }
      for (int Di = 0; Di < nKpf; Di++) {
        int D = Di;
        if constexpr (DSkew > 0) {
          D += (sg * DSkew) % nKpf;
          if (D >= nKpf) {
            D -= nKpf;
          }
        }
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

    // Load launches: seed tA_max/tA_sum with the *final* values mode 0 computed, so the
    // per-block softmax below has nothing left to discover. Every launch partitions the same
    // FragARow the same way (same MMA, same subgroup layout), so element i of this thread's
    // fragment is the same Q row in every launch and a flat per-thread slot needs no
    // coordinate math. tA_sum is seeded too, so the epilogue's division is unchanged.
    if constexpr (ShareSoftmaxStats && StaticScoreMode >= 1) {
      const int stat_base = thr_id * int(FragARow{}.size());
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA_max.size(); i++) {
        const ElementA m = stats_wg_ptr[stat_base + i];
        // A fully-masked row (no unmasked keys anywhere, possible under causal/local) keeps
        // max == lowest(), and exp2(scale*S - lowest()) would overflow to inf here -- the
        // online path never evaluates that because such a row's blocks are all -inf and its
        // sum stays 0, which the epilogue special-cases. Substituting 0 keeps the
        // exponentials finite; the row's sum is still 0, so the epilogue emits 0 as before.
        tA_max(i) = (m == cutlass::platform::numeric_limits<ElementA>::lowest()) ? ElementA(0) : m;
        tA_sum(i) = stats_wg_ptr[kStatsPerWG + stat_base + i];
      }
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    constexpr bool SkipSplitBarrier =
        (NoSplitBarrierMode == 1) || (NoSplitBarrierMode == 2 && ScoreBlock2D && StaticScoreMode >= 1);

    constexpr bool VPfNext =
        (VPfNextMode == 1) || (VPfNextMode == 2 && ScoreBlock2D && StaticScoreMode >= 1);

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
        } else if constexpr (QKAccSplit > 1) {
          // Identical loads in an identical order to the plain path below -- the only change
          // is which accumulator each DPAS targets. Head-dim chunks are dealt round-robin
          // into QKAccSplit independent accumulators, so the serial DPAS chain each one
          // carries is QKAccSplit times shorter, and QKAccSplit chains are in flight at
          // once. The partials are summed at the end of the K block: QKAccSplit-1 fragment
          // adds against nD DPAS, so the overhead shrinks as head_dim grows.
          // The loop shape below is deliberately identical to the plain path's -- one
          // CUTLASS_PRAGMA_UNROLL loop over nD, same serpentine index, same two copies. An
          // earlier version stepped whole rounds of QKAccSplit in a dynamic outer loop with
          // a scalar tail; that broke the unroll the copy fragments depend on and spill went
          // 640 B/thread -> 8.5 KB, doubling GEMM1's time. Only the accumulator index may
          // change. QKAccSplit is required to be a power of two so `Di & (QKAccSplit-1)` is
          // a constant in each unrolled body: a non-constant index would put the
          // accumulator array in scratch, which is the same failure by another route.
          const int nD = size<4>(tKgK);
          FragS tSrS_part[QKAccSplit - 1];
          CUTLASS_PRAGMA_UNROLL
          for (int s = 0; s < QKAccSplit - 1; s++) {
            clear(tSrS_part[s]);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int Di = 0; Di < nD; Di++) {
            const int D = (ZigzagD && ((K_grp / KGrp) & 1)) ? (nD - 1 - Di) : Di;
            copy(copy_q, tQgQ(_, _, _, D), tQrQ);
            reorder(tQrQ, tSrQ);
            copy(copy_k_cache, tKgK_cache(_, _, _, grp_page_idx[0], D), tKrK);
            reorder(tKrK, tSrK);
            // Slot 0 is the group accumulator itself, so a partial round needs no tail: any
            // chunk count lands somewhere valid and every slot is summed in below.
            const int slot = Di & (QKAccSplit - 1);
            if (slot == 0) {
              cute::gemm(mma_qk, tSrQ, tSrK, tSrS_grp[0]);
            } else {
              cute::gemm(mma_qk, tSrQ, tSrK, tSrS_part[slot - 1]);
            }
          }
          CUTLASS_PRAGMA_UNROLL
          for (int s = 0; s < QKAccSplit - 1; s++) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragS{}.size(); i++) {
              tSrS_grp[0](i) += tSrS_part[s](i);
            }
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
            int D = (ZigzagD && ((K_grp / KGrp) & 1)) ? (nD - 1 - Di) : Di;
            if constexpr (DSkew > 0) {
              // Rotate each subgroup's start point in the head-dim walk. GEMM1's D loop is a
              // reduction into tSrS, so any permutation of it is valid; only the fp summation
              // order changes. Without this every subgroup requests the *same* K chunk at the
              // same instant -- SubgroupLayoutQK splits Q only, so all 32 issue one identical
              // address and serialize on one cache line. Skewing makes the 32 in-flight
              // requests hit distinct chunks instead. Costs no registers, unlike every other
              // way of reducing K pressure. Subtract-if instead of modulo: nD is a dynamic
              // int, so % would emit a division inside the hot loop.
              // KSkew advances the phase per K block as well, so the 32-way arrangement is
              // not merely spread but also moves in time; with a fixed rotation every K
              // block reproduces the identical collision pattern.
              D += (int(thr_id / intel::sg_size) * DSkew + (KSkew > 0 ? (K_grp / KGrp) * KSkew : 0)) % nD;
              if (D >= nD) {
                D -= nD;
              }
            }
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
          // One block ahead when VPfNext, clamped: on the final block next_page_idx is
          // derived from K+1 == kblocks_cache, i.e. one past the page table, and an
          // out-of-range physical page here hangs the device (measured RC=124 unclamped).
          int v_pf_idx = page_idx;
          if constexpr (VPfNext) {
            const int Knext = cute::min(K + 1, k_end - 1);
            v_pf_idx = PagedKV ? get_physical_k_tile(Knext, l_coord, seq_len_kv_cache) : Knext;
          }
          CUTLASS_PRAGMA_UNROLL
          for (int VVi = 0; VVi < VTiles; VVi++) {
            // Prefetch only -- the tile index need not be a compile-time constant here,
            // unlike the consume loop below where it indexes the accumulator.
            int VV = VVi;
            if constexpr (VPfSkew > 0) {
              VV = (VVi + int(thr_id / intel::sg_size) * VPfSkew) % VTiles;
            }
            prefetch(prefetch_v_cache, pVgV_cache(_, _, _, VV, v_pf_idx));
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

        /* K prefetch, early placement -- see FMHA_PREFILL_PF_EARLY. Identical to the
           block at the end of the loop body; exactly one of the two is compiled in.
           next_page_idx already points at the next block here (it is advanced above,
           before the score store), so the addresses are the same either way. */
        if constexpr (!(ScoreBlock2D && StaticScoreMode >= 1) && PfEarly) {
          const int nPf = size<4>(pKgK);
          const bool rev = PfZigzag && ZigzagD && (((K_grp / KGrp) + 1) & 1);
          const int pf_skew =
              (DSkew > 0)
                  ? (int(thr_id / intel::sg_size) * DSkew + (KSkew > 0 ? (K_grp / KGrp + 1) * KSkew : 0)) % nPf
                  : 0;
          for (int Di = 0; Di < nPf; Di++) {
            int D = rev ? (nPf - 1 - Di) : Di;
            if constexpr (DSkew > 0) {
              D += pf_skew;
              if (D >= nPf) {
                D -= nPf;
              }
            }
            prefetch(prefetch_k_cache, pKgK_cache(_, _, _, next_page_idx, D));
          }
        }

        // Store-only launch: the scores are on their way to the scratch and no output
        // tile belongs to this launch, so skip softmax and GEMM2 entirely. Leaving the
        // O accumulator untouched is the point -- it is what frees the registers.
        if constexpr (!(SplitStore && StaticScoreMode == 0)) {
          /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
          ElementS softmax_scale = params.scale;
          // With shared statistics the load launches already hold the final row max, so the
          // exponentiation is all that is left: no row reductions, and -- the point of the
          // whole variant -- no rescale of the O accumulator, since the max cannot move.
          // Kept as a separate `if constexpr` around only the softmax call rather than a
          // branch that assigns a `rescale` variable: default-constructing FragSRow here
          // perturbs register allocation for the entire loop, in every launch (measured:
          // mode 0 went 18.4 -> 51.7 ms).
          if constexpr (ShareSoftmaxStats && StaticScoreMode >= 1) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrS.size(); i++) {
              tSrS(i) = sycl::native::exp2(softmax_scale * tSrS(i) - broadcast<0>(tA_max, tSrS, i));
            }
            reorder(tSrS, tArP);

            /* GEMM 2: A += P * V, split in v dimension. No rescale. */
            CUTLASS_PRAGMA_UNROLL
            for (int VV = 0; VV < VTiles; VV++) {
              copy(copy_v_cache, tVgV_cache(_, _, _, VV, page_idx), tVrV);
              reorder(tVrV, tArV);
              cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
            }
          } else {
            apply_relative_bias(
                tSrS,
                K,
                softmax_scale,
                mma_qk,
                seq_len_kv_cache,
                q_head,
                q_token_offset,
                blk_qv,
                thr_id,
                full_tile_offset);
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
        }

        /* K prefetch */
        if constexpr (!(ScoreBlock2D && StaticScoreMode >= 1) && !PfEarly) {
          const int nPf = size<4>(pKgK);
          // Match the head-dim order the *next* group will walk in, so its first
          // chunk is the first one prefetched rather than the last.
          const bool rev = PfZigzag && ZigzagD && (((K_grp / KGrp) + 1) & 1);
          // Skew the prefetch order to match the skewed consume order. Two reasons: the
          // 32 subgroups otherwise issue the same prefetch address in the same cycle (the
          // same collision D_SKEW fixes for the loads), and with a skewed walk a fixed
          // 0..nPf-1 prefetch hands each subgroup its first-needed chunk at a different,
          // mostly wrong, position. Prefetch touches no fragment and feeds no DPAS, so
          // reordering it is free even where reordering a load would not be.
          // Phase must track the *next* K group, since that is what these lines prefetch.
          const int pf_skew =
              (DSkew > 0)
                  ? (int(thr_id / intel::sg_size) * DSkew + (KSkew > 0 ? (K_grp / KGrp + 1) * KSkew : 0)) % nPf
                  : 0;
          for (int Di = 0; Di < nPf; Di++) {
            int D = rev ? (nPf - 1 - Di) : Di;
            if constexpr (DSkew > 0) {
              D += pf_skew;
              if (D >= nPf) {
                D -= nPf;
              }
            }
            prefetch(prefetch_k_cache, pKgK_cache(_, _, _, next_page_idx, D));
          }
        }

        if constexpr (!SkipSplitBarrier) {
          barrier_wait(ScopeWorkgroup);
        }
      }
    }

    // Mode 0 publishes the finished row statistics for the load launches. Placed after the
    // whole K loop, so these are the final values; the launch boundary is the synchronization
    // (a later kernel cannot start before this one retires), so no barrier or fence is needed.
    if constexpr (ShareSoftmaxStats && StaticScoreMode == 0) {
      const int stat_base = thr_id * int(FragARow{}.size());
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA_max.size(); i++) {
        stats_wg_ptr[stat_base + i] = tA_max(i);
        stats_wg_ptr[kStatsPerWG + stat_base + i] = tA_sum(i);
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
