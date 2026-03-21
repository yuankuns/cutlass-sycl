/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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

/*
 * SLM GEMM Example 2: Two-stage fused GEMM with GEMM2 using TN layout.
 *
 * Demonstrates the SDPA backward dQ-like pattern:
 *   GEMM1 (NT): S[kBlockN, kBlockM] = A[kBlockN, D] * B1[kBlockM, D]^T
 *   slm_reorder_save: S accumulator f32 -> f16, save to SLM
 *   GEMM2 (TN): C[kBlockM, D] += S^T[kBlockM, kBlockN] * B2[kBlockN, D]
 *     A from SLM (transposed read), B from global
 *   Loop over n_blocks, accumulating C across iterations.
 *
 * Per workgroup (m_block):
 *   B1[kBlockM, D] fixed, C[kBlockM, D] output
 *   Loop n_block = 0..N/kBlockN-1:
 *     A_n[kBlockN, D], B2_n[kBlockN, D] advance
 *     S = A_n * B1^T        → [kBlockN, kBlockM]
 *     C += S^T * B2_n       → [kBlockM, D]
 *
 * Computes: C[M,D] = B1[M,D] * A[N,D]^T * B2[N,D]
 *
 * Key difference from slm_gemm_example (NN GEMM2):
 *   - GEMM2 reads SLM transposed: S^T[kBlockM, kBlockN]
 *   - SLM stored as S[kBlockN, kBlockM] kBlockM-contiguous (non-transposed)
 *   - For TN load, S^T(m,k)=S(k,m)=smem[k*kBlockM+m], consecutive m contiguous
 *   - B2 presented as B2t[D, kBlockN] col-major for MMA TT atom
 */

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "../sdpa_bwd/sdpa_util.hpp"

using namespace cute;

// ============================================================
// Trait struct
// ============================================================
template <class T_, int kBlockM_, int kBlockN_, int kHeadDim_, int kNSGs_,
          int AtomLayoutM1_ = 4, int AtomLayoutM2_ = 2>
struct SlmGemmTrait2 {
    using DType = T_;
    using VType = float;
    static constexpr int kBlockM  = kBlockM_;
    static constexpr int kBlockN  = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kNSGs    = kNSGs_;
    static constexpr int SubgroupSize = 16;
    static constexpr int AtomLayoutM1 = AtomLayoutM1_;
    static constexpr int AtomLayoutM2 = AtomLayoutM2_;

    using MMA_Atom_ARCH = XE_DPAS_TT<8, VType, DType>;
    using _K = Int<MMA_Atom_ARCH::K>;

    // GEMM1 (NT): S[kBlockN, kBlockM] = A[kBlockN, D] * B1[kBlockM, D]
    using SubgroupLayout1 = Layout<Shape<Int<AtomLayoutM1_>, Int<kNSGs / AtomLayoutM1_>, _1>>;
    using TileShape1      = Layout<Shape<Int<kBlockN>, Int<kBlockM>, _K>>;
    using TiledMma1 = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                              TileShape1,
                                              SubgroupLayout1>::TiledMMA;

    // GEMM2 (TN): C[kBlockM, kHeadDim] += S^T[kBlockM, kBlockN] * B2t[kHeadDim, kBlockN]
    // K-reduction dimension = kBlockN
    using SubgroupLayout2 = Layout<Shape<Int<AtomLayoutM2_>, Int<kNSGs / AtomLayoutM2_>, _1>>;
    using TileShape2      = Layout<Shape<Int<kBlockM>, Int<kHeadDim>, _K>>;
    using TiledMma2 = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                              TileShape2,
                                              SubgroupLayout2>::TiledMMA;

    static constexpr int smem_size = kBlockN * kBlockM * sizeof(DType);
};

// ============================================================
// create_reg: allocate register fragment (sdpa_backward.hpp style)
// ============================================================
template<typename T, class Trait, class MTensor, class TiledMMA>
auto
create_reg(Trait const &trait, MTensor const &C, TiledMMA const &tiled_mma) {
    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));
    auto thr_mma = tiled_mma.get_slice(local_id);

    Tensor cC = make_identity_tensor(C.shape());
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto copy_c = make_block_2d_copy_D(tiled_mma, C);
    auto thr_copy_c = copy_c.get_slice(local_id);

    if constexpr(is_same_v<T, float>) {
        auto r32 = thr_mma.partition_sg_fragment_C(
            make_identity_tensor(select<0,1>(tile_mnk)));
        return r32;
    } else {
        auto r16 = thr_copy_c.partition_sg_fragment_S(gC);
        return r16;
    }
}

// ============================================================
// gemm_kernel: both A, B from global memory with prefetch
// ============================================================
template<bool clear_acc, class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
gemm_kernel(Trait &trait,
            Tensor<Engine0, Layout0> const& A,
            Tensor<Engine1, Layout1> const& B,
            SubgroupTensor<Engine2, Layout2, TVLayout2> &acc,
            TiledMMA const &mma) {
    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0,2>(tile_mnk), make_coord(0,_));
    Tensor gB = local_tile(cB, select<1,2>(tile_mnk), make_coord(0,_));

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    auto thr_mma    = mma.get_slice(local_id);
    auto thr_copy_a = copy_a.get_slice(local_id);
    auto thr_copy_b = copy_b.get_slice(local_id);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_prefetch_A = prefetch_a.get_slice(local_id);
    auto thr_prefetch_B = prefetch_b.get_slice(local_id);
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    const int prefetch_dist = 3;
    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_prefetch = 0;

    if constexpr(clear_acc) clear(acc);

    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        barrier_arrive(barrier_scope);
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);
        gemm(mma, tCrA, tCrB, acc);
        barrier_wait(barrier_scope);
    }
}

// ============================================================
// slm_reorder_save: M-packed VNNI scatter store to SLM.
//   SLM layout: M-packed VNNI uint32[kBlockN][kBlockM/2].
//   Each row = one N-value's M-packed VNNI data.
//   VNNI pair j for N-row n: {S[n, m_2j], S[n, m_2j+1]} as two fp16.
//   row_stride = (kBlockM/2) * 4 bytes.
//
//   Works directly on f32 accumulator r (no CuTe reorder).
//   Process 2 N-rows per iteration: f32→f16 of 2 GRFs into 1 GRF.
//   In the resulting d32 SIMD16 view:
//     Lanes 0-7:  {S[row_even, col_2l], S[row_even, col_2l+1]}
//     Lanes 8-15: {S[row_odd,  col_2(l-8)], S[row_odd, col_2(l-8)+1]}
//   d32 scatter store to M-packed VNNI positions.
//   SubtileM/2 = 8 iterations per col-group × 2 col-groups = 16 d32 stores.
// ============================================================
template<class Trait,
         class Engine0, class Layout0,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
slm_reorder_save(Trait &trait,
                 Tensor<Engine0, Layout0> &mC,
                 uint32_t slm_offset,
                 SubgroupTensor<Engine2, Layout2, TVLayout2> &r,
                 TiledMMA const &tiled_mma) {
    using T = typename Trait::DType;

#ifdef __SYCL_DEVICE_ONLY__
    static_assert(sizeof(T) == 2, "Expected fp16/bf16");
    constexpr int kBlockN  = Trait::kBlockN;
    constexpr int kBlockM  = Trait::kBlockM;
    constexpr int SubgroupSize = Trait::SubgroupSize;
    constexpr int AtomLayoutM1 = Trait::AtomLayoutM1;
    constexpr int SubtileM = kBlockN / AtomLayoutM1;       // 16 N-rows per SG

    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));
    int sg_idx = local_id / SubgroupSize;
    int lane   = local_id % SubgroupSize;
    int sg_m   = sg_idx % AtomLayoutM1;
    int sg_n   = sg_idx / AtomLayoutM1;

    // M-packed VNNI: uint32[kBlockN][kBlockM/2]
    constexpr int AtomLayoutN1 = Trait::kNSGs / AtomLayoutM1;
    constexpr int SubtileN = kBlockM / AtomLayoutN1;       // 32 M-cols per SG
    constexpr int col_groups = SubtileN / SubgroupSize;     // 2
    constexpr uint32_t row_stride = (kBlockM / 2) * 4;     // 128 bytes per N-row

    // Per-lane N-row offset within each 2-row group:
    //   lanes 0-7:  even N-row (offset 0)
    //   lanes 8-15: odd  N-row (offset 1)
    uint32_t n_row_offset = lane >> 3;

    CUTLASS_PRAGMA_UNROLL
    for (int cg = 0; cg < col_groups; ++cg) {
        // M-pair for this col-group:
        //   sg_n * (SubtileN/2) + cg * (SubgroupSize/2) + (lane & 7)
        uint32_t m_pair = sg_n * (SubtileN / 2) + cg * (SubgroupSize / 2) + (lane & 7);
        uint32_t addr_col = slm_offset + m_pair * 4;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < SubtileM / 2; ++i) {
            auto &src = *recast_ptr<intel::float2>(&r(cg * SubtileM + 2 * i));
            uint32_t n_row = sg_m * SubtileM + 2 * i + n_row_offset;
            uint32_t addr = addr_col + n_row * row_stride;

        intel::float2 packed = {};
        asm volatile(
            "{\n"
            ".decl MP_HF v_type=G type=HF num_elts=64 alias=<%0,0>\n"
            ".decl MP_F  v_type=G type=F  num_elts=32 alias=<%1,0>\n"
            "mov (M1, 16) MP_HF(0,0)<1>  MP_F(0,0)<1;1,0>\n"
            "mov (M1, 16) MP_HF(0,16)<1> MP_F(1,0)<1;1,0>\n"
            "lsc_store.slm (M1, 16) flat[%2]:a32 %0:d32\n"
            "}\n"
            : "+rw"(packed)
            : "rw"(src), "rw"(addr)
        );
        }
    }
#else
    CUTE_INVALID_CONTROL_PATH("Inline ASM SLM store requires Xe device");
#endif
}

// ============================================================
// load_slm_tn: TN d32x8 load from M-packed VNNI SLM.
//   SLM layout: M-packed VNNI uint32[kBlockN][kBlockM/2].
//   Each lane reads its N-row (= K-col of GEMM2 MMA) across M-pairs.
//   d32x8 reads 8 consecutive M-pairs = 16 M-values.
//   VnniPairs_2 = SubtileM_2/2 = 16. 16/8 = 2 d32x8 loads per k-tile.
//   De-VNNI unpack: each d32 = {lo_fp16, hi_fp16} = 2 M-values.
// ============================================================
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2>
CUTLASS_DEVICE void
load_slm_tn(Trait &trait,
            uint32_t slm_offset,
            Tensor<Engine0, Layout0> const& sA,
            Tensor<Engine1, Layout1> const& tCgA_tile,
            SubgroupTensor<Engine2, Layout2, TVLayout2> &tCrA) {
    using T = typename Trait::DType;

#ifdef __SYCL_DEVICE_ONLY__
    static_assert(sizeof(T) == 2);
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int SubgroupSize = Trait::SubgroupSize;
    constexpr int AtomLayoutM2 = Trait::AtomLayoutM2;
    constexpr int SubtileM_2 = kBlockM / AtomLayoutM2;     // 32 M-rows per SG
    constexpr int VnniPairs_2 = SubtileM_2 / 2;            // 16 VNNI M-pairs
    constexpr uint32_t row_stride = (kBlockM / 2) * 4;     // 128 bytes per N-row
    constexpr int D32X8_BURST = 8;

    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));
    int sg_idx = local_id / SubgroupSize;
    int sg_m_2 = sg_idx % AtomLayoutM2;

    // partition_A first coord gives the lane's K value (= N-row index)
    auto [m0, k0] = tCgA_tile(0);

    // Address: N-row = k0, M-pair block = sg_m_2 * VnniPairs_2
    uint32_t a_addr = slm_offset
                    + uint32_t(k0) * row_stride
                    + sg_m_2 * VnniPairs_2 * 4;

    static_assert(VnniPairs_2 % D32X8_BURST == 0);
    constexpr int TN_BURSTS = VnniPairs_2 / D32X8_BURST;   // 2

    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < TN_BURSTS; ++g) {
        // d32x8 loads 8 VNNI M-pairs directly into tCrA — no unpack needed.
        // Each d32 = {fp16_m_even, fp16_m_odd} which is exactly how tCrA
        // stores consecutive fp16 elements (2 per uint32 slot).
        auto &dst = *recast_ptr<intel::uint8>(&tCrA(g * 16));
        uint32_t ag = a_addr + g * D32X8_BURST * 4;

        asm volatile(
            "lsc_load.slm (M1, 16) %0:d32x8 flat[%1]:a32"
            : "=rw"(dst)
            : "rw"(ag)
        );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Inline ASM SLM load requires Xe device");
#endif
}

// ============================================================
// gemm_slm_tn_kernel: A from SLM (TN read), B from global
// ============================================================
template<bool clear_acc, class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class Engine3, class Layout3, class TVLayout3,
         class TiledMMA>
void
gemm_slm_tn_kernel(Trait &trait,
                   Tensor<Engine0, Layout0> const& A,
                   uint32_t slm_offset,
                   Tensor<Engine1, Layout1> &sA,
                   Tensor<Engine2, Layout2> const& B,
                   SubgroupTensor<Engine3, Layout3, TVLayout3> &acc,
                   TiledMMA const &mma) {
    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0,2>(tile_mnk), make_coord(0,_));
    Tensor gB = local_tile(cB, select<1,2>(tile_mnk), make_coord(0,_));

    auto copy_b = make_block_2d_copy_B(mma, B);

    auto thr_mma    = mma.get_slice(local_id);
    auto thr_copy_b = copy_b.get_slice(local_id);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tCgA = thr_mma.partition_A(gA);

    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_prefetch_B = prefetch_b.get_slice(local_id);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    const int prefetch_dist = 3;
    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_prefetch = 0;

    if constexpr(clear_acc) clear(acc);

    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        barrier_arrive(barrier_scope);

        load_slm_tn(trait, slm_offset, sA, tCgA(_,_,_,k_tile), tCrA);

        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
        reorder(tBrB, tCrB);
        gemm(mma, tCrA, tCrB, acc);

        barrier_wait(barrier_scope);
    }
}

// ============================================================
// mha_reorder_copy: convert f32 acc -> f16, write to global
// ============================================================
template<class Trait, class TiledMma,
         class Engine0, class Layout0, class TVLayout0,
         class Engine1, class Layout1>
void
mha_reorder_copy(Trait &trait, TiledMma &tiled_mma,
                 SubgroupTensor<Engine0, Layout0, TVLayout0> &r,
                 Tensor<Engine1, Layout1> &m) {
    auto r16 = create_reg<typename Trait::DType>(trait, m, tiled_mma);
    reorder(r, r16);

    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));
    auto copy_c = make_block_2d_copy_D(tiled_mma, m);
    auto thr_copy_c = copy_c.get_slice(local_id);
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor cC = make_identity_tensor(m.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    Tensor tCgC = thr_copy_c.partition_D(gC);
    copy(copy_c, r16, tCgC);
}

// ============================================================
// Device kernel: workgroup = m_block, inner loop over n_blocks
//   GEMM1 (NT) -> SLM -> GEMM2 (TN), accumulate rdC across n_blocks
// ============================================================
template<class Trait>
void
slm_gemm2_device(Trait trait,
                 typename Trait::DType const *A_ptr,   // [N_total, D] row-major
                 typename Trait::DType const *B1_ptr,  // [M_total, D] row-major
                 typename Trait::DType const *B2_ptr,  // [N_total, D] row-major
                 typename Trait::DType *C_ptr,         // [M_total, D] row-major
                 typename Trait::DType *scratch_ptr,
                 int M_total, int N_total) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kBlockM  = Trait::kBlockM;
    constexpr int kBlockN  = Trait::kBlockN;
    constexpr int kHeadDim = Trait::kHeadDim;

    // Workgroup = m_block
    const int m_block = BlockIdxX();
    auto group = compat::get_nd_item<1>().get_group();
    auto smem  = compat::local_mem<T[kBlockN * kBlockM]>();
    uint32_t slm_offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem));

    // B1: [kBlockM, kHeadDim] row-major — fixed for this workgroup
    Tensor mB1 = make_tensor(make_gmem_ptr(B1_ptr + m_block * kBlockM * kHeadDim),
                             make_layout(
                                 make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                                 make_stride(Int<kHeadDim>{}, _1{})));

    // Scratch proxy for GEMM1 output: [kBlockN, kBlockM] row-major
    Tensor mScratch = make_tensor(make_gmem_ptr(scratch_ptr),
                                  make_layout(
                                      make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                                      make_stride(Int<kBlockM>{}, _1{})));

    // SLM: S[kBlockN, kBlockM] non-transposed, kBlockM contiguous
    Tensor sC1 = make_tensor(make_smem_ptr(smem),
                             make_layout(
                                 make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                                 make_stride(Int<kBlockM>{}, _1{})));

    // GEMM2 A proxy: S^T[kBlockM, kBlockN] col-major (stride 1, kBlockM)
    // proxy(m,k) = S(k,m). Shape [kBlockM, kBlockN] with stride (1, kBlockM).
    Tensor mScratchT = make_tensor(make_gmem_ptr(scratch_ptr),
                                   make_layout(
                                       make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                       make_stride(_1{}, Int<kBlockM>{})));

    // SLM for GEMM2 A: same memory, transposed view
    Tensor sC1T = make_tensor(make_smem_ptr(smem),
                              make_layout(
                                  make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                  make_stride(_1{}, Int<kBlockM>{})));

    // Output C: [kBlockM, kHeadDim] row-major — for this workgroup
    Tensor mC = make_tensor(make_gmem_ptr(C_ptr + m_block * kBlockM * kHeadDim),
                            make_layout(
                                make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                                make_stride(Int<kHeadDim>{}, _1{})));

    typename Trait::TiledMma1 tiled_mma1;
    typename Trait::TiledMma2 tiled_mma2;

    // Accumulator for GEMM2 — persists across n_blocks
    auto rdC = create_reg<V>(trait, mC, tiled_mma2);
    clear(rdC);

    const int max_n_block = N_total / kBlockN;

    // A and B2 advance per n_block
    Tensor mA = make_tensor(make_gmem_ptr(A_ptr),
                            make_layout(
                                make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
                                make_stride(Int<kHeadDim>{}, _1{})));
    // B2 as B2t[D, kBlockN] col-major for MMA TT B operand
    Tensor mB2t = make_tensor(make_gmem_ptr(B2_ptr),
                              make_layout(
                                  make_shape(Int<kHeadDim>{}, Int<kBlockN>{}),
                                  make_stride(_1{}, Int<kHeadDim>{})));

    for (int n_block = 0; n_block < max_n_block; ++n_block) {
        // GEMM1: rS[kBlockN, kBlockM] = A_n[kBlockN, D] * B1[kBlockM, D]
        {
            auto rS = create_reg<V>(trait, mScratch, tiled_mma1);
            gemm_kernel<true>(trait, mA, mB1, rS, tiled_mma1);
            slm_reorder_save(trait, mScratch, slm_offset, rS, tiled_mma1);
        }

        sycl::group_barrier(group);

        // GEMM2: rdC[kBlockM, D] += S^T[kBlockM, kBlockN] * B2t[D, kBlockN]
        gemm_slm_tn_kernel<false>(trait, mScratchT, slm_offset, sC1T, mB2t, rdC, tiled_mma2);

        sycl::group_barrier(group);

        // Advance A and B2 to next n_block
        mA.data()   = mA.data()   + int(kBlockN * kHeadDim);
        mB2t.data() = mB2t.data() + int(kBlockN * kHeadDim);
    }

    // Write accumulated result to global memory
    mha_reorder_copy(trait, tiled_mma2, rdC, mC);
}

// ============================================================
// Kernel name
// ============================================================
template<class...> class slmGemm2KernelName;

// ============================================================
// Launch function
// ============================================================
template<class T, int kBlockM, int kBlockN, int kHeadDim, int kNSGs,
         int AtomLayoutM1, int AtomLayoutM2>
void
launch_slm_gemm2(const T *A_d, const T *B1_d, const T *B2_d,
                 T *C_d, T *scratch_d, int M_total, int N_total) {
    auto trait = SlmGemmTrait2<T, kBlockM, kBlockN, kHeadDim, kNSGs,
                               AtomLayoutM1, AtomLayoutM2>{};

    int num_m_blocks = M_total / kBlockM;
    auto dimGrid  = compat::dim3(num_m_blocks, 1, 1);
    auto dimBlock = compat::dim3(kNSGs * trait.SubgroupSize, 1, 1);

    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(trait.smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy{
        dimGrid, dimBlock, launch_props, kernel_props};

    auto event = compat::experimental::launch<
        slm_gemm2_device<decltype(trait)>,
        slmGemm2KernelName<decltype(trait)>>(
        policy, trait, A_d, B1_d, B2_d, C_d, scratch_d, M_total, N_total);

    EventManager::getInstance().addEvent(event);
    compat::wait_and_throw();
}

// ============================================================
// CPU reference — mirrors the GPU kernel structure:
//   C[M_total, D] = B1[M_total, D'] * A[N_total, D']^T * B2[N_total, D]
//   For each m_block (workgroup):
//     B1_blk = &B1[m_block * kBlockM, 0]  shape [kBlockM, D]
//     C_blk  = &C [m_block * kBlockM, 0]  shape [kBlockM, D]
//     for each n_block:
//       A_blk  = &A [n_block * kBlockN, 0]  shape [kBlockN, D]
//       B2_blk = &B2[n_block * kBlockN, 0]  shape [kBlockN, D]
//       GEMM1: S[kBlockN, kBlockM] = A_blk * B1_blk^T   (f32)
//       S_f16 = half(S)
//       GEMM2: C_blk += S_f16^T * B2_blk                (f32)
//     C_blk = half(C_blk)
// ============================================================
template<class T>
void
cpu_reference(const T *A, const T *B1, const T *B2, T *C_ref,
              int M_total, int N_total, int D, int kBlockM, int kBlockN) {
    const int num_m_blocks = M_total / kBlockM;
    const int num_n_blocks = N_total / kBlockN;
    const int A_r_stride   = D;
    const int B1_r_stride  = D;
    const int B2_r_stride  = D;
    const int C_r_stride   = D;
    const int S_r_stride   = kBlockM;  // S[kBlockN, kBlockM] row-major

    for (int mb = 0; mb < num_m_blocks; ++mb) {
        const T *B1_blk = B1  + mb * kBlockM * B1_r_stride;
        T       *C_blk  = C_ref + mb * kBlockM * C_r_stride;

        std::vector<float> C_f32(kBlockM * D, 0.0f);

        const T *A_blk  = A;
        const T *B2_blk = B2;

        for (int nb = 0; nb < num_n_blocks; ++nb) {
            // GEMM1: S[kBlockN, kBlockM] = A_blk[kBlockN, D] * B1_blk[kBlockM, D]^T
            std::vector<float> S_f32(kBlockN * kBlockM, 0.0f);
            for (int n = 0; n < kBlockN; ++n) {
                for (int m = 0; m < kBlockM; ++m) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; ++d)
                        sum += float(A_blk[n * A_r_stride + d]) *
                               float(B1_blk[m * B1_r_stride + d]);
                    S_f32[n * S_r_stride + m] = sum;
                }
            }

            // Truncate S to f16
            std::vector<T> S_f16(kBlockN * kBlockM);
            for (int i = 0; i < kBlockN * kBlockM; ++i)
                S_f16[i] = T(S_f32[i]);

            // GEMM2: C_f32[kBlockM, D] += S_f16^T[kBlockM, kBlockN] * B2_blk[kBlockN, D]
            for (int m = 0; m < kBlockM; ++m) {
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int n = 0; n < kBlockN; ++n)
                        sum += float(S_f16[n * S_r_stride + m]) *
                               float(B2_blk[n * B2_r_stride + d]);
                    C_f32[m * D + d] += sum;
                }
            }

            // Advance A, B2 to next n_block
            A_blk  += kBlockN * A_r_stride;
            B2_blk += kBlockN * B2_r_stride;
        }

        for (int i = 0; i < kBlockM * D; ++i)
            C_blk[i] = T(C_f32[i]);
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv) {
    using T = cute::half_t;

    constexpr int kBlockM  = 64;
    constexpr int kBlockN  = 64;
    constexpr int kHeadDim = 64;
    constexpr int kNSGs    = 8;
    constexpr int AtomLayoutM1 = 4;
    constexpr int AtomLayoutM2 = 2;

    int M_total = 128;
    int N_total = 256;
    if (argc >= 2) sscanf(argv[1], "%d", &M_total);
    if (argc >= 3) sscanf(argv[2], "%d", &N_total);

    M_total = (M_total / kBlockM) * kBlockM;
    N_total = (N_total / kBlockN) * kBlockN;
    if (M_total == 0) M_total = kBlockM;
    if (N_total == 0) N_total = kBlockN;

    const int D = kHeadDim;

    printf("SLM GEMM Example 2 (TN GEMM2)\n");
    printf("  M_total=%d  N_total=%d  D=%d  kBlockM=%d  kBlockN=%d\n",
           M_total, N_total, D, kBlockM, kBlockN);
    printf("  num_m_blocks=%d (workgroups)  num_n_blocks=%d (inner loop)\n",
           M_total / kBlockM, N_total / kBlockN);
    printf("  C[%d,%d] = B1[%d,%d] * A[%d,%d]^T * B2[%d,%d]\n",
           M_total, D, M_total, D, N_total, D, N_total, D);

    // Host allocations
    std::vector<T> h_A(N_total * D);
    std::vector<T> h_B1(M_total * D);
    std::vector<T> h_B2(N_total * D);
    std::vector<T> h_C(M_total * D);
    std::vector<T> h_C_ref(M_total * D);

    srand(42);
    for (auto &v : h_A)  v = T(float(rand() % 21 - 10) / 10.0f);
    for (auto &v : h_B1) v = T(float(rand() % 21 - 10) / 10.0f);
    for (auto &v : h_B2) v = T(float(rand() % 21 - 10) / 10.0f);

    cpu_reference(h_A.data(), h_B1.data(), h_B2.data(), h_C_ref.data(),
                  M_total, N_total, D, kBlockM, kBlockN);

    auto d_A       = compat::malloc<T>(N_total * D);
    auto d_B1      = compat::malloc<T>(M_total * D);
    auto d_B2      = compat::malloc<T>(N_total * D);
    auto d_C       = compat::malloc<T>(M_total * D);
    auto d_scratch = compat::malloc<T>(kBlockN * kBlockM);

    compat::memcpy<T>(d_A,  h_A.data(),  N_total * D);
    compat::memcpy<T>(d_B1, h_B1.data(), M_total * D);
    compat::memcpy<T>(d_B2, h_B2.data(), N_total * D);
    compat::wait_and_throw();

    launch_slm_gemm2<T, kBlockM, kBlockN, kHeadDim, kNSGs,
                     AtomLayoutM1, AtomLayoutM2>(
        d_A, d_B1, d_B2, d_C, d_scratch, M_total, N_total);

    compat::memcpy<T>(h_C.data(), d_C, M_total * D);
    compat::wait_and_throw();

    float atol = 3e-3f;
    float rtol = 3e-3f;
    printf("Result: ");
    verify(h_C_ref.data(), h_C.data(), M_total / kBlockM, kBlockM, D, atol, rtol);

    return 0;
}
