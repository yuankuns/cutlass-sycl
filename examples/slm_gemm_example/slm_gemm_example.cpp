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
 * SLM GEMM Example: Two-stage fused GEMM through Shared Local Memory (SLM).
 *
 * Demonstrates the pattern from SDPA backward:
 *   GEMM1 (NT): S[N,BM] = B1[N,D] * A_blk[BM,D]   (both from global, block_2d_copy)
 *   slm_reorder_save: S accumulator f32 -> f16, save to SLM
 *   GEMM2 (SLM+global): C[N,D] += SLM[N,BM] * B2t[D,BM]  (A from SLM via partition_A)
 *   Loop over m_blocks, accumulating C across iterations.
 *
 * Computes: C[N,D] = B1[N,D] * A[M,D]^T * B2[M,D]  (with intermediate f16 truncation)
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
// Trait struct (FAKernel-like)
// ============================================================
template <class T_, int kBlockM_, int kBlockN_, int kHeadDim_, int kNSGs_,
          int AtomLayoutM1_ = 4, int AtomLayoutM2_ = 2>
struct SlmGemmTrait {
    using DType = T_;
    using VType = float;
    static constexpr int kBlockM  = kBlockM_;
    static constexpr int kBlockN  = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kNSGs    = kNSGs_;
    static constexpr int SubgroupSize = 16;

    using MMA_Atom_ARCH = XE_DPAS_TT<8, VType, DType>;
    using _K = Int<MMA_Atom_ARCH::K>;

    // GEMM1: S[kBlockN, kBlockM] = B1[kBlockN, D] * A_blk[kBlockM, D]
    using SubgroupLayout1 = Layout<Shape<Int<AtomLayoutM1_>, Int<kNSGs / AtomLayoutM1_>, _1>>;
    using TileShape1      = Layout<Shape<Int<kBlockN>, Int<kBlockM>, _K>>;
    using TiledMma1 = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                              TileShape1,
                                              SubgroupLayout1>::TiledMMA;

    // GEMM2: C[kBlockN, kHeadDim] += SLM[kBlockN, kBlockM] * B2t[kHeadDim, kBlockM]
    using SubgroupLayout2 = Layout<Shape<Int<AtomLayoutM2_>, Int<kNSGs / AtomLayoutM2_>, _1>>;
    using TileShape2      = Layout<Shape<Int<kBlockN>, Int<kHeadDim>, _K>>;
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
// slm_reorder_save: reorder f32 acc -> f16, save to SLM
//   partition_D with local_id gives per-subgroup base coords;
//   add sg lane id explicitly for per-lane SLM addressing.
// ============================================================
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
slm_reorder_save(Trait &trait,
                 Tensor<Engine0, Layout0> &mC,
                 Tensor<Engine1, Layout1> &sC,
                 SubgroupTensor<Engine2, Layout2, TVLayout2> &r,
                 TiledMMA const &tiled_mma) {
    auto r16 = create_reg<typename Trait::DType>(trait, mC, tiled_mma);
    reorder(r, r16);

    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));
    int sg_lane = compat::get_nd_item<1>().get_sub_group().get_local_id();

    auto copy_c = make_block_2d_copy_D(tiled_mma, mC);
    auto thr_copy_c = copy_c.get_slice(local_id);
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor cC = make_identity_tensor(mC.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    Tensor tCgC = thr_copy_c.partition_D(gC);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCgC); ++i) {
        auto [m, n] = tCgC(i);
        sC(m, n + sg_lane) = r16(i);
    }
}

// ============================================================
// load_slm: load one k-tile of A from SLM into MMA fragment
//   partition_A with local_id is lane-aware — coords include
//   per-lane offset, so sA(m, k) addresses correctly.
// ============================================================
template<class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2>
CUTLASS_DEVICE void
load_slm(Tensor<Engine0, Layout0> const& sA,
         Tensor<Engine1, Layout1> const& tCgA_tile,
         SubgroupTensor<Engine2, Layout2, TVLayout2> &tCrA) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCgA_tile); ++i) {
        auto [m, k] = tCgA_tile(i);
        tCrA(i) = sA(m, k);
    }
}

// ============================================================
// gemm_slm_kernel: A from SLM (via load_slm), B from global
// ============================================================
template<bool clear_acc, class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class Engine3, class Layout3, class TVLayout3,
         class TiledMMA>
void
gemm_slm_kernel(Trait &trait,
                Tensor<Engine0, Layout0> const& A,
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

        load_slm(sA, tCgA(_,_,_,k_tile), tCrA);

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
// Device kernel: for each n_block (workgroup), loop m_blocks
//   GEMM1 -> SLM -> GEMM2, accumulate across m_blocks
//   Mirrors sdpa_backward.hpp dq_dk_dv_1colblock pattern.
// ============================================================
template<class Trait>
void
slm_gemm_device(Trait trait,
                typename Trait::DType const *A_ptr,   // [N_total, D] row-major (K-like)
                typename Trait::DType const *B1_ptr,  // [M_total, D] row-major (Q-like)
                typename Trait::DType const *B2_ptr,  // [M_total, D] row-major (dO-like)
                typename Trait::DType *C_ptr,         // [N_total, D] row-major (dV-like)
                typename Trait::DType *scratch_ptr,
                int M_total, int N_total) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kBlockM  = Trait::kBlockM;
    constexpr int kBlockN  = Trait::kBlockN;
    constexpr int kHeadDim = Trait::kHeadDim;

    // n_block = workgroup index (like BlockIdxX in sdpa_backward.hpp)
    const int n_block = BlockIdxX();
    auto group = compat::get_nd_item<1>().get_group();
    auto smem  = compat::local_mem<T[kBlockN * kBlockM]>();

    // A (K-like): [kBlockN, kHeadDim] row-major — fixed for this n_block
    Tensor mA = make_tensor(make_gmem_ptr(A_ptr + n_block * kBlockN * kHeadDim),
                            make_layout(
                                make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
                                make_stride(Int<kHeadDim>{}, _1{})));

    // Scratch proxy: same shape/stride as SLM
    Tensor mScratch = make_tensor(make_gmem_ptr(scratch_ptr),
                                  make_layout(
                                      make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                                      make_stride(Int<kBlockM>{}, _1{})));

    // SLM tensor: [kBlockN, kBlockM]
    Tensor sC1 = make_tensor(make_smem_ptr(smem),
                             make_layout(
                                 make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                                 make_stride(Int<kBlockM>{}, _1{})));

    // Output C (dV-like): [kBlockN, kHeadDim] row-major — for this n_block
    Tensor mC = make_tensor(make_gmem_ptr(C_ptr + n_block * kBlockN * kHeadDim),
                            make_layout(
                                make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
                                make_stride(Int<kHeadDim>{}, _1{})));

    typename Trait::TiledMma1 tiled_mma1;
    typename Trait::TiledMma2 tiled_mma2;

    // Accumulator for GEMM2 — persists across m_blocks
    auto rdC = create_reg<V>(trait, mC, tiled_mma2);
    clear(rdC);

    const int max_m_block = M_total / kBlockM;

    // B1 (Q-like) and B2 (dO-like): advance per m_block
    Tensor mB1 = make_tensor(make_gmem_ptr(B1_ptr),
                             make_layout(
                                 make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                                 make_stride(Int<kHeadDim>{}, _1{})));
    Tensor mB2t = make_tensor(make_gmem_ptr(B2_ptr),
                              make_layout(
                                  make_shape(Int<kHeadDim>{}, Int<kBlockM>{}),
                                  make_stride(_1{}, Int<kHeadDim>{})));

    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        // GEMM1: rS[kBlockN, kBlockM] = A[kBlockN, D] * B1_blk[kBlockM, D]
        {
            auto rS = create_reg<V>(trait, mScratch, tiled_mma1);
            gemm_kernel<true>(trait, mA, mB1, rS, tiled_mma1);
            slm_reorder_save(trait, mScratch, sC1, rS, tiled_mma1);
        }

        sycl::group_barrier(group);

        // GEMM2: rdC[kBlockN, D] += SLM[kBlockN, kBlockM] * B2t[D, kBlockM]
        gemm_slm_kernel<false>(trait, mScratch, sC1, mB2t, rdC, tiled_mma2);

        sycl::group_barrier(group);

        // Advance B1 and B2t pointers to next m_block
        mB1.data()  = mB1.data()  + int(kBlockM * kHeadDim);
        mB2t.data() = mB2t.data() + int(kBlockM * kHeadDim);
    }

    // Write accumulated result to global memory
    mha_reorder_copy(trait, tiled_mma2, rdC, mC);
}

// ============================================================
// Kernel name
// ============================================================
template<class...> class slmGemmKernelName;

// ============================================================
// Launch function
// ============================================================
template<class T, int kBlockM, int kBlockN, int kHeadDim, int kNSGs,
         int AtomLayoutM1, int AtomLayoutM2>
void
launch_slm_gemm(const T *A_d, const T *B1_d, const T *B2_d,
                T *C_d, T *scratch_d, int M_total, int N_total) {
    auto trait = SlmGemmTrait<T, kBlockM, kBlockN, kHeadDim, kNSGs,
                              AtomLayoutM1, AtomLayoutM2>{};

    int num_n_blocks = N_total / kBlockN;
    auto dimGrid  = compat::dim3(num_n_blocks, 1, 1);
    auto dimBlock = compat::dim3(kNSGs * trait.SubgroupSize, 1, 1);

    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(trait.smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy{
        dimGrid, dimBlock, launch_props, kernel_props};

    auto event = compat::experimental::launch<
        slm_gemm_device<decltype(trait)>,
        slmGemmKernelName<decltype(trait)>>(
        policy, trait, A_d, B1_d, B2_d, C_d, scratch_d, M_total, N_total);

    EventManager::getInstance().addEvent(event);
    compat::wait_and_throw();
}

// ============================================================
// CPU reference — mirrors the GPU kernel structure:
//   All matrices row-major with row stride = D (or kBlockM for S).
//   for each n_block:
//     A_blk  = &A [n_block * kBlockN, 0]    shape [kBlockN, D]
//     C_blk  = &C [n_block * kBlockN, 0]    shape [kBlockN, D]
//     for each m_block:
//       B1_blk = &B1[m_block * kBlockM, 0]  shape [kBlockM, D]
//       B2_blk = &B2[m_block * kBlockM, 0]  shape [kBlockM, D]
//       GEMM1: S [kBlockN, kBlockM] = A_blk * B1_blk^T   (f32 acc)
//       S_f16 = half(S)                                    (slm truncation)
//       GEMM2: C_blk += S_f16 * B2_blk                    (f32 acc)
//     C_blk = half(C_blk)
// ============================================================
template<class T>
void
cpu_reference(const T *A, const T *B1, const T *B2, T *C_ref,
              int M_total, int N_total, int D, int kBlockM, int kBlockN) {
    const int num_n_blocks = N_total / kBlockN;
    const int num_m_blocks = M_total / kBlockM;
    const int A_r_stride  = D;        // row stride for A  [N_total, D]
    const int B1_r_stride = D;        // row stride for B1 [M_total, D]
    const int B2_r_stride = D;        // row stride for B2 [M_total, D]
    const int C_r_stride  = D;        // row stride for C  [N_total, D]
    const int S_r_stride  = kBlockM;  // row stride for S  [kBlockN, kBlockM]

    for (int nb = 0; nb < num_n_blocks; ++nb) {
        const T *A_blk = A    + nb * kBlockN * A_r_stride;
        T       *C_blk = C_ref + nb * kBlockN * C_r_stride;

        std::vector<float> C_f32(kBlockN * D, 0.0f);

        const T *B1_blk = B1;
        const T *B2_blk = B2;

        for (int mb = 0; mb < num_m_blocks; ++mb) {
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

            // Truncate S to f16 (matches kernel slm_reorder_save)
            std::vector<T> S_f16(kBlockN * kBlockM);
            for (int i = 0; i < kBlockN * kBlockM; ++i)
                S_f16[i] = T(S_f32[i]);

            // GEMM2: C_f32[kBlockN, D] += S_f16[kBlockN, kBlockM] * B2_blk[kBlockM, D]
            for (int n = 0; n < kBlockN; ++n) {
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int m = 0; m < kBlockM; ++m)
                        sum += float(S_f16[n * S_r_stride + m]) *
                               float(B2_blk[m * B2_r_stride + d]);
                    C_f32[n * D + d] += sum;
                }
            }

            // Advance B1, B2 to next m_block
            B1_blk += kBlockM * B1_r_stride;
            B2_blk += kBlockM * B2_r_stride;
        }

        // Convert accumulated f32 result to f16
        for (int i = 0; i < kBlockN * D; ++i)
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

    int M_total = 256;
    int N_total = 128;
    if (argc >= 2) sscanf(argv[1], "%d", &M_total);
    if (argc >= 3) sscanf(argv[2], "%d", &N_total);

    // Round to block sizes
    M_total = (M_total / kBlockM) * kBlockM;
    N_total = (N_total / kBlockN) * kBlockN;
    if (M_total == 0) M_total = kBlockM;
    if (N_total == 0) N_total = kBlockN;

    const int D = kHeadDim;

    printf("SLM GEMM Example\n");
    printf("  M_total=%d  N_total=%d  D=%d  kBlockM=%d  kBlockN=%d\n",
           M_total, N_total, D, kBlockM, kBlockN);
    printf("  num_n_blocks=%d (workgroups)  num_m_blocks=%d (inner loop)\n",
           N_total / kBlockN, M_total / kBlockM);
    printf("  C[%d,%d] = A[%d,%d] * B1[%d,%d]^T * B2[%d,%d]\n",
           N_total, D, N_total, D, M_total, D, M_total, D);

    // Host allocations
    // A (K-like): [N_total, D], B1 (Q-like): [M_total, D], B2 (dO-like): [M_total, D]
    std::vector<T> h_A(N_total * D);
    std::vector<T> h_B1(M_total * D);
    std::vector<T> h_B2(M_total * D);
    std::vector<T> h_C(N_total * D);
    std::vector<T> h_C_ref(N_total * D);

    // Random init
    srand(42);
    for (auto &v : h_A)  v = T(float(rand() % 21 - 10) / 10.0f);
    for (auto &v : h_B1) v = T(float(rand() % 21 - 10) / 10.0f);
    for (auto &v : h_B2) v = T(float(rand() % 21 - 10) / 10.0f);

    // CPU reference
    cpu_reference(h_A.data(), h_B1.data(), h_B2.data(), h_C_ref.data(),
                  M_total, N_total, D, kBlockM, kBlockN);

    // Device allocations
    auto d_A       = compat::malloc<T>(N_total * D);
    auto d_B1      = compat::malloc<T>(M_total * D);
    auto d_B2      = compat::malloc<T>(M_total * D);
    auto d_C       = compat::malloc<T>(N_total * D);
    auto d_scratch = compat::malloc<T>(kBlockN * kBlockM);

    // Copy to device
    compat::memcpy<T>(d_A,  h_A.data(),  N_total * D);
    compat::memcpy<T>(d_B1, h_B1.data(), M_total * D);
    compat::memcpy<T>(d_B2, h_B2.data(), M_total * D);
    compat::wait_and_throw();

    // Launch: one workgroup per n_block
    launch_slm_gemm<T, kBlockM, kBlockN, kHeadDim, kNSGs,
                    AtomLayoutM1, AtomLayoutM2>(
        d_A, d_B1, d_B2, d_C, d_scratch, M_total, N_total);

    // Copy back
    compat::memcpy<T>(h_C.data(), d_C, N_total * D);
    compat::wait_and_throw();

    // Verify
    float atol = 3e-3f;
    float rtol = 3e-3f;
    printf("Result: ");
    verify(h_C_ref.data(), h_C.data(), N_total / kBlockN, kBlockN, D, atol, rtol);

    return 0;
}
