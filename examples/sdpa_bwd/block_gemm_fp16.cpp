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

#include <random>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/reference/device/tensor_compare.h"

using namespace cute;

template<typename T>
void
random_init(int seed, T *dst, size_t N, int a = -1, int b = 1) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(a, b);
    for (int i = 0; i < N; ++i) {
        dst[i] = static_cast<T>(dis(gen));
    }
}

template<class T1, class T2>
void print_t(T1 m, T2 g) {
    print(m);
    for (int i = 0; i < size(g); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10.7f ", (float)m(g(i)));
    }
    print("\n");
}

bool isclose(float a, float b, float atol, float rtol) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

template<typename T, typename V>
bool allclose(T *refe, V *test, int L, int M, int N, float atol, float rtol) {
    size_t err = 0;
    size_t count = L * M * N;
    bool flag = true;
    for (int l = 0; l < L; ++l) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float expect = (float)refe[l * M * N + m * N + n];
                float value = (float)test[l * M * N + m * N + n];
                if (not isclose(expect, value, atol, rtol)) {
                    printf("(%d, %d, %d) expect: %f value: %f ratio %f\n", l, m, n, expect, value, value / expect);
                    err++;
                }
                if (err > 20)
                    return false;
                if (isnan(value) or isinf(value)) {
                    printf("\x1B[31m %f detected \x1B[0m at (%d, %d, %d)\n", value, l, m, n);
                    exit(1);
                }
            }
        }
    }
    float ratio = static_cast<float>(count - err) / static_cast<float>(count);
    // printf("c=%f (%ld)\n", ratio, err);
    // printf("CHECK SUM SUCCESS\n");
    return ratio > 0.99f;
}

static constexpr char strSUCCESS[] = "\x1B[32mPASS\x1B[0m";
static constexpr char strFAILURE[] = "\x1B[31mFAIL\x1B[0m";
template<typename T, typename V>
void verify(T *refe, V *test, int l, int m, int n, float atol, float rtol) {
    bool close = allclose(refe, test, l, m, n, atol, rtol);
    printf("allclose %s\n", close ? strSUCCESS : strFAILURE);
}

template <class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2>
CUTLASS_DEVICE void
manual_atomic_add(Tensor <Engine0, Layout0>& m_tile,
                  Tensor <Engine1, Layout1>& g_tile,
                  Tensor <Engine2, Layout2>& r_tile) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    const int local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int ki = 0; ki < size(g_tile); ++ki) {
        auto [m, n, l] = g_tile(ki);
        cutlass::atomicAdd(&m_tile(m, n + local_id, 0), r_tile(ki));
    }
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void verify_with_cutlass(
    TA const* d_A,
    TB const* d_B,
    TC* ref_d_C,
    int m,
    int n,
    int k,
    int l,
    Alpha alpha,
    Beta beta,
    char transA,
    char transB
    ) {
    cutlass::TensorRef ref_A_T(d_A, cutlass::layout::ColumnMajor::packed({m, k}));
    cutlass::TensorRef ref_A_N(d_A, cutlass::layout::RowMajor::packed({m, k}));

    cutlass::TensorRef ref_B_T(d_B, cutlass::layout::ColumnMajor::packed({k, n}));
    cutlass::TensorRef ref_B_N(d_B, cutlass::layout::RowMajor::packed({k, n}));

    cutlass::TensorRef ref_C(ref_d_C, cutlass::layout::RowMajor::packed({m, n}));
    cutlass::TensorRef ref_D(ref_d_C, cutlass::layout::RowMajor::packed({m, n}));

    if (transA == 'T' && transB == 'N') {
        cutlass::reference::device::GemmComplex(
            {m, n, k},
            alpha,
            ref_A_T,
            cutlass::ComplexTransform::kNone,
            ref_B_N,
            cutlass::ComplexTransform::kNone,
            beta,
            ref_C,
            ref_D,
            float(0),  // accumulator
            l,     // batch_count
            m * k, // batch_stride_A
            k * n, // batch_stride_B
            m * n, // batch_stride_C
            m * n  // batch_stride_D
            );
    } else if (transA == 'T' && transB == 'T') {
        cutlass::reference::device::GemmComplex(
            {m, n, k},
            alpha,
            ref_A_T,
            cutlass::ComplexTransform::kNone,
            ref_B_T,
            cutlass::ComplexTransform::kNone,
            beta,
            ref_C,
            ref_D,
            float(0),  // accumulator
            l,     // batch_count
            m * k, // batch_stride_A
            k * n, // batch_stride_B
            m * n, // batch_stride_C
            m * n  // batch_stride_D
            );
    } else if (transA == 'N' && transB == 'T') {
        cutlass::reference::device::GemmComplex(
            {m, n, k},
            alpha,
            ref_A_N,
            cutlass::ComplexTransform::kNone,
            ref_B_T,
            cutlass::ComplexTransform::kNone,
            beta,
            ref_C,
            ref_D,
            float(0),  // accumulator
            l,     // batch_count
            m * k, // batch_stride_A
            k * n, // batch_stride_B
            m * n, // batch_stride_C
            m * n  // batch_stride_D
            );
    } else if (transA == 'N' && transB == 'N') {
        cutlass::reference::device::GemmComplex(
            {m, n, k},
            alpha,
            ref_A_N,
            cutlass::ComplexTransform::kNone,
            ref_B_N,
            cutlass::ComplexTransform::kNone,
            beta,
            ref_C,
            ref_D,
            float(0),  // accumulator
            l,     // batch_count
            m * k, // batch_stride_A
            k * n, // batch_stride_B
            m * n, // batch_stride_C
            m * n  // batch_stride_D
            );
    } else {
        assert(false && "Not implemented");
    }


    // CUTLASS on SYCL uses the compatibility library compat for e.g. default in-order queue
    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    // bool passed = cutlass::reference::device::BlockCompareEqual(
    //     ref_d_C, d_C, m * n);

    // return passed;
}


template <class ProblemShape, class CtaTiler,
          class TA, class TiledCopyA,
          class TB, class TiledCopyB,
          class TC, class TiledCopyC,
          class TiledMma, bool DirectCopy, bool AtomicAdd>
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, int stages,
            TA const* A, TiledCopyA,
            TB const* B, TiledCopyB,
            TC      * C, TiledCopyC, TiledMma mma)
{
    auto [M, N, K, L] = shape_MNK;
    auto [bM, bN, bK] = cta_tiler;
    auto A_shape = make_shape(bM, K, 1);
    auto B_shape = make_shape(bN, K, 1);
    auto C_shape = make_shape(bM, bN, 1);

    auto dA = make_stride(K, _1{}, _0{});
    auto dB = make_stride(K, _1{}, _0{});
    auto dC = make_stride(N, _1{}, _0{});

    // Get the appropriate blocks for this thread block
    int m_coord = BlockIdxX();
    int n_coord = BlockIdxY();
    int l_coord = BlockIdxZ();

    int A_offset = m_coord * bM * K + l_coord * M * K;
    int B_offset = n_coord * bN * K + l_coord * N * K;
    int C_offset = m_coord * bM * N + n_coord * bN + l_coord * M * N;

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A + A_offset), make_layout(A_shape, dA));
    auto mB = make_tensor(make_gmem_ptr(B + B_offset), make_layout(B_shape, dB));
    auto mC = make_tensor(make_gmem_ptr(C + C_offset), make_layout(C_shape, dC));

    auto copy_a = TiledCopyA{mA};
    auto copy_b = TiledCopyB{mB};
    auto copy_c = TiledCopyC{mC};

    Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
    Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,l)
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)

    Tensor gA = local_tile(mA_coord, select<0, 2>(cta_tiler), make_coord(0, _, 0));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_coord, select<1, 2>(cta_tiler), make_coord(0, _, 0));  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC_coord, select<0, 1>(cta_tiler), make_coord(0, 0, 0));  // (BLK_M,BLK_N)

    //
    // Define A/B partitioning and C accumulators
    //

    TiledMma tiled_mma;
    constexpr int sg_size = 16;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * sg_size;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    // Partition global counting tensors for MMA
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = make_tensor<TA>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<TB>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

    ThrCopy thr_copy_a = copy_a.get_slice(compat::local_id::x());
    ThrCopy thr_copy_b = copy_b.get_slice(compat::local_id::x());

    // Retile registers for copies
    Tensor tArA = thr_copy_a.retile_D(tCrA);
    Tensor tBrB = thr_copy_b.retile_D(tCrB);

    // Retile global counting tensors for copies
    Tensor tAgA = thr_copy_a.retile_S(tCgA);
    Tensor tBgB = thr_copy_b.retile_S(tCgB);

    //
    // PREFETCH
    //

    // constexpr int Num_SGs = size(tiled_mma);
    static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

    static constexpr auto BLK_M = get<0>(CtaTiler{});
    static constexpr auto BLK_N = get<1>(CtaTiler{});
    static constexpr auto BLK_K = get<2>(CtaTiler{});

    auto prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(copy_a);
    auto prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(copy_b);
    int thread_idx = int(ThreadIdxX());
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

    // Partition global tile for prefetch
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    int prefetch_k = 0;

    // Clear the accumulators
    Tensor tCrC = partition_fragment_C(tiled_mma, take<0,2>(cta_tiler));
    clear(tCrC);

    constexpr int barrier_scope = 2;
    int k_tile_count = ceil_div(get<2>(shape_MNK), get<2>(cta_tiler));

    // CUTLASS_PRAGMA_UNROLL
    // for (; prefetch_k < stages; prefetch_k++) {
    //     prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
    //     prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    // }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
        barrier_arrive(barrier_scope);
        // Copy gmem to rmem for the first k_tile
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        // if (prefetch_k < k_tile_count) {
        //     prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        //     prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        // }

        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);

    }
    //
    // Epilogue
    //
    if constexpr(DirectCopy) {
        copy(copy_c, tCrC, tCgC);
    } else if constexpr(AtomicAdd) {
        auto thr_layout_atom = Layout<Shape<_4, _128>>{};
        auto val_layout_atom = Layout<Shape<_16, _1>>{};
        auto tiled_atom = make_tiled_copy(
            Copy_Atom<XE_ATOMIC<TC>, TC>{},
            thr_layout_atom,
            val_layout_atom);
        auto thr_atom_add = tiled_atom.get_thread_slice(ThreadIdxX());
        Tensor thr_tile_atom_D = thr_atom_add.partition_D(mC(_, _, 0));
        auto tensor_frag = thr_atom_add.retile_S(tCrC);
        copy(tiled_atom, tensor_frag, thr_tile_atom_D);
    } else {
        manual_atomic_add(mC, tCgC, tCrC);
    }

}

// Setup params for a NT GEMM
template <bool DirectCopy, bool AtomicAdd, class TA, class TB, class TC>
void
gemm_nt(int m, int n, int k, int l,
        TA const* A, int ldA,
        TB const* B, int ldB,
        TC      * C, int ldC)
{
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto L = int(l);
    auto prob_shape = make_shape(M, N, K, L);                     // (M, N, K, L)

    // Define CTA tile sizes (static)
    auto bM = Int<256>{};
    auto bN = Int<256>{};
    auto bK = Int< 32>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<2>{};  // Pipeline
    using StrideR = Stride<int, _1, _0>;
    // Define the thread layouts (static)
    TiledCopy copyA = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, StrideR>, TA>{},
                                      Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                      Layout<Shape<_32,_2>>{});              // Val layout  32x2

    TiledCopy copyB = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideR>, TB>{},
                                      Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                      Layout<Shape<_16,_1>>{});              // Val layout  16x1
    // TiledCopy copyC = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N, decltype(dC)>, TC>{},
    TiledCopy copyC = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N, StrideR>, TC>{},
                                      Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                      Layout<Shape<_8,_1>>{});              // Val layout  8x1

    TiledMMA mmaC = TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<decltype(cta_tiler)>,
                                   Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA{};

    auto dimBlock = compat::dim3(size(mmaC));
    auto dimGrid  = compat::dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)), size(L));

    constexpr int SubgroupSize = 16;
    constexpr int smem_size = 0;
    auto kernel_props = [] {
        return compat::experimental::kernel_properties{
            sycl::ext::oneapi::experimental::sub_group_size<SubgroupSize>
        };
    }();
    compat::experimental::launch_properties launch_props {
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::launch_policy policy{
        dimGrid, dimBlock, launch_props, kernel_props
    };
    auto event = compat::experimental::launch<
        gemm_device<decltype(prob_shape), decltype(cta_tiler),
                    TA, decltype(copyA),
                    TB, decltype(copyB),
                    TC, decltype(copyC), decltype(mmaC), DirectCopy, AtomicAdd>>(
                        policy, prob_shape, cta_tiler, bP,
                        A, copyA,
                        B, copyB,
                        C, copyC, mmaC);

    EventManager::getInstance().addEvent(event);
}

template <bool DirectCopy, bool AtomicAdd, class TA, class TB, class TC>
void
gemm(char transA, char transB, int m, int n, int k, int l,
     TA const* A, int ldA,
     TB const* B, int ldB,
     TC      * C, int ldC)
{
    return gemm_nt<DirectCopy, AtomicAdd>(m, n, k, l, A, ldA, B, ldB, C, ldC);
}


int main(int argc, char** argv)
{
    int m = 8192;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 8192;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 8192;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    int l = 2;
    if (argc >= 5)
        sscanf(argv[4], "%d", &l);

    char transA = 'T';
    if (argc >= 6)
        sscanf(argv[5], "%c", &transA);

    char transB = 'N';
    if (argc >= 7)
        sscanf(argv[6], "%c", &transB);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = float;
    using TI = float;

    TI alpha = TI(1.0f);
    TI beta  = TI(0.0f);

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "L = " << l << std::endl;
    std::cout << "C = A^" << transA << " B^" << transB << std::endl;

    std::vector<TA> h_A(m * k * l);
    std::vector<TB> h_B(n * k * l);
    std::vector<TC> test_gemm_direct(m * n * l);
    std::vector<TC> test_gemm_atomic(m * n * l);
    std::vector<TC> test_gemm_manual(m * n * l);
    std::vector<TC> refe_gemm(m * n * l);
    int seed = 123;
    random_init(seed + 1, h_A.data(), m * k * l);
    random_init(seed + 2, h_B.data(), n * k * l);

    printf("A: ");
    for (int i = 0; i < 16; ++i) {
        printf("%7.4f ", (float)h_A[i]);
    }
    printf("\n");
    printf("B: ");
    for (int i = 0; i < 16; ++i) {
        printf("%7.4f ", (float)h_B[i]);
    }
    printf("\n");

    auto d_A = compat::malloc<TA>(m * k * l);
    auto d_B = compat::malloc<TB>(k * n * l);
    auto d_C = compat::malloc<TC>(m * n * l); // direct copy output
    auto d_D = compat::malloc<TC>(m * n * l);
    auto d_E = compat::malloc<TC>(m * n * l); // atomic add output
    auto d_F = compat::malloc<TC>(m * n * l); // manual atomic add output

    compat::memcpy<TA>(d_A, h_A.data(), m * k * l);
    compat::memcpy<TB>(d_B, h_B.data(), k * n * l);

    int ldA = 0, ldB = 0, ldC = m;

    if (transA == 'N') {
        ldA = m;
    } else if (transA == 'T') {
        ldA = k;
    } else {
        assert(false);
    }

    if (transB == 'N') {
        ldB = k;
    } else if (transB == 'T') {
        ldB = n;
    } else {
        assert(false);
    }

    ldC = n;

    verify_with_cutlass(
        d_A,
        d_B,
        d_D,
        m,
        n,
        k,
        l,
        alpha,
        beta,
        transA,
        transB
        );
    // Run once
    // direct
    gemm<true, false>(transA, transB, m, n, k, l,
                      d_A, ldA,
                      d_B, ldB,
                      d_C, ldC);
    compat::wait_and_throw();

    // XE_ATOMIC
    gemm<false, true>(transA, transB, m, n, k, l,
                      d_A, ldA,
                      d_B, ldB,
                      d_E, ldC);
    compat::wait_and_throw();

    // manual
    gemm<false, false>(transA, transB, m, n, k, l,
                      d_A, ldA,
                      d_B, ldB,
                      d_F, ldC);
    compat::wait_and_throw();

    compat::memcpy<TC>(refe_gemm.data(), d_D, m * n * l);
    compat::memcpy<TC>(test_gemm_direct.data(), d_C, m * n * l);
    compat::memcpy<TC>(test_gemm_atomic.data(), d_E, m * n * l);
    compat::memcpy<TC>(test_gemm_manual.data(), d_F, m * n * l);
    compat::wait_and_throw();

    float atol = 1e-3;
    float rtol = 1e-3;

    printf("verify direct copy\n");
    verify(refe_gemm.data(), test_gemm_direct.data(), l, m, n, atol, rtol);
    printf("verify atomic add\n");
    verify(refe_gemm.data(), test_gemm_atomic.data(), l, m, n, atol, rtol);
    printf("verify manual atomic add\n");
    verify(refe_gemm.data(), test_gemm_manual.data(), l, m, n, atol, rtol);
    double tflops = (2.0 * m * n * k * l) * 1e-12;

    const int timing_iterations = 100;
    GPU_Clock timer;

    // Timing iterations
    /*
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(transA, transB, m, n, k, l,
             alpha,
             d_A, ldA,
             d_B, ldB,
             beta,
             d_C, ldC);
    }
    compat::wait();
    double cute_time = timer.seconds() / timing_iterations;
    double gio = l * (m * k * sizeof(TA) + n * k * sizeof(TB) + m * n * sizeof(TC)) * 1e-9;
    printf("CUTE_GEMM:     [%6.1f]TFlop/s  [%6.1f]GB/s    (%6.4f)ms\n", tflops / cute_time, gio / cute_time, cute_time*1000);
    */
    return 0;
}
