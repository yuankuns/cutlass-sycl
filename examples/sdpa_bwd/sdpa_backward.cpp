#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <cassert>
#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cnpy.h"
#include "sdpa_util.hpp"
#include "params.hpp"


void read_args(int argc, char**argv, int n, int64_t *p) {
    if (argc >= n + 1)
        sscanf(argv[n], "%ld", p);
}

void
debug_info() {
    print("block idx (%d,%d,%d) dim (%d,%d,%d) thread idx (%d,%d,%d) dim (%d,%d,%d)\n",
          BlockIdxX(), BlockIdxY(), BlockIdxZ(),
          GridDimX(), GridDimY(), GridDimZ(),
          ThreadIdxX(), ThreadIdxY(), ThreadIdxZ(),
          BlockDimX(), BlockDimY(), BlockDimZ());
}

template<class Engine, class Layout>
void print_t(Tensor<Engine, Layout> &r) {
    print(r);
    for (int i = 0; i < size(r); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10.7f ", (float)r(i));
    }
    print("\n");
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

template<class T>
void print_t_2d(T t) {
    static_assert(rank(t) == 2, "Only support 2D Tensor");
    print(t);
    for (int i = 0; i <  size < 0>(t); ++i) {
        print("\n(%03d): ", i);
        for (int j = 0; j < size<1>(t); ++j) {
            print("%10.7f ", (float)t(i,j));
        }
    }
    print("\n");
}

template<class T>
void print_d(T t) {
    print(t);
    for (int i = 0; i < size(t); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10u ", t(i));
    }
    print("\n");
}

template<class T>
void print_c(T t) {
    print(t);
    for (int i = 0; i < size(t); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print(t(i));
    }
    print("\n");
}

using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>; // batch, num_head_q,num_head_kv,seq_len_qo,seq_len_kv,head_size_qk,head_size_vo

template <typename T>
struct OPS_tobf16{
    template <class Tensor>
    auto operator()(Tensor &src){
        cutlass::NumericConverter<
            T, float, cutlass::FloatRoundStyle::round_toward_zero> converter;
        auto dst = make_tensor_like<T>(src);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src); ++i) {
            dst(i) = converter(src(i));
        }
        return dst;
    }
};

template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout),
                                     get<1>(layout)),
                         get<2>(layout));
    return l;
}

constexpr int tid = 0;
constexpr int bid = 0;

const bool
is_cur_thread() {
    return cute::thread(tid, bid);
}

template <typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
CUTLASS_DEVICE void
apply_mask_causal(Tensor<Engine0, Layout0> &tensor,
                   Tensor<Engine1, Layout1> &rC,
                   int m_offset, int n_offset, int diagonal_offset = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    int sg_local_id = sg.get_local_id();
    int sg_group_id = sg.get_group_id();
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<1>(tensor); ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(tensor); ++m) {
            int x = n_offset + get<1>(rC_2d(m, n)) + sg_local_id + diagonal_offset;
            int y = m_offset + get<0>(rC_2d(m, n));
            if (x > y) {
                tensor(m, n) = -INFINITY;
            }
        }
    }
    return;
}

template<typename T, class Trait, class MTensor, class TiledMMA>
auto
create_reg(Trait const &trait,
           MTensor const &C,
           TiledMMA const &tiled_mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    Tensor cC = make_identity_tensor(C.shape());   // (M,N)
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));  // (BLK_M,BLK_N)
    auto copy_c = make_block_2d_copy_D(tiled_mma, C);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    if constexpr(is_same_v<T, float>) {
        auto r32 = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_mnk))); // allocate C fragment storage
        return r32;
    } else {
        auto r16 = thr_copy_c.partition_sg_fragment_S(gC);
        return r16;
    }
}

template<class Trait, class MTensor, class TiledMMA>
auto
create_reg_SdP(Trait const &trait,
               MTensor const &C,
               TiledMMA const &tiled_mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    Tensor cC = make_identity_tensor(C.shape());   // (M,N)
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));  // (BLK_M,BLK_N)
    auto copy_c = make_block_2d_copy_D(tiled_mma, C);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto r32 = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_mnk))); // allocate C fragment storage
    return r32;
}

template<class Trait, class MTensor, class TiledMMA>
auto
create_reg_SdP_16b(Trait const & trait,
                   MTensor const &C,
                   TiledMMA const & tiled_mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);
    Tensor cC = make_identity_tensor(C.shape());   // (M,N)
    auto tile_mnk = tiled_mma.tile_mnk();
    auto copy_c = make_block_2d_copy_D(tiled_mma, C);
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));  // (BLK_M,BLK_N)
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto r16 = thr_copy_c.partition_sg_fragment_S(gC);
    return r16;
}

template<bool clear_acc, class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
gemm_kernel(Trait &trait,
            Tensor<Engine0, Layout0> const& A,         // (M,K)
            Tensor<Engine1, Layout1> const& B,         // (N,K)
            SubgroupTensor<Engine2, Layout2, TVLayout2> & acc,
            TiledMMA const & mma,
            const int m_block,
            const int n_block) {
    // -----
    // Setup
    // -----

    /* Get workgroup and local IDs */
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;

    /* Create proxy coordinate tensors for each global tensor */
    Tensor cA = make_identity_tensor(A.shape());   // (M,K)
    Tensor cB = make_identity_tensor(B.shape());   // (N,K)

    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0,2>(tile_mnk), make_coord(m_block,_));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(cB, select<1,2>(tile_mnk), make_coord(n_block,_));  // (BLK_N,BLK_K,k)

    /* Create block 2D TiledCopies */
    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
    auto thr_mma    =    mma.get_slice(first_thread_in_sg_idx);
    auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
    auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);

    /* Register fragments for MMA */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

    /* Register fragments for copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    /* Partition global tensor (proxies) for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    /* Partition C */
    // Tensor tCrC = partition_fragment_C(mma, select<0,1>(tile_mnk));

    /* Create prefetch TiledCopy instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    auto thr_prefetch_A = prefetch_a.get_slice(first_thread_in_sg_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(first_thread_in_sg_idx);

    /* Partition global tensor (proxies) for prefetch */
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    /* Prefetch distance, in units of k tiles */
    const int prefetch_dist = 3;

    // ------
    // Kernel
    // ------

    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_prefetch = 0;
    /* Clear the accumulators */
    if constexpr(clear_acc)
        clear(acc);

    /* Warm up loops with prefetch to L1 */
    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
    }

    /* Main loop */
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        /* Split barrier keeping threads loosely together */
        barrier_arrive(barrier_scope);

        /* Copy A/B from global memory (ideally L1 cache) to registers */
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        /* Prefetch A/B tiles to L1 */
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

        /* Shuffle data from copy fragments to MMA fragments */
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        /* Accumulate C += A * B */
        gemm(mma, tCrA, tCrB, acc);

        /* Other half of split barrier */
        barrier_wait(barrier_scope);
    }
}

template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TVLayout2, class TiledMMA>
void
gemm_SdP(Trait &trait,
         Tensor<Engine0, Layout0> const& A,         // (M,K)
         Tensor<Engine1, Layout1> const& B,         // (N,K)
         SubgroupTensor<Engine2, Layout2, TVLayout2> & rSdP,
         TiledMMA const & mma,
          const int m_block,
          const int n_block) {
    gemm_kernel<true>(trait, A, B, rSdP, mma, m_block, n_block);
}

template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TVLayout2, class TiledMMA>
void
gemm_dKV(Trait &trait,
         Tensor<Engine0, Layout0> const& A,         // (M,K)
         Tensor<Engine1, Layout1> const& B,         // (N,K)
         SubgroupTensor<Engine2, Layout2, TVLayout2> & rdKV,
         TiledMMA const & mma,
          const int m_block,
          const int n_block) {
    gemm_kernel<false>(trait, A, B, rdKV, mma, m_block, n_block);
}

template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TiledMMA>
void
gemm_dQ(Trait &trait,
        Tensor<Engine0, Layout0> const& A,         // (M,K)
        Tensor<Engine1, Layout1> const& B,         // (N,K)
        Tensor<Engine2, Layout2> const& C,         // (M,N)
        TiledMMA const & mma,
        const int m_block,
        const int n_block) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto tile_mnk = mma.tile_mnk();
    Tensor cC = make_identity_tensor(C.shape());   // (M,N)
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(m_block, n_block));  // (BLK_M,BLK_N)
    auto thr_mma = mma.get_slice(first_thread_in_sg_idx);
    auto tCrC = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_mnk))); // allocate C fragment storage
    Tensor tCgC = thr_mma.partition_C(gC);
    gemm_kernel<true>(trait, A, B, tCrC, mma, m_block, n_block);
    int local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i <  size(tCgC); ++i) {
        auto [m, n] = tCgC(i);
        cutlass::atomicAdd(&C(m, n + local_id), tCrC(i));
    }
}

template <class Trait, class TiledMma,
          class Engine0, class Layout0, class TVLayout0,
          class Engine1, class Layout1>
void
mha_copy(Trait & trait, TiledMma &tiled_mma,
         SubgroupTensor<Engine0, Layout0, TVLayout0> &r,
         Tensor<Engine1, Layout1> &m,
         int m_block, int n_block) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto copy_c = make_block_2d_copy_D(tiled_mma, m);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor cC = make_identity_tensor(m.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(m_block, n_block));
    Tensor tCgC = thr_copy_c.partition_D(gC);
    copy(copy_c, r, tCgC);
}

template <class Trait, class TiledMma,
          class Engine0, class Layout0, class TVLayout0,
          class Engine1, class Layout1>
void
mha_reorder_copy(Trait & trait, TiledMma &tiled_mma,
                 SubgroupTensor<Engine0, Layout0, TVLayout0> &r,
                 Tensor<Engine1, Layout1> &m,
                 int m_block, int n_block) {
    auto r16 = create_reg<typename Trait::DType>(trait, m, tiled_mma);
    reorder(r, r16);
    mha_copy(trait, tiled_mma, r16, m, m_block, n_block);
}

template <bool Is_even_MN, class TileCopy,
          class Engine0, class Layout0,
          class Engine1, class Layout1>
CUTLASS_DEVICE void
mha_save(TileCopy &tile_copy,
         Tensor<Engine0, Layout0> &src,
         Tensor<Engine1, Layout1> &dst) {
    static_assert(Layout0::rank == 5, "Only support Tensor with 5 ranks");
    static_assert(Layout0::rank == Layout1::rank, "Only support same rank Tensor");
    if constexpr(Is_even_MN) {
        copy(tile_copy, src, dst);
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<3>(dst); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<4>(dst); ++n) {
                auto src_block = src(_, _, _, m, n);
                auto dst_block = dst(_, _, _, m, n);
                copy(tile_copy, src_block, dst_block);
            }
        }
    }
}

template <bool Is_even_MN, class TileCopy,
          class Engine0, class Layout0,
          class Engine1, class Layout1>
CUTLASS_DEVICE void
mha_load(TileCopy &tile_copy,
         Tensor<Engine0, Layout0> &src,
         Tensor<Engine1, Layout1> &dst) {
    static_assert(Layout0::rank == 5, "Only support Tensor with 5 ranks");
    static_assert(Layout0::rank == Layout1::rank, "Only support same rank Tensor");
    if constexpr(Is_even_MN) {
        copy(tile_copy, src, dst);
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<3>(src); ++m) {
            auto src_block = src(_, _, _, m, _);
            auto dst_block = dst(_, _, _, m, _);
            copy(tile_copy, src_block, dst_block);
        }
    }
}

template <class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2>
CUTLASS_DEVICE void
mha_atomic_add(Tensor<Engine0, Layout0>& m_tile,
               Tensor<Engine1, Layout1>& g_tile,
               Tensor<Engine2, Layout2>& r_tile,
               const int local_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<3>(g_tile); ++ni) {
        auto g = g_tile(_, _, _, ni);
        auto r = r_tile(_, _, _, ni);
        CUTLASS_PRAGMA_UNROLL
        for (int ki = 0; ki < size(g); ++ki) {
            auto [m, n, l] = g(ki);
            cutlass::atomicAdd(&m_tile(m, n + local_id, 0), r(ki));
        }
    }
}
template<bool Is_even_M, class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void
load_1colvec(Tensor0 &reg, Tensor1 &mT, Tensor2 &coord_row,
             int tail_m = 0) {
    if constexpr(Is_even_M) {
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(reg); ++mi) {
            reg(mi) = mT(get<0>(coord_row(mi)));
        }
    } else {
        for (int mi = 0; mi < size(reg); ++mi) {
            int row = get<0>(coord_row(mi));
            if (row < tail_m) {
                reg(mi) = mT(row);
            }
        }
    }
}

template<typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_layout(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_1>{});  // ((2, 2), MMA_M, MMA_N, Tile_M, M, N)
    auto l2 = make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                          make_layout(get<2>(l)));
    return l2;
}

template<class Engine0, class Layout0, class Engine1, class Layout1>
CUTLASS_DEVICE void
scale_apply_exp2(Tensor<Engine0, Layout0> &tensor,
                 Tensor<Engine1, Layout1> &max,
                 const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * M_LOG2E;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

template<class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void softmax_backward(Tensor0 &P, Tensor1 &dP_sum, Tensor2 &dP, const float scale) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(dP); ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int mj = 0; mj < size<1>(dP); ++mj) {
            dP(mi, mj) = P(mi, mj) * (dP(mi, mj) - dP_sum(mi)) * scale;
        }
    }
}

template<bool Is_even_N, bool Seq_parallel, class Trait>
void
dq_dk_dv_1colblock(Trait &trait, Param<typename Trait::DType> &param,
                   const int bidb, const int bidh, const int bidhkv, const int n_block,
                   const int tail_n = 0) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kNSGs = Trait::kNSGs;
    constexpr int SubgroupSize = Trait::SubgroupSize;
    constexpr int AtomLayoutMdQ = Trait::AtomLayoutMdQ;
    constexpr bool is_causal = Trait::is_causal;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    const int local_id = sg.get_local_id();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t q2_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k2_offset = bofst.k_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t v2_offset = bofst.v_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t dk2_offset = bofst.dk_offset(bidb, bidh, 0);
    const index_t dv2_offset = bofst.dv_offset(bidb, bidh, 0);
    const index_t o2_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t dq2_offset = bofst.dq_offset(bidb, bidh, 0);
    const index_t lse_offset = bofst.lse_offset(bidb, bidh, 0);
    // buff offset
    const index_t pb_offset = (bidb * param.num_head_q * param.seq_len_kv_pad * kBlockM
                               + bidh * param.seq_len_kv_pad * kBlockM
                               + n_block * kBlockN * kBlockM) * 2;
    const index_t dsb_offset = pb_offset + kBlockN * kBlockM;

    const index_t s_offset = bofst.ps_offset(bidb, bidh, 0, n_block * kBlockN);
    const index_t s2_offset = bofst.ps_offset(bidb, bidh, 0, 0);

    const auto block_n_dim = tail_n == 0 ? Int<kBlockN>{} : ((tail_n + 1) & ~1);
    auto shapeO = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapeQtOt2 = make_shape(Int<kHeadDim>{}, kBlockM);
    auto shapeSP2 = make_shape(kBlockM, block_n_dim);
    auto shapePt2 = make_shape(block_n_dim, kBlockM);

    using Shape12 = Shape<
        std::conditional_t<Is_even_N, Int<kBlockN>, int>, Int<kHeadDim>>;
    using Shape22 = Shape<
        Int <kHeadDim>,
        std::conditional_t<Is_even_N, Int<kBlockN>, int>>;
    auto shapeQ2 = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapedQ2 = Shape<Int<kBlockM>, Int<kHeadDim>>{};
    Shape12 shapeKtV2;
    Shape22 shapeK2;
    if constexpr(Is_even_N) {
        shapeKtV2 = make_shape(Int<kBlockN>{}, Int<kHeadDim>{});
        shapeK2 = make_shape(Int<kHeadDim>{}, Int<kBlockN>{});
    } else {
        shapeKtV2 = make_shape(tail_n, Int<kHeadDim>{});
        shapeK2 = make_shape(Int<kHeadDim>{}, tail_n);
    }

    Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q2_offset),
                            make_layout(
                                shapeQ2,
                                make_stride(param.q_r_stride, _1{})));
    Tensor mKt = make_tensor(make_gmem_ptr(param.k_ptr + k2_offset),
                             make_layout(
                                 shapeKtV2,
                                 make_stride(param.k_r_stride, _1{})));
    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + o2_offset),
                             make_layout(
                                 shapeO,
                                 make_stride(param.o_r_stride, _1{})));
    Tensor mV = make_tensor(make_gmem_ptr(param.v_ptr + v2_offset),
                            make_layout(
                                shapeKtV2,
                                make_stride(param.v_r_stride, _1{})));
   // intermediate buffer
    Tensor mP = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                            make_layout(
                                shapeSP2,
                                make_stride(block_n_dim, _1{})));
    Tensor mPt = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                             make_layout(
                                 shapePt2,
                                 make_stride(_1{}, block_n_dim)));
    Tensor mdOt = make_tensor(make_gmem_ptr(param.do_ptr + o2_offset),
                              make_layout(
                                  shapeQtOt2,
                                  make_stride(_1{}, param.o_r_stride)));
    Tensor mK = make_tensor(make_gmem_ptr(param.k_ptr + k2_offset),
                            make_layout(
                                shapeK2,
                                make_stride(_1{}, param.k_r_stride)));
    Tensor mdPt = make_tensor(make_gmem_ptr(param.pb_ptr + dsb_offset),
                              make_layout(
                                  shapePt2,
                                  make_stride(_1{}, block_n_dim)));
    Tensor mQt = make_tensor(make_gmem_ptr(param.q_ptr + q2_offset),
                              make_layout(
                                  shapeQtOt2,
                                  make_stride(_1{}, param.q_r_stride)));

    Tensor mLSE = make_tensor(make_gmem_ptr(param.lse_ptr + lse_offset),
                              make_layout(
                                  Shape<Int<kBlockM>>{},
                                  Stride<_1>{}));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + lse_offset),
                                make_layout(
                                    Shape<Int<kBlockM>>{},
                                    Stride<_1>{}));

    Tensor mdV = make_tensor(make_gmem_ptr(param.dv_ptr + dv2_offset),
                             make_layout(
                                 make_shape(param.seq_len_kv, Int<kHeadDim>{}),
                                 make_stride(param.dv_r_stride, _1{})));
    Tensor mdP = make_tensor(make_gmem_ptr(param.pb_ptr + dsb_offset),
                              make_layout(
                                  shapeSP2,
                                  make_stride(block_n_dim, _1{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq2_offset),
                                  make_layout(
                                      shapedQ2,
                                      make_stride(param.dq_r_stride, _1{})));
    Tensor mdK = make_tensor(make_gmem_ptr(param.dk_ptr + dk2_offset),
                              make_layout(
                                  make_shape(param.seq_len_kv, Int<kHeadDim>{}),
                                  make_stride(param.dk_r_stride, _1{})));
#ifdef _DEBUG_
    Tensor mS = make_tensor(make_gmem_ptr(param.s_ptr + s2_offset),
                             make_layout(
                                 make_shape(param.seq_len_q_pad, param.seq_len_kv_pad),
                                 make_stride(param.s_r_stride, _1{})));
    Tensor mdPd = make_tensor(make_gmem_ptr(param.dp_ptr + s2_offset),
                               make_layout(
                                   make_shape(param.seq_len_q_pad, param.seq_len_kv_pad),
                                   make_stride(param.s_r_stride, _1{})));
#endif

    typename Trait::TiledMmaSdP2 tiled_mma_sdp;
    typename Trait::TiledMmadKV2 tiled_mma_dkv;
    typename Trait::TiledMmadQ2 tiled_mma_dq;

    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);

    // for lse read
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}); // same buffer as accS
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    static_assert(decltype(size<0>(taccScS))::value == 8);
    Tensor taccScS_rc = logical_divide(taccScS, Shape<_1>{});
    Tensor taccScS_row = logical_divide(taccScS, Shape<_1>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<V>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    // static_assert(size<0>(tSrS) * size<1>(tSrS) == size<0>(lse) && "row of acc and lse not match");
    // misc

    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    const int tail_m = param.seq_len_q % kBlockM;

    auto rdV = create_reg<V>(trait,
                             mdV,
                             tiled_mma_dkv);
    auto rdK = create_reg<V>(trait,
                             mdK,
                             tiled_mma_dkv);
    clear(rdV);
    clear(rdK);
    // clear accumulator
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        const bool Is_even_M = not ((m_block == max_m_block - 1) and (tail_m != 0));
        if (not Is_even_M) {
            mQ = make_tensor(make_gmem_ptr(mQ.data()),
                             make_layout(
                                 make_shape(tail_m, Int<kHeadDim>{}),
                                 make_stride(param.q_r_stride, _1{})));
            mdO = make_tensor(make_gmem_ptr(mdO.data()),
                             make_layout(
                                 make_shape(tail_m, Int<kHeadDim>{}),
                                 make_stride(param.o_r_stride, _1{})));
            mdOt = make_tensor(make_gmem_ptr(mdOt.data()),
                               make_layout(
                                   make_shape(Int<kHeadDim>{}, tail_m),
                                   make_stride(_1{}, param.o_r_stride)));
            mdQaccum = make_tensor(make_gmem_ptr(mdQaccum.data()),
                                   make_layout(
                                       shapedQ2,
                                       make_stride(param.dq_r_stride, _1{})));
            mQt = make_tensor(make_gmem_ptr(mQt.data()),
                              make_layout(
                                  make_shape(Int<kHeadDim>{}, tail_m),
                                  make_stride(_1{}, param.q_r_stride)));
        }
        {
        auto rS = create_reg<V>(trait,
                                mP,
                                tiled_mma_sdp);
        clear(rS);
        // S=QKt
        gemm_SdP(trait, mQ, mKt, rS,
                 tiled_mma_sdp, 0, 0);
        Tensor scores = make_tensor(rS.data(), convert_layout_acc_layout(rS.layout()));
        if constexpr(is_causal) {
            apply_mask_causal(scores, taccScS_rc, m_block * kBlockM, n_block * kBlockN, param.seq_len_q - param.seq_len_kv);
        }

        if (Is_even_M) {
            load_1colvec<true>(lse, mLSE, taccScS_row);
        } else {
            load_1colvec<false>(lse, mLSE, taccScS_row, tail_m);
        }

        Tensor dP_sum = make_fragment_like(lse);

        if (Is_even_M)
            load_1colvec<true>(dP_sum, mdPsum, taccScS_row);
        else
            load_1colvec<false>(dP_sum, mdPsum, taccScS_row, tail_m);

        // P=softmax(S,lse)
        scale_apply_exp2(scores, lse, param.scale_softmax_log2);
        mha_reorder_copy(trait, tiled_mma_sdp, rS, mP, 0, 0);
#ifdef _DEBUG_
        mha_reorder_copy(trait, tiled_mma_sdp, rS, mS, m_block, n_block); // debug
#endif
        auto rdP = create_reg<V>(trait,
                                 mdP,
                                 tiled_mma_sdp);
        clear(rdP);
        // dP=dO*Vt
        gemm_SdP(trait, mdO, mV, rdP,
                 tiled_mma_sdp, 0, 0);
        Tensor dS = make_tensor(rdP.data(), scores.layout());
        // dS=P(dP-sum_row(P))*scale
        softmax_backward(scores, dP_sum, dS, param.scale_softmax);
        mha_reorder_copy(trait, tiled_mma_sdp, rdP, mdP, 0, 0); // copy dP to internal buff
#ifdef _DEBUG_
        mha_reorder_copy(trait, tiled_mma_sdp, rdP, mdPd, m_block, n_block); // debug
#endif
        }
        // dV=Pt*dO
        gemm_dKV(trait, mPt, mdOt, rdV,
                 tiled_mma_dkv, 0, 0);
        // dQ=dP*K
        gemm_dQ(trait, mdP, mK, mdQaccum,
                tiled_mma_dq, 0, 0);

        // dK=dPt*Q
        gemm_dKV(trait, mdPt, mQt, rdK,
                 tiled_mma_dkv, 0, 0);
        // update ptr/atom copy
        mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
        mdO.data() = mdO.data() + int(kBlockM * param.o_r_stride);
        mdOt.data() = mdOt.data() + int(kBlockM * param.o_r_stride);
        mdQaccum.data() = mdQaccum.data() + int(kBlockM * param.dq_r_stride);
        mQt.data() = mQt.data() + int(kBlockM * param.q_r_stride);
        mLSE.data() = mLSE.data() + int(kBlockM);
        mdPsum.data() = mdPsum.data() + int(kBlockM);

    }
    mha_reorder_copy(trait, tiled_mma_dkv, rdV, mdV, n_block, 0);
    mha_reorder_copy(trait, tiled_mma_dkv, rdK, mdK, n_block, 0);
}

template<bool Is_even_M, class T>
void
compute_o_dot_do(T &trait, Param<typename T::DType> &param,
                 const int m_block, const int bidb, const int bidh) {
    // The thread index.
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    constexpr int SubgroupSize = T::SubgroupSize;
    using DType = typename T::DType;
    using VType = typename T::VType;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t o_offset = bofst.o_offset(bidb, bidh, m_block * kBlockM);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, m_block * kBlockM);

    using ShapeO = Shape<
        std::conditional_t <Is_even_M, Int<kBlockM>, int>,
        Int<kHeadDim>>;
    using ShapeP = Shape<
        std::conditional_t <Is_even_M, Int<kBlockM>, int>>;
    ShapeO O_shape;
    ShapeP dP_shape;
    if constexpr(Is_even_M) {
        O_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
        dP_shape = make_shape(Int<kBlockM>{});
    } else {
        O_shape = make_shape(param.tail_m, Int<kHeadDim>{});
        dP_shape = make_shape(param.tail_m);
    }
    auto dQ_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});

    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                             make_layout(
                                 O_shape,
                                 make_stride(param.o_r_stride, _1{})));
    Tensor mO = make_tensor(make_gmem_ptr(param.o_ptr + o_offset),
                            make_layout(
                                O_shape,
                                make_stride(param.o_r_stride, _1{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                                      make_stride(param.dq_r_stride, _1{})));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + dpsum_offset),
                                make_layout(
                                    dP_shape,
                                    Stride<_1>{}));

    auto tileload_odo = make_tiled_copy(Copy_Atom<UniversalCopy<DType>, DType>{},
                                        Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>,
                                        Stride<Int<SubgroupSize>, _1>>{},
                                        Layout<Shape<_1, _1>>{});
    auto tileload_dq = make_tiled_copy(Copy_Atom<UniversalCopy<VType>, VType>{},
                                        Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>>{},
                                        Layout<Shape<_1, _1>>{});
    auto thr_load_odo = tileload_odo.get_thread_slice(ThreadIdxX());
    auto thr_load_dq = tileload_dq.get_thread_slice(ThreadIdxX());

    Tensor thr_tile_do_S = thr_load_odo.partition_S(mdO);
    Tensor thr_tile_o_S = thr_load_odo.partition_S(mO);
    Tensor thr_tile_dq_D = thr_load_dq.partition_D(mdQaccum);
    Tensor rdQ = make_fragment_like(thr_tile_dq_D);
    Tensor rdO = make_fragment_like<DType>(rdQ);
    Tensor rO = make_fragment_like<DType>(rdQ);
    clear(rdQ);
    copy(tileload_dq, rdQ, thr_tile_dq_D);

    Tensor cO = make_identity_tensor(dQ_shape);
    Tensor tcO = thr_load_odo.partition_S(cO);
    Tensor tcO_row = logical_divide(tcO, Shape<_1>{})(make_coord(0, 0), _, 0);
    Tensor rdO_2d = make_tensor(rdO.data(),
                                convert_layout_2d_layout(rdO.layout()));
    Tensor rO_2d = make_tensor(rO.data(),
                               convert_layout_2d_layout(rO.layout()));
    if constexpr(Is_even_M) {
        copy(tileload_odo, thr_tile_do_S, rdO);
        copy(tileload_odo, thr_tile_o_S, rO);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
            float accum = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(rdO_2d); ++ni) {
                accum = accum + (float)rdO_2d(mi, ni) * (float)rO_2d(mi, ni);
            }
            accum = sycl::reduce_over_group(sg, accum, sycl::plus<>());
            if (sg.get_local_id() == 0) {
                mdPsum(get<0>(tcO_row(mi))) = accum;
            }
        }
    } else {
        for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
            if (get<0>(tcO_row(mi)) < param.tail_m) {
                copy(tileload_odo, thr_tile_do_S(_, mi, _), rdO(_, mi, _));
                copy(tileload_odo, thr_tile_o_S(_, mi, _), rO(_, mi, _));
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
            float accum = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(rdO_2d); ++ni) {
                accum = accum + (float)rdO_2d(mi, ni) * (float)rO_2d(mi, ni);
            }
            accum = sycl::reduce_over_group(sg, accum, sycl::plus<>());
            if (sg.get_local_id() == 0 and get<0>(tcO_row(mi)) < param.tail_m)
                mdPsum(get<0>(tcO_row(mi))) = accum;
        }
    }
}

template<class T>
void
compute_o_dot_do2(T & trait, Param<typename T::DType> param) {
    // The block index for the M dimension.
    const int m_block = BlockIdxX();
    // The block index for the batch.
    const int bidb = BlockIdxZ();
    // The block index for the head.
    const int bidh = BlockIdxY();;
    const int mid = ThreadIdxX() / 16; // 1 row per subgroup
    const int hid = ThreadIdxX() % 16; // 1 thread per col
    auto sg = compat::get_nd_item<1>().get_sub_group();
    // The thread index.
    // const int tidx = threadIdx.x;
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    using DType = typename T::DType;

    auto bofst = Boffset(param);

    const index_t o_offset = bofst.o_offset(bidb, bidh, m_block * kBlockM);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, m_block * kBlockM);
    const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);

    const DType *o_ptr = param.o_ptr + o_offset;
    const DType *do_ptr = param.do_ptr + o_offset;
    float *dpsum_ptr = param.odo_ptr + dpsum_offset;
    float *dqaccum_ptr = param.dqaccum_ptr + dq_offset;

    int tail_m = param.seq_len_q - m_block * kBlockM;
    int m = ThreadIdxX();
    if (m < tail_m) {
        float dP_sum_cur = 0.0f;
        for (int h = 0; h < kHeadDim; ++h) {
            float o_val = static_cast<float>(o_ptr[m * param.o_r_stride + h]);
            float do_val = static_cast<float>(do_ptr[m * param.o_r_stride + h]);
            dP_sum_cur += o_val * do_val;
            dqaccum_ptr[m * param.dq_r_stride + h] = 0.0f;
        }
        dpsum_ptr[m] = dP_sum_cur;
    }
}

template<class T>
void
mha_backward_seq(T trait,
                 Param<typename T::DType> param) {
    const int bidb = BlockIdxZ();
    const int bidhq = BlockIdxY();
    const int bidhkv = bidhq / param.num_qh_per_kvh;
    // const int max_n_block = ceil_div(param.seq_len_kv, trait.kBlockN);
    for (int n_block = 0; n_block < param.n_block; ++n_block)
        if (param.tail_n > 0 and n_block == param.n_block - 1)
            dq_dk_dv_1colblock<false, false>(trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
        else
            dq_dk_dv_1colblock<true, false>(trait, param, bidb, bidhq, bidhkv, n_block);
}

template<class T>
void
mha_backward_parallel(T trait,
                      Param<typename T::DType> param) {
    const int bidb = BlockIdxZ();
    const int bidhq = BlockIdxY();
    const int n_block = BlockIdxX();
    const int bidhkv = bidhq / param.num_qh_per_kvh;
    if (param.tail_n > 0 and n_block == param.n_block - 1)
        dq_dk_dv_1colblock<false, true>(trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
    else
        dq_dk_dv_1colblock<true, true>(trait, param, bidb, bidhq, bidhkv, n_block);
}

template<class T>
void
mha_dot_do_o(T trait,
             Param<typename T::DType> param) {
    // The block index for the M dimension.
    const int m_block = BlockIdxX();
    // The block index for the batch.
    const int bidb = BlockIdxZ();
    // The block index for the head.
    const int bidh = BlockIdxY();;
    if (m_block == param.m_block - 1 and param.tail_m > 0) {
        compute_o_dot_do<false>(trait, param, m_block, bidb, bidh);
    } else {
        compute_o_dot_do<true>(trait, param, m_block, bidb, bidh);
    }
}

template <class T>
void
convert_dq(T &trait, Param<typename T::DType> &param, int m_block, int bidb, int bidh) {
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    using DType = typename T::DType;
    using VType = typename T::VType;

    auto bofst = Boffset(param);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);
    VType * dQaccum = param.dqaccum_ptr + dq_offset;
    DType * dQ = param.dq_ptr + q_offset;

    int tail_m = param.seq_len_q - m_block * kBlockM;
    int m = ThreadIdxX();
    if (m < tail_m) {
        for (int h = 0; h < kHeadDim; ++h) {
            dQ[m * param.q_r_stride + h] = static_cast<DType>(dQaccum[m * param.dq_r_stride + h]);
        }
    }
}

template <bool Is_even_M, class T>
void
convert_dq(T &trait, Param<typename T::DType> &param, int m_block, int bidb, int bidh) {
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    using DType = typename T::DType;
    using VType = typename T::VType;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;

    auto bofst = Boffset(param);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);
    using ShapeQ = Shape<
        std::conditional_t<Is_even_M, Int<kBlockM>, int>,
        Int<kHeadDim>, _1>;
    ShapeQ shapeQ;
    if constexpr (Is_even_M) {
        shapeQ = make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, _1{});
    } else {
        shapeQ = make_shape(param.tail_m, Int<kHeadDim>{}, _1{});
    }

    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                      make_stride(param.dq_r_stride, _1{}, _1{})));
    Tensor mdQ = make_tensor(make_gmem_ptr(param.dq_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{}, _1{})));

    auto tile_dq = typename T::TileShapedQ{};

    auto tileloaddQ = typename T::TiledLoaddQ{mdQaccum};
    auto tilesavedQ = typename T::TiledSavedV{mdQ};


    typename T::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_slice(first_thread_in_sg_idx);

    Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
    Tensor gdQ = local_tile(mQ_coord, select<0, 1>(tile_dq), make_coord(_, _, 0)); // dump dQ

    Tensor tdQgdQ = thr_mma_dq.partition_C(gdQ); // save to dq
    Tensor tdQrdQaccum = partition_fragment_C(tiled_mma_dq,
                                              make_shape(get<0>(tile_dq),
                                                         get<1>(tile_dq),
                                                         ceil_div(Int<kBlockM>{}, get<0>(tile_dq)),
                                                         ceil_div(Int<kHeadDim>{}, get<1>(tile_dq))));

    Tensor tdQrdQ = make_fragment_like<DType>(tdQrdQaccum);
    if constexpr(Is_even_M) {
        mha_load<true>(tileloaddQ, tdQgdQ, tdQrdQaccum);
    } else {
        mha_load<false>(tileloaddQ, tdQgdQ, tdQrdQaccum);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tdQrdQ); ++i) {
        tdQrdQ(i) = static_cast<DType>(tdQrdQaccum(i));
    }
    if constexpr(Is_even_M) {
        mha_save<true>(tilesavedQ, tdQrdQ, tdQgdQ);
    } else {
        mha_save<false>(tilesavedQ, tdQrdQ, tdQgdQ);
    }
}

template<class T>
void
mhd_convert_dq(T trait,
               Param<typename T::DType>  param) {
    // The block index for the M dimension.
    const int m_block = BlockIdxX();
    // The block index for the batch.
    const int bidb = BlockIdxZ();
    // The block index for the head.
    const int bidh = BlockIdxY();
    if (param.tail_m > 0 and m_block == param.m_block - 1) {
        convert_dq<false>(trait, param, m_block, bidb, bidh);
    } else {
        convert_dq<true>(trait, param, m_block, bidb, bidh);
    }
}

template<class...> class mhaodoDeviceName;
template<class...> class mhabwdDeviceName;
template<class...> class mhacvtDeviceName;

template<typename T, class ProblemShape, int kBlockM, int kBlockN,
         int kHeadDim, int kNSGs, int AtomLayoutMSdP, int AtomLayoutNdKV,
         int AtomLayoutMdQ, bool is_causal, bool is_bhsd>
void launch_mha_backward_headdim(ProblemShape problem_shape,
                                 const T *do_d,
                                 const T *o_d,
                                 const T *q_d,
                                 const T *k_d,
                                 const T *v_d,
                                 const float *lse_d,
                                 float *odo_d,
                                 float *dqaccum_d,
                                 T *dq_d,
                                 T *dk_d,
                                 T *dv_d,
                                 T *s_d,
                                 T *dp_d,
                                 const int seq_len_q_pad,
                                 const int seq_len_kv_pad) {
    auto trait = FAKernel<T, kHeadDim, kBlockM, kBlockN, kNSGs,
                          AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                          is_causal>{};

    const int BATCH = get<0>(problem_shape);
    const int NUM_HEAD_Q = get<1>(problem_shape);
    const int NUM_HEAD_KV = get<2>(problem_shape);
    const int SEQ_LEN_Q = get<3>(problem_shape);
    const int SEQ_LEN_KV = get<4>(problem_shape);
    const int N_BLOCK = ceil_div(SEQ_LEN_KV, kBlockN);
    const int tail_n = SEQ_LEN_KV % kBlockN;
    const int M_BLOCK = ceil_div(SEQ_LEN_Q, kBlockM);
    const int tail_m = SEQ_LEN_Q % kBlockM;
    T * pbuff = compat::malloc<T>(BATCH * NUM_HEAD_Q * seq_len_kv_pad * 2 * kBlockM);
    auto param = Param<T>(do_d, o_d, q_d, k_d, v_d, lse_d, odo_d,
                          dqaccum_d, dq_d, dk_d, dv_d, s_d, dp_d, pbuff,
                          1 / sqrt(static_cast<float>(kHeadDim)));
    param.batch = BATCH;
    param.num_head_q = NUM_HEAD_Q;
    param.num_head_kv = NUM_HEAD_KV;
    param.num_qh_per_kvh = NUM_HEAD_Q / NUM_HEAD_KV;
    param.seq_len_q = SEQ_LEN_Q;
    param.seq_len_kv = SEQ_LEN_KV;
    param.head_dim = kHeadDim;
    param.n_block = N_BLOCK;
    param.tail_n = tail_n;
    param.m_block = M_BLOCK;
    param.tail_m = tail_m;
    param.seq_len_kv_pad = seq_len_kv_pad;
    param.seq_len_q_pad = seq_len_q_pad;
    if constexpr(is_bhsd) {
        setup_bhsd_stride(param);
    } else {
        setup_bshd_stride(param);
    }
#ifdef _BENCH_
    printf("Launching mha backward kernel with 50 times\n");
    int count = 50;
    auto start = std::chrono::high_resolution_clock::now();
    auto dur0 = start - start;
    auto dur1 = start - start;
    auto dur2 = start - start;
    for (int iii=0; iii < count; ++iii) {
    auto start0 = std::chrono::high_resolution_clock::now();
#endif
    auto dimGrid0 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
    auto dimBlock0 = compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
    compat::experimental::launch_properties launch_props0{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    compat::experimental::kernel_properties kernel_props0{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy0{dimGrid0, dimBlock0, launch_props0, kernel_props0};
    auto event0 = compat::experimental::launch<
        mha_dot_do_o<decltype(trait)>,
        mhaodoDeviceName<decltype(trait)>>(policy0,
                                           trait,
                                           param);
    EventManager::getInstance().addEvent(event0);
    compat::wait_and_throw();
#ifdef _BENCH_
    auto end0 = std::chrono::high_resolution_clock::now();
    dur0 += end0 - start0;

    auto start1 = std::chrono::high_resolution_clock::now();
#endif
    auto dimGrid1 = compat::dim3(size(param.n_block),
                                 size(param.num_head_q), size(param.batch));
    assert((param.num_head_q % param.num_head_kv == 0) && "num_head_q must be dividable by num_head_kv");
    assert((param.num_head_q >= param.num_head_kv) && "num_head_q must be bigger than or equal to num_head_kv");
    auto dimBlock1 = compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
    // auto dimBlock = compat::dim3(size(trait.tiled_mma_sdp));

    compat::experimental::launch_properties launch_props1{
        sycl::ext::oneapi::experimental::work_group_scratch_size(trait.smem_size),
    };
    compat::experimental::kernel_properties kernel_props1{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy1{dimGrid1, dimBlock1, launch_props1, kernel_props1};
    auto event1 = compat::experimental::launch<
        mha_backward_parallel<decltype(trait)>,
        mhabwdDeviceName<decltype(trait)>>(policy1,
                                           trait,
                                           param);
    EventManager::getInstance().addEvent(event1);
    compat::wait_and_throw();
#ifdef _BENCH_
    auto end1 = std::chrono::high_resolution_clock::now();
    dur1 += end1 - start1;

    auto start2 = std::chrono::high_resolution_clock::now();
#endif
    auto dimGrid2 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
    auto dimBlock2 = compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
    compat::experimental::launch_properties launch_props2{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    compat::experimental::kernel_properties kernel_props2{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy2{dimGrid2, dimBlock2, launch_props2, kernel_props2};
    auto event2 = compat::experimental::launch<
        mhd_convert_dq<decltype(trait)>,
        mhacvtDeviceName<decltype(trait)>>(policy2,
                                           trait,
                                           param);
    EventManager::getInstance().addEvent(event2);
    compat::wait_and_throw();
#ifdef _BENCH_
    auto end2 = std::chrono::high_resolution_clock::now();
    dur2 += end2 - start2;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - start;
    printf("mha dot do kernel time, odo :%f mha bwd: %f dq: %f total: %f us\n",
           std::chrono::duration_cast<std::chrono::microseconds>(dur0).count() / float(count),
           std::chrono::duration_cast<std::chrono::microseconds>(dur1).count() / float(count),
           std::chrono::duration_cast<std::chrono::microseconds>(dur2).count() / float(count),
           std::chrono::duration_cast<std::chrono::microseconds>(dur).count() / float(count));
#endif
}

template<typename T, class ProblemShape, int kMPad, int kNPad, bool is_causal, bool is_bhsd>
void launch_mha_backward(ProblemShape problem_shape,
                         const T *do_d,
                         const T *o_d,
                         const T *q_d,
                         const T *k_d,
                         const T *v_d,
                         const float *lse_d,
                         float *odo_d,
                         float *dqaccum_d,
                         T *dq_d,
                         T *dk_d,
                         T *dv_d,
                         T *s_d,
                         T *dp_d,
                         const int seq_len_q_pad,
                         const int seq_len_kv_pad) {
    const int headdim = get<5>(problem_shape);
    if (headdim == 64) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 32;
        constexpr int kHeadDim = 64;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 4;
        constexpr int AtomLayoutNdKV = 2;
        constexpr int AtomLayoutMdQ = 2;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 96) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 64;
        constexpr int kHeadDim = 96;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 2;
        constexpr int AtomLayoutNdKV = 4;
        constexpr int AtomLayoutMdQ = 4;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 128) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 64;
        constexpr int kHeadDim = 128;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 2;
        constexpr int AtomLayoutNdKV = 4;
        constexpr int AtomLayoutMdQ = 4;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 192) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 32;
        constexpr int kHeadDim = 192;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 4;
        constexpr int AtomLayoutNdKV = 2;
        constexpr int AtomLayoutMdQ = 2;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 256) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 32;
        constexpr int kHeadDim = 256;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 4;
        constexpr int AtomLayoutNdKV = 2;
        constexpr int AtomLayoutMdQ = 2;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else {
        assert(false && "only support headdim 64,96,128,192,256");
    }
}

int main(int argc, char**argv) {
    // using T = cute::bfloat16_t;
    using T = cute::half_t;
    using V = float;
    std::string data_file = "mha.npz";
    // read qkv
    cnpy::NpyArray q_npy = cnpy::npz_load(data_file, "q");
    cnpy::NpyArray k_npy = cnpy::npz_load(data_file, "k");
    cnpy::NpyArray v_npy = cnpy::npz_load(data_file, "v");

    // read s and p for debug
    cnpy::NpyArray s_npy = cnpy::npz_load(data_file, "s");
    cnpy::NpyArray p_npy = cnpy::npz_load(data_file, "p");

    // read grad output
    cnpy::NpyArray do_npy = cnpy::npz_load(data_file, "grad");
    cnpy::NpyArray o_npy = cnpy::npz_load(data_file, "out");

    // read lse
    cnpy::NpyArray lse_npy = cnpy::npz_load(data_file, "lse");
    // read odo
    cnpy::NpyArray odo_npy = cnpy::npz_load(data_file, "odo");

    // read grad reference
    cnpy::NpyArray dq_npy = cnpy::npz_load(data_file, "q_grad");
    cnpy::NpyArray dk_npy = cnpy::npz_load(data_file, "k_grad");
    cnpy::NpyArray dv_npy = cnpy::npz_load(data_file, "v_grad");
    cnpy::NpyArray dp_npy = cnpy::npz_load(data_file, "p_grad");
    cnpy::NpyArray ds_npy = cnpy::npz_load(data_file, "s_grad");

    // read shape
    cnpy::NpyArray shape = cnpy::npz_load(data_file, "shape");

    int64_t BATCH = shape.data<int>()[0];
    int64_t NUM_HEAD_Q = shape.data<int>()[1];
    int64_t NUM_HEAD_KV = shape.data<int>()[2];
    int64_t SEQ_LEN_QO = shape.data<int>()[3];
    int64_t SEQ_LEN_KV = shape.data<int>()[4];
    int64_t HEAD_SIZE_QK = shape.data<int>()[5];
    int64_t HEAD_SIZE_VO = shape.data<int>()[6];
    bool is_causal = shape.data<int>()[7];
    bool is_bhsd = shape.data<int>()[8];
    assert(HEAD_SIZE_QK == HEAD_SIZE_VO && "only support head_size_qk==head_size_vo");
    constexpr int kBlockN = 64;
    constexpr int kBlockM = 64;
    int64_t SEQ_LEN_QO_PAD = ceil_div(SEQ_LEN_QO, kBlockM) * kBlockM;
    int64_t SEQ_LEN_KV_PAD = ceil_div(SEQ_LEN_KV, kBlockN) * kBlockN;
    printf("batch %d nh_q %d nh_k %d sq_q %d(%d) sq_k %d(%d) hd_q %d hd_v %d causal %d bhsd %d\n", BATCH, NUM_HEAD_Q, NUM_HEAD_KV, SEQ_LEN_QO, SEQ_LEN_QO_PAD, SEQ_LEN_KV, SEQ_LEN_KV_PAD, HEAD_SIZE_QK, HEAD_SIZE_VO, is_causal, is_bhsd);
    // read_args(argc, argv, 1, &BATCH);
    // read_args(argc, argv, 2, &NUM_HEAD_Q);
    // read_args(argc, argv, 3, &NUM_HEAD_KV);
    // read_args(argc, argv, 4, &SEQ_LEN_QO);
    // read_args(argc, argv, 5, &SEQ_LEN_KV);
    // read_args(argc, argv, 6, &HEAD_SIZE_QK);
    // read_args(argc, argv, 7, &HEAD_SIZE_VO);

    // alloc qkv
    T *q_d = compat::malloc<T>(q_npy.num_vals);
    T *k_d = compat::malloc<T>(k_npy.num_vals);
    T *v_d = compat::malloc<T>(v_npy.num_vals);

    // alloc ps
    T *p_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    T *s_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);

    // alloc lse, odo
    V *lse_d = compat::malloc<V>(lse_npy.num_vals);
    V *odo_d = compat::malloc<V>(odo_npy.num_vals);

    // alloc grad output
    T *do_d = compat::malloc<T>(do_npy.num_vals);
    T *o_d = compat::malloc<T>(o_npy.num_vals);

    // alloc grad test on device
    T *dq_d = compat::malloc<T>(dq_npy.num_vals);
    V *dqaccum_d = compat::malloc<V>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);
    T *dk_d = compat::malloc<T>(dk_npy.num_vals);
    T *dv_d = compat::malloc<T>(dv_npy.num_vals);
    T *dp_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    // copy qkv
    compat::memcpy<T>(q_d, q_npy.data<T>(), q_npy.num_vals);
    compat::memcpy<T>(k_d, k_npy.data<T>(), k_npy.num_vals);
    compat::memcpy<T>(v_d, v_npy.data<T>(), v_npy.num_vals);

    // copy grad output
    compat::memcpy<T>(do_d, do_npy.data<T>(), do_npy.num_vals);
    compat::memcpy<T>(o_d, o_npy.data<T>(), o_npy.num_vals);

    // copy lse
    compat::memcpy<V>(lse_d, lse_npy.data<V>(), lse_npy.num_vals);

    // copy odo
    // compat::memcpy<V>(odo_d, odo_npy.data<V>(), odo_npy.num_vals);

    auto problem_shape = ProblemShapeRegular(BATCH, NUM_HEAD_Q, NUM_HEAD_KV,
                                             SEQ_LEN_QO, SEQ_LEN_KV, HEAD_SIZE_QK, HEAD_SIZE_VO);
    if (is_bhsd) {
        if (is_causal)
            launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, true, true>(
                problem_shape,
                do_d, o_d,
                q_d, k_d, v_d,
                lse_d, odo_d,
                dqaccum_d, dq_d, dk_d, dv_d,
                s_d, dp_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);
        else
            launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, false, true>(
                problem_shape,
                do_d, o_d,
                q_d, k_d, v_d,
                lse_d, odo_d,
                dqaccum_d, dq_d, dk_d, dv_d,
                s_d, dp_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);
    } else {
        if (is_causal) {
            launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, true, false>(
                problem_shape,
                do_d, o_d,
                q_d, k_d, v_d,
                lse_d, odo_d,
                dqaccum_d, dq_d, dk_d, dv_d,
                s_d, dp_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);
        } else {
            launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, false, false>(
                problem_shape,
                do_d, o_d,
                q_d, k_d, v_d,
                lse_d, odo_d,
                dqaccum_d, dq_d, dk_d, dv_d,
                s_d, dp_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);
        }
    }
    float atol = 3e-3f;
    float rtol = 3e-3f;

    std::vector<V> odo_test(odo_npy.num_vals);
    compat::memcpy<V>(odo_test.data(), odo_d, odo_test.size());
    compat::wait_and_throw();
    printf("odo val: ");
    verify(odo_npy.data<V>(), odo_test.data(), BATCH, NUM_HEAD_Q, SEQ_LEN_QO, atol, rtol);

#ifdef _DEBUG_
    std::vector<T> s_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    compat::memcpy<T>(s_test.data(), s_d, s_test.size());
    compat::wait_and_throw();
    printf("P val: ");
    verify(p_npy.data<T>(), s_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_QO_PAD, SEQ_LEN_KV, SEQ_LEN_KV_PAD, atol, rtol);

    std::vector<T> dp_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    compat::memcpy<T>(dp_test.data(), dp_d, dp_test.size());
    compat::wait_and_throw();
    printf("dS val: ");
    verify(ds_npy.data<T>(), dp_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_QO_PAD, SEQ_LEN_KV, SEQ_LEN_KV_PAD, atol, rtol);
#endif

    std::vector<T> dv_test(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_VO);
    compat::memcpy<T>(dv_test.data(), dv_d, dv_test.size());
    compat::wait_and_throw();
    printf("dV val: ");
    verify(dv_npy.data<T>(), dv_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_KV, HEAD_SIZE_VO, atol, rtol);

    std::vector<T> dk_test(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_QK);
    compat::memcpy<T>(dk_test.data(), dk_d, dk_test.size());
    compat::wait_and_throw();
    printf("dK val: ");
    verify(dk_npy.data<T>(), dk_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_KV, HEAD_SIZE_QK, atol, rtol);

    std::vector<V> dqaccum_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);
    compat::memcpy<V>(dqaccum_test.data(), dqaccum_d, dqaccum_test.size());
    compat::wait_and_throw();
    printf("dQaccum val: ");
    if (is_bhsd) {
        verify<T, V, true>(dq_npy.data<T>(), dqaccum_test.data(), BATCH, NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_QO_PAD, HEAD_SIZE_QK, atol, rtol);
    } else {
        verify<T, V, false>(dq_npy.data<T>(), dqaccum_test.data(), BATCH, NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_QO_PAD, HEAD_SIZE_QK, atol, rtol);
    }

    std::vector<T> dq_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    compat::memcpy<T>(dq_test.data(), dq_d, dq_test.size());
    compat::wait_and_throw();
    printf("dQ val: ");
    verify(dq_npy.data<T>(), dq_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, HEAD_SIZE_QK, atol, rtol);

}
