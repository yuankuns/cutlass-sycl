#include <sycl/sycl.hpp>
#include <syclcompat.hpp>
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

template<class T>
void print_t(T t) {
    print(t);
    for (int i = 0; i < size(t); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10.7f ", (float)t(i));
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

constexpr int tid = 0;
constexpr int bid = 16;


template<class T, class ThrMma,
         class YTensor, class CopyY,
         class FragTensor,
         class AuxTensor,
         class SumTensor>
void softmax_bwd_partial_sum(T &trait,
                             ThrMma &thr_mma,
                             YTensor &gP,
                             CopyY &copy_y,
                             FragTensor &tCrC,
                             AuxTensor &tCrCy,
                             SumTensor &sum_row) {
    // copy y
    Tensor tCgy = thr_mma.partition_C(gP);
    copy(copy_y, tCgy, tCrCy);

    // calculate sum of dy*y
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrC); ++i) {
        auto dy_col = tCrC(_, _, i);
        auto y_col = tCrCy(_, _, i);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(dy_col); ++j) {
            dy_col(j) = dy_col(j) * y_col(j);
            sum_row(j) = sum_row(j) + dy_col(j);
        }
    }
}

template<class T,
         class FragTensor, class AuxTensor,
         class STensor>
void softmax_bwd_last(T &trait,
                      FragTensor &tCrC,
                      AuxTensor &tCrCy,
                      STensor &sum_buf) {
    // copy y
    float inv_scale = 1.0f / sqrtf(
        static_cast<float>(trait.HEAD_SIZE_QK));
    for (int i = 0; i < size<2>(tCrC); ++i) {
        auto y_col = tCrCy(_, _, i);
        auto ydy_col = tCrC(_, _, i);
        for (int j = 0; j < size(y_col); ++j) {
            auto sum_val = sum_buf(j);
            ydy_col(j) = (ydy_col(j) - y_col(j) * sum_val) * inv_scale;
        }
    }
}

template<int k_tile = 0, typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename Tensor4,
         typename Tensor5, typename Tensor6,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
CUTLASS_DEVICE void gemm_ker(Tensor0 &tCrC, Tensor1 &tCrA, Tensor2 &tCrB,
              Tensor3 &tAgA, Tensor4 &tArA,
              Tensor5 &tBgB, Tensor6 &tBrB, TiledMma &tiled_mma,
              TiledCopyA &copy_a, TiledCopyB &copy_b,
              ThrCopyA &thr_copy_a, ThrCopyB &thr_copy_b) {
    constexpr int barrier_scope = 2;
    for (int k = 0; k < size<3>(tAgA); ++k) {
        barrier_arrive(barrier_scope);
        cute::copy(copy_a, tAgA(_, _, _, k), tArA);
        cute::copy(copy_b, tBgB(_, _, _, k), tBrB);
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);
    }
}

template<class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void load_1colvec(Tensor0 &reg, Tensor1 &mT, Tensor2 &coord_row) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size(reg); ++mi) {
        reg(mi) = mT(get<0>(coord_row(mi)));
    }
}
template<typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_layout(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_1>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
}

template<class Engine0, class Layout0, class Engine1, class Layout1>
CUTLASS_DEVICE void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, const float scale) {
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

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
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
template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class AccT>
void gemm_kernel(T &trait, ProblemShape &shape_mnkl,
                 ThrMma &thr_mma,
                 typename T::DType const *A, AStride dA, TiledCopyA,
                 typename T::DType const *B, BStride dB, TiledCopyB,
                 AccT &tCrC, const int m_coord, const int n_coord, const int l_coord,
                 bool debug) {

    auto A_shape = select<0,2,3>(shape_mnkl);
    auto B_shape = select<1,2,3>(shape_mnkl);

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_layout(A_shape, dA));
    auto mB = make_tensor(make_gmem_ptr(B), make_layout(B_shape, dB));

    auto copy_a = TiledCopyA{mA};
    auto copy_b = TiledCopyB{mB};

    Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
    Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,l)


    // Get the appropriate blocks for this thread block
    // int m_coord = BlockIdxX();
    // int n_coord = BlockIdxY();
    // int l_coord = BlockIdxZ();
    // auto cta_coord = make_coord(m_coord, n_coord, l_coord);  // (m,n,k)

    Tensor gA = local_tile(mA_coord, select<0, 2>(trait.tile_mnk), make_coord(m_coord, _, l_coord));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_coord, select<1, 2>(trait.tile_mnk), make_coord(n_coord, _, l_coord));  // (BLK_N,BLK_K,k)

    // Partition global counting tensors for MMA
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);

    Tensor tCrA = make_tensor<typename T::DType>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<typename T::DType>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

    ThrCopy thr_copy_a = copy_a.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_b = copy_b.get_slice(syclcompat::local_id::x());

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
    static constexpr auto ATOM_M = get<1>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

    static constexpr auto BLK_M = get<0>(T::tile_mnk);
    static constexpr auto BLK_N = get<1>(T::tile_mnk);
    static constexpr auto BLK_K = get<2>(T::tile_mnk);

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
    // clear(tCrC);

    constexpr int barrier_scope = 2;
    int k_tile_count = ceil_div(get<2>(shape_mnkl), get<2>(trait.tile_mnk));
    auto stages = trait.bP;
    CUTLASS_PRAGMA_UNROLL
    for (; prefetch_k < stages; prefetch_k++) {
        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
        barrier_arrive(barrier_scope);
        // Copy gmem to rmem for the first k_tile
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }

        cute::gemm(trait.tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);

    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class AccT>
void gemm_kernel_bb(T &trait, ProblemShape &shape_mnkl,
                    ThrMma &thr_mma,
                    typename T::DType const *A, AStride dA, TiledCopyA,
                    typename T::DType const *B, BStride dB, TiledCopyB,
                    AccT &tCrC, const int m_coord, const int n_coord, const int lq_coord, const int lk_coord,
                    bool debug) {

    auto A_shape = select<0,2,3>(shape_mnkl);
    auto B_shape = select<1,2,4>(shape_mnkl);

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_layout(A_shape, dA));
    auto mB = make_tensor(make_gmem_ptr(B), make_layout(B_shape, dB));

    auto copy_a = TiledCopyA{mA};
    auto copy_b = TiledCopyB{mB};

    Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
    Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,lh)


    // Get the appropriate blocks for this thread block
    // int m_coord = BlockIdxX();
    // int n_coord = BlockIdxY();
    // int l_coord = BlockIdxZ();
    // auto cta_coord = make_coord(m_coord, n_coord, l_coord);  // (m,n,k)

    Tensor gA = local_tile(mA_coord, select<0, 2>(trait.tile_mnk), make_coord(m_coord, _, lq_coord));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_coord, select<1, 2>(trait.tile_mnk), make_coord(n_coord, _, lk_coord));  // (BLK_N,BLK_K,k)

    // Partition global counting tensors for MMA
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);

    Tensor tCrA = make_tensor<typename T::DType>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<typename T::DType>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

    ThrCopy thr_copy_a = copy_a.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_b = copy_b.get_slice(syclcompat::local_id::x());

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
    static constexpr auto ATOM_M = get<1>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

    static constexpr auto BLK_M = get<0>(T::tile_mnk);
    static constexpr auto BLK_N = get<1>(T::tile_mnk);
    static constexpr auto BLK_K = get<2>(T::tile_mnk);

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
    // clear(tCrC);

    constexpr int barrier_scope = 2;
    int k_tile_count = ceil_div(get<2>(shape_mnkl), get<2>(trait.tile_mnk));
    auto stages = trait.bP;
    CUTLASS_PRAGMA_UNROLL
    for (; prefetch_k < stages; prefetch_k++) {
        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
        barrier_arrive(barrier_scope);
        // Copy gmem to rmem for the first k_tile
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }

        cute::gemm(trait.tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);

    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class CStride, class TiledCopyC>
void gemm(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
          typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
          typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
          typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
          const int m_coord, const int l_coord,
          bool debug) {
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    OPS_tobf16<typename T::DType> op;
    if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
        static constexpr int n_coord = 0;
        Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, l_coord));  // (BLK_M,BLK_N)
        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
        clear(tCrC);
        gemm_kernel(trait, shape_mnkl, thr_mma,
                    A, dA, tiledcopy_A,
                    B, dB, tiledcopy_B,
                    tCrC,
                    m_coord, n_coord, l_coord,
                    debug);
        auto tCrC_bf16 = op(tCrC);
        copy(copy_c, tCrC_bf16, tCgC);
    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class CStride, class TiledCopyC>
void gemm_dq(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
             typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
             typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
             typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
             const int m_coord, const int lq_coord, const int lk_coord,
             bool debug) {
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    OPS_tobf16<typename T::DType> op;
    if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
        static constexpr int n_coord = 0;
        Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, lq_coord));  // (BLK_M,BLK_N)
        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
        clear(tCrC);
        gemm_kernel_bb(trait, shape_mnkl, thr_mma,
                       A, dA, tiledcopy_A,
                       B, dB, tiledcopy_B,
                       tCrC,
                       m_coord, n_coord, lq_coord, lk_coord,
                       debug);
        auto tCrC_bf16 = op(tCrC);
        copy(copy_c, tCrC_bf16, tCgC);
    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class CStride, class TiledCopyC>
void gemm_dkv(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
              typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
              typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
              typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
              const int m_coord, const int lh_coord, const int lb_coord,
              bool debug) {
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    OPS_tobf16<typename T::DType> op;
    if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
        static constexpr int n_coord = 0;
        Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, lh_coord + lb_coord * trait.max_block_n));  // (BLK_M,BLK_N)
        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
        clear(tCrC);
        for (int h_q = lh_coord * trait.block_n_chunk; h_q < (lh_coord + 1) * trait.block_n_chunk; ++h_q) {
            int l_coord = h_q + lb_coord * trait.NUM_HEAD_Q;
            gemm_kernel(trait, shape_mnkl, thr_mma,
                        A, dA, tiledcopy_A,
                        B, dB, tiledcopy_B,
                        tCrC,
                        m_coord, n_coord, l_coord,
                        debug);
        }
        auto tCrC_bf16 = op(tCrC);
        copy(copy_c, tCrC_bf16, tCgC);
    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class CStride, class TiledCopyCaux,
         class FragTensor,class AuxTensor, class SumTensor>
void gemm_dp(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
             typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
             typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
             typename T::DType const *Caux, CStride dC, TiledCopyCaux tiledcopy_Caux,
             FragTensor &tCrC, AuxTensor &tCrCy, SumTensor &sum_row,
             const int m_coord, const int lq_coord, const int lk_coord,
             bool debug) {
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mCaux = make_tensor(make_gmem_ptr(Caux), make_layout(C_shape, dC));
    auto copy_caux = TiledCopyCaux{mCaux};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    static constexpr int n_coord = 0;
    Tensor gCaux = local_tile(mC_coord, select<0, 1>(trait.tile_mnk),
                              make_coord(m_coord, n_coord, lq_coord)); // aux
    gemm_kernel_bb(trait, shape_mnkl, thr_mma,
                A, dA, tiledcopy_A,
                B, dB, tiledcopy_B,
                tCrC,
                m_coord, n_coord, lq_coord, lk_coord, true);
    // softmax partial reduce
    softmax_bwd_partial_sum(trait, thr_mma, gCaux, copy_caux, tCrC, tCrCy, sum_row);
}

template<typename T, class ProblemShape, class ThrMma,
         class CStride, class TiledCopyC,
         class FragTensor,
         class AuxTensor, class SumTensor>
void softmax_bwd(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
                 typename T::DType *C, CStride dC, TiledCopyC tiledcopy_C,
                 FragTensor &tCrC,
                 AuxTensor &tCrCy, SumTensor &sum_row,
                 const int m_coord, const int l_coord) {
    const int n_coord = 0;
    OPS_tobf16<typename T::DType> op;
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk),
                            make_coord(m_coord, n_coord, l_coord)); // dy
    Tensor tCgC = thr_mma.partition_C(gC);
    // y*dy

    softmax_bwd_last(trait, tCrC, tCrCy, sum_row);
    auto tCrC_bf16 = op(tCrC);
    copy(copy_c, tCrC_bf16, tCgC);
}

template<int NUM_SG, class Tensor, class STensor>
void reduce_row(Tensor &t, STensor &sram) {
    auto group = syclcompat::get_nd_item<1>().get_group();
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    const auto sg_local_id = sg.get_local_id();
    const auto sg_group_id = sg.get_group_id();
    const auto sg_group_id_N = sg_group_id % NUM_SG;
    const auto sg_group_id_M = sg_group_id / NUM_SG;
    auto stensor = sram(_, _, sg_group_id_M);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i< size(t); ++i) {
        t(i) = reduce_over_group(sg, t(i), sycl::plus<>());
    }

    if (sg_local_id == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(t); ++i) {
            stensor(i, sg_group_id_N) = t(i);
        }
    }
    // have to wait here
    sycl::group_barrier(group);
    if (sg_local_id == 0) {
        for (int i = 0; i < size(t); ++i) {
            t(i) = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < NUM_SG; ++j) {
                t(i) += stensor(i, j);
            }
        }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(t); ++i) {
        t(i) = sycl::group_broadcast(sg, t(i), 0);
    }
}

template<typename T, typename V>
void copy_tensor(T &src, V &dst) {
    // static_assert(size(src) == size(dst));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(src); ++i) {
        dst(i) = src(i);
    }
}

/*
template<class Trait>
void
dq_dk_dv_1colblock(Trait &trait, Param<typename Trait::DType> &param,
                   const int bidb, const int bidh, const int n_block) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kBlockK = Trait::kBlockK;
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    // auto dA = make_stride(K, Int<1>{}, _0);
    // auto dB = make_stride(Int<1>{}, N, _0);
    auto bofst = Boffset(param);
    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidh, 0);
    const index_t s_offset = bofst.ps_offset(bidb, bidh, 0, 0);

    Shape shapeQ = make_shape(param.seq_len_q, Int<kHeadDim>{}, Int<1>{});
        // Shape<Int<kBlockM>, Int<kHeadDim>>{};
    Shape shapeK = make_shape(param.seq_len_kv, Int<kHeadDim>{}, Int<1>{});
        // Shape<Int<kBlockN>, Int<kHeadDim>>{};
    Shape shapeS = make_shape(param.seq_len_q, param.seq_len_kv, Int<1>{});
        // Shape<Int<kBlockM>, Int<kBlockN>>{};
    Tensor mQ = make_tensor(make_gmem_ptr(param.q + q_offset), make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{}, _0{})));
    Tensor mK = make_tensor(make_gmem_ptr(param.k + k_offset), make_layout(
                                shapeK,
                                make_stride(param.k_r_stride, _1{}, _0{})));
    Tensor mS = make_tensor(make_gmem_ptr(param.s + s_offset), make_layout(
                                shapeS,
                                make_stride(param.s_r_stride, _1{}, _0{})));

    Shape tile_sdp = typename Trait::TileShapeSdP{};

    auto tileloadQ = typename Trait::TiledLoadQ{mQ};
    auto tileloadK = typename Trait::TiledLoadK{mK};
    auto tilesaveS = typename Trait::TiledSaveS{mS};

    Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
    Tensor mK_coord = cute::get_xe_tensor(shapeK);
    Tensor mS_coord = cute::get_xe_tensor(shapeS);


    typename Trait::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);


    cutlass::NumericConverter<T, float> converter;
    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        Tensor gQ = local_tile(mQ_coord, select<0, 2>(tile_sdp), make_coord(m_block, _, 0));
        Tensor gK = local_tile(mK_coord, select<1, 2>(tile_sdp), make_coord(n_block, _, 0));
        Tensor gS = local_tile(mS_coord, select<0, 1>(tile_sdp), make_coord(m_block, n_block, 0)); // debug

        Tensor tSgQ = thr_mma_sdp.partition_A(gQ);
        Tensor tSgK = thr_mma_sdp.partition_B(gK);
        Tensor tSgS = thr_mma_sdp.partition_C(gS); // debug

        Tensor tSrQ = make_tensor<T>(make_fragment_layout(tileloadQ, tSgQ(_,_,_,0).shape()));
        Tensor tSrK = make_tensor<T>(make_fragment_layout(tileloadK, tSgK(_,_,_,0).shape()));

        ThrCopy thr_copy_q = tileloadQ.get_slice(syclcompat::local_id::x());
        ThrCopy thr_copy_k = tileloadK.get_slice(syclcompat::local_id::x());

        // Retile registers for copies
        Tensor tQrQ = thr_copy_q.retile_D(tSrQ);
        Tensor tKrK = thr_copy_k.retile_D(tSrK);

        // Retile global counting tensors for copies
        Tensor tQgQ = thr_copy_q.retile_S(tSgQ);
        Tensor tKgK = thr_copy_k.retile_S(tSgK);

        Tensor tSrS = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrS);

        constexpr int k_tile = ceil_div(kHeadDim, kBlockK);
        gemm_ker<k_tile>(tSrS, tSrQ, tSrK, tQgQ, tQrQ, tKgK, tKrK, tiled_mma_sdp, tileloadQ, tileloadK, thr_copy_q, thr_copy_k);
        // if (m_block == 0 and cute::thread(0, 0)) {
        //     print("tSrS: ");
        //     print_t(tSrS);
        // }
        auto tSrSl = make_tensor_like<T>(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
            tSrSl(i) = converter(tSrS(i));
        }
        copy(tilesaveS, tSrSl, tSgS);
    }

}
*/

template<class Trait>
void
dq_dk_dv_1colblock2(Trait &trait, Param<typename Trait::DType> &param,
                    const int bidb, const int bidh, const int n_block) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kBlockK = Trait::kBlockK;
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto group = syclcompat::get_nd_item<1>().get_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidh, n_block * kBlockN);
    const index_t v_offset = bofst.v_offset(bidb, bidh, n_block * kBlockN);
    const index_t o_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t do_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t dv_offset = bofst.v_offset(bidb, bidh, n_block * kBlockN);
    const index_t lse_offset = bofst.lse_offset(bidb, bidh, 0);
    const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, 0);

    // buff offset
    const index_t pb_offset = bidb * param.num_head_q * kBlockM * kBlockN + bidh * kBlockM * kBlockN;

    const index_t s_offset = bofst.ps_offset(bidb, bidh, 0, n_block * kBlockN);
    const index_t dp_offset = bofst.ps_offset(bidb, bidh, 0, n_block * kBlockN);

    Shape shapeQ = Shape<Int<kBlockM>, Int<kHeadDim>, _1>{};
    Shape shapeK = Shape<Int<kBlockN>, Int<kHeadDim>, _1>{};
    Shape shapeV = Shape<Int<kBlockN>, Int<kHeadDim>, _1>{};
    Shape shapeO = Shape<Int<kBlockM>, Int<kHeadDim>, _1>{};
    Shape shapeOt = Shape<Int<kHeadDim>, Int<kBlockM>, _1>{};

    Shape shapeSP = Shape<Int<kBlockM>, Int<kBlockN>, _1>{};
    Shape shapedP = Shape<Int<kBlockM>, Int<kBlockN>, _1>{};

    Shape shapePt = Shape<Int<kBlockN>, Int<kBlockM>, _1>{};

    Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{}, _0{})));
    Tensor mK = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                            make_layout(
                                shapeK,
                                make_stride(param.k_r_stride, _1{}, _0{})));
    Tensor mV = make_tensor(make_gmem_ptr(param.v_ptr + v_offset),
                            make_layout(
                                shapeV,
                                make_stride(param.v_r_stride, _1{}, _0{})));
    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + do_offset),
                             make_layout(
                                 shapeO,
                                 make_stride(param.o_r_stride, _1{}, _0{})));
    // intermediate buffer
    Tensor mP = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                            make_layout(
                                shapeSP,
                                make_stride(Int<kBlockN>{}, _1{}, _1{})));
    Tensor mPt = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                             make_layout(
                                 shapePt,
                                 make_stride(_1{}, Int<kBlockN>{}, _1{})));
    Tensor mdOt = make_tensor(make_gmem_ptr(param.do_ptr+do_offset),
                              make_layout(
                                  shapeOt,
                                  make_stride(_1{}, param.o_r_stride, _0{})));
    Tensor mLSE = make_tensor(make_gmem_ptr(param.lse_ptr + lse_offset),
                              make_layout(
                                  Shape<Int<kBlockM>>{},
                                  Stride<_1>{}));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + dpsum_offset),
                                make_layout(
                                    Shape<Int<kBlockM>>{},
                                    Stride<_1>{}));

    Tensor mdV = make_tensor(make_gmem_ptr(param.dv_ptr + dv_offset), make_layout(
                             shapeV,
                             make_stride(param.v_r_stride, _1{}, _1{})));
    Tensor mS = make_tensor(make_gmem_ptr(param.s_ptr + s_offset), make_layout(
                                shapeSP,
                                make_stride(param.s_r_stride, _1{}, _0{})));
    Tensor mdP = make_tensor(make_gmem_ptr(param.dp_ptr + s_offset), make_layout(
                                shapeSP,
                                make_stride(param.s_r_stride, _1{}, _0{})));


    Shape tile_sdp = typename Trait::TileShapeSdP{};
    Shape tile_dkv = typename Trait::TileShapedKV{};

    auto tileloadQ = typename Trait::TiledLoadQ{mQ};
    auto tileloadK = typename Trait::TiledLoadK{mK};
    auto tileloaddO = typename Trait::TiledLoaddO{mdO};
    auto tileloadV = typename Trait::TiledLoadV{mV};
    auto tileloadPt = typename Trait::TiledLoadPt{mPt};
    auto tileloaddOt = typename Trait::TiledLoaddOt{mdOt}; // load dO as operand B for dV=Pt*dO

    auto tilesaveP = typename Trait::TiledSaveS{mP}; // to internal buffer
    auto tilesavedV = typename Trait::TiledSavedV{mdV};

    auto tilesaveS = typename Trait::TiledSaveS{mS};
    auto tilesavedP = typename Trait::TiledSavedP{mdP};

    Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
    Tensor mK_coord = cute::get_xe_tensor(shapeK);
    Tensor mV_coord = cute::get_xe_tensor(shapeV);
    Tensor mdO_coord = cute::get_xe_tensor(shapeO);
    Tensor mdOt_coord = cute::get_xe_tensor(shapeOt);
    Tensor mdV_coord = cute::get_xe_tensor(shapeV);

    Tensor mS_coord = cute::get_xe_tensor(shapeSP);
    Tensor mPt_coord = cute::get_xe_tensor(shapePt);
    Tensor mdP_coord = cute::get_xe_tensor(shapedP);

    typename Trait::TiledMmaSdP tiled_mma_sdp;
    typename Trait::TiledMmadKV tiled_mma_dkv;

    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);
    auto thr_mma_dkv = tiled_mma_dkv.get_slice(first_thread_in_sg_idx);

    Tensor gQ = local_tile(mQ_coord, select<0, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gK = local_tile(mK_coord, select<1, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gdO = local_tile(mdO_coord, select<0, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gV = local_tile(mV_coord, select<1, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gPt = local_tile(mPt_coord, select<0, 2>(tile_dkv), make_coord(0, _, 0)); // load Pt
    Tensor gdOt = local_tile(mdOt_coord, select<1, 2>(tile_dkv), make_coord(0, _, 0));

    Tensor gP = local_tile(mS_coord, select<0, 1>(tile_sdp), make_coord(0, 0, 0)); // dump P
    Tensor gdV = local_tile(mdV_coord, select<0, 1>(tile_dkv), make_coord(0, 0, 0));

    Tensor gS = local_tile(mS_coord, select<0, 1>(tile_sdp), make_coord(0, 0, 0)); // debug
    Tensor gdP = local_tile(mdP_coord, select<0, 1>(tile_sdp), make_coord(0, 0, 0)); // debug

    Tensor tSgQ = thr_mma_sdp.partition_A(gQ);
    Tensor tSgK = thr_mma_sdp.partition_B(gK);
    Tensor tdPgdO = thr_mma_sdp.partition_A(gdO);
    Tensor tdPgV = thr_mma_sdp.partition_B(gV);
    Tensor tdVgPt = thr_mma_dkv.partition_A(gPt);
    Tensor tdVgdOt = thr_mma_dkv.partition_B(gdOt);

    Tensor tPgP = thr_mma_sdp.partition_C(gP); // internal buffer
    Tensor tdVgdV = thr_mma_dkv.partition_C(gdV); // save to dv

    Tensor tSgS = thr_mma_sdp.partition_C(gS); // debug
    Tensor tdPgdP = thr_mma_sdp.partition_C(gdP); // debug

    Tensor tSrQ = make_tensor<T>(make_fragment_layout(tileloadQ, tSgQ(_,_,_,0).shape()));
    Tensor tSrK = make_tensor<T>(make_fragment_layout(tileloadK, tSgK(_,_,_,0).shape()));
    Tensor tdPrdO = make_tensor<T>(make_fragment_layout(tileloaddO, tdPgdO(_,_,_,0).shape()));
    Tensor tdPrV = make_tensor<T>(make_fragment_layout(tileloadV, tdPgV(_,_,_,0).shape()));
    Tensor tdVrPt = make_tensor<T>(make_fragment_layout(tileloadPt, tdVgPt(_,_,_,0).shape()));
    Tensor tdVrdOt = make_tensor<T>(make_fragment_layout(tileloaddOt, tdVgdOt(_,_,_,0).shape()));

    ThrCopy thr_copy_q = tileloadQ.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_k = tileloadK.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_do = tileloaddO.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_v = tileloadV.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_pt = tileloadPt.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_dot = tileloaddOt.get_slice(syclcompat::local_id::x());

    // Retile registers for copies
    Tensor tQrQ = thr_copy_q.retile_D(tSrQ);
    Tensor tKrK = thr_copy_k.retile_D(tSrK);
    Tensor tdOrdO = thr_copy_do.retile_D(tdPrdO);
    Tensor tVrV = thr_copy_v.retile_D(tdPrV);
    Tensor tPtrPt = thr_copy_pt.retile_D(tdVrPt);
    Tensor tdOtrdOt = thr_copy_dot.retile_D(tdVrdOt);

    // Retile global counting tensors for copies
    Tensor tQgQ = thr_copy_q.retile_S(tSgQ);
    Tensor tKgK = thr_copy_k.retile_S(tSgK);
    Tensor tdOgdO = thr_copy_do.retile_S(tdPgdO);
    Tensor tVgV = thr_copy_v.retile_S(tdPgV);
    Tensor tPtgPt = thr_copy_pt.retile_S(tdVgPt);
    Tensor tdOtgdOt = thr_copy_dot.retile_S(tdVgdOt);

    Tensor tSrS = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPrdP = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdVrdV = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    // for lse read
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}); // same buffer as accS
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    static_assert(decltype(size<0>(taccScS))::value == 8);
    Tensor taccScS_row = logical_divide(taccScS, Shape<_1>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<V>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    static_assert(size<0>(tSrS) * size<1>(tSrS) == size<0>(lse) && "row of acc and lse not match");
    // misc

    const int max_m_block = param.seq_len_q / kBlockM;
    const int tail_m = param.seq_len_q % kBlockM;

    constexpr int k_tile = ceil_div(kHeadDim, kBlockK);

    // clear accumulator
    clear(tdVrdV);
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        clear(tSrS);
        clear(tdPrdP);
        // if ((m_block == 0) and (cute::thread(0, 0))) {
        //     print(tQrQ);
        //     print("\n");
        //     print(tKrK);
        //     print("\n");
        //     print(tSrS);
        //     print("\n");
        // }
        // S=QKt
        gemm_ker<k_tile>(tSrS, tSrQ, tSrK, tQgQ, tQrQ, tKgK, tKrK, tiled_mma_sdp, tileloadQ, tileloadK, thr_copy_q, thr_copy_k);
        // if (cute::thread(0, 0)) {
        //     print("S(%d, %d):", m_block, n_block);
        //     print_t(tSrS);
        //     print("\n");
        // }
        load_1colvec(lse, mLSE, taccScS_row);
        Tensor dP_sum = make_fragment_like(lse);
        load_1colvec(dP_sum, mdPsum, taccScS_row);
        Tensor scores = make_tensor(tSrS.data(), convert_layout_acc_layout(tSrS.layout()));
        // P=softmax(S,lse)
        scale_apply_exp2(scores, lse, param.scale_softmax_log2);
        // if (cute::thread(0, 0)) {
        //     print("P(%d, %d): ", m_block, n_block);
        //     print_t(tSrS);
        //     print("\n");
        // }
        auto tSrSl = convert_type<T>(tSrS);
        copy(tilesaveP, tSrSl, tPgP); // save P to internal buffers
        copy(tilesaveS, tSrSl, tSgS); // save P to external tensor for verification
        // if (m_block == 0 and cute::thread(0, 0)) {
        //     print("scores");
        //     print_t(scores);
        //     print("\n");
        // }
        // dP=dO*Vt
        gemm_ker<k_tile>(tdPrdP, tdPrdO, tdPrV, tdOgdO, tdOrdO, tVgV, tVrV, tiled_mma_sdp, tileloaddO, tileloadV, thr_copy_do, thr_copy_v);
        Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
        // if (m_block == 0 and cute::thread(0, 0)) {
        //     print("scores/P");
        //     print_t(scores);
        //     print("\ndPsum");
        //     print_t(dP_sum);
        //     print("\ndP");
        //     print_t(dS);
        //     print("\n");
        // }
        // dS=P(dP-sum_row(P))*scale
        softmax_backward(scores, dP_sum, dS, param.scale_softmax);
        // if (m_block == 0  and cute::thread(0, 0)) {
        //     print("dS");
        //     print_t(dS);
        //     print("\n");
        // }

        cute::copy(tileloadPt, tPtgPt(_, _, _, 0), tdVrPt);
        cute::copy(tileloaddOt, tdOtgdOt(_, _, _, 0), tdVrdOt);
        // if (cute::thread(0, 0)) { //
        //     // print("P");
        //     // print_t(tSrS);
        //     print("\nPt(%d, %d):", m_block, n_block);
        //     print_t(tdVrPt);
        //     print("\ndOt(%d, %d):", m_block, n_block);
        //     print_t(tdVrdOt);
        //     print("\n size k: ");
        //     print(size<3>(tPtgPt));
        //     print("\n");
        // }
        auto tdPrdPl = convert_type<T>(tdPrdP);
        copy(tilesavedP, tdPrdPl, tdPgdP); // save dP to external tensor for verification

        // dV=Pt*dO
        gemm_ker(tdVrdV, tdVrPt, tdVrdOt, tPtgPt, tPtrPt, tdOtgdOt, tdOtrdOt, tiled_mma_dkv, tileloadPt, tileloaddOt, thr_copy_pt, thr_copy_dot);
        // if (cute::thread(0, 0)) {
        //     print("\ndV: ");
        //     print_t(tdVrdV);
        // }
        // update ptr/atom copy
        mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
        mdO.data() = mdO.data() + int(kBlockM * param.o_r_stride);
        mdOt.data() = mdOt.data() + int(kBlockM * param.o_r_stride);
        mS.data() = mS.data() + int(kBlockM * param.s_r_stride); // debug
        mdP.data() = mdP.data() + int(kBlockM * param.s_r_stride); // debug
        mLSE.data() = mLSE.data() + int(kBlockM);
        mdPsum.data() = mdPsum.data() + int(kBlockM);

        tileloadQ = typename Trait::TiledLoadQ{mQ};
        tileloaddO = typename Trait::TiledLoaddO{mdO};
        tileloaddOt = typename Trait::TiledLoaddOt{mdOt};

        tilesaveS = typename Trait::TiledSaveS{mS}; // debug
        tilesavedP = typename Trait::TiledSaveS{mdP}; // debug
    }
    int m_block = max_m_block;
    // tail case
    if (tail_m > 0) {
        const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);
        const index_t k_offset = bofst.k_offset(bidb, bidh, n_block * kBlockN);
        const index_t do_offset = bofst.o_offset(bidb, bidh, m_block * kBlockM);
        const index_t s_offset = bofst.ps_offset(bidb, bidh, m_block * kBlockM, n_block * kBlockN);
        const index_t lse_offset = bofst.lse_offset(bidb, bidh, m_block * kBlockM);

        Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                                make_layout(
                                    make_shape(tail_m, Int<kHeadDim>{}, _1{}),
                                    make_stride(param.q_r_stride, _1{}, _0{})));
        Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + do_offset),
                                 make_layout(
                                     make_shape(tail_m, Int<kHeadDim>{}, _1{}),
                                     make_stride(param.o_r_stride, _1{}, _0{})));
        Tensor mdOt = make_tensor(make_gmem_ptr(param.do_ptr+do_offset),
                                  make_layout(
                                      make_shape(Int<kHeadDim>{}, tail_m, _1{}),
                                      make_stride(_1{}, param.o_r_stride, _0{})));
        Tensor mK = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                                make_layout(
                                    shapeK,
                                    make_stride(param.k_r_stride, _1{}, _0{})));
        Tensor mS = make_tensor(make_gmem_ptr(param.s_ptr + s_offset),
                                make_layout(
                                    make_shape(tail_m, Int<kBlockN>{}, _1{}),
                                    make_stride(param.s_r_stride, _1{}, _0{}))); // debug
        Tensor mdP = make_tensor(make_gmem_ptr(param.dp_ptr + s_offset),
                                make_layout(
                                    make_shape(tail_m, Int<kBlockN>{}, _1{}),
                                    make_stride(param.s_r_stride, _1{}, _0{}))); // debug

        auto tileloadQ = typename Trait::TiledLoadQ{mQ};
        auto tileloadK = typename Trait::TiledLoadK{mK};
        auto tileloaddO = typename Trait::TiledLoaddO{mdO};
        auto tileloaddOt = typename Trait::TiledLoaddOt{mdOt};
        auto tilesaveS = typename Trait::TiledSaveS{mS};
        auto tilesavedP = typename Trait::TiledSavedP{mdP};

        clear(tSrS);
        clear(tdPrdP);
        // S=QKt
        gemm_ker<k_tile>(tSrS, tSrQ, tSrK, tQgQ, tQrQ, tKgK, tKrK, tiled_mma_sdp, tileloadQ, tileloadK, thr_copy_q, thr_copy_k);
        load_1colvec(lse, mLSE, taccScS_row);
        Tensor dP_sum = make_fragment_like(lse);
        load_1colvec(dP_sum, mdPsum, taccScS_row);
        Tensor scores = make_tensor(tSrS.data(), convert_layout_acc_layout(tSrS.layout()));
        // P=softmax(S,lse)
        scale_apply_exp2(scores, lse, param.scale_softmax_log2);
        auto tSrSl = convert_type<T>(tSrS);
        copy(tilesaveP, tSrSl, tPgP); // save P to internal buffers
        copy(tilesaveS, tSrSl, tSgS); // save P to external tensor for verification
        // dP=dO*Vt
        gemm_ker<k_tile>(tdPrdP, tdPrdO, tdPrV, tdOgdO, tdOrdO, tVgV, tVrV, tiled_mma_sdp, tileloaddO, tileloadV, thr_copy_do, thr_copy_v);
        Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
        // dS=P(dP-sum_row(P))*scale
        softmax_backward(scores, dP_sum, dS, param.scale_softmax);
        auto tdPrdPl = convert_type<T>(tdPrdP);
        copy(tilesavedP, tdPrdPl, tdPgdP);
        // dV=Pt*dO
        gemm_ker(tdVrdV, tdVrPt, tdVrdOt, tPtgPt, tPtrPt, tdOtgdOt, tdOtrdOt, tiled_mma_dkv, tileloadPt, tileloaddOt, thr_copy_pt, thr_copy_dot);
        // gemm_ker(tdVrdV, tdVrPt, tdVrdOt, tPtgPt, tPtrPt, tdOtgdOt, tdOtrdOt, tiled_mma_dkv, tileloadPt, tileloaddOt, thr_copy_pt, thr_copy_dot);
        // if (cute::thread(0, 0)) {
        //     print("\nfinal dV: ");
        //     print_t(tdVrdV);
        // }
    }
    auto tdVrdVl = convert_type<T>(tdVrdV);
    copy(tilesavedV, tdVrdVl, tdVgdV);
}

/*
template<class Trait>
void
dq_dk_dv_1colblock3(Trait &trait, Param<typename Trait::DType> &param,
                    const int bidb, const int bidh, const int n_block) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kBlockK = Trait::kBlockK;
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    // auto dA = make_stride(K, Int<1>{}, _0);
    // auto dB = make_stride(Int<1>{}, N, _0);
    auto bofst = Boffset(param);
    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidh, 0);
    const index_t s_offset = bofst.ps_offset(bidb, bidh, 0, 0);

    Shape shapeQ = make_shape(param.seq_len_q, Int<kHeadDim>{}, Int<1>{});
        // Shape<Int<kBlockM>, Int<kHeadDim>>{};
    Shape shapeK = make_shape(param.seq_len_kv, Int<kHeadDim>{}, Int<1>{});
        // Shape<Int<kBlockN>, Int<kHeadDim>>{};
    Shape shapeSP = make_shape(param.seq_len_q, param.seq_len_kv, Int<1>{});
        // Shape<Int<kBlockM>, Int<kBlockN>>{};
    Tensor mQ = make_tensor(make_gmem_ptr(param.q + q_offset), make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{}, _0{})));
    Tensor mK = make_tensor(make_gmem_ptr(param.k + k_offset), make_layout(
                                shapeK,
                                make_stride(param.k_r_stride, _1{}, _0{})));
    Tensor mS = make_tensor(make_gmem_ptr(param.s + s_offset), make_layout(
                                shapeSP,
                                make_stride(param.s_r_stride, _1{}, _0{})));

    Shape tile_sdp = typename Trait::TileShapeSdP{};

    auto tileloadQ = typename Trait::TiledLoadQ{mQ};
    auto tileloadK = typename Trait::TiledLoadK{mK};
    auto tilesaveS = typename Trait::TiledSaveS{mS};

    Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
    Tensor mK_coord = cute::get_xe_tensor(shapeK);
    Tensor mS_coord = cute::get_xe_tensor(shapeSP);


    typename Trait::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);


    Tensor gQ = local_tile(mQ_coord, select<0, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gK = local_tile(mK_coord, select<1, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gS = local_tile(mS_coord, select<0, 1>(tile_sdp), make_coord(0, 0, 0)); // debug

    Tensor tSgQ = thr_mma_sdp.partition_A(gQ);
    Tensor tSgK = thr_mma_sdp.partition_B(gK);
    Tensor tSgS = thr_mma_sdp.partition_C(gS); // debug

    Tensor tSrQ = make_tensor<T>(make_fragment_layout(tileloadQ, tSgQ(_,_,_,0).shape()));
    Tensor tSrK = make_tensor<T>(make_fragment_layout(tileloadK, tSgK(_,_,_,0).shape()));

    ThrCopy thr_copy_q = tileloadQ.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_k = tileloadK.get_slice(syclcompat::local_id::x());

    // Retile registers for copies
    Tensor tQrQ = thr_copy_q.retile_D(tSrQ);
    Tensor tKrK = thr_copy_k.retile_D(tSrK);

    // Retile global counting tensors for copies
    Tensor tQgQ = thr_copy_q.retile_S(tSgQ);
    Tensor tKgK = thr_copy_k.retile_S(tSgK);

    Tensor tSrS = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});

    cutlass::NumericConverter<T, float> converter;
    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    constexpr int k_tile = ceil_div(kHeadDim, kBlockK);

    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        gQ = local_tile(mQ_coord, select<0, 2>(tile_sdp), make_coord(m_block, _, 0));
        gK = local_tile(mK_coord, select<1, 2>(tile_sdp), make_coord(n_block, _, 0));
        gS = local_tile(mS_coord, select<0, 1>(tile_sdp), make_coord(m_block, n_block, 0)); // debug

        tSgQ = thr_mma_sdp.partition_A(gQ);
        tSgK = thr_mma_sdp.partition_B(gK);
        tSgS = thr_mma_sdp.partition_C(gS); // debug

        thr_copy_q = tileloadQ.get_slice(syclcompat::local_id::x());
        thr_copy_k = tileloadK.get_slice(syclcompat::local_id::x());

        tQrQ = thr_copy_q.retile_D(tSrQ);
        tKrK = thr_copy_k.retile_D(tSrK);

        tQgQ = thr_copy_q.retile_S(tSgQ);
        tKgK = thr_copy_k.retile_S(tSgK);
        clear(tSrS);

        gemm_ker<k_tile>(tSrS, tSrQ, tSrK, tQgQ, tQrQ, tKgK, tKrK, tiled_mma_sdp, tileloadQ, tileloadK, thr_copy_q, thr_copy_k);
        // if (m_block == 0 and cute::thread(0, 0)) {
        //     print("tSrS: ");
        //     print_t(tSrS);
        // }
        auto tSrSl = make_tensor_like<T>(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
            tSrSl(i) = converter(tSrS(i));
        }
        copy(tilesaveS, tSrSl, tSgS);
    }

}
*/

template<class T>
void
mha_backward(T trait,
             Param<typename T::DType> param) {
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    // auto thr_mma = trait.tiled_mma.get_slice(first_thread_in_sg_idx);
    // std::is_same_v<decltype(copy_Pst), int>;
    OPS_tobf16<typename T::DType> op;
    const int bidb = BlockIdxZ();
    const int bidh = BlockIdxY();
    const int max_n_block = ceil_div(param.seq_len_kv, trait.kBlockN);
    for (int n_block = 0; n_block < max_n_block; ++n_block)
        dq_dk_dv_1colblock2(trait, param, bidb, bidh, n_block);
}

template<typename T, class ProblemShape>
void launch_mha_backward(ProblemShape problem_shape,
                         const T *do_d,
                         const T *o_d,
                         const T *q_d,
                         const T *k_d,
                         const T *v_d,
                         const float *lse_d,
                         const float *odo_d,
                         T *dq_d,
                         T *dk_d,
                         T *dv_d,
                         T *s_d,
                         T *dp_d) {
    // skip if head != 128
    if (get<5>(problem_shape) != 128)
        return;
    constexpr int numSGs = 8;
    constexpr int kHeadDim = 128;
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kBlockK = 32;
    auto trait = FAKernel<T, kHeadDim, kBlockM, kBlockN, kBlockK, numSGs>();
    T * pbuff = syclcompat::malloc<T>(get<0>(problem_shape) * get<1>(problem_shape) * kBlockM * kBlockN);
    auto param = Param<T>(do_d, o_d, q_d, k_d, v_d, lse_d, odo_d,
                          dq_d, dk_d, dv_d, s_d, dp_d, pbuff,
                          1 / sqrt(static_cast<float>(kHeadDim)));
    param.batch = get<0>(problem_shape);
    param.num_head_q = get<1>(problem_shape);
    param.num_head_kv = get<2>(problem_shape);
    param.seq_len_q = get<3>(problem_shape);
    param.seq_len_kv = get<4>(problem_shape);
    param.head_dim = kHeadDim;
    setup_bhsd_stride(param);
    auto dimGrid = syclcompat::dim3(size(1), size(param.num_head_q), size(param.batch));
    assert((trait.num_head_q % trait.num_head_kv == 0) && "num_head_q must be dividable by num_head_kv");
    assert((trait.num_head_q >= trait.num_head_kv) && "num_head_q must be bigger than or equal to num_head_kv");
    auto dimBlock = syclcompat::dim3(size(numSGs * trait.SubgroupSize), size(1), size(1));
    // auto dimBlock = syclcompat::dim3(size(trait.tiled_mma_sdp));

    syclcompat::experimental::launch_properties launch_props{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    syclcompat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    syclcompat::experimental::launch_policy policy{dimGrid, dimBlock, launch_props, kernel_props};
    auto event = syclcompat::experimental::launch<
        mha_backward<decltype(trait)>>(policy,
                                       trait,
                                       param);
    EventManager::getInstance().addEvent(event);
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

    // alloc qkv
    T *q_d = syclcompat::malloc<T>(q_npy.num_vals);
    T *k_d = syclcompat::malloc<T>(k_npy.num_vals);
    T *v_d = syclcompat::malloc<T>(v_npy.num_vals);

    // alloc ps
    T *p_d = syclcompat::malloc<T>(p_npy.num_vals);
    T *s_d = syclcompat::malloc<T>(s_npy.num_vals);

    // alloc lse, odo
    V *lse_d = syclcompat::malloc<V>(lse_npy.num_vals);
    V *odo_d = syclcompat::malloc<V>(odo_npy.num_vals);

    // alloc grad output
    T *do_d = syclcompat::malloc<T>(do_npy.num_vals);
    T *o_d = syclcompat::malloc<T>(o_npy.num_vals);

    // alloc grad test on device
    T *dq_d = syclcompat::malloc<T>(dq_npy.num_vals);
    T *dk_d = syclcompat::malloc<T>(dk_npy.num_vals);
    T *dv_d = syclcompat::malloc<T>(dv_npy.num_vals);
    T *dp_d = syclcompat::malloc<T>(dp_npy.num_vals);

    // copy qkv
    syclcompat::memcpy<T>(q_d, q_npy.data<T>(), q_npy.num_vals);
    syclcompat::memcpy<T>(k_d, k_npy.data<T>(), k_npy.num_vals);
    syclcompat::memcpy<T>(v_d, v_npy.data<T>(), v_npy.num_vals);

    // copy grad output
    syclcompat::memcpy<T>(do_d, do_npy.data<T>(), do_npy.num_vals);
    syclcompat::memcpy<T>(o_d, o_npy.data<T>(), o_npy.num_vals);

    // copy lse
    syclcompat::memcpy<V>(lse_d, lse_npy.data<V>(), lse_npy.num_vals);

    // copy odo
    syclcompat::memcpy<V>(odo_d, odo_npy.data<V>(), odo_npy.num_vals);

    int64_t BATCH = shape.data<int>()[0];
    int64_t NUM_HEAD_Q = shape.data<int>()[1];
    int64_t NUM_HEAD_KV = shape.data<int>()[2];
    int64_t SEQ_LEN_QO = shape.data<int>()[3];
    int64_t SEQ_LEN_KV = shape.data<int>()[4];
    int64_t HEAD_SIZE_QK = shape.data<int>()[5];
    int64_t HEAD_SIZE_VO = shape.data<int>()[6];
    bool is_causal = shape.data<int>()[7];
    printf("batch %d nh_q %d nh_k %d sq_q %d sq_k %d hd_q %d hd_v %d\n", BATCH, NUM_HEAD_Q, NUM_HEAD_KV, SEQ_LEN_QO, SEQ_LEN_KV, HEAD_SIZE_QK, HEAD_SIZE_VO);
    // read_args(argc, argv, 1, &BATCH);
    // read_args(argc, argv, 2, &NUM_HEAD_Q);
    // read_args(argc, argv, 3, &NUM_HEAD_KV);
    // read_args(argc, argv, 4, &SEQ_LEN_QO);
    // read_args(argc, argv, 5, &SEQ_LEN_KV);
    // read_args(argc, argv, 6, &HEAD_SIZE_QK);
    // read_args(argc, argv, 7, &HEAD_SIZE_VO);

    auto problem_shape = ProblemShapeRegular(BATCH, NUM_HEAD_Q, NUM_HEAD_KV, SEQ_LEN_QO, SEQ_LEN_KV, HEAD_SIZE_QK, HEAD_SIZE_VO);
    launch_mha_backward<T, decltype(problem_shape)>(
        problem_shape,
        do_d, o_d,
        q_d, k_d, v_d,
        lse_d, odo_d,
        dq_d, dk_d, dv_d,
        s_d, dp_d);

    float atol = 1e-3f;
    float rtol = 1e-3f;
    std::vector<T> s_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * SEQ_LEN_KV);
    syclcompat::memcpy<T>(s_test.data(), s_d, s_test.size());
    syclcompat::wait_and_throw();
    printf("P val\n");
    verify(p_npy.data<T>(), s_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_KV, atol, rtol);
    std::vector<T> dp_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * SEQ_LEN_KV);
    syclcompat::memcpy<T>(dp_test.data(), dp_d, dp_test.size());
    syclcompat::wait_and_throw();
    printf("dS val\n");
    verify(ds_npy.data<T>(), dp_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_KV, atol, rtol);
    // for (int m = 0; m < 4; ++m) {
    //     for (int n = 0; n < 64; ++n) {
    //         if (n % 16 == 0)
    //             printf("\n(%03d,%03d): ", m, n);
    //         printf("%7.4f ", (float)s_test[m * SEQ_LEN_KV + n], ());
    //     }
    //     printf("\n");
    // }
    syclcompat::wait_and_throw();
    std::vector<T> dv_test(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_VO);
    syclcompat::memcpy<T>(dv_test.data(), dv_d, dv_test.size());
    printf("dV val\n");
    verify(dv_npy.data<T>(), dv_test.data(), BATCH * NUM_HEAD_KV, SEQ_LEN_KV, HEAD_SIZE_VO, atol, rtol);
    // syclcompat::wait_and_throw();
    // std::vector<T> dps_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * SEQ_LEN_KV);
    // syclcompat::memcpy<T>(dps_test.data(), dps_d, dps_test.size());
    // printf("verify dPs: ");
    // verify(dps_npy.data<T>(), dps_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_KV, atol, rtol);
    // syclcompat::wait_and_throw();
    // std::vector<T> dk_test(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_QK);
    // syclcompat::memcpy<T>(dk_test.data(), dk_d, dk_test.size());
    // printf("verify dK: ");
    // verify(dk_npy.data<T>(), dk_test.data(), BATCH * NUM_HEAD_KV, SEQ_LEN_KV, HEAD_SIZE_QK, atol, rtol);

    // syclcompat::wait_and_throw();
    // std::vector<T> dq_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    // syclcompat::memcpy<T>(dq_test.data(), dq_d, dq_test.size());
    // printf("verify dQ: ");
    // verify(dq_npy.data<T>(), dq_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, HEAD_SIZE_QK, atol, rtol);
}
