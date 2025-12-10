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

using namespace cute;

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

template <class T_, int kHeadDim_, int kBlockM_, int kBlockN_, int kNSGs_,
          int AtomLayoutMSdP_ = 2, int AtomLayoutNdKV_ = 2, int AtomLayoutMdQ_ = 2,
          bool is_causal_ = false>
struct FAKernel {
    /*
      Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
      K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
      V BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
      P BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
      O BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
    */
    using DType = T_;
    using VType = float; // accumulation
    using MMA_Atom_ARCH = std::conditional_t<
        std::is_same_v<DType, cutlass::half_t>,
        MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
        MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>>;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kNSGs = kNSGs_;
    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static constexpr int AtomLayoutNdKV = AtomLayoutNdKV_;
    static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
    static constexpr bool is_causal = is_causal_;
    using MMA_Atom2 = XE_DPAS_TT<8, VType, DType>;
    using _K = Int<MMA_Atom2::K * 2>;
    using SubgroupLayoutSdP = Layout<Shape<Int<AtomLayoutMSdP>, Int<kNSGs / AtomLayoutMSdP>, _1>>;
    using SubgroupLayoutSdP2 = Layout<Shape<_1, _8, _1>,
                                      Stride<_8, _1, _0>>;
    using SubgroupLayoutdKV = Layout<Shape<Int<AtomLayoutNdKV>, Int<kNSGs / AtomLayoutNdKV>, _1>>;
    using SubgroupLayoutdKV2 = Layout<Shape<_8, _1, _1>,
                                     Stride<_1, _1, _0>>;
    using SubgroupLayoutdQ = Layout<Shape<Int<AtomLayoutMdQ>, Int<kNSGs / AtomLayoutMdQ>, _1>>;
    using SubgroupLayoutdQ2 = Layout<Shape<_8, _1, _1>,
                                     Stride<_1, _0, _0>>;
    using TileShapeSdP = Tile<Int<kBlockM>, Int<kBlockN>, _16>;
    using TileShapeSdP2 = Layout<Shape<Int<kBlockM>, Int<kBlockN>, _64>>;

    static_assert(size<0>(TileShapeSdP{}) <= kBlockM && "tile size M must be smaller than or equal to kBlockM");
    static_assert(kBlockM % size<0>(TileShapeSdP{}) == 0 && "kBlockM must be dividable by tile size M");
    static_assert(size<1>(TileShapeSdP{}) <= kBlockN && "tile size N must be smaller than or equal to kBlockN");
    static_assert(kBlockN % size<1>(TileShapeSdP{}) == 0 && "kBlockN must be dividable by tile size N ");

    using TileShapedKV = Tile<Int<kBlockN>, Int<16 * kNSGs / AtomLayoutNdKV>, Int<kBlockN>>;
    static_assert(size<0>(TileShapedKV{}) <= kBlockN && "tile size M must be smaller than or equal to kBlockN");
    static_assert(kBlockN % size<0>(TileShapedKV{}) == 0 && "kBlockN must be dividable by tile size M");
    static_assert(size<1>(TileShapedKV{}) <= kHeadDim && "tile size N must be smaller than or equal to kHeadDim");
    static_assert(kHeadDim % size<1>(TileShapedKV{}) == 0 && "kHeadDim must be dividable by tile size N");
    using TileShapedKV2 = Layout<Shape<Int<kBlockN>, Int<kHeadDim>, Int<kBlockM>>>;
    using TileShapedQ = Tile<Int<kBlockM>, Int<16 * kNSGs / AtomLayoutMdQ>, Int<kBlockN>>;
    using TileShapedQ2 = Layout<Shape<Int<kBlockM>, Int<kHeadDim>, _K>>;
    static_assert(size<0>(TileShapedQ{}) <= kBlockM && "tile size M must be smaller than or equal to kBlockM");
    static_assert(kBlockM % size<0>(TileShapedQ{}) == 0 && "kBlockM must dividable by tile size M");
    static_assert(size<1>(TileShapedQ{}) <= kHeadDim && "tile size N must be smaller than or equal to kHeadDim");
    static_assert(kHeadDim % size<1>(TileShapedQ{}) == 0 && "kHeadDim must be dividable by tile size N");

    using TiledMmaSdP = typename TiledMMAHelper<MMA_Atom<MMA_Atom2>,
                                                TileShapeSdP2,
                                                SubgroupLayoutSdP2>::TiledMMA;
    using TiledMmadKV = typename TiledMMAHelper<MMA_Atom<MMA_Atom2>,
                                                TileShapedKV2,
                                                SubgroupLayoutdKV2>::TiledMMA;
    using TiledMmadQ = typename TiledMMAHelper<MMA_Atom<MMA_Atom2>,
                                               TileShapedQ2,
                                               SubgroupLayoutdQ2>::TiledMMA;
    static constexpr auto bP = Int<2>{}; // Pipeline

    using StrideR = cute::tuple<long, cute::C<1>>;
    using StrideC = cute::tuple<cute::C<1>, long>;

    // for load Q and Kt in S=QKt
    using TiledLoadQ = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideR>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                    Layout<Shape<_16,_1>>{}));              // Val layout  16x1
    using TiledLoadKt = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_16,_1>>{}));              // Val layout  16x1

    // for load dO and Vt in dP=dO*Vt
    using TiledLoaddO = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                     Layout<Shape<_16,_1>>{}));              // Val layout  16x1

    using TiledLoadV = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideR>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_16,_1>>{}));              // Val layout  16x1

    // for load Pt and dO in dV=Pt*dO
    using TiledLoadPt = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideC>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 m-major
                                     Layout<Shape<_16,_1>>{})); // // Val layout  8x1
    using TiledLoaddOt = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_V, StrideC>, DType>{}, // should be V here
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_16,_1>>{})); // val layout 16x1

    // for load dP, K and dQ in dQ=dP*K
    using TiledLoaddP = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x8x16_LD_N, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                     Layout<Shape<_8,_1>>{})); // val layout 16x1
    using TiledLoadK = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x8x16_LD_N, StrideC>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_8,_1>>{})); // val layout 16x1

    using TiledLoaddQ = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U32x8x16_LD_N, StrideR>, VType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_8,_1>>{})); // val layout 8x1

    //  for load dPt, Q in dK=dPt*Q
    using TiledLoaddPt = decltype(make_tiled_copy(
                                      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideC>, DType>{},
                                      Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                      Layout<Shape<_16,_1>>{}));              // Val layout  16x1
    using TiledLoadQt = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideC>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_16,_1>>{}));              // Val layout  16x1

    // for save S in S=QKt and P
    using TiledSaveS = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    // for save dP in dP=dO*Vt
    using TiledSavedP = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    // for save dV in dV=Pt*dO
    using TiledSavedV = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_8,_1>>{})); // Val layout  8x1
    // for save dQ in dQ=dP*K
    using TiledSavedQ = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N, StrideR>, VType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_8,_1>>{})); // val layout 8x1
    // for save dK=dPt*Q
    using TiledSavedK = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_8,_1>>{})); // Val layout  8x1

    static constexpr int SubgroupSize = 16;
    static constexpr int smem_size = 0;

    FAKernel() {}
};

using index_t = uint64_t;

template<typename T>
struct Param {
    Param(const T *dO,
          const T *o,
          const T *q,
          const T *k,
          const T *v,
          const float *lse,
          float *odo,
          float *dqaccum,
          T *dq,
          T *dk,
          T *dv,
          T *s,
          T *dp,
          T *pb,
          const float softmax_scale)
        : do_ptr(dO),
          o_ptr(o),
          q_ptr(q),
          k_ptr(k),
          v_ptr(v),
          lse_ptr(lse),
          odo_ptr(odo),
          dqaccum_ptr(dqaccum),
          dq_ptr(dq),
          dk_ptr(dk),
          dv_ptr(dv),
          s_ptr(s),
          dp_ptr(dp),
          pb_ptr(pb),
          scale_softmax(softmax_scale),
          scale_softmax_log2(softmax_scale * M_LOG2E),
          is_bhsd(true) {}
    // read only
    const T *do_ptr;
    const T *o_ptr;
    const T *q_ptr;
    const T *k_ptr;
    const T *v_ptr;
    const float *lse_ptr;
    const float scale_softmax;
    const float scale_softmax_log2;
    // write
    float *odo_ptr;
    float *dqaccum_ptr;
    T *dq_ptr;
    T *dk_ptr;
    T *dv_ptr;
    T *s_ptr;
    T *dp_ptr;
    T *pb_ptr;

    // const dimension
    int batch;
    int num_head_q;
    int num_head_kv;
    int seq_len_q;
    int seq_len_q_pad;
    int seq_len_kv;
    int seq_len_kv_pad;
    int head_dim;
    int n_block;
    int tail_n;
    int m_block;
    int tail_m;
    int num_qh_per_kvh;
    int q_r_stride;
    int q_h_stride;
    int q_b_stride;

    int k_r_stride;
    int k_h_stride;
    int k_b_stride;

    int dk_r_stride;
    int dk_h_stride;
    int dk_b_stride;

    int v_r_stride;
    int v_h_stride;
    int v_b_stride;

    int dv_r_stride;
    int dv_h_stride;
    int dv_b_stride;

    int o_r_stride;
    int o_h_stride;
    int o_b_stride;

    int s_r_stride;
    int s_s_stride;
    int s_b_stride;

    int dq_r_stride;
    int dq_h_stride;
    int dq_b_stride;
    /*
     * input output layout
     * true batch, numhead, seqlen, headsize
     * false batch, seqlen, numhead, headsize
     */
    bool is_bhsd;
};

template<typename T>
struct Boffset {
    Boffset(Param<T> &param_) : param(param_) {}
    index_t q_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.q_b_stride + h_id * param.q_h_stride + s_id * param.q_r_stride;
    }
    index_t k_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.k_b_stride + h_id * param.k_h_stride + s_id * param.k_r_stride;
    }
    index_t v_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.v_b_stride + h_id * param.v_h_stride + s_id * param.v_r_stride;
    }
    index_t dk_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.dk_b_stride + h_id * param.dk_h_stride + s_id * param.dk_r_stride;
    }
    index_t dv_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.dv_b_stride + h_id * param.dv_h_stride + s_id * param.dv_r_stride;
    }
    index_t ps_offset(const index_t b_id, const index_t h_id,
                      const index_t sq_id, const index_t sk_id) {
        return b_id * param.s_b_stride +
            h_id * param.s_s_stride +
            sq_id * param.s_r_stride + sk_id;
    }
    index_t lse_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.seq_len_q * param.num_head_q + h_id * param.seq_len_q + s_id;
    }

    index_t o_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.o_b_stride + h_id * param.o_h_stride + s_id * param.o_r_stride;
    }

    index_t dq_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.dq_b_stride + h_id * param.dq_h_stride + s_id * param.dq_r_stride;
    }
    Param<T> &param;
};

// for debug
template<typename T>
void setup_bhsd_stride(Param<T> &param) {
    param.q_r_stride = param.head_dim;
    param.q_h_stride = param.seq_len_q * param.head_dim;
    param.q_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    // param.dq_r_stride = param.head_dim;
    // param.dq_h_stride = param.seq_len_q * param.head_dim;
    // param.dq_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    param.k_r_stride = param.head_dim;
    param.k_h_stride = param.seq_len_kv * param.head_dim;
    param.k_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dk_r_stride = param.head_dim;
    param.dk_h_stride = param.seq_len_kv * param.head_dim;
    param.dk_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.v_r_stride = param.head_dim;
    param.v_h_stride = param.seq_len_kv * param.head_dim;
    param.v_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dv_r_stride = param.head_dim;
    param.dv_h_stride = param.seq_len_kv * param.head_dim;
    param.dv_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.o_r_stride = param.head_dim;
    param.o_h_stride = param.seq_len_q * param.head_dim;
    param.o_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    // param.do_r_stride = param.head_dim;
    // param.do_h_stride = param.seq_len_q * param.head_dim;
    // param.do_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;
    param.s_r_stride = param.seq_len_kv_pad;
    param.s_s_stride = param.seq_len_q_pad * param.seq_len_kv_pad;
    param.s_b_stride = param.num_head_q * param.seq_len_q_pad * param.seq_len_kv_pad;

    param.dq_r_stride = param.head_dim;
    param.dq_h_stride = param.seq_len_q_pad * param.head_dim;
    param.dq_b_stride = param.num_head_q * param.seq_len_q_pad * param.head_dim;
}

template<typename T>
void setup_bshd_stride(Param<T> &param) {
    param.q_r_stride = param.num_head_q * param.head_dim;
    param.q_h_stride = param.head_dim;
    param.q_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    // param.dq_r_stride = param.head_dim;
    // param.dq_h_stride = param.seq_len_q * param.head_dim;
    // param.dq_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    param.k_r_stride = param.num_head_kv * param.head_dim;
    param.k_h_stride = param.head_dim;
    param.k_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dk_r_stride = param.num_head_q * param.head_dim;
    param.dk_h_stride = param.head_dim;
    param.dk_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.v_r_stride = param.num_head_kv * param.head_dim;
    param.v_h_stride = param.head_dim;
    param.v_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dv_r_stride = param.num_head_q * param.head_dim;
    param.dv_h_stride = param.head_dim;
    param.dv_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.o_r_stride = param.num_head_q * param.head_dim;
    param.o_h_stride = param.head_dim;
    param.o_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    // param.do_r_stride = param.head_dim;
    // param.do_h_stride = param.seq_len_q * param.head_dim;
    // param.do_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;
    param.s_r_stride = param.seq_len_kv_pad;
    param.s_s_stride = param.seq_len_q_pad * param.seq_len_kv_pad;
    param.s_b_stride = param.num_head_q * param.seq_len_q_pad * param.seq_len_kv_pad;

    param.dq_r_stride = param.num_head_q * param.head_dim;
    param.dq_h_stride = param.head_dim;
    param.dq_b_stride = param.num_head_q * param.seq_len_q_pad * param.head_dim;
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
constexpr int bid = 0;

const bool
is_cur_thread() {
    return cute::thread(tid, bid);
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

template <class CVT, class T0, class T1>
CUTLASS_DEVICE auto convert_type(CVT &cvt, T0 &src, T1 &dst) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(src); ++i) {
        dst(i) = cvt(src(i));
    }
    return dst;
}

template<class Engine, class Layout, class TiledMMA>
auto
create_regreuse(Tensor<Engine, Layout> &C,
                TiledMMA const &mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * 16;
    Tensor cC = make_identity_tensor(C.shape());   // (M,N)
    auto wg_tile = mma.tile_mnk();
    Tensor gC = local_tile(cC, select<0, 1>(wg_tile), make_coord(0, 0));       // (BLK_M,BLK_N)
    auto copy_c = make_block_2d_copy_D(mma, C);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto r16 = thr_copy_c.partition_sg_fragment_S(gC);
    return r16;
}

template <class ATensor, class BTensor, class CTensor,
          class DTensor, class ETensor, class TiledMMA>
void
gemm_SdP(ATensor   const& A,         // (M,K)
         BTensor   const& B,         // (N,K)
         CTensor        & C,         // (M,N)
         DTensor        & D,         // (M,N)
         ETensor        & rSdP,         // (M,N)
         TiledMMA const & mma,
         int wg_m,
         int wg_n)
{
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
  auto sg = compat::get_nd_item<1>().get_sub_group();
  auto first_thread_in_sg_idx = sg.get_group_linear_id() * 16;

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  Tensor cC = make_identity_tensor(C.shape());   // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);
  auto copy_d = make_block_2d_copy_D(mma, D);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma    =    mma.get_slice(first_thread_in_sg_idx);
  auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
  auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);
  auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);

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
  // Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));
  auto tCrC = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(wg_tile))); // allocate C fragment storage
  Tensor tCgC = thr_copy_c.partition_D(gC);    /* also matches copy_c's source layout */

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

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;
  /* Clear the accumulators */
  clear(tCrC);

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
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }
  /* Write C to global memory */
  reorder(tCrC, rSdP);
  copy(copy_c, rSdP, tCgC);
  copy(copy_d, rSdP, tCgC);
}

template <class ATensor, class BTensor, class CTensor,
          class TiledMMA>
void
gemm_kernel(ATensor   const& A,         // (M,K)
            BTensor   const& B,         // (N,K)
            CTensor        & C,         // (M,N)
            TiledMMA const & mma,
            int wg_m,
            int wg_n,
            bool debug)
{
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
  auto sg = compat::get_nd_item<1>().get_sub_group();
  auto first_thread_in_sg_idx = sg.get_group_linear_id() * 16;

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  Tensor cC = make_identity_tensor(C.shape());   // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma    =    mma.get_slice(first_thread_in_sg_idx);
  auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
  auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);
  auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);

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
  // Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));
  auto tCrC = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(wg_tile))); // allocate C fragment storage
  Tensor tCgC = thr_copy_c.partition_D(gC);    /* also matches copy_c's source layout */

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

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;
  /* Clear the accumulators */
  clear(tCrC);

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
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }
  /* Write C to global memory */
  auto tCrC16bit = thr_copy_c.partition_sg_fragment_S(gC);
  reorder(tCrC, tCrC16bit);
  copy(copy_c, tCrC16bit, tCgC);
}

template <class ATensor, class BTensor,
          class CTensor, class DTensor,
          class TiledMMA>
void
gemm_dkv(ATensor   const& A,         // (M,K)
         BTensor   const& B,         // (N,K)
         CTensor        & rSdP,         // (M,N)
         DTensor        & acc,         // (M,N)
         TiledMMA const & mma,
         int wg_m,
         int wg_n, bool debug = false)
{
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * 16;
    // auto first_thread_in_sg_idx = ThreadIdxX();
  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  // Tensor cC = make_identity_tensor(C.shape());   // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  // Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  // auto copy_c = make_block_2d_copy_D(mma, C);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma    =    mma.get_slice(first_thread_in_sg_idx);
  auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
  auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);
  // auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);

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
  // Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));
  // auto tCrC = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(wg_tile))); // allocate C fragment storage
  // Tensor tCgC = thr_copy_c.partition_D(gC);    /* also matches copy_c's source layout */

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

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;
  /* Clear the accumulators */
  // clear(tCrC);

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
    // reorder(rSdP, tCrA); //transpose fail here

    /* Accumulate C += A * B */
    gemm(mma, tCrA, tCrB, acc);
    // gemm(mma, tCrA, tCrB, acc);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }
  /* Write C to global memory */
}

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
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
    auto item = compat::get_nd_item<1>().get_group();
    const int local_id = ThreadIdxX();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t v_offset = bofst.v_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t dk_offset = bofst.dk_offset(bidb, bidh, n_block * kBlockN);
    const index_t dv_offset = bofst.dv_offset(bidb, bidh, n_block * kBlockN);
    const index_t o_offset = bofst.o_offset(bidb, bidh, 0);
    // buff offset
    const index_t pb_offset = (bidb * param.num_head_q * param.seq_len_kv_pad * kBlockM
                               + bidh * param.seq_len_kv_pad * kBlockM
                               + n_block * kBlockN * kBlockM) * 2;
    const index_t dsb_offset = pb_offset + kBlockN * kBlockM;

    const index_t s_offset = bofst.ps_offset(bidb, bidh, 0, n_block * kBlockN);

    const auto block_n_dim = tail_n == 0 ? Int<kBlockN>{} : ((tail_n + 1) & ~1);
    using Shape1 = Shape<
        std::conditional_t<Is_even_N, Int<kBlockN>, int>,
        Int<kHeadDim>>;
    using Shape2 = Shape<
        Int <kHeadDim>,
        std::conditional_t<Is_even_N, Int<kBlockN>, int>>;
    auto shapeQ = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapedQ = Shape<Int<kBlockM>, Int<kHeadDim>>{};
    Shape1 shapeKtV;
    Shape2 shapeK;
    if constexpr(Is_even_N) {
        shapeKtV = make_shape(Int<kBlockN>{}, Int<kHeadDim>{});
        shapeK = make_shape(Int<kHeadDim>{}, Int<kBlockN>{});
    } else {
        shapeKtV = make_shape(tail_n, Int<kHeadDim>{});
        shapeK = make_shape(Int<kHeadDim>{}, tail_n);
    }
    auto shapeO = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapeQtOt = make_shape(Int<kHeadDim>{}, kBlockM);
    auto shapeSP = make_shape(kBlockM, block_n_dim);
    auto shapePt = make_shape(block_n_dim, kBlockM);

    Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{})));
    Tensor mKt = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                            make_layout(
                                shapeKtV,
                                make_stride(param.k_r_stride, _1{})));
    Tensor mP = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                            make_layout(
                                shapeSP,
                                make_stride(block_n_dim, _1{})));
    Tensor mPt = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                             make_layout(
                                 shapePt,
                                 make_stride(_1{}, block_n_dim)));
    Tensor mdOt = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                              make_layout(
                                  shapeQtOt,
                                  make_stride(_1{}, param.o_r_stride)));

    Tensor mdV = make_tensor(make_gmem_ptr(param.dv_ptr + dv_offset),
                             make_layout(
                                 shapeKtV,
                                 make_stride(param.dv_r_stride, _1{})));
#ifdef _DEBUG_
    Tensor mS = make_tensor(make_gmem_ptr(param.s_ptr + s_offset), make_layout(
                                shapeSP,
                                make_stride(param.s_r_stride, _1{})));
#endif

    typename Trait::TiledMmaSdP tiled_mma_sdp;
    typename Trait::TiledMmadKV tiled_mma_dkv;


    auto tile_dkv = tiled_mma_dkv.tile_mnk();

    Tensor cdV = make_identity_tensor(mdV.shape());
    Tensor gdV = local_tile(cdV, tile_dkv, make_coord(0,0,0), Step<_1, _1, X>{});  // (BLK_M,BLK_N,k)
    auto thr_mma_dkv = tiled_mma_dkv.get_slice(local_id);
    auto tdVrdV = thr_mma_dkv.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_dkv))); // allocate C fragment storage
    auto copy_dv = make_block_2d_copy_D(tiled_mma_dkv, mdV);
    auto thr_copy_dv = copy_dv.get_slice(local_id);
    Tensor tdVgdV = thr_copy_dv.partition_D(gdV);    /* also matches copy_c's source layout */
    // if (is_cur_thread() and n_block == 0) {
    //     print("tile_dkv:\n");
    //     print(tile_dkv);
    //     print("\ntdKgdK:");
    //     print(tdKgdK);
    //     print("\n");
    //     print(tdKgdK(0));
    //     print("\n");
    // }
    cutlass::NumericConverter<T, float> converter;
    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    const int tail_m = param.seq_len_q % kBlockM;
    clear(tdVrdV);
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        const bool Is_even_M = not ((m_block == max_m_block - 1) and (tail_m != 0));
        if (not Is_even_M) {
            mQ = make_tensor(make_gmem_ptr(mQ.data()),
                             make_layout(
                                 make_shape(tail_m, Int<kHeadDim>{}),
                                 make_stride(param.q_r_stride, _1{})));
            mdOt = make_tensor(make_gmem_ptr(mdOt.data()),
                               make_layout(
                                   make_shape(Int<kHeadDim>{}, tail_m),
                                   make_stride(_1{}, param.o_r_stride)));
#ifdef _DEBUG_
            mS = make_tensor(make_gmem_ptr(mS.data()),
                             make_layout(
                                 make_shape(tail_m, block_n_dim),
                                 make_stride(param.s_r_stride, _1{}))); // debug
#endif
        }
        // P=Q*Kt
        auto rSdP = create_regreuse(mP, tiled_mma_sdp);
        gemm_SdP(mQ, mKt, mS, mP, rSdP, tiled_mma_sdp, 0, 0);
        sycl::group_barrier(group);
        // dV=Pt*dO
        gemm_dkv(mPt, mdOt, rSdP, tdVrdV, tiled_mma_dkv, 0, 0);
        // update ptr/atom copy
        mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
        mdOt.data() = mdOt.data() + int(kBlockM * param.o_r_stride);
#ifdef _DEBUG_
        mS.data() = mS.data() + int(kBlockM * param.s_r_stride); // debug
#endif
    }
    auto tdVrdV16bit = thr_copy_dv.partition_sg_fragment_S(gdV);
    reorder(tdVrdV, tdVrdV16bit);
    copy(copy_dv, tdVrdV16bit, tdVgdV);
}

template<class T>
void
mha_backward_seq(T trait,
                 Param<typename T::DType> param) {
    const int bidb = BlockIdxZ();
    const int bidhq = BlockIdxY();
    const int bidhkv = bidhq / param.num_qh_per_kvh;
    for (int n_block = 0; n_block < param.n_block; ++n_block) {
    if (param.tail_n > 0 and n_block == param.n_block - 1)
        dq_dk_dv_1colblock<false, false>(trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
    else
        dq_dk_dv_1colblock<true, false>(trait, param, bidb, bidhq, bidhkv, n_block);
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
    auto dimGrid1 = compat::dim3(size(1),
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
        mha_backward_seq<decltype(trait)>,
        mhabwdDeviceName<decltype(trait)>>(policy1,
                                           trait,
                                           param);
    EventManager::getInstance().addEvent(event1);
    compat::wait_and_throw();
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
    if (headdim == 128) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 128;
        constexpr int kHeadDim = 128;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 2;
        constexpr int AtomLayoutNdKV = 4;
        constexpr int AtomLayoutMdQ = 4;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        // static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
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
    cnpy::NpyArray p_npy = cnpy::npz_load(data_file, "p");

    // read grad output
    cnpy::NpyArray do_npy = cnpy::npz_load(data_file, "do");

    // read grad reference
    cnpy::NpyArray dq_npy = cnpy::npz_load(data_file, "dq");
    cnpy::NpyArray dk_npy = cnpy::npz_load(data_file, "dk");
    cnpy::NpyArray dv_npy = cnpy::npz_load(data_file, "dv");
    cnpy::NpyArray dp_npy = cnpy::npz_load(data_file, "dp");

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
    // alloc grad output
    T *do_d = compat::malloc<T>(do_npy.num_vals);

    // alloc grad test on device
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

    auto problem_shape = ProblemShapeRegular(BATCH, NUM_HEAD_Q, NUM_HEAD_KV,
                                             SEQ_LEN_QO, SEQ_LEN_KV, HEAD_SIZE_QK, HEAD_SIZE_VO);
    if (is_bhsd) {
            launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, false, true>(
                problem_shape,
                do_d, nullptr,
                q_d, k_d, v_d,
                nullptr, nullptr,
                dqaccum_d, nullptr, dk_d, dv_d,
                s_d, dp_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);
    }
    float atol = 10e-3f;
    float rtol = 10e-3f;

    std::vector<T> p_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    compat::memcpy<T>(p_test.data(), s_d, p_test.size());
    compat::wait_and_throw();
    printf("p val: ");
    verify(p_npy.data<T>(), p_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_QO_PAD, SEQ_LEN_KV, SEQ_LEN_KV_PAD, atol, rtol);

    std::vector<T> dv_test(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_VO);
    compat::memcpy<T>(dv_test.data(), dv_d, dv_test.size());
    compat::wait_and_throw();
    printf("dV val: ");
    verify(dv_npy.data<T>(), dv_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_KV, HEAD_SIZE_VO, atol, rtol);
}
