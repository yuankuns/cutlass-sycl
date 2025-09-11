#pragma once
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>
#include <cute/tensor.hpp>
using namespace cute;

template <class T_, int kHeadDim_, int kBlockM_, int kBlockN_, int kBlockK_, int kNSGs_>
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
    static constexpr int kBlockK = kBlockK_;
    static constexpr int kNSGs = kNSGs_;
    // using SubgroupLayout = Layout<Shape<Int<kNSGs>, _1, _1>, Stride<_1, _1, _1>>;
    static constexpr int AtomLayoutMSdP = 4;
    using SubgroupLayout=Layout<Shape<Int<AtomLayoutMSdP>, Int<kNSGs / AtomLayoutMSdP>, _1>>;
    static_assert(16 *AtomLayoutMSdP == kBlockM);
    static_assert(32 *kNSGs / AtomLayoutMSdP == kBlockN);
    static_assert(kBlockK == 32);
    using TileShapeMSdP = Tile<Int<16 * AtomLayoutMSdP>, Int<32 * kNSGs / AtomLayoutMSdP>, _32>;

    // using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
    // using TileShapeMSdP = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
    using TiledMmaSdP = typename TiledMMAHelper<MMA_Atom_ARCH,
                                                Layout<TileShapeMSdP>,
                                                SubgroupLayout>::TiledMMA;
    static constexpr auto bP = Int<2>{}; // Pipeline

    using StrideR = cute::tuple<long, cute::C<1>>;
    using StrideC = cute::tuple<cute::C<1>, long>;

    using TiledCopyQ = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideR>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                    Layout<Shape<_16,_1>>{}));              // Val layout  16x1
    using TiledCopyK = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideR>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_16,_1>>{}));              // Val layout  16x1

    using TiledCopydO = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                     Layout<Shape<_16,_1>>{}));              // Val layout  16x1

    using TiledCopyV = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideR>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_16,_1>>{}));              // Val layout  16x1
    using TiledCopyS = decltype(make_tiled_copy(
                                    Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    using TiledCopydP = decltype(make_tiled_copy(
                                     Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
                                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                     Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    // static constexpr auto tiled_mma_sdp = TiledMmaSdP{};
    /*
      shape
      Pt BATCH,NUM_HEAD_Q,SEQ_LEN_KV,SEQ_LEN_QO
      dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
      dV BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
      M SEQ_LEN_KV
      N HEAD_SIZE_VO
      K SEQ_LEN_QO
      dV=Pt*dO
    */
    // using CopyPt = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride0>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
    //                     Layout<Shape<_16,_1>>{})); // Val layout  16x1
    // using CopygO = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_32,_2>>{})); // Val layout  32x2);
    // using CopygV = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_8,_1>>{})); // Val layout  8x1
    /*
      dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
      dV BATCH,NUM_HEAD_KV,HEAD_SIZE_VO,SEQ_LEN_KV
      dPs BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
      M SEQ_LEN_QO
      N SEQ_LEN_KV
      K HEAD_SIZE_VO
      dPs=dO*Vt
    */

    // using CopygOA = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride1>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
    //                     Layout<Shape<_32,_2>>{}));              // Val layout  32x2
    // using CopyVt = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride1>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_16,_1>>{})); //Val layout 16x1
    // using CopygP = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    // using CopyP = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_LD_N, Stride1>, DType>{},
    //                     Layout<Shape<_1, _16>>{},
    //                     Layout<Shape<_8, _1>> {}));

    /*
     * dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
     * Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
     * dK BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
     * M SEQ_LEN_KV
     * N HEAD_SIZE_QK
     * K SEQ_LEN_QO
     * dK=dPt*Q
     */
    // copy_pst already defined at line 103
    // using CopyQ = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_32,_2>>{}));              // Val layout  16x1
    // using CopygK = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    /*
     * dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
     * K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
     * dQ BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
     * M SEQ_LEN_QO
     * N HEAD_SIZE_QK
     * K SEQ_LEN_KV
     * dQ=dP*K
     */
    // using CopygPA = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride1>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
    //                     Layout<Shape<_32,_2>>{}));              // Val layout  32x2
    // using CopyK = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_32,_2>>{}));              // Val layout  32x2
    // using CopygQ = decltype(
    //     make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
    //                     Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
    //                     Layout<Shape<_8,_1>>{}));              // Val layout  8x1

    // static constexpr TiledMMA mmaC = TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<decltype(tile_mnk)>,
    //                                                 Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA{};  // 256x128x16 TiledMMA
    // using TiledMma = decltype(mmaC);
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
          const float *odo,
          T *dq,
          T *dk,
          T *dv,
          T *s,
          T *dp,
          const float softmax_scale)
        : do_ptr(dO),
          o_ptr(o),
          q_ptr(q),
          k_ptr(k),
          v_ptr(v),
          lse_ptr(lse),
          odo_ptr(odo),
          dq_ptr(dq),
          dk_ptr(dk),
          dv_ptr(dv),
          s_ptr(s),
          dp_ptr(dp),
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
    const float *odo_ptr;
    const float scale_softmax;
    const float scale_softmax_log2;
    // write
    T *dq_ptr;
    T *dk_ptr;
    T *dv_ptr;
    T *s_ptr;
    T *dp_ptr;

    // const dimension
    int batch;
    int num_head_q;
    int num_head_kv;
    int seq_len_q;
    int seq_len_kv;
    int head_dim;
    int q_r_stride;
    int q_h_stride;
    int q_b_stride;

    int k_r_stride;
    int k_h_stride;
    int k_b_stride;

    int v_r_stride;
    int v_h_stride;
    int v_b_stride;

    int o_r_stride;
    int o_h_stride;
    int o_b_stride;

    int s_r_stride;
    int s_s_stride;
    int s_b_stride;
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

    // param.dk_r_stride = param.head_dim;
    // param.dk_h_stride = param.seq_len_kv * param.head_dim;
    // param.dk_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.v_r_stride = param.head_dim;
    param.v_h_stride = param.seq_len_kv * param.head_dim;
    param.v_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    // param.dv_r_stride = param.head_dim;
    // param.dv_h_stride = param.seq_len_kv * param.head_dim;
    // param.dv_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.o_r_stride = param.head_dim;
    param.o_h_stride = param.seq_len_q * param.head_dim;
    param.o_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    // param.do_r_stride = param.head_dim;
    // param.do_h_stride = param.seq_len_q * param.head_dim;
    // param.do_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;
    param.s_r_stride = param.seq_len_kv;
    param.s_s_stride = param.seq_len_q * param.seq_len_kv;
    param.s_b_stride = param.num_head_q * param.seq_len_q * param.seq_len_kv;
}
