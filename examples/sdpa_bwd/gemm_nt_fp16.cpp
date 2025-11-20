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

template<class T>
void print_t(T r) {
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

template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout),
                                     get<1>(layout)),
                         get<2>(layout));
    return l;
}

constexpr int tid = 17;
constexpr int bid = 0;

template<typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename Tensor4,
         typename Tensor5, typename Tensor6,
         typename Tensor7, typename Tensor8,
         typename TiledMma, typename TileMNK,
         typename TiledCopyA, typename TiledCopyB>
CUTLASS_DEVICE void
gemm_ker(Tensor0 &tCrCmn, Tensor1 &tCrA, Tensor2 &tCrB,
         Tensor3 &tAgAmk, Tensor4 &tArA, Tensor5 &gA,
         Tensor6 &tBgBnk, Tensor7 &tBrB, Tensor8 &gB,
         TiledMma &tiled_mma, TileMNK &tile_mnk,
         TiledCopyA &copy_a, TiledCopyB &copy_b) {
    constexpr int barrier_scope = 2;
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<3>(tAgAmk); ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n <  size<3>(tBgBnk); ++n) {
            auto tCrC = tCrCmn(_, _, _, m, n);
            auto tAgA = tAgAmk(_, _, _, m, _);
            auto tBgB = tBgBnk(_, _, _, n, _);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<3>(tAgA); ++k) {
                barrier_arrive(barrier_scope);
                cute::copy(copy_a, tAgA(_, _, _, k), tArA);
                cute::copy(copy_b, tBgB(_, _, _, k), tBrB);
                cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
                barrier_wait(barrier_scope);
            }
        }
    }
}

template<typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename Tensor4,
         typename Tensor5, typename Tensor6,
         typename TiledMma, typename TileMNK,
         typename TiledCopyA, typename TiledCopyB>
CUTLASS_DEVICE void
gemm_ker(Tensor0 &tCrC, Tensor1 &tCrA, Tensor2 &tCrB,
         Tensor3 &tAgA, Tensor4 &tArA,
         Tensor5 &tBgB, Tensor6 &tBrB,
         TiledMma &tiled_mma, TileMNK &tile_mnk,
         TiledCopyA &copy_a, TiledCopyB &copy_b) {
    constexpr int barrier_scope = 2;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<3>(tAgA); ++k) {
        barrier_arrive(barrier_scope);
        cute::copy(copy_a, tAgA(_, _, _, k), tArA);
        cute::copy(copy_b, tBgB(_, _, _, k), tBrB);
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);
    }
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


template <class CVT, class T0, class T1>
CUTLASS_DEVICE auto convert_type(CVT &cvt, T0 &src, T1 &dst) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(src); ++i) {
        dst(i) = cvt(src(i));
    }
    return dst;
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
    const int local_id = sg.get_local_id();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t v_offset = bofst.v_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t dk_offset = bofst.dk_offset(bidb, bidh, n_block * kBlockN);
    const index_t dv_offset = bofst.dv_offset(bidb, bidh, n_block * kBlockN);
    const index_t o_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, 0);
    const index_t lse_offset = bofst.lse_offset(bidb, bidh, 0);
    // buff offset
    const index_t pb_offset = bidb * param.num_head_q * param.seq_len_kv_pad * kBlockM
        + bidh * param.seq_len_kv_pad * kBlockM + n_block * kBlockN * kBlockM;

    const index_t s_offset = bofst.ps_offset(bidb, bidh, 0, n_block * kBlockN);

    const auto block_n_dim = tail_n == 0 ? Int<kBlockN>{} : tail_n;
    using Shape1 = Shape<
        std::conditional_t<Is_even_N, Int<kBlockN>, int>,
        Int <kHeadDim>, Int<1>>;
    using Shape2 = Shape<
        Int <kHeadDim>,
        std::conditional_t<Is_even_N, Int<kBlockN>, int>,
        Int<1>>;
    auto shapeQ = make_shape(kBlockM, Int<kHeadDim>{}, _1{});
    auto shapedQ = Shape<Int<kBlockM>, Int<kHeadDim>, _1>{};
    Shape1 shapeKtV;
    Shape2 shapeK;
    if constexpr(Is_even_N) {
        shapeKtV = make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, _1{});
        shapeK = make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, _1{});
    } else {
        shapeKtV = make_shape(tail_n, Int<kHeadDim>{}, _1{});
        shapeK = make_shape(Int<kHeadDim>{}, tail_n, _1{});
    }
    auto shapeO = make_shape(kBlockM, Int<kHeadDim>{}, _1{});
    auto shapeQtOt = make_shape(Int<kHeadDim>{}, kBlockM, _1{});


    auto shapeSP = make_shape(kBlockM, block_n_dim, _1{});

    auto shapePt = make_shape(block_n_dim, kBlockM, _1{});

    Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{}, _1{})));
    Tensor mKt = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                            make_layout(
                                shapeKtV,
                                make_stride(param.k_r_stride, _1{}, _1{})));
#ifdef _DEBUG_
    Tensor mS = make_tensor(make_gmem_ptr(param.s_ptr + s_offset), make_layout(
                                shapeSP,
                                make_stride(param.s_r_stride, _1{}, _1{})));
#endif

    auto tile_sdp = typename Trait::TileShapeSdP{};

    auto tileloadQ = typename Trait::TiledLoadQ{mQ};
    auto tileloadKt = typename Trait::TiledLoadKt{mKt};

    auto tilesaveS = typename Trait::TiledSaveS{mS}; // debug

    Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
    Tensor mKtV_coord = cute::get_xe_tensor(shapeKtV);
    Tensor mSP_coord = cute::get_xe_tensor(shapeSP);

    typename Trait::TiledMmaSdP tiled_mma_sdp;

    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);

    Tensor gQ = local_tile(mQ_coord, select<0, 2>(tile_sdp), make_coord(0,_,0));
    Tensor gKtV = local_tile(mKtV_coord, select<1, 2>(tile_sdp), make_coord(0,_,0));

    Tensor gSP = local_tile(mSP_coord, select<0, 1>(tile_sdp), make_coord(0,0,0)); // dump P

    Tensor tSgQ = thr_mma_sdp.partition_A(gQ);
    Tensor tSgKt = thr_mma_sdp.partition_B(gKtV);

    Tensor tPgP = thr_mma_sdp.partition_C(gSP); // save P to internal buffer

    Tensor tSrQ = make_tensor<T>(make_fragment_layout(tileloadQ, tSgQ(_,_,_,0).shape()));
    Tensor tSrKt = make_tensor<T>(make_fragment_layout(tileloadKt, tSgKt(_,_,_,0).shape()));

    ThrCopy thr_copy_q = tileloadQ.get_slice(compat::local_id::x());
    ThrCopy thr_copy_kt = tileloadKt.get_slice(compat::local_id::x());

    // Retile registers for copies
    Tensor tQrQ = thr_copy_q.retile_D(tSrQ);
    Tensor tKtrKt = thr_copy_kt.retile_D(tSrKt);

    // Retile global counting tensors for copies
    Tensor tQgQ = thr_copy_q.retile_S(tSgQ);
    Tensor tKtgKt = thr_copy_kt.retile_S(tSgKt);

    Tensor tSrS = partition_fragment_C(tiled_mma_sdp,
                                       Shape<Int<kBlockM>, Int<kBlockN>>{});

    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    const int tail_m = param.seq_len_q % kBlockM;
    cutlass::NumericConverter<T, float> converter;
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        const bool Is_even_M = not ((m_block == max_m_block - 1) and (tail_m != 0));
        if (not Is_even_M) {
            mQ = make_tensor(make_gmem_ptr(mQ.data()),
                             make_layout(
                                 make_shape(tail_m, Int<kHeadDim>{}, _1{}),
                                 make_stride(param.q_r_stride, _1{}, _1{})));
            mS = make_tensor(make_gmem_ptr(mS.data()),
                             make_layout(
                                 make_shape(tail_m, block_n_dim, _1{}),
                                 make_stride(param.s_r_stride, _1{}, _1{}))); // debug
            tileloadQ = typename Trait::TiledLoadQ{mQ};
            tilesaveS = typename Trait::TiledSaveS{mS};
        }
        clear(tSrS);
        // S=QKt
        gemm_ker(tSrS, tSrQ, tSrKt, tQgQ, tQrQ, tKtgKt, tKtrKt,
                 tiled_mma_sdp, tile_sdp, tileloadQ, tileloadKt);
        auto tSrSl = make_tensor_like<T>(tSrS);
        convert_type(converter, tSrS, tSrSl);
#ifdef _DEBUG_
        if (cute::thread(0, 16) and m_block == 0 and n_block == 16) {
            print("P:\n");
            print_t(tSrSl);
        }
#endif
        copy(tilesaveS, tSrSl, tPgP);
        // mha_save<Is_even_N>(tilesaveS, tSrSl, tPgP); // save P to external tensor for verification

        mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
        mS.data() = mS.data() + int(kBlockM * param.s_r_stride); // debug

        tileloadQ = typename Trait::TiledLoadQ{mQ};
        tilesaveS = typename Trait::TiledSaveS{mS}; // debug
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


template<class...> class mhabwdDeviceName;

template<typename T, class ProblemShape, int kBlockM, int kBlockN,
         int kHeadDim, int kNSGs, int AtomLayoutMSdP, int AtomLayoutNdKV,
         int AtomLayoutMdQ, bool is_causal, bool is_bhsd>
void launch_mha_backward_headdim(ProblemShape problem_shape,
                                 const T *q_d,
                                 const T *k_d,
                                 T *s_d,
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
    T * pbuff = compat::malloc<T>(BATCH * NUM_HEAD_Q * seq_len_kv_pad * kBlockM);
    auto param = Param<T>(nullptr, nullptr, q_d, k_d, nullptr, nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr, s_d, nullptr, pbuff,
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

    auto dimGrid1 = compat::dim3(size(param.n_block),
                                 size(param.num_head_q), size(param.batch));
    assert((param.num_head_q % param.num_head_kv == 0) && "num_head_q must be dividable by num_head_kv");
    assert((param.num_head_q >= param.num_head_kv) && "num_head_q must be bigger than or equal to num_head_kv");
    auto dimBlock1 = compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
    // auto dimBlock = compat::dim3(size(trait.tiled_mma_sdp));

    compat::experimental::launch_properties launch_props1{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
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
                         const T *q_d,
                         const T *k_d,
                         T *s_d,
                         const int seq_len_q_pad,
                         const int seq_len_kv_pad) {
    int headdim = get<5>(problem_shape);
    if (headdim == 128) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 32;
        constexpr int kHeadDim = 128;
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
            q_d, k_d, s_d,
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
    // alloc qkv
    T *q_d = compat::malloc<T>(q_npy.num_vals);
    T *k_d = compat::malloc<T>(k_npy.num_vals);

    // alloc ps
    T *s_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);

    // copy qk
    compat::memcpy<T>(q_d, q_npy.data<T>(), q_npy.num_vals);
    compat::memcpy<T>(k_d, k_npy.data<T>(), k_npy.num_vals);


    auto problem_shape = ProblemShapeRegular(BATCH, NUM_HEAD_Q, NUM_HEAD_KV,
                                             SEQ_LEN_QO, SEQ_LEN_KV, HEAD_SIZE_QK, HEAD_SIZE_VO);
    launch_mha_backward<T, decltype(problem_shape),
                        kBlockM, kBlockN, false, true>(
                            problem_shape,
                            q_d, k_d,
                            s_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);

    float atol = 1e-3f;
    float rtol = 1e-3f;

    std::vector<T> s_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    compat::memcpy<T>(s_test.data(), s_d, s_test.size());
    compat::wait_and_throw();
    printf("S val: ");
    verify(s_npy.data<T>(), s_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, SEQ_LEN_QO_PAD, SEQ_LEN_KV, SEQ_LEN_KV_PAD, atol, rtol);

}
