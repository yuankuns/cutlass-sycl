#include "sdpa_kernel.hpp"
#include "cnpy.h"
#include "sdpa_util.hpp"

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
        constexpr int kBlockN = 64;
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

    // init dqaccum
    compat::fill(dqaccum_d, 0.0f, BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);

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
