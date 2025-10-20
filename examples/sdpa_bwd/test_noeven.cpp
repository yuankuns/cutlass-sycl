#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <cassert>
#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
// #include "cnpy.h"

using index_t = uint64_t;
using namespace cute;


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

template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout),
                                     get<1>(layout)),
                         get<2>(layout));
    return l;
}

const int tid = 16;
const int bid = 0;

template<class T, class V, int kBlockM, int kBlockN, int kHeadDim>
void
test_noeven(V * q_ptr, T *v_ptr,
            int q_r_stride,
            int num_head_q, int seq_len_q, int seq_len_q_pad) {
    // The block index for the M dimension.
    const int m_block = BlockIdxX();
    // The block index for the batch.
    const int bidb = BlockIdxZ();
    // The block index for the head.
    const int bidh = BlockIdxY();;
    // The thread index.
    constexpr int kNSGs = 8;
    constexpr int SubgroupSize = 16;


    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * SubgroupSize;

    const index_t q_offset = bidb * num_head_q * seq_len_q_pad * kHeadDim
        + bidh * seq_len_q_pad * kHeadDim + m_block * kBlockM * kHeadDim;
    const index_t v_offset = bidb * num_head_q * seq_len_q * kHeadDim
        + bidh * seq_len_q * kHeadDim + m_block * kBlockM * kHeadDim;

    Shape shapeQ = Shape<Int<kBlockM>, Int<kHeadDim>, _1>{};
    Shape shape2 = make_shape(1, Int <kHeadDim>{}, _1{}); // only load 1
    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(q_r_stride, _1{}, _1{})));
    Tensor mV = make_tensor(make_gmem_ptr(v_ptr + v_offset),
                            make_layout(
                                shape2,
                                make_stride(q_r_stride, _1{}, _1{})));
    constexpr int val_m = 1;
    constexpr int val_n = 1;
    auto tiled_load =
        make_tiled_copy(Copy_Atom<UniversalCopy<V>, V>{},
                        Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>>{},
                        Layout<Shape<Int<val_m>, Int<val_n>>>{});
    auto tiled_save =
        make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                        Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>>{},
                        Layout<Shape<Int<val_m>, Int<val_n>>>{});



    auto thr_copy_load = tiled_load.get_thread_slice(ThreadIdxX());
    auto thr_copy_save = tiled_save.get_thread_slice(ThreadIdxX());
    Tensor thr_tile_load_S = thr_copy_load.partition_S(mQ);
    Tensor thr_tile_save_D = thr_copy_save.partition_D(mV);
    Tensor t32 = make_fragment_like(thr_tile_load_S);
    // make coordination
    Tensor cQ = make_identity_tensor(shapeQ);
    Tensor tcS = thr_copy_load.partition_S(cQ);
    Tensor tcS_row = logical_divide(tcS, Shape<_1>{})(make_coord(0,0,),_,0, 0);
    // end of row coord
    copy(tiled_load, thr_tile_load_S, t32);
    auto t16 = make_tensor_like<T>(t32);
    for (int i = 0; i < size(t16); ++i) {
        t16(i) = static_cast<T>(t32(i));
    }
    // correct
    for (int mi = 0; mi < size<1>(t16); ++mi) {
        if (get<0>(tcS_row(mi)) < seq_len_q) {
            copy(tiled_save, t16(_, mi, _, _), thr_tile_save_D(_,mi,_,_));
        }
    }
}

template<class T, class V, int kBlockM, int kBlockN, int kHeadDim>
void launch_test_noeven(V * q_ptr, T * v_ptr,
                        int num_head_q, int seq_len_q, int seq_len_q_pad,
                        int batch) {
    constexpr int kNSGs = 8;
    constexpr int SubgroupSize = 16;
    const int M_BLOCK = ceil_div(seq_len_q, kBlockM);
    auto dimGrid0 = compat::dim3(size(M_BLOCK), size(num_head_q), size(batch));
    auto dimBlock0 = compat::dim3(size(kNSGs * SubgroupSize), size(1), size(1));
    compat::experimental::launch_properties launch_props0{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    compat::experimental::kernel_properties kernel_props0{
        sycl::ext::oneapi::experimental::sub_group_size<SubgroupSize>};
    compat::experimental::launch_policy policy0{dimGrid0, dimBlock0, launch_props0, kernel_props0};
    auto event0 = compat::experimental::launch<
        test_noeven<T, V, kBlockM, kBlockN, kHeadDim>>(policy0,
                                                       q_ptr, v_ptr,
                                                       kHeadDim, num_head_q,
                                                       seq_len_q, seq_len_q_pad);
    EventManager::getInstance().addEvent(event0);
}

int main(int argc, char **argv) {
    using T = cute::half_t;;
    using V = float;

    int64_t BATCH = 4; // shape.data<int>()[0];
    int64_t NUM_HEAD_Q = 4; // shape.data<int>()[1];
    int64_t NUM_HEAD_KV = 4; // shape.data<int>()[2];
    int64_t SEQ_LEN_QO = 1; // shape.data<int>()[3];
    int64_t SEQ_LEN_KV = 1; // shape.data<int>()[4];
    int64_t PAD = 64;
    constexpr int kBlockN = 32;
    constexpr int kBlockM = 64;
    int64_t SEQ_LEN_QO_PAD = ceil_div(SEQ_LEN_QO, kBlockM) * kBlockM;
    int64_t HEAD_SIZE_QK = 96; // shape.data<int>()[5];
    int64_t HEAD_SIZE_VO = 96; // shape.data<int>()[6];
    bool is_causal = false;// shape.data<int>()[7];
    bool is_bhsd = true; // shape.data<int>()[8];
    std::vector<V> q_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);
    V *q_d = compat::malloc<V>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);
    T *v_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    int64_t k_stride = SEQ_LEN_QO_PAD * HEAD_SIZE_QK;
    int64_t j_stride = HEAD_SIZE_QK;
    // init host value
    for (int k = 0; k < BATCH * NUM_HEAD_Q; ++k) {
        for (int j = 0; j < SEQ_LEN_QO_PAD; ++j) {
            for (int i = 0; i < HEAD_SIZE_QK; ++i) {
                q_h[k * k_stride + j * j_stride + i] = k * 1000 + j * 100 + i;
            }
        }
    }
    // output expect
    // j=0 i=0~95, so should be 0~95,0~95,0~95,0~95 etc.
    // copy to device
    compat::memcpy<V>(q_d, q_h.data(), BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);

    if (HEAD_SIZE_QK == 64) {
        constexpr int kHeadDim = 64;
        launch_test_noeven<T, V, kBlockM, kBlockN, kHeadDim>(
            q_d, v_d,
            NUM_HEAD_Q, SEQ_LEN_QO,
            SEQ_LEN_QO_PAD,
            BATCH);
    } else if (HEAD_SIZE_QK == 96) {
        constexpr int kHeadDim = 96;
        launch_test_noeven<T, V, kBlockM, kBlockN, kHeadDim>(
            q_d, v_d,
            NUM_HEAD_Q, SEQ_LEN_QO,
            SEQ_LEN_QO_PAD,
            BATCH);
    } else if (HEAD_SIZE_QK == 128) {
        constexpr int kHeadDim = 128;
        launch_test_noeven<T, V, kBlockM, kBlockN, kHeadDim>(
            q_d, v_d,
            NUM_HEAD_Q, SEQ_LEN_QO,
            SEQ_LEN_QO_PAD,
            BATCH);
    } else if (HEAD_SIZE_QK == 192) {
        constexpr int kHeadDim = 192;
        launch_test_noeven<T, V, kBlockM, kBlockN, kHeadDim>(
            q_d, v_d,
            NUM_HEAD_Q, SEQ_LEN_QO,
            SEQ_LEN_QO_PAD,
            BATCH);
    } else if (HEAD_SIZE_QK == 256) {
        constexpr int kHeadDim = 256;
        launch_test_noeven<T, V, kBlockM, kBlockN, kHeadDim>(
            q_d, v_d,
            NUM_HEAD_Q, SEQ_LEN_QO,
            SEQ_LEN_QO_PAD,
            BATCH);
    } else {
        assert(false && "only support headdim 64,96,128,192,256");
    }
    compat::wait_and_throw();
    std::vector<T> v_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    compat::memcpy<T>(v_h.data(), v_d, BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    compat::wait_and_throw();
    // print output
    for (int b = 0; b < 2; ++b) {
        for (int h = 0; h < 2; ++h) {
            print("BATCH %d HEAD %d\n", b, h);
            for (int s = 0; s < SEQ_LEN_QO; ++s) {
                print("SEQ %d:\n", s);
                for (int i = 0; i < HEAD_SIZE_QK; ++i) {
                    int64_t idx = b * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK +
                        h * SEQ_LEN_QO * HEAD_SIZE_QK +
                        s * HEAD_SIZE_QK + i;
                    print("%10.7f ", (float)v_h[idx]);
                    if (i % 16 == 15)
                        print("\n");
                }
            }
            print("\n");
        }
    }
}
