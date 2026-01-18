#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <cassert>
#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"

using namespace cute;
template<class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2>
void
save_slm(Tensor<Engine0, Layout0> &s,
         Tensor<Engine1, Layout1> const& g,
         SubgroupTensor<Engine2, Layout2, TVLayout2> const& r) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int local_id = sg.get_local_id();
    for (int i = 0; i < size(g); ++i) {
        auto [mi, ni] = g(i);
        // s(mi, ni + local_id) = r(i);
        auto &sv = *reinterpret_cast<const cute::intel::storage_vector_t<half_t, 32>*>(&r(i));
        int *payload = reinterpret_cast<int *>(&s(mi, ni + local_id));
#ifdef __SYCL_DEVICE_ONLY__
        __asm__(
            "lsc_store.slm (M1,1) flat[%1]:a32 %0:d16u32\n"
            :: "rw" (sv), "rw.u"(payload)
            );
#endif
    }
}

template<typename T, int SubgroupSize, class MTensor, class TiledMMA>
auto
create_reg(MTensor const &C,
           TiledMMA const &tiled_mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * SubgroupSize;
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

void
test_lsc(half_t* pb_ptr) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int SubgroupSize = 16;
    auto smem = compat::local_mem<half_t[kBlockM * kBlockN]>();

    int pb_offset = BlockIdxX() * kBlockN * kBlockM;

    Tensor mPt = make_tensor(make_gmem_ptr(pb_ptr + pb_offset),
                             make_layout(
                                 Shape<Int<kBlockN>, Int<kBlockM>>{},
                                 Stride<Int<kBlockM>, _1>{}));
    Tensor sPt = make_tensor(make_smem_ptr(smem),
                             make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{},
                                         Stride<Int<kBlockM>, _1>{}));

    using TiledMma = typename TiledMMAHelper<
        MMA_Atom<XE_DPAS_TT<8, float, half_t>>,
        Layout<Shape<Int<kBlockN>, Int<kBlockM>, Int<16>>>,
        Layout<Shape<Int<2>, Int<4>, Int<1>>>
        >::TiledMMA;
    TiledMma tiled_mma;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int local_id = sg.get_local_id();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * SubgroupSize;
    auto copy_c = make_block_2d_copy_D(tiled_mma, mPt);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto tile_mnk = tiled_mma.tile_mnk();

    Tensor cC = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0,0));
    Tensor tCgC = thr_copy_c.partition_D(gC);
    auto rPt = create_reg<half_t, SubgroupSize>(mPt, tiled_mma);
    save_slm(sPt, tCgC, rPt);
}

template<class...> class testlscName;

int main() {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int pb_size = kBlockM * kBlockN * 100; // 10 blocks
    constexpr int kNSGs = 8;
    constexpr int SubgroupSize = 16;
    half_t* pb_ptr = compat::malloc<half_t>(pb_size);
    auto dimGrid = compat::dim3(size(10), size(1), size(1));
    auto dimBlock = compat::dim3(size(kNSGs * SubgroupSize), size(1), size(1));

    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(kBlockM * kBlockN * sizeof(half_t)),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<SubgroupSize>};
    compat::experimental::launch_policy policy{dimGrid, dimBlock, launch_props, kernel_props};
    auto event = compat::experimental::launch<
        test_lsc>(policy, pb_ptr);
    EventManager::getInstance().addEvent(event);
    compat::wait_and_throw();
    return 0;
}
