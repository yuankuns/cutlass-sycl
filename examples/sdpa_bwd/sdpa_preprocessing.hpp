#pragma once

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
    using ThreadLayout = Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>,
                                Stride<Int<SubgroupSize>, _1>>;
    using ValueLayout = std::conditional_t<
        kHeadDim == 96,
        Layout<Shape<_1, _2>>,
        std::conditional_t<
            kHeadDim == 192,
            Layout<Shape<_1, _4>>,
            Layout<Shape<_1, Int<kHeadDim / SubgroupSize>>>>>;
    using OdOType = cutlass::AlignedArray<DType, size(ValueLayout{})>;
    using OdOAtom = Copy_Atom<UniversalCopy<OdOType>, DType>;
    using dQType = cutlass::AlignedArray<VType, size(ValueLayout{})>;
    using dQAtom = Copy_Atom<UniversalCopy<dQType>, VType>;

    auto tileload_odo = make_tiled_copy(OdOAtom{},
                                        ThreadLayout{},
                                        ValueLayout{});
    auto tileload_dq = make_tiled_copy(dQAtom{},
                                       ThreadLayout{},
                                       ValueLayout{});

    auto thr_load_odo = tileload_odo.get_thread_slice(ThreadIdxX());
    auto thr_load_dq = tileload_dq.get_thread_slice(ThreadIdxX());

    Tensor thr_tile_do_S = thr_load_odo.partition_S(mdO);
    Tensor thr_tile_o_S = thr_load_odo.partition_S(mO);
    Tensor thr_tile_dq_D = thr_load_dq.partition_D(mdQaccum);
    Tensor rdQ = make_fragment_like(thr_tile_dq_D);
    Tensor rdO = make_fragment_like<DType>(rdQ);
    Tensor rO = make_fragment_like<DType>(rdQ);
    Tensor cO = make_identity_tensor(dQ_shape);
    Tensor tcO = thr_load_odo.partition_S(cO);
    Tensor tcO_row = logical_divide(tcO, Shape<_1>{})(make_coord(0, 0), _, 0);
    Layout rdO_layout = rdO.layout();
    Tensor rdO_2d = make_tensor(rdO.data(),
                                make_layout(get<1>(rdO_layout),
                                            make_layout(get<0>(rdO_layout),
                                                        get<2>(rdO_layout))));
    Tensor rO_2d = make_tensor(rO.data(),
                               rdO_2d.layout());

    constexpr int NumValperCol = size<0>(rdO_2d);
    auto smem = compat::local_mem<VType[kNSGs * SubgroupSize * NumValperCol]>();
    auto stensor = make_tensor(make_smem_ptr(smem),
                               make_layout(
                                   Shape<
                                   Int<NumValperCol>,
                                   Int<kNSGs>,
                                   Int<SubgroupSize>>{}));
    clear(rdO_2d);
    clear(rO_2d);
    if constexpr(Is_even_M) {
        copy(tileload_odo, thr_tile_do_S, rdO);
        copy(tileload_odo, thr_tile_o_S, rO);
    } else {
        for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
            if (get<0>(tcO_row(mi)) < param.tail_m) {
                copy(tileload_odo, thr_tile_do_S(_, mi, _), rdO(_, mi, _));
                copy(tileload_odo, thr_tile_o_S(_, mi, _), rO(_, mi, _));
            }
        }
    }
    int sg_group_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
        float accum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(rdO_2d); ++ni) {
            accum = accum + (float)rdO_2d(mi, ni) * (float)rO_2d(mi, ni);
        }
        stensor(mi, sg_group_id, sg_local_id) = accum;
    }
    // sycl::group_barrier(group);
    if (sg_local_id == 0) {
        float accum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NumValperCol; ++mi) {
            float accum = 0.0f;
            // reduce within subgroup
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < SubgroupSize; ++ni) {
                accum += stensor(mi, sg_group_id, ni);
            }
            if constexpr(Is_even_M) {
                mdPsum(get<0>(tcO_row(mi))) = accum;
            } else {
                if (get<0>(tcO_row(mi)) < param.tail_m) {
                    mdPsum(get<0>(tcO_row(mi))) = accum;
                }
            }
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
        Int<kHeadDim>>;
    ShapeQ shapeQ;
    if constexpr (Is_even_M) {
        shapeQ = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
    } else {
        shapeQ = make_shape(param.tail_m, Int<kHeadDim>{});
    }

    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                      make_stride(param.dq_r_stride, _1{})));
    Tensor mdQ = make_tensor(make_gmem_ptr(param.dq_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{})));

    typename T::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_slice(first_thread_in_sg_idx);

    auto tile_dq = tiled_mma_dq.tile_mnk();

    auto tileloaddQ = make_block_2d_copy_C(tiled_mma_dq, mdQaccum);
    auto tilesavedQ = make_block_2d_copy_D(tiled_mma_dq, mdQ);

    auto thr_load_dQ = tileloaddQ.get_slice(first_thread_in_sg_idx);
    auto thr_save_dQ = tilesavedQ.get_slice(first_thread_in_sg_idx);

    Tensor gdQaccum = local_tile(make_identity_tensor(mdQaccum.shape()),
                                 select<0, 1>(tile_dq), make_coord(0,0)); // read dQaccum
    Tensor gdQ = local_tile(make_identity_tensor(mdQ.shape()),
                            select<0, 1>(tile_dq), make_coord(0,0)); // dump dQ
    Tensor tdQgdQaccum = thr_load_dQ.partition_S(gdQaccum); // load from dqaccum
    auto tdQrdQaccum = thr_load_dQ.partition_sg_fragment_D(gdQaccum); // register for dqaccum
    auto tdQrdQ = thr_save_dQ.partition_sg_fragment_S(gdQ); // register for dq
    Tensor tdQgdQ = thr_save_dQ.partition_D(gdQ); // save to dq

    copy(tileloaddQ, tdQgdQaccum, tdQrdQaccum);
    reorder(tdQrdQaccum, tdQrdQ);
    copy(tilesavedQ, tdQrdQ, tdQgdQ);
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
