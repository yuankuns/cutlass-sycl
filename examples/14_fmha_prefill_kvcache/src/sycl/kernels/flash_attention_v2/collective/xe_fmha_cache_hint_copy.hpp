/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

/*! \file
    \brief Block-2D copy atoms that carry an explicit LSC cache hint.

    Motivation. At head_dim=512 this FMHA prefill kernel is cache-bandwidth bound on
    GEMM1's operand re-reads, and the score workspace round-trip competes for the same
    L1. The obvious lever -- tell the score store not to allocate in L1, so it stops
    evicting Q/K -- was previously believed to be out of reach from kernel code, on the
    grounds that only the *legacy* SPIRV copy path (copy_xe_legacy_spirv.hpp) exposes
    CacheControl and the modern block-2D atoms have no cacheopts plumbing.

    That reading was wrong in a useful way. The modern atoms in
    cute/arch/copy_xe_2d.hpp are themselves plain inline asm emitting LSC mnemonics,
    and an LSC cache hint is nothing but a mnemonic suffix -- XE_PREFETCH_2D already
    hardcodes `.ca.ca`. So the hint is reachable without touching any shared header:
    declare copy ops here that emit the same instruction with a different suffix, and
    opt them into CuTe's machinery by specializing the two hooks it keys on
    (is_xe_block_2d_atom and Copy_Traits). Both are templates, so specializing them for
    *our* op types is an additive, kernel-local change.

    Suffix is `.<L1>.<L3>`, applied to the load/store as `lsc_*_block2d.ugm.<L1>.<L3>`:
      uc = uncached (do not allocate)     ca = cached          st = streaming
      wt = write-through                  wb = write-back      rw/none = default
    The defaults below are:
      - store `.uc.wb`: score writes bypass L1 but still land in L2, so the reader
        launch can still hit them there instead of going to DRAM.
      - load  `.st.ca`: score reads are consumed exactly once, so marking them
        streaming keeps them from evicting the Q/K lines we do want resident.
    Both are overridable per build via FMHA_PREFILL_SCORE_{LOAD,STORE}_HINT.
*/

#pragma once

#include "cute/arch/copy_xe_2d.hpp"
#include "cute/atom/copy_traits_xe_2d.hpp"

// The hint suffixes, as bare tokens (`st.ca`, `uc.wb`, `uc.uc`, ...). The mnemonic is built
// by token pasting, so a typo is a *compile* error rather than a silently unhinted
// instruction -- which matters, because the whole variant is unmeasurable if the hint gets
// dropped. Verify with:
//   strings <binary> | grep -oE "lsc_(load|store)_block2d\.ugm(\.[a-z]{2}\.[a-z]{2})?"
// One suffix serves every hinted load here (score loads and, when enabled, the K/V VNNI
// loads); the store has its own. Keeping them as one knob each is deliberate -- what is
// being measured is which *surface* gets hinted, selected by the mainloop's enable flags.
#ifndef FMHA_PREFILL_SCORE_LOAD_HINT
#define FMHA_PREFILL_SCORE_LOAD_HINT st.ca
#endif
#ifndef FMHA_PREFILL_SCORE_STORE_HINT
#define FMHA_PREFILL_SCORE_STORE_HINT uc.wb
#endif

#define FMHA_LSC_STR_(x) #x
#define FMHA_LSC_STR(x) FMHA_LSC_STR_(x)
#define FMHA_LSC_LOAD_MNEMONIC \
  "lsc_load_block2d.ugm." FMHA_LSC_STR(FMHA_PREFILL_SCORE_LOAD_HINT) " (M1, 1)  %0:d%2.%3x%4x%5nn flat[%1+(0,0)]"
#define FMHA_LSC_LOAD_VNNI_MNEMONIC \
  "lsc_load_block2d.ugm." FMHA_LSC_STR(FMHA_PREFILL_SCORE_LOAD_HINT) " (M1, 1)  %0:d%2.%3x%4x%5nt flat[%1+(0,0)]"
#define FMHA_LSC_LOAD_TRANSPOSE_MNEMONIC \
  "lsc_load_block2d.ugm." FMHA_LSC_STR(FMHA_PREFILL_SCORE_LOAD_HINT) " (M1, 1)  %0:d%2.%3x%4tn flat[%1+(0,0)]"
#define FMHA_LSC_STORE_MNEMONIC \
  "lsc_store_block2d.ugm." FMHA_LSC_STR(FMHA_PREFILL_SCORE_STORE_HINT) " (M1, 1) flat[%1+(0,0)] %0:d%2.%3x%4nn"

namespace cutlass::fmha::collective {

//
// Loads
//

// Hinted load, `.st.ca` by default. Mirrors cute::XE_LOAD_2D exactly -- same base class, so
// Height/Width/BlockCount/Transposing are derived identically -- and differs only in the
// mnemonic suffix.
template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D_HINTED : cute::XE_Copy_Op_2D_Base<Bits, Height, Width, Width / BlockWidth> {
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, T* dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace cute::intel;
    auto& dv = *reinterpret_cast<cute::intel::storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
    asm(FMHA_LSC_LOAD_MNEMONIC
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width / BlockWidth), "P"(BlockWidth), "P"(Height));
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }

  using PREFETCH = cute::XE_PREFETCH_2D<Bits, Height, Width>;
};

// Hinted VNNI load. This is the one the selector picks for the fp16 B operand (K and V), so
// it is the variant that matters for the K-request-redundancy experiment; the mnemonic is
// identical to the plain load apart from the trailing `nt` transform flag.
template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D_VNNI_HINTED : cute::XE_Copy_Op_2D_Base<Bits, Height, Width, Width / BlockWidth> {
  static_assert(Bits == 8 || Bits == 16, "Unsupported data size");

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, T* dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace cute::intel;
    auto& dv = *reinterpret_cast<cute::intel::storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
    asm(FMHA_LSC_LOAD_VNNI_MNEMONIC
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width / BlockWidth), "P"(BlockWidth), "P"(Height));
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }

  using PREFETCH = cute::XE_PREFETCH_2D<Bits, Height, Width>;
};

// Hinted transposing load. This is what the selector picks for the K-cache operand in this
// kernel's layout (K arrives as (d, k) and the DPAS wants it transposed), so it is the op the
// K experiment actually goes through.
template <int Bits, int Height, int Width>
struct XE_LOAD_2D_TRANSPOSE_HINTED : cute::XE_Copy_Op_2D_Base<Bits, Height, Width, 1, true> {
  static_assert(Bits == 32 || Bits == 64, "Unsupported data size");
  static_assert(Width <= 8, "Width exceeds hardware limits");
  static_assert(Bits != 64 || (Height == 8 && Width < 4), "Unsupported D64 transpose block size");

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, T* dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace cute::intel;
    auto& dv = *reinterpret_cast<cute::intel::storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
    asm(FMHA_LSC_LOAD_TRANSPOSE_MNEMONIC : "=rw"(dv) : "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height));
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }

  using PREFETCH = cute::XE_PREFETCH_2D<Bits, Height, Width>;
};

//
// Stores
//

// Hinted store, `.uc.wb` by default: the score scratch is written once by the store launch
// and read once by each load launch, so allocating it in L1 only evicts Q/K, while keeping
// it in L3 still saves the reader a DRAM trip.
template <int Bits, int Height, int Width>
struct XE_STORE_2D_HINTED : cute::XE_Copy_Op_2D_Base<Bits, Height, Width> {
  static_assert(Height <= 8, "Height exceeds hardware limits");

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, const T* src) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace cute::intel;
    auto& sv = *reinterpret_cast<const cute::intel::storage_vector_t<T, Width * Height * Bits / sg_size>*>(src);
    asm(FMHA_LSC_STORE_MNEMONIC ::"rw"(sv), "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height));
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

//
// Builders
//

// Map a stock op onto its hinted equivalent. Only the two geometries CuTe's selector can
// pick for the score surface are listed; anything else is a compile error rather than a
// silent fallback to the unhinted op.
template <class Op>
struct with_cache_hint;

template <int B, int H, int W, int BW>
struct with_cache_hint<cute::XE_LOAD_2D<B, H, W, BW>> {
  using type = XE_LOAD_2D_HINTED<B, H, W, BW>;
};

template <int B, int H, int W, int BW>
struct with_cache_hint<cute::XE_LOAD_2D_VNNI<B, H, W, BW>> {
  using type = XE_LOAD_2D_VNNI_HINTED<B, H, W, BW>;
};

template <int B, int H, int W>
struct with_cache_hint<cute::XE_LOAD_2D_TRANSPOSE<B, H, W>> {
  using type = XE_LOAD_2D_TRANSPOSE_HINTED<B, H, W>;
};

template <int B, int H, int W>
struct with_cache_hint<cute::XE_STORE_2D<B, H, W>> {
  using type = XE_STORE_2D_HINTED<B, H, W>;
};

// make_block_2d_copy_C/D with a cache hint. These mirror the stock builders exactly -- same
// ValType, same MMAType, same selector call, so the same geometry is chosen -- and then swap
// the op for its hinted twin before building the TiledCopy. Running the stock selector
// rather than hardcoding a geometry is what keeps this correct if the tile shape changes.
template <class TiledMMA, class GTensor>
CUTE_HOST_DEVICE auto make_block_2d_copy_D_hinted(TiledMMA const& mma, GTensor const& gmem) {
  using ValType = typename GTensor::value_type;
  using MMAType = typename TiledMMA::ValTypeD;
  auto cD = cute::make_identity_tensor(cute::select<0, 1>(mma.tile_mnk()));
  auto op = cute::block_2d_selector<ValType, MMAType, true>(mma.get_slice(0).atom_partition_C(cD).layout(), gmem.stride());
  using HintedOp = typename with_cache_hint<decltype(op)>::type;
  return cute::make_block_2d_copy_CD<ValType>(HintedOp{}, mma, gmem.stride()).with(gmem);
}

// Hinted B operand. Used for the K/V loads: at head_dim=512 all 32 subgroups load the whole
// K block, so K's L1 request volume is 32x its footprint, and those requests evict the Q
// lines that must survive the block. A streaming hint asks the cache not to retain K -- the
// only way to cut that interference that does not move any data (SLM staging and N-splitting
// were both measured and lost).
template <class TiledMMA, class GTensor>
CUTE_HOST_DEVICE auto make_block_2d_copy_B_hinted(TiledMMA const& mma, GTensor const& gmem) {
  using ValType = typename GTensor::value_type;
  using MMAType = typename TiledMMA::ValTypeB;
  auto cB = cute::make_identity_tensor(cute::select<1, 2>(mma.tile_mnk()));
  auto op = cute::block_2d_selector<ValType, MMAType>(mma.get_slice(0).atom_partition_B(cB).layout(), gmem.stride());
  using HintedOp = typename with_cache_hint<decltype(op)>::type;
  return cute::make_block_2d_copy_B<ValType>(HintedOp{}, mma, gmem.stride()).with(gmem);
}

template <class TiledMMA, class GTensor>
CUTE_HOST_DEVICE auto make_block_2d_copy_C_hinted(TiledMMA const& mma, GTensor const& gmem) {
  using ValType = typename GTensor::value_type;
  using MMAType = typename TiledMMA::ValTypeC;
  auto cC = cute::make_identity_tensor(cute::select<0, 1>(mma.tile_mnk()));
  auto op = cute::block_2d_selector<ValType, MMAType>(mma.get_slice(0).atom_partition_C(cC).layout(), gmem.stride());
  using HintedOp = typename with_cache_hint<decltype(op)>::type;
  return cute::make_block_2d_copy_CD<ValType>(HintedOp{}, mma, gmem.stride()).with(gmem);
}

}  // namespace cutlass::fmha::collective

namespace cute {

// Opt our ops into the block-2D machinery. make_block_2d_copy_* static_asserts on this
// trait, and Copy_Traits supplies the Src/Dst/Ref layouts; both simply forward to the
// stock atom of the same geometry, since the cache hint changes no data layout.
template <int B, int H, int W, int BW>
struct is_xe_block_2d_atom<cutlass::fmha::collective::XE_LOAD_2D_HINTED<B, H, W, BW>> : std::true_type {};

template <int B, int H, int W, int BW>
struct is_xe_block_2d_atom<cutlass::fmha::collective::XE_LOAD_2D_VNNI_HINTED<B, H, W, BW>> : std::true_type {};

template <int B, int H, int W>
struct is_xe_block_2d_atom<cutlass::fmha::collective::XE_LOAD_2D_TRANSPOSE_HINTED<B, H, W>> : std::true_type {};

template <int B, int H, int W>
struct is_xe_block_2d_atom<cutlass::fmha::collective::XE_STORE_2D_HINTED<B, H, W>> : std::true_type {};

template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<cutlass::fmha::collective::XE_LOAD_2D_HINTED<CopyBits, Height, Width, BlockWidth>,
                   XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<cutlass::fmha::collective::XE_LOAD_2D_HINTED<CopyBits, Height, Width, BlockWidth>,
                         XMode, YMode, ValType, TiledStrides> {
  using Super = Xe2DLoadTraitsBase<cutlass::fmha::collective::XE_LOAD_2D_HINTED<CopyBits, Height, Width, BlockWidth>,
                                   XMode, YMode, ValType, TiledStrides>;
  using Super::Super;

  using DstLayout = XeInterleavedLayout<Layout<Shape<Int<BlockWidth>, Int<Height>, Int<Width / BlockWidth>>,
                                               Stride<_1, Int<Width>, Int<BlockWidth>>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;
  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));
};

// VNNI load traits. Same DstLayout as cute's XE_LOAD_2D_VNNI: the hint changes no layout,
// but the VNNI layout differs from the plain load's, so it cannot share the block above.
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<cutlass::fmha::collective::XE_LOAD_2D_VNNI_HINTED<CopyBits, Height, Width, BlockWidth>,
                   XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<cutlass::fmha::collective::XE_LOAD_2D_VNNI_HINTED<CopyBits, Height, Width, BlockWidth>,
                         XMode, YMode, ValType, TiledStrides> {
  using Super =
      Xe2DLoadTraitsBase<cutlass::fmha::collective::XE_LOAD_2D_VNNI_HINTED<CopyBits, Height, Width, BlockWidth>,
                         XMode, YMode, ValType, TiledStrides>;
  using Super::Super;

  static constexpr int BV = 32 / CopyBits;

  using DstLayout =
      XeInterleavedLayout<Layout<Shape<Int<BV>, Int<BlockWidth>, Int<Height / BV>, Int<Width / BlockWidth>>,
                                 Stride<Int<Width>, _1, Int<Width * BV>, Int<BlockWidth>>>,
                          CopyBits,
                          sizeof_bits_v<ValType>>;
  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));
};

// Transposing load traits, mirroring cute's XE_LOAD_2D_TRANSPOSE layout.
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width>
struct Copy_Traits<cutlass::fmha::collective::XE_LOAD_2D_TRANSPOSE_HINTED<CopyBits, Height, Width>,
                   XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<cutlass::fmha::collective::XE_LOAD_2D_TRANSPOSE_HINTED<CopyBits, Height, Width>,
                         XMode, YMode, ValType, TiledStrides> {
  using Super = Xe2DLoadTraitsBase<cutlass::fmha::collective::XE_LOAD_2D_TRANSPOSE_HINTED<CopyBits, Height, Width>,
                                   XMode, YMode, ValType, TiledStrides>;
  using Super::Super;

  using DstLayout = XeInterleavedLayout<Layout<Shape<Int<Height>, Int<Width>>, Stride<Int<Width>, _1>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;
  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));
};

// The store traits cannot simply inherit from Copy_Traits<XE_STORE_2D<...>>: copy_unpack
// is a hidden friend that names that base's own `Op`, so an inherited one would emit the
// *unhinted* instruction and the hint would silently vanish. Mirror the stock body instead,
// with Op bound to our type. (The load side is safe to inherit from Xe2DLoadTraitsBase,
// because that base is already parameterized on Op.)
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width>
struct Copy_Traits<cutlass::fmha::collective::XE_STORE_2D_HINTED<CopyBits, Height, Width>,
                   XMode, YMode, ValType, TiledStrides>
    : Xe2DTraitsBase<cutlass::fmha::collective::XE_STORE_2D_HINTED<CopyBits, Height, Width>,
                     XMode, YMode, ValType, TiledStrides> {
  using SrcLayout = XeInterleavedLayout<Layout<Shape<Int<Width>, Int<Height>>>, CopyBits, sizeof_bits_v<ValType>>;
  using RefLayout = SrcLayout;
  using DstLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));

  using Op = cutlass::fmha::collective::XE_STORE_2D_HINTED<CopyBits, Height, Width>;
  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Traits = typename Super::Traits;
  using ThrID = typename Super::ThrID;

  using Super::Super;

  template <class SEngine, class SLayout, class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const& traits, Tensor<SEngine, SLayout> const& src, Tensor<DEngine, DLayout>& dst) {
    using SType = typename SEngine::value_type;
    constexpr auto SBits = sizeof_bits_v<SType>;

    static_assert(is_counting_layout_v<DLayout>, "Destination tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<SEngine>, "Source tensor must be in registers.");
    static_assert(size(SLayout{}) * SBits == size<1>(SrcLayout{}), "Source tensor size does not match copy atom size.");
    static_assert(
        size(DLayout{}) * SBits == size<1>(DstLayout{}), "Destination tensor size does not match copy atom size.");

    traits.template update_payload<SBits>(dst.data().coord_);
    Op::copy(traits.payload, recast_ptr<int_byte_t<bits_to_bytes(Super::ValBits)>>(&*src.data()));
  }
};

}  // namespace cute
