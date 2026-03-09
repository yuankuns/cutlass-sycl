// Bug reproducer: inline asm exp2 variants validation across headdims.
//
// Compares multiple inline asm approaches for scale+exp2 against pure C++ reference
// for headdim=192/256, with real DPAS GEMM for realistic register pressure.
//
// Variants tested:
//   copy32     - float8 init copy + (M1,32)              -- WRONG on 192/256
//   elem16     - per-element local float + (M1,16)       -- correct
//   c_ref      - pure C++ sycl::exp2                     -- reference
//   mad16exp32 - float8 init copy, mad(M1,16) + exp(M1,32) -- diagnostic
//   recast16   - recast_ptr<float8> alias + (M1,16)
//   copy16     - 8 separate float locals + (M1,16)       -- correct
//   recast32   - recast_ptr<float8> alias + (M1,32)      -- WRONG on 192/256
//
// Build:
//   source /opt/intel/oneapi/setvars.sh && cd build
//   ninja bug_reproducer
//   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./examples/sdpa_bwd/bug_reproducer

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <cute/tensor.hpp>
#include "params.hpp"
#include "sdpa_backward.hpp"

#include <random>
#include <cmath>
#include <vector>

using namespace cute;

// headdim=192: kBlockN=32, kBlockM=64, kNSGs=8, AtomLayoutMSdP=4
using Trait192 = FAKernel<cutlass::bfloat16_t, 192, 64, 32, 8, 4, 2, 2, false>;
// headdim=256: kBlockN=32, kBlockM=64, kNSGs=8, AtomLayoutMSdP=4
using Trait256 = FAKernel<cutlass::bfloat16_t, 256, 64, 32, 8, 4, 2, 2, false>;

template<typename Trait>
struct BugReproKernel {
    const cutlass::bfloat16_t* Kt;
    const cutlass::bfloat16_t* Q;
    float* out_c_ref;        // pure C++ exp2 -- reference
    float* out_copy16;       // 8 separate float locals in one asm block + (M1,16) -- correct
    float* out_copy32;       // float8 init copy + (M1,32) -- WRONG
    float* out_elem16;       // per-element local float + (M1,16) -- correct
    float* out_mad16exp32;   // float8 init copy, mad(M1,16) + exp(M1,32) -- diagnostic
    float* out_recast16;     // recast_ptr<float8> alias + (M1,16)
    float* out_recast32;     // recast_ptr<float8> alias + (M1,32) -- WRONG

    CUTLASS_DEVICE void operator()(sycl::nd_item<1>) const {
#if defined(__SYCL_DEVICE_ONLY__)
        auto trait = Trait{};
        typename Trait::TiledMmaSdP tiled_mma{};

        constexpr int kBlockN  = Trait::kBlockN;
        constexpr int kHeadDim = Trait::kHeadDim;
        constexpr int kBlockM  = Trait::kBlockM;

        auto mKt = make_tensor(make_gmem_ptr(Kt),
                               make_layout(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
                                           make_stride(Int<1>{}, Int<kBlockN>{})));
        auto mQ  = make_tensor(make_gmem_ptr(Q),
                               make_layout(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                                           make_stride(Int<1>{}, Int<kBlockM>{})));

        // Run real GEMM to create realistic register allocation pressure.
        auto rS0 = create_reg<float>(trait, mKt, tiled_mma);
        auto rS1 = create_reg<float>(trait, mKt, tiled_mma);
        auto rS2 = create_reg<float>(trait, mKt, tiled_mma);
        auto rS3 = create_reg<float>(trait, mKt, tiled_mma);
        auto rS4 = create_reg<float>(trait, mKt, tiled_mma);
        auto rS5 = create_reg<float>(trait, mKt, tiled_mma);
        auto rS6 = create_reg<float>(trait, mKt, tiled_mma);
        auto rS7 = create_reg<float>(trait, mKt, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS0, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS1, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS2, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS3, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS4, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS5, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS6, tiled_mma);
        gemm_SdP(trait, mKt, mQ, rS7, tiled_mma);

        auto s0 = make_tensor(rS0.data(), convert_layout_acc_layout(rS0.layout()));
        auto s1 = make_tensor(rS1.data(), convert_layout_acc_layout(rS1.layout()));
        auto s2 = make_tensor(rS2.data(), convert_layout_acc_layout(rS2.layout()));
        // s6 is not used for output; it adds register pressure.
        auto s3 = make_tensor(rS3.data(), convert_layout_acc_layout(rS3.layout()));
        auto s4 = make_tensor(rS4.data(), convert_layout_acc_layout(rS4.layout()));
        auto s5 = make_tensor(rS5.data(), convert_layout_acc_layout(rS5.layout()));
        auto s6 = make_tensor(rS6.data(), convert_layout_acc_layout(rS6.layout()));
        auto s7 = make_tensor(rS7.data(), convert_layout_acc_layout(rS7.layout()));

        auto lid = int(compat::get_nd_item<1>().get_local_id(0));
        if (cute::thread0()){
            print(s0);
            print("\n");
        }
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s0); ++ni) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(s0); ++mi) {
                float v = float((lid * 16 + ni * size<0>(s0) + mi) % 17 - 8) * 0.5f + 0.1f;
                s0(mi, ni) = v;
                s1(mi, ni) = v;
                s2(mi, ni) = v;
                s3(mi, ni) = v;
                s4(mi, ni) = v;
                s5(mi, ni) = v;
                s6(mi, ni) = v;
                s7(mi, ni) = v;
            }
        }

        const float scale   = sycl::rsqrt(float(kHeadDim));
        const float neg_max = 0.f;

        // copy32: float8 init copy + (M1,32) -- WRONG
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s0); ++ni) {
            auto t = s0(_, ni);
            intel::float8 v = {t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7)};
            __asm__ volatile (
                "{\n"
                ".decl VALS_%= v_type=G type=F num_elts=128 alias=<%0,0>\n"
                ".decl SCALE_%= v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX_%= v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1,32) VALS_%=(0,0)<1>  VALS_%=(0,0)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,32) VALS_%=(0,32)<1> VALS_%=(0,32)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,32) VALS_%=(0,64)<1> VALS_%=(0,64)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,32) VALS_%=(0,96)<1> VALS_%=(0,96)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,0)<1>  VALS_%=(0,0)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,32)<1> VALS_%=(0,32)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,64)<1> VALS_%=(0,64)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,96)<1> VALS_%=(0,96)<1;1,0>\n"
                "}\n"
                : "+rw"(v) : "rw"(scale), "rw"(neg_max)
            );
            t(0)=v[0]; t(1)=v[1]; t(2)=v[2]; t(3)=v[3];
            t(4)=v[4]; t(5)=v[5]; t(6)=v[6]; t(7)=v[7];
        }

        // elem16: per-element local float + (M1,16) -- correct
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s1); ++ni) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(s1); ++mi) {
                float val = s1(mi, ni);
                __asm__ volatile (
                    "{\n"
                    ".decl VAL_%=     v_type=G type=F num_elts=16 alias=<%0,0>\n"
                    ".decl SCALE_%=   v_type=G type=F num_elts=16 alias=<%1,0>\n"
                    ".decl NEG_MAX_%= v_type=G type=F num_elts=16 alias=<%2,0>\n"
                    "mad (M1,16) VAL_%=(0,0)<1> VAL_%=(0,0)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                    "exp (M1,16) VAL_%=(0,0)<1> VAL_%=(0,0)<1;1,0>\n"
                    "}\n"
                    : "+rw"(val) : "rw"(scale), "rw"(neg_max)
                );
                s1(mi, ni) = val;
            }
        }

        // c_ref: pure C++ sycl::exp2 -- reference
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s2); ++ni) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(s2); ++mi) {
                s2(mi, ni) = sycl::exp2(s2(mi, ni) * scale + neg_max);
            }
        }

        // mad16exp32: float8 init copy, mad(M1,16) + exp(M1,32) -- diagnostic
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s7); ++ni) {
            auto t = s7(_, ni);
            intel::float8 v = {t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7)};
            __asm__ volatile (
                "{\n"
                ".decl VALS_%= v_type=G type=F num_elts=128 alias=<%0,0>\n"
                ".decl SCALE_%= v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX_%= v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1,16) VALS_%=(0,0)<1>   VALS_%=(0,0)<1;1,0>   SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,16)<1>  VALS_%=(0,16)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,32)<1>  VALS_%=(0,32)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,48)<1>  VALS_%=(0,48)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,64)<1>  VALS_%=(0,64)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,80)<1>  VALS_%=(0,80)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,96)<1>  VALS_%=(0,96)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,112)<1> VALS_%=(0,112)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,0)<1>  VALS_%=(0,0)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,32)<1> VALS_%=(0,32)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,64)<1> VALS_%=(0,64)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,96)<1> VALS_%=(0,96)<1;1,0>\n"
                "}\n"
                : "+rw"(v) : "rw"(scale), "rw"(neg_max)
            );
            t(0)=v[0]; t(1)=v[1]; t(2)=v[2]; t(3)=v[3];
            t(4)=v[4]; t(5)=v[5]; t(6)=v[6]; t(7)=v[7];
        }

        // recast16: recast_ptr<intel::float8> alias + (M1,16)
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s3); ++ni) {
            auto t = s3(_, ni);
            intel::float8& v8 = *recast_ptr<intel::float8>(&t(0));
            __asm__ volatile (
                "{\n"
                ".decl VALS_%= v_type=G type=F num_elts=128 alias=<%0,0>\n"
                ".decl SCALE_%= v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX_%= v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1,16) VALS_%=(0,0)<1>  VALS_%=(0,0)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,16)<1> VALS_%=(0,16)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,32)<1> VALS_%=(0,32)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,48)<1> VALS_%=(0,48)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,64)<1> VALS_%=(0,64)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,80)<1> VALS_%=(0,80)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,96)<1>  VALS_%=(0,96)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,16) VALS_%=(0,112)<1> VALS_%=(0,112)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,0)<1>  VALS_%=(0,0)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,16)<1> VALS_%=(0,16)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,32)<1> VALS_%=(0,32)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,48)<1> VALS_%=(0,48)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,64)<1> VALS_%=(0,64)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,80)<1> VALS_%=(0,80)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,96)<1>  VALS_%=(0,96)<1;1,0>\n"
                "exp (M1,16) VALS_%=(0,112)<1> VALS_%=(0,112)<1;1,0>\n"
                "}\n"
                : "+rw"(v8) : "rw"(scale), "rw"(neg_max)
            );
        }

        // copy16: 8 separate float locals in one asm block + (M1,16) -- correct
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s4); ++ni) {
            auto t = s4(_, ni);
            float v0=t(0), v1=t(1), v2=t(2), v3=t(3),
                  v4=t(4), v5=t(5), v6=t(6), v7=t(7);
            __asm__ volatile (
                "{\n"
                ".decl V0_%= v_type=G type=F num_elts=16 alias=<%0,0>\n"
                ".decl V1_%= v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl V2_%= v_type=G type=F num_elts=16 alias=<%2,0>\n"
                ".decl V3_%= v_type=G type=F num_elts=16 alias=<%3,0>\n"
                ".decl V4_%= v_type=G type=F num_elts=16 alias=<%4,0>\n"
                ".decl V5_%= v_type=G type=F num_elts=16 alias=<%5,0>\n"
                ".decl V6_%= v_type=G type=F num_elts=16 alias=<%6,0>\n"
                ".decl V7_%= v_type=G type=F num_elts=16 alias=<%7,0>\n"
                ".decl SC_%= v_type=G type=F num_elts=16 alias=<%8,0>\n"
                ".decl NM_%= v_type=G type=F num_elts=16 alias=<%9,0>\n"
                "mad (M1,16) V0_%=(0,0)<1> V0_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "mad (M1,16) V1_%=(0,0)<1> V1_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "mad (M1,16) V2_%=(0,0)<1> V2_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "mad (M1,16) V3_%=(0,0)<1> V3_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "mad (M1,16) V4_%=(0,0)<1> V4_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "mad (M1,16) V5_%=(0,0)<1> V5_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "mad (M1,16) V6_%=(0,0)<1> V6_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "mad (M1,16) V7_%=(0,0)<1> V7_%=(0,0)<1;1,0> SC_%=(0,0)<0;1,0> NM_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V0_%=(0,0)<1> V0_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V1_%=(0,0)<1> V1_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V2_%=(0,0)<1> V2_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V3_%=(0,0)<1> V3_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V4_%=(0,0)<1> V4_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V5_%=(0,0)<1> V5_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V6_%=(0,0)<1> V6_%=(0,0)<1;1,0>\n"
                "exp (M1,16) V7_%=(0,0)<1> V7_%=(0,0)<1;1,0>\n"
                "}\n"
                : "+rw"(v0),"+rw"(v1),"+rw"(v2),"+rw"(v3),
                  "+rw"(v4),"+rw"(v5),"+rw"(v6),"+rw"(v7)
                : "rw"(scale), "rw"(neg_max)
            );
            t(0)=v0; t(1)=v1; t(2)=v2; t(3)=v3;
            t(4)=v4; t(5)=v5; t(6)=v6; t(7)=v7;
        }

        // recast32: recast_ptr<intel::float8> alias + (M1,32) -- WRONG
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s5); ++ni) {
            auto t = s5(_, ni);
            intel::float8& v8 = *recast_ptr<intel::float8>(&t(0));
            __asm__ volatile (
                "{\n"
                ".decl VALS_%= v_type=G type=F num_elts=128 alias=<%0,0>\n"
                ".decl SCALE_%= v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX_%= v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1,32) VALS_%=(0,0)<1>  VALS_%=(0,0)<1;1,0>  SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,32) VALS_%=(0,32)<1> VALS_%=(0,32)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,32) VALS_%=(0,64)<1> VALS_%=(0,64)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "mad (M1,32) VALS_%=(0,96)<1> VALS_%=(0,96)<1;1,0> SCALE_%=(0,0)<0;1,0> NEG_MAX_%=(0,0)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,0)<1>  VALS_%=(0,0)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,32)<1> VALS_%=(0,32)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,64)<1> VALS_%=(0,64)<1;1,0>\n"
                "exp (M1,32) VALS_%=(0,96)<1> VALS_%=(0,96)<1;1,0>\n"
                "}\n"
                : "+rw"(v8) : "rw"(scale), "rw"(neg_max)
            );
        }

        const int elts = size(s0);
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(s0); ++ni) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(s0); ++mi) {
                int idx = lid * elts + ni * size<0>(s0) + mi;
                out_c_ref[idx]        = s2(mi, ni);
                out_copy16[idx]       = s4(mi, ni);
                out_copy32[idx]       = s0(mi, ni);
                out_elem16[idx]       = s1(mi, ni);
                out_mad16exp32[idx]   = s7(mi, ni);
                out_recast16[idx]     = s3(mi, ni);
                out_recast32[idx]     = s5(mi, ni);
            }
        }
#endif
    }
};

template<typename Trait>
void run_test(sycl::queue& q, const char* label) {
    constexpr int kBlockN  = Trait::kBlockN;
    constexpr int kBlockM  = Trait::kBlockM;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kNSGs    = Trait::kNSGs;
    constexpr int total    = kBlockN * kBlockM;
    constexpr int threads  = kNSGs * 16;

    auto* Kt_d    = sycl::malloc_device<cutlass::bfloat16_t>(kBlockN * kHeadDim, q);
    auto* Q_d     = sycl::malloc_device<cutlass::bfloat16_t>(kBlockM * kHeadDim, q);
    auto* c_ref_d      = sycl::malloc_device<float>(total, q);
    auto* copy16_d     = sycl::malloc_device<float>(total, q);
    auto* copy32_d     = sycl::malloc_device<float>(total, q);
    auto* elem16_d     = sycl::malloc_device<float>(total, q);
    auto* mad16exp32_d = sycl::malloc_device<float>(total, q);
    auto* recast16_d   = sycl::malloc_device<float>(total, q);
    auto* recast32_d   = sycl::malloc_device<float>(total, q);

    std::vector<cutlass::bfloat16_t> Kt_h(kBlockN * kHeadDim), Q_h(kBlockM * kHeadDim);
    { std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.f, 1.f);
      for (auto& x : Kt_h) x = cutlass::bfloat16_t(dist(rng));
      for (auto& x : Q_h)  x = cutlass::bfloat16_t(dist(rng)); }
    q.memcpy(Kt_d, Kt_h.data(), Kt_h.size() * sizeof(cutlass::bfloat16_t)).wait();
    q.memcpy(Q_d,  Q_h.data(),  Q_h.size()  * sizeof(cutlass::bfloat16_t)).wait();

    q.submit([&](sycl::handler& h) {
        BugReproKernel<Trait> k{Kt_d, Q_d, c_ref_d, copy16_d, copy32_d, elem16_d, mad16exp32_d, recast16_d, recast32_d};
        h.parallel_for(sycl::nd_range<1>(threads, threads),
                       sycl::ext::oneapi::experimental::properties{
                           sycl::ext::oneapi::experimental::sub_group_size<16>},
                       k);
    }).wait();

    std::vector<float> copy32_h(total), elem16_h(total), c_ref_h(total),
                       mad16exp32_h(total), recast16_h(total), copy16_h(total), recast32_h(total);
    q.memcpy(copy32_h.data(),     copy32_d,     total * sizeof(float)).wait();
    q.memcpy(elem16_h.data(),     elem16_d,     total * sizeof(float)).wait();
    q.memcpy(c_ref_h.data(),      c_ref_d,      total * sizeof(float)).wait();
    q.memcpy(mad16exp32_h.data(), mad16exp32_d, total * sizeof(float)).wait();
    q.memcpy(recast16_h.data(),   recast16_d,   total * sizeof(float)).wait();
    q.memcpy(copy16_h.data(),     copy16_d,     total * sizeof(float)).wait();
    q.memcpy(recast32_h.data(),   recast32_d,   total * sizeof(float)).wait();

    int mm_copy32=0, mm_elem16=0, mm_r16=0, mm_copy16=0, mm_r32=0, mm_m16e32=0;
    for (int i = 0; i < total; ++i) {
        float ref = c_ref_h[i];
        auto bad = [&](float v){ return std::abs(v-ref)/(std::abs(ref)+1e-8f) > 0.01f; };
        if (bad(copy32_h[i]))     ++mm_copy32;
        if (bad(elem16_h[i]))     ++mm_elem16;
        if (bad(mad16exp32_h[i])) ++mm_m16e32;
        if (bad(recast16_h[i]))   ++mm_r16;
        if (bad(copy16_h[i]))     ++mm_copy16;
        if (bad(recast32_h[i]))   ++mm_r32;
    }

    printf("\n%s\n", label);
    printf("  copy32     (float8 init copy, M1,32):                   %d/%d mismatches\n", mm_copy32,  total);
    printf("  elem16     (per-element local, M1,16):                  %d/%d mismatches\n", mm_elem16,  total);
    printf("  mad16exp32 (float8, mad M1,16 + exp M1,32):            %d/%d mismatches\n", mm_m16e32,  total);
    printf("  recast16   (recast_ptr<float8> alias, M1,16):           %d/%d mismatches\n", mm_r16,     total);
    printf("  copy16     (8 separate locals, M1,16 in one asm block): %d/%d mismatches\n", mm_copy16,  total);
    printf("  recast32   (recast_ptr<float8> alias, M1,32):           %d/%d mismatches\n", mm_r32,     total);

    printf("  idx  %-12s %-12s %-12s %-12s %-12s %-12s\n", "c_ref","copy32","mad16exp32","recast16","copy16","recast32");
    for (int i = 0; i < 8; ++i) {
        auto flag = [&](float v){ return std::abs(v-c_ref_h[i])/(std::abs(c_ref_h[i])+1e-8f)>0.01f ? "✗":"✓"; };
        printf("  %3d  %-12g %-12g %-12g %-12g %-12g %-12g  %s%s%s%s%s\n", i,
               c_ref_h[i], copy32_h[i], mad16exp32_h[i], recast16_h[i], copy16_h[i], recast32_h[i],
               flag(copy32_h[i]), flag(mad16exp32_h[i]), flag(recast16_h[i]), flag(copy16_h[i]), flag(recast32_h[i]));
    }

    sycl::free(Kt_d, q); sycl::free(Q_d, q);
    sycl::free(copy32_d, q); sycl::free(elem16_d, q); sycl::free(c_ref_d, q);
    sycl::free(mad16exp32_d, q); sycl::free(recast16_d, q); sycl::free(copy16_d, q); sycl::free(recast32_d, q);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    printf("Driver: %s\n", q.get_device().get_info<sycl::info::device::driver_version>().c_str());

    run_test<Trait192>(q, "headdim=192, scores=(Int<8>,Int<2>), ScaleExpHelper<Int<8>>");
    run_test<Trait256>(q, "headdim=256, scores=(Int<8>,Int<2>), ScaleExpHelper<Int<8>>");

    return 0;
}

