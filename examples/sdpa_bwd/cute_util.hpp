#pragma once
// Helper struct for template specialization of scale_exp operations
#ifdef __SYCL_DEVICE_ONLY__
template<typename T>
struct ScaleExpHelper {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, float scale, float neg_max_scaled) {
        // Generic implementation using loop
        static constexpr int M = T::value;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < M; ++mi) {
            float val = tensor(mi);
            __asm__ volatile (
                "{\n"
                ".decl VAL v_type=G type=F num_elts=16 alias=<%0,0>\n"
                ".decl SCALE v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1, 16) VAL(0,0)<1> VAL(0,0)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "exp (M1, 16) VAL(0,0)<1> VAL(0,0)<1;1,0>\n"
                "}\n"
                : "+rw"(val)
                : "rw"(scale), "rw"(neg_max_scaled)
            );
            tensor(mi) = val;
        }
    }
};

template<>
struct ScaleExpHelper<Int<1>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, float scale, float neg_max_scaled) {
        float& val = *recast_ptr<float>(&tensor(0));
        __asm__ volatile (
            "{\n"
            ".decl VAL v_type=G type=F num_elts=16 alias=<%0,0>\n"
            ".decl SCALE v_type=G type=F num_elts=16 alias=<%1,0>\n"
            ".decl NEG_MAX v_type=G type=F num_elts=16 alias=<%2,0>\n"
            "mad (M1, 16) VAL(0,0)<1> VAL(0,0)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
            "exp (M1, 16) VAL(0,0)<1> VAL(0,0)<1;1,0>\n"
            "}\n"
            : "+rw"(val)
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

template<>
struct ScaleExpHelper<Int<2>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, float scale, float neg_max_scaled) {
        intel::float2& vals = *recast_ptr<intel::float2>(&tensor(0));
        __asm__ volatile (
            "{\n"
            ".decl VALS v_type=G type=F num_elts=32 alias=<%0,0>\n"
            ".decl SCALE v_type=G type=F num_elts=16 alias=<%1,0>\n"
            ".decl NEG_MAX v_type=G type=F num_elts=16 alias=<%2,0>\n"
            "mad (M1, 16) VALS(0,0)<1>  VALS(0,0)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
            "mad (M1, 16) VALS(0,16)<1> VALS(0,16)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
            "exp (M1, 16) VALS(0,0)<1>  VALS(0,0)<1;1,0>\n"
            "exp (M1, 16) VALS(0,16)<1> VALS(0,16)<1;1,0>\n"
            "}\n"
            : "+rw"(vals)
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

template<>
struct ScaleExpHelper<Int<4>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, float scale, float neg_max_scaled) {
        intel::float4& vals = *recast_ptr<intel::float4>(&tensor(0));
        __asm__ volatile (
            "{\n"
            ".decl VALS v_type=G type=F num_elts=64 alias=<%0,0>\n"
            ".decl SCALE v_type=G type=F num_elts=16 alias=<%1,0>\n"
            ".decl NEG_MAX v_type=G type=F num_elts=16 alias=<%2,0>\n"
            "mad (M1, 16) VALS(0,0)<1>  VALS(0,0)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
            "mad (M1, 16) VALS(0,16)<1> VALS(0,16)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
            "mad (M1, 16) VALS(0,32)<1> VALS(0,32)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
            "mad (M1, 16) VALS(0,48)<1> VALS(0,48)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
            "exp (M1, 16) VALS(0,0)<1>  VALS(0,0)<1;1,0>\n"
            "exp (M1, 16) VALS(0,16)<1> VALS(0,16)<1;1,0>\n"
            "exp (M1, 16) VALS(0,32)<1> VALS(0,32)<1;1,0>\n"
            "exp (M1, 16) VALS(0,48)<1> VALS(0,48)<1;1,0>\n"
            "}\n"
            : "+rw"(vals)
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

template<>
struct ScaleExpHelper<Int<8>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, float scale, float neg_max_scaled) {
        // Process 8 elements one at a time to avoid GRF aliasing issues
        // This is safer than the float8 approach which can fail for certain tensor layouts
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
            float val = tensor(i);
            __asm__ volatile (
                "{\n"
                ".decl VAL v_type=G type=F num_elts=16 alias=<%0,0>\n"
                ".decl SCALE v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1, 16) VAL(0,0)<1> VAL(0,0)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "exp (M1, 16) VAL(0,0)<1> VAL(0,0)<1;1,0>\n"
                "}\n"
                : "+rw"(val)
                : "rw"(scale), "rw"(neg_max_scaled)
            );
            tensor(i) = val;
        }
    }
};

template<>
struct ScaleExpHelper<Int<32>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, float scale, float neg_max_scaled) {
        // Tensor layout is ((_8,_4)):((_1,_8)) meaning:
        // - Elements 0-7 are contiguous (stride=1)
        // - Elements 8-15 start at offset 8 (stride=1 within group)
        // - Elements 16-23 start at offset 16, etc.
        // So we can directly recast each group of 8 consecutive elements to intel::float8
        #pragma unroll
        for (int group = 0; group < 4; group++) {
            int offset = group * 8;
            intel::float8& vals = *recast_ptr<intel::float8>(&tensor(offset));
            __asm__ volatile (
                "{\n"
                ".decl VALS v_type=G type=F num_elts=128 alias=<%0,0>\n"
                ".decl SCALE v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1, 16) VALS(0,0)<1>   VALS(0,0)<1;1,0>   SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "mad (M1, 16) VALS(0,16)<1>  VALS(0,16)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "mad (M1, 16) VALS(0,32)<1>  VALS(0,32)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "mad (M1, 16) VALS(0,48)<1>  VALS(0,48)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "mad (M1, 16) VALS(0,64)<1>  VALS(0,64)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "mad (M1, 16) VALS(0,80)<1>  VALS(0,80)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "mad (M1, 16) VALS(0,96)<1>  VALS(0,96)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "mad (M1, 16) VALS(0,112)<1> VALS(0,112)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<1;1,0>\n"
                "exp (M1, 16) VALS(0,0)<1>   VALS(0,0)<1;1,0>\n"
                "exp (M1, 16) VALS(0,16)<1>  VALS(0,16)<1;1,0>\n"
                "exp (M1, 16) VALS(0,32)<1>  VALS(0,32)<1;1,0>\n"
                "exp (M1, 16) VALS(0,48)<1>  VALS(0,48)<1;1,0>\n"
                "exp (M1, 16) VALS(0,64)<1>  VALS(0,64)<1;1,0>\n"
                "exp (M1, 16) VALS(0,80)<1>  VALS(0,80)<1;1,0>\n"
                "exp (M1, 16) VALS(0,96)<1>  VALS(0,96)<1;1,0>\n"
                "exp (M1, 16) VALS(0,112)<1> VALS(0,112)<1;1,0>\n"
                "}\n"
                : "+rw"(vals)
                : "rw"(scale), "rw"(neg_max_scaled)
            );
        }
    }
};
#endif // end of __SYCL_DEVICE_ONLY__
