if (CUTLASS_ENABLE_SYCL)
  if (SYCL_INTEL_TARGET)
    cutlass_example_add_executable(
      sdpa_backward
      sdpa_backward.cpp
      cnpy.cpp
    )
    cutlass_example_add_executable(
      gemm_bf16
      gemm_bf16.cpp
      cnpy.cpp
    )
    cutlass_example_add_executable(
      gemm_fp16
      gemm_fp16.cpp
      cnpy.cpp
    )
    target_link_options(sdpa_backward PUBLIC "-lz")
    target_link_options(gemm_bf16 PUBLIC "-lz")
    target_link_options(gemm_fp16 PUBLIC "-lz")
  endif()
endif()
