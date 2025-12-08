set(TEST_GROUPS --groups=128)

cutlass_example_add_executable(
  sdpa_backward
  sdpa_backward.cpp
  cnpy.cpp
)
cutlass_example_add_executable(
  test_noeven
  test_noeven.cpp
)
cutlass_example_add_executable(
  gemm_fp16
  gemm_fp16.cpp
)
cutlass_example_add_executable(
  block_gemm_fp16
  block_gemm_fp16.cpp
)
cutlass_example_add_executable(
  gemm_bf16
  gemm_bf16.cpp
)
cutlass_example_add_executable(
  reg_reuse_dq
  reg_reuse_dq.cpp
  cnpy.cpp
)
cutlass_example_add_executable(
  reg_reuse_dk
  reg_reuse_dk.cpp
  cnpy.cpp
)
cutlass_example_add_executable(
  reg_reuse_dv
  reg_reuse_dv.cpp
  cnpy.cpp
)
cutlass_example_add_executable(
  last_block_bug
  last_block_bug.cpp
  cnpy.cpp
)
cutlass_example_add_executable(
  xe_gemm_test
  xe_gemm_test.cpp
)

target_compile_definitions(reg_reuse_dq PRIVATE
  SYCL_INTEL_TARGET=1
)
target_link_options(sdpa_backward PUBLIC "-lz")
target_link_options(reg_reuse_dq PUBLIC "-lz")
target_link_options(reg_reuse_dk PUBLIC "-lz")
target_link_options(reg_reuse_dv PUBLIC "-lz")
target_link_options(last_block_bug PUBLIC "-lz")
set_target_properties(sdpa_backward PROPERTIES CXX_COMPILER_LAUNCHER "IGC_VectorAliasBBThreshold=10000" )
set_target_properties(sdpa_backward PROPERTIES CXX_LINKER_LAUNCHER "IGC_VectorAliasBBThreshold=10000" )
