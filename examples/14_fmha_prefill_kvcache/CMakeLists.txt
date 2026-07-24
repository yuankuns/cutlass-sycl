# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(FMHA_PREFILL_EXAMPLE_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(FMHA_PREFILL_SRC_DIR "${FMHA_PREFILL_EXAMPLE_ROOT}/src")

set(FMHA_STANDALONE_HEAD_DIMS "64;72;96;128;192;256;512" CACHE STRING "Head dimensions to instantiate")
set(FMHA_STANDALONE_OPT_LEVEL "-O2" CACHE STRING "SYCL device optimization level for FMHA prefill kernels")
set(FMHA_STANDALONE_PAGED_HD512_TILE_Q "256" CACHE STRING "Paged head_dim=512 Q tile size")
set(FMHA_STANDALONE_PAGED_HD512_TILE_KV "64" CACHE STRING "Paged head_dim=512 KV tile size")
set(FMHA_STANDALONE_PAGED_HD512_TILE_PV "32" CACHE STRING "Paged head_dim=512 PV output tile size")
set(FMHA_STANDALONE_PAGED_HD512_TILE_O "256" CACHE STRING "Paged head_dim=512 output V tile size")
set(FMHA_STANDALONE_PAGED_HD512_NUM_SG "32" CACHE STRING "Paged head_dim=512 subgroups per work-group")
option(FMHA_STANDALONE_ENABLE_TESTS "Register standalone FMHA prefill CTest cases" ON)
set(FMHA_STANDALONE_TEST_WARMUP "1" CACHE STRING "Warmup iterations for CTest cases")
set(FMHA_STANDALONE_TEST_ITERS "5" CACHE STRING "Measured iterations for CTest cases")
set(FMHA_STANDALONE_TEST_TIMEOUT "300" CACHE STRING "Timeout in seconds for each CTest case")
set(FMHA_STANDALONE_TEST_ATOL "0.08" CACHE STRING "Absolute tolerance for CTest correctness checks")
set(FMHA_STANDALONE_TEST_RTOL "0.08" CACHE STRING "Relative tolerance for CTest correctness checks")

set(FMHA_PREFILL_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_PREFILL_NP_HEAD_DIMS 64 72 96 128)

set(FMHA_PREFILL_TILED_Q_64 128)
set(FMHA_PREFILL_TILED_KV_64 64)
set(FMHA_PREFILL_NUM_SG_64 8)
set(FMHA_PREFILL_TILED_Q_96 128)
set(FMHA_PREFILL_TILED_KV_96 64)
set(FMHA_PREFILL_NUM_SG_96 8)
set(FMHA_PREFILL_TILED_Q_128 128)
set(FMHA_PREFILL_TILED_KV_128 64)
set(FMHA_PREFILL_NUM_SG_128 16)
set(FMHA_PREFILL_TILED_Q_192 256)
set(FMHA_PREFILL_TILED_KV_192 64)
set(FMHA_PREFILL_NUM_SG_192 32)
set(FMHA_PREFILL_TILED_Q_256 256)
set(FMHA_PREFILL_TILED_KV_256 64)
set(FMHA_PREFILL_NUM_SG_256 32)
set(FMHA_PREFILL_TILED_Q_512 ${FMHA_STANDALONE_PAGED_HD512_TILE_Q})
set(FMHA_PREFILL_TILED_KV_512 ${FMHA_STANDALONE_PAGED_HD512_TILE_KV})
set(FMHA_PREFILL_NUM_SG_512 ${FMHA_STANDALONE_PAGED_HD512_NUM_SG})

set(FMHA_PREFILL_TILED_Q_NP_64 128)
set(FMHA_PREFILL_TILED_KV_NP_64 64)
set(FMHA_PREFILL_NUM_SG_NP_64 8)
set(FMHA_PREFILL_TILED_Q_NP_72 256)
set(FMHA_PREFILL_TILED_KV_NP_72 64)
set(FMHA_PREFILL_NUM_SG_NP_72 16)
set(FMHA_PREFILL_TILED_Q_NP_96 256)
set(FMHA_PREFILL_TILED_KV_NP_96 64)
set(FMHA_PREFILL_NUM_SG_NP_96 16)
set(FMHA_PREFILL_TILED_Q_NP_128 256)
set(FMHA_PREFILL_TILED_KV_NP_128 32)
set(FMHA_PREFILL_NUM_SG_NP_128 16)

set(FMHA_PREFILL_TEMPLATE "${FMHA_PREFILL_SRC_DIR}/sycl/xe_fmha_fwd_prefill_kernel.cpp.in")
set(FMHA_PREFILL_GENERATED)
set(FMHA_STANDALONE_HEAD_DIM_DEFS)

foreach(HEAD_DIM ${FMHA_STANDALONE_HEAD_DIMS})
  if(HEAD_DIM IN_LIST FMHA_PREFILL_PAGED_HEAD_DIMS)
    set(EMIT_PAGED 1)
    set(TILED_Q ${FMHA_PREFILL_TILED_Q_${HEAD_DIM}})
    set(TILED_KV ${FMHA_PREFILL_TILED_KV_${HEAD_DIM}})
    set(NUM_SG ${FMHA_PREFILL_NUM_SG_${HEAD_DIM}})
  else()
    set(EMIT_PAGED 0)
    set(TILED_Q 128)
    set(TILED_KV 64)
    set(NUM_SG 8)
  endif()

  if(HEAD_DIM IN_LIST FMHA_PREFILL_NP_HEAD_DIMS)
    set(EMIT_NP 1)
    set(TILED_Q_NP ${FMHA_PREFILL_TILED_Q_NP_${HEAD_DIM}})
    set(TILED_KV_NP ${FMHA_PREFILL_TILED_KV_NP_${HEAD_DIM}})
    set(NUM_SG_NP ${FMHA_PREFILL_NUM_SG_NP_${HEAD_DIM}})
  else()
    set(EMIT_NP 0)
    set(TILED_Q_NP 128)
    set(TILED_KV_NP 64)
    set(NUM_SG_NP 8)
  endif()

  if(NOT EMIT_PAGED AND NOT EMIT_NP)
    message(FATAL_ERROR "Unsupported standalone FMHA prefill HEAD_DIM=${HEAD_DIM}")
  endif()

  math(EXPR TILED_O "(( ${HEAD_DIM} + 31 ) / 32) * 32")
  if(HEAD_DIM STREQUAL "512")
    set(TILED_O ${FMHA_STANDALONE_PAGED_HD512_TILE_O})
    set(TILED_PV ${FMHA_STANDALONE_PAGED_HD512_TILE_PV})
  else()
    set(TILED_PV 32)
  endif()

  set(GENERATED_FILE "${CMAKE_CURRENT_BINARY_DIR}/generated/xe_fmha_fwd_prefill_kernel_${HEAD_DIM}.cpp")
  configure_file("${FMHA_PREFILL_TEMPLATE}" "${GENERATED_FILE}" @ONLY)
  list(APPEND FMHA_PREFILL_GENERATED "${GENERATED_FILE}")
  list(APPEND FMHA_STANDALONE_HEAD_DIM_DEFS "FMHA_STANDALONE_HAS_HD_${HEAD_DIM}=1")
endforeach()

add_library(fmha_prefill_kvcache_kernels STATIC
  ${FMHA_PREFILL_GENERATED}
  standalone_runtime.cpp
)

target_compile_features(fmha_prefill_kvcache_kernels PUBLIC cxx_std_20)
target_compile_definitions(fmha_prefill_kvcache_kernels PUBLIC SGL_KERNEL_STANDALONE_NO_TORCH)
target_compile_options(fmha_prefill_kvcache_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${FMHA_STANDALONE_OPT_LEVEL}>
)
target_include_directories(fmha_prefill_kvcache_kernels PUBLIC
  "${FMHA_PREFILL_EXAMPLE_ROOT}"
  "${FMHA_PREFILL_SRC_DIR}"
  "${CUTLASS_EXAMPLES_COMMON_SOURCE_DIR}"
  "${CUTLASS_EXAMPLES_UTILS_DIR}"
  "${CUTLASS_APPLICATIONS_DIR}"
)
target_link_libraries(fmha_prefill_kvcache_kernels PUBLIC
  CUTLASS
  cutlass_tools_util_includes
)

add_sycl_to_target(TARGET fmha_prefill_kvcache_kernels)

add_executable(fmha_prefill_kvcache main.cpp)
target_compile_features(fmha_prefill_kvcache PUBLIC cxx_std_20)
target_compile_definitions(fmha_prefill_kvcache PRIVATE ${FMHA_STANDALONE_HEAD_DIM_DEFS})
target_include_directories(fmha_prefill_kvcache PRIVATE
  "${FMHA_PREFILL_EXAMPLE_ROOT}"
  "${FMHA_PREFILL_SRC_DIR}"
  "${CUTLASS_EXAMPLES_COMMON_SOURCE_DIR}"
  "${CUTLASS_EXAMPLES_UTILS_DIR}"
  "${CUTLASS_APPLICATIONS_DIR}"
)
target_link_libraries(fmha_prefill_kvcache PRIVATE fmha_prefill_kvcache_kernels)

add_sycl_to_target(TARGET fmha_prefill_kvcache)

add_dependencies(cutlass_examples fmha_prefill_kvcache)

install(
  TARGETS fmha_prefill_kvcache
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TESTING AND FMHA_STANDALONE_ENABLE_TESTS)
  include("${CMAKE_CURRENT_LIST_DIR}/tests/model_cases.cmake")
endif()
