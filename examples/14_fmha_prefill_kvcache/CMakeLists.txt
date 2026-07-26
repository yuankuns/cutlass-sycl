# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(FMHA_PREFILL_EXAMPLE_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(FMHA_PREFILL_SRC_DIR "${FMHA_PREFILL_EXAMPLE_ROOT}/src")

set(FMHA_STANDALONE_HEAD_DIMS "64;72;80;96;128;192;256;512" CACHE STRING "Head dimensions to instantiate")
set(FMHA_STANDALONE_OPT_LEVEL "-O2" CACHE STRING "SYCL device optimization level for FMHA prefill kernels")
option(FMHA_STANDALONE_ENABLE_TESTS "Register standalone FMHA prefill CTest cases" ON)
set(FMHA_STANDALONE_TEST_WARMUP "1" CACHE STRING "Warmup iterations for CTest cases")
set(FMHA_STANDALONE_TEST_ITERS "5" CACHE STRING "Measured iterations for CTest cases")
set(FMHA_STANDALONE_TEST_TIMEOUT "300" CACHE STRING "Timeout in seconds for each CTest case")
set(FMHA_STANDALONE_TEST_ATOL "0.08" CACHE STRING "Absolute tolerance for CTest correctness checks")
set(FMHA_STANDALONE_TEST_RTOL "0.08" CACHE STRING "Relative tolerance for CTest correctness checks")

set(FMHA_PREFILL_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_PREFILL_NP_HEAD_DIMS 64 72 80 96 128 192)

set(FMHA_PREFILL_TILED_Q_64 128)
set(FMHA_PREFILL_TILED_KV_64 64)
set(FMHA_PREFILL_NUM_SG_64 8)
set(FMHA_PREFILL_TILED_Q_96 128)
set(FMHA_PREFILL_TILED_KV_96 64)
set(FMHA_PREFILL_NUM_SG_96 8)
set(FMHA_PREFILL_TILED_Q_128 256)
set(FMHA_PREFILL_TILED_KV_128 32)
set(FMHA_PREFILL_NUM_SG_128 16)
set(FMHA_PREFILL_TILED_Q_192 256)
set(FMHA_PREFILL_TILED_KV_192 64)
set(FMHA_PREFILL_NUM_SG_192 32)
set(FMHA_PREFILL_TILED_Q_256 256)
set(FMHA_PREFILL_TILED_KV_256 64)
set(FMHA_PREFILL_NUM_SG_256 32)
set(FMHA_PREFILL_TILED_Q_512 256 CACHE STRING
    "Paged prefill Q tile extent for HEAD_DIM=512")
set(FMHA_PREFILL_TILED_KV_512 64 CACHE STRING
    "Paged prefill KV tile extent for HEAD_DIM=512")
set(FMHA_PREFILL_TILED_D_512 32 CACHE STRING
    "Paged prefill QK inner head tile extent for HEAD_DIM=512")
set(FMHA_PREFILL_TILED_PV_512 32 CACHE STRING
    "Paged prefill PV output sub-tile extent for HEAD_DIM=512")
set(FMHA_PREFILL_NUM_SG_512 32 CACHE STRING
    "Paged prefill subgroup count for HEAD_DIM=512")
set(FMHA_PREFILL_QK_SG_M_512 32 CACHE STRING
    "Paged prefill Q*K subgroup layout M/Q extent for HEAD_DIM=512")
set(FMHA_PREFILL_QK_SG_N_512 1 CACHE STRING
    "Paged prefill Q*K subgroup layout N/K extent for HEAD_DIM=512")
set(FMHA_PREFILL_QK_SG_K_512 1 CACHE STRING
    "Paged prefill Q*K subgroup layout inner extent for HEAD_DIM=512")
set(FMHA_PREFILL_TILED_OUT_512 256 CACHE STRING
    "Paged prefill output tile head extent for HEAD_DIM=512")
set(FMHA_PREFILL_NUM_KV_SPLITS_512 1 CACHE STRING
    "Paged prefill split-K partition count for HEAD_DIM=512")
set(FMHA_PREFILL_USE_EXPLICIT_PV_LAYOUT_512 OFF CACHE BOOL
    "Use explicit P*V subgroup layout instead of deriving it from Q*K for HEAD_DIM=512")
set(FMHA_PREFILL_ENABLE_SCORE_BLOCK2D_512 ON CACHE BOOL
    "Enable two-kernel block2D score reuse for the HEAD_DIM=512 paged identity fast path")
set(FMHA_PREFILL_PV_SG_M_512 32 CACHE STRING
    "Paged prefill P*V subgroup layout M extent for HEAD_DIM=512")
set(FMHA_PREFILL_PV_SG_N_512 1 CACHE STRING
    "Paged prefill P*V subgroup layout N/V extent for HEAD_DIM=512")
set(FMHA_PREFILL_PV_SG_K_512 1 CACHE STRING
    "Paged prefill P*V subgroup layout K extent for HEAD_DIM=512")

set(FMHA_PREFILL_TILED_Q_NP_64 128)
set(FMHA_PREFILL_TILED_KV_NP_64 64)
set(FMHA_PREFILL_NUM_SG_NP_64 8)
set(FMHA_PREFILL_TILED_Q_NP_72 256)
set(FMHA_PREFILL_TILED_KV_NP_72 64)
set(FMHA_PREFILL_NUM_SG_NP_72 16)
set(FMHA_PREFILL_TILED_Q_NP_80 256)
set(FMHA_PREFILL_TILED_KV_NP_80 64)
set(FMHA_PREFILL_NUM_SG_NP_80 16)
set(FMHA_PREFILL_TILED_Q_NP_96 256)
set(FMHA_PREFILL_TILED_KV_NP_96 64)
set(FMHA_PREFILL_NUM_SG_NP_96 16)
set(FMHA_PREFILL_TILED_Q_NP_128 256)
set(FMHA_PREFILL_TILED_KV_NP_128 32)
set(FMHA_PREFILL_NUM_SG_NP_128 16)
set(FMHA_PREFILL_TILED_Q_NP_192 256)
set(FMHA_PREFILL_TILED_KV_NP_192 32)
set(FMHA_PREFILL_NUM_SG_NP_192 16)

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

  math(EXPR TILED_OUT "((${HEAD_DIM} + 31) / 32) * 32")
  if(DEFINED FMHA_PREFILL_TILED_OUT_${HEAD_DIM})
    set(TILED_OUT ${FMHA_PREFILL_TILED_OUT_${HEAD_DIM}})
  endif()
  set(TILED_D 32)
  if(DEFINED FMHA_PREFILL_TILED_D_${HEAD_DIM})
    set(TILED_D ${FMHA_PREFILL_TILED_D_${HEAD_DIM}})
  endif()
  set(TILED_PV 32)
  if(DEFINED FMHA_PREFILL_TILED_PV_${HEAD_DIM})
    set(TILED_PV ${FMHA_PREFILL_TILED_PV_${HEAD_DIM}})
  endif()
  set(QK_SG_M ${NUM_SG})
  if(DEFINED FMHA_PREFILL_QK_SG_M_${HEAD_DIM})
    set(QK_SG_M ${FMHA_PREFILL_QK_SG_M_${HEAD_DIM}})
  endif()
  set(QK_SG_N 1)
  if(DEFINED FMHA_PREFILL_QK_SG_N_${HEAD_DIM})
    set(QK_SG_N ${FMHA_PREFILL_QK_SG_N_${HEAD_DIM}})
  endif()
  set(QK_SG_K 1)
  if(DEFINED FMHA_PREFILL_QK_SG_K_${HEAD_DIM})
    set(QK_SG_K ${FMHA_PREFILL_QK_SG_K_${HEAD_DIM}})
  endif()
  math(EXPR QK_SG_STRIDE_M "${QK_SG_N} * ${QK_SG_K}")
  set(QK_SG_STRIDE_N ${QK_SG_K})
  set(QK_SG_STRIDE_K 1)
  set(USE_EXPLICIT_PV_LAYOUT 1)
  if(DEFINED FMHA_PREFILL_USE_EXPLICIT_PV_LAYOUT_${HEAD_DIM})
    if(FMHA_PREFILL_USE_EXPLICIT_PV_LAYOUT_${HEAD_DIM})
      set(USE_EXPLICIT_PV_LAYOUT 1)
    else()
      set(USE_EXPLICIT_PV_LAYOUT 0)
    endif()
  endif()
  set(PV_SG_M ${NUM_SG})
  if(DEFINED FMHA_PREFILL_PV_SG_M_${HEAD_DIM})
    set(PV_SG_M ${FMHA_PREFILL_PV_SG_M_${HEAD_DIM}})
  endif()
  set(PV_SG_N 1)
  if(DEFINED FMHA_PREFILL_PV_SG_N_${HEAD_DIM})
    set(PV_SG_N ${FMHA_PREFILL_PV_SG_N_${HEAD_DIM}})
  endif()
  set(PV_SG_K 1)
  if(DEFINED FMHA_PREFILL_PV_SG_K_${HEAD_DIM})
    set(PV_SG_K ${FMHA_PREFILL_PV_SG_K_${HEAD_DIM}})
  endif()
  math(EXPR PV_SG_STRIDE_M "${PV_SG_N} * ${PV_SG_K}")
  set(PV_SG_STRIDE_N ${PV_SG_K})
  set(PV_SG_STRIDE_K 1)
  set(NUM_KV_SPLITS 1)
  if(DEFINED FMHA_PREFILL_NUM_KV_SPLITS_${HEAD_DIM})
    set(NUM_KV_SPLITS ${FMHA_PREFILL_NUM_KV_SPLITS_${HEAD_DIM}})
  endif()
  set(ENABLE_SCORE_BLOCK2D 0)
  if(HEAD_DIM STREQUAL "512" AND FMHA_PREFILL_ENABLE_SCORE_BLOCK2D_512)
    set(ENABLE_SCORE_BLOCK2D 1)
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
