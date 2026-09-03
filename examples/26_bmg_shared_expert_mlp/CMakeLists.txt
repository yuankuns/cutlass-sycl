# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_SHARED_EXPERT_MLP_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_SHARED_EXPERT_MLP_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  26_bmg_shared_expert_mlp
  26_bmg_shared_expert_mlp.cpp
  TEST_COMMAND_OPTIONS
  BMG_SHARED_EXPERT_MLP_QUICK_ARGS
)

# The two GEMMs use oneMKL's bf16 x bf16 -> bf16 (fp32 compute) gemm/gemm_batch,
# mirroring torch.mm / torch.bmm. Link discovery follows 20_bmg_dflash.
if (TARGET MKL::MKL_SYCL::BLAS)
  target_link_libraries(26_bmg_shared_expert_mlp PRIVATE MKL::MKL_SYCL::BLAS)
elseif (TARGET MKL::MKL_SYCL)
  target_link_libraries(26_bmg_shared_expert_mlp PRIVATE MKL::MKL_SYCL)
else()
  find_library(
    BMG_SHARED_EXPERT_MLP_MKL_SYCL_BLAS
    NAMES mkl_sycl_blas
    HINTS
      "$ENV{MKLROOT}/lib"
      "/opt/intel/oneapi/mkl/latest/lib"
      "/opt/intel/oneapi/latest/lib"
  )
  if (BMG_SHARED_EXPERT_MLP_MKL_SYCL_BLAS)
    target_link_libraries(26_bmg_shared_expert_mlp PRIVATE ${BMG_SHARED_EXPERT_MLP_MKL_SYCL_BLAS})
  else()
    message(FATAL_ERROR "26_bmg_shared_expert_mlp requires libmkl_sycl_blas for the oneMKL bf16 GEMMs")
  endif()
endif()

get_filename_component(BMG_SHARED_EXPERT_MLP_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(BMG_SHARED_EXPERT_MLP_COMPILER_ROOT "${BMG_SHARED_EXPERT_MLP_COMPILER_BIN_DIR}/.." ABSOLUTE)
get_filename_component(BMG_SHARED_EXPERT_MLP_ONEAPI_ROOT "${BMG_SHARED_EXPERT_MLP_COMPILER_ROOT}/../.." ABSOLUTE)
get_filename_component(BMG_SHARED_EXPERT_MLP_COMPILER_VERSION "${BMG_SHARED_EXPERT_MLP_COMPILER_ROOT}" NAME)
set(BMG_SHARED_EXPERT_MLP_COMPILER_LIB_DIR "${BMG_SHARED_EXPERT_MLP_COMPILER_ROOT}/lib")
set(BMG_SHARED_EXPERT_MLP_ONEAPI_LIB_DIR "${BMG_SHARED_EXPERT_MLP_ONEAPI_ROOT}/${BMG_SHARED_EXPERT_MLP_COMPILER_VERSION}/lib")
set(BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIB_DIRS)
if (EXISTS "${BMG_SHARED_EXPERT_MLP_COMPILER_LIB_DIR}/libsycl.so" OR EXISTS "${BMG_SHARED_EXPERT_MLP_COMPILER_LIB_DIR}/libiomp5.so")
  list(APPEND BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIB_DIRS "${BMG_SHARED_EXPERT_MLP_COMPILER_LIB_DIR}")
endif()
if (EXISTS "${BMG_SHARED_EXPERT_MLP_ONEAPI_LIB_DIR}/libur_adapter_level_zero.so" OR
    EXISTS "${BMG_SHARED_EXPERT_MLP_ONEAPI_LIB_DIR}/libur_adapter_level_zero.so.0")
  list(APPEND BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIB_DIRS "${BMG_SHARED_EXPERT_MLP_ONEAPI_LIB_DIR}")
endif()
if (BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIB_DIRS)
  set_property(TARGET 26_bmg_shared_expert_mlp APPEND PROPERTY BUILD_RPATH ${BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIB_DIRS})
endif()
list(JOIN BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIB_DIRS ":" BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIBRARY_PATH)

set_tests_properties(
  ctest_examples_26_bmg_shared_expert_mlp
  PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "LD_LIBRARY_PATH=${BMG_SHARED_EXPERT_MLP_SYCL_RUNTIME_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
)
