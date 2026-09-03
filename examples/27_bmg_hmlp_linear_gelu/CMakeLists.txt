# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_HMLP_LINEAR_GELU_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_HMLP_LINEAR_GELU_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  27_bmg_hmlp_linear_gelu
  27_bmg_hmlp_linear_gelu.cpp
  TEST_COMMAND_OPTIONS
  BMG_HMLP_LINEAR_GELU_QUICK_ARGS
)

# The Linear is delegated to oneMKL row-major GEMM (same dependency handling as
# examples/20_bmg_dflash).
if (TARGET MKL::MKL_SYCL::BLAS)
  target_link_libraries(27_bmg_hmlp_linear_gelu PRIVATE MKL::MKL_SYCL::BLAS)
elseif (TARGET MKL::MKL_SYCL)
  target_link_libraries(27_bmg_hmlp_linear_gelu PRIVATE MKL::MKL_SYCL)
else()
  find_library(
    BMG_HMLP_LINEAR_GELU_MKL_SYCL_BLAS
    NAMES mkl_sycl_blas
    HINTS
      "$ENV{MKLROOT}/lib"
      "/opt/intel/oneapi/mkl/latest/lib"
      "/opt/intel/oneapi/latest/lib"
  )
  if (BMG_HMLP_LINEAR_GELU_MKL_SYCL_BLAS)
    target_link_libraries(27_bmg_hmlp_linear_gelu PRIVATE ${BMG_HMLP_LINEAR_GELU_MKL_SYCL_BLAS})
  else()
    message(FATAL_ERROR "27_bmg_hmlp_linear_gelu requires libmkl_sycl_blas for the oneMKL GEMM")
  endif()
endif()

get_filename_component(BMG_HMLP_LG_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(BMG_HMLP_LG_COMPILER_ROOT "${BMG_HMLP_LG_COMPILER_BIN_DIR}/.." ABSOLUTE)
get_filename_component(BMG_HMLP_LG_ONEAPI_ROOT "${BMG_HMLP_LG_COMPILER_ROOT}/../.." ABSOLUTE)
get_filename_component(BMG_HMLP_LG_COMPILER_VERSION "${BMG_HMLP_LG_COMPILER_ROOT}" NAME)
set(BMG_HMLP_LG_COMPILER_LIB_DIR "${BMG_HMLP_LG_COMPILER_ROOT}/lib")
set(BMG_HMLP_LG_ONEAPI_LIB_DIR "${BMG_HMLP_LG_ONEAPI_ROOT}/${BMG_HMLP_LG_COMPILER_VERSION}/lib")
set(BMG_HMLP_LG_SYCL_RUNTIME_LIB_DIRS)
if (EXISTS "${BMG_HMLP_LG_COMPILER_LIB_DIR}/libsycl.so" OR EXISTS "${BMG_HMLP_LG_COMPILER_LIB_DIR}/libiomp5.so")
  list(APPEND BMG_HMLP_LG_SYCL_RUNTIME_LIB_DIRS "${BMG_HMLP_LG_COMPILER_LIB_DIR}")
endif()
if (EXISTS "${BMG_HMLP_LG_ONEAPI_LIB_DIR}/libur_adapter_level_zero.so" OR
    EXISTS "${BMG_HMLP_LG_ONEAPI_LIB_DIR}/libur_adapter_level_zero.so.0")
  list(APPEND BMG_HMLP_LG_SYCL_RUNTIME_LIB_DIRS "${BMG_HMLP_LG_ONEAPI_LIB_DIR}")
endif()
if (BMG_HMLP_LG_SYCL_RUNTIME_LIB_DIRS)
  set_property(TARGET 27_bmg_hmlp_linear_gelu APPEND PROPERTY BUILD_RPATH ${BMG_HMLP_LG_SYCL_RUNTIME_LIB_DIRS})
endif()
list(JOIN BMG_HMLP_LG_SYCL_RUNTIME_LIB_DIRS ":" BMG_HMLP_LG_SYCL_RUNTIME_LIBRARY_PATH)

if (TEST ctest_examples_27_bmg_hmlp_linear_gelu)
  set_tests_properties(
    ctest_examples_27_bmg_hmlp_linear_gelu
    PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "LD_LIBRARY_PATH=${BMG_HMLP_LG_SYCL_RUNTIME_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
  )
endif()
