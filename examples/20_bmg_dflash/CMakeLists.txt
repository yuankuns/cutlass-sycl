# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_DFLASH_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_DFLASH_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  20_bmg_dflash_device_guard
  20_bmg_dflash_device_guard.cpp
  TEST_COMMAND_OPTIONS
  BMG_DFLASH_QUICK_ARGS
)

cutlass_example_add_executable(
  20_bmg_dflash_cache_path
  20_bmg_dflash_cache_path.cpp
  TEST_COMMAND_OPTIONS
  BMG_DFLASH_QUICK_ARGS
)

if (TARGET MKL::MKL_SYCL::BLAS)
  target_link_libraries(20_bmg_dflash_cache_path PRIVATE MKL::MKL_SYCL::BLAS)
elseif (TARGET MKL::MKL_SYCL)
  target_link_libraries(20_bmg_dflash_cache_path PRIVATE MKL::MKL_SYCL)
else()
  find_library(
    BMG_DFLASH_MKL_SYCL_BLAS
    NAMES mkl_sycl_blas
    HINTS
      "$ENV{MKLROOT}/lib"
      "/opt/intel/oneapi/mkl/latest/lib"
      "/opt/intel/oneapi/latest/lib"
  )
  if (BMG_DFLASH_MKL_SYCL_BLAS)
    target_link_libraries(20_bmg_dflash_cache_path PRIVATE ${BMG_DFLASH_MKL_SYCL_BLAS})
  else()
    message(FATAL_ERROR "20_bmg_dflash_cache_path requires libmkl_sycl_blas for oneMKL GEMM")
  endif()
endif()

cutlass_example_add_executable(
  20_bmg_scatter_mamba_states_after_mtp_verify
  20_bmg_scatter_mamba_states_after_mtp_verify.cpp
  TEST_COMMAND_OPTIONS
  BMG_DFLASH_QUICK_ARGS
)

get_filename_component(BMG_DFLASH_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(BMG_DFLASH_COMPILER_ROOT "${BMG_DFLASH_COMPILER_BIN_DIR}/.." ABSOLUTE)
get_filename_component(BMG_DFLASH_ONEAPI_ROOT "${BMG_DFLASH_COMPILER_ROOT}/../.." ABSOLUTE)
get_filename_component(BMG_DFLASH_COMPILER_VERSION "${BMG_DFLASH_COMPILER_ROOT}" NAME)
set(BMG_DFLASH_COMPILER_LIB_DIR "${BMG_DFLASH_COMPILER_ROOT}/lib")
set(BMG_DFLASH_ONEAPI_LIB_DIR "${BMG_DFLASH_ONEAPI_ROOT}/${BMG_DFLASH_COMPILER_VERSION}/lib")
set(BMG_DFLASH_SYCL_RUNTIME_LIB_DIRS)
if (EXISTS "${BMG_DFLASH_COMPILER_LIB_DIR}/libsycl.so" OR EXISTS "${BMG_DFLASH_COMPILER_LIB_DIR}/libiomp5.so")
  list(APPEND BMG_DFLASH_SYCL_RUNTIME_LIB_DIRS "${BMG_DFLASH_COMPILER_LIB_DIR}")
endif()
if (EXISTS "${BMG_DFLASH_ONEAPI_LIB_DIR}/libur_adapter_level_zero.so" OR
    EXISTS "${BMG_DFLASH_ONEAPI_LIB_DIR}/libur_adapter_level_zero.so.0")
  list(APPEND BMG_DFLASH_SYCL_RUNTIME_LIB_DIRS "${BMG_DFLASH_ONEAPI_LIB_DIR}")
endif()
if (BMG_DFLASH_SYCL_RUNTIME_LIB_DIRS)
  foreach(_bmg_dflash_target
      20_bmg_dflash_device_guard
      20_bmg_dflash_cache_path
      20_bmg_scatter_mamba_states_after_mtp_verify)
    set_property(TARGET ${_bmg_dflash_target} APPEND PROPERTY BUILD_RPATH ${BMG_DFLASH_SYCL_RUNTIME_LIB_DIRS})
  endforeach()
endif()
list(JOIN BMG_DFLASH_SYCL_RUNTIME_LIB_DIRS ":" BMG_DFLASH_SYCL_RUNTIME_LIBRARY_PATH)

foreach(_bmg_dflash_test
    ctest_examples_20_bmg_dflash_device_guard
    ctest_examples_20_bmg_dflash_cache_path
    ctest_examples_20_bmg_scatter_mamba_states_after_mtp_verify)
  set_tests_properties(
    ${_bmg_dflash_test}
    PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "LD_LIBRARY_PATH=${BMG_DFLASH_SYCL_RUNTIME_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
  )
endforeach()
