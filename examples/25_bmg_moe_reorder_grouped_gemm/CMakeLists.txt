# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_MOE_REORDER_GROUPED_GEMM_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_MOE_REORDER_GROUPED_GEMM_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/25_bmg_moe_reorder_grouped_gemm.cpp")
  cutlass_example_add_executable(
    25_bmg_moe_reorder_grouped_gemm
    25_bmg_moe_reorder_grouped_gemm.cpp
    TEST_COMMAND_OPTIONS
    BMG_MOE_REORDER_GROUPED_GEMM_QUICK_ARGS
  )
endif()
