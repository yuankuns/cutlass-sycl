# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_RELATIVE_HELPERS_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_RELATIVE_HELPERS_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  18_bmg_rel_proj_small_t
  18_bmg_rel_proj_small_t.cpp
  TEST_COMMAND_OPTIONS
  BMG_RELATIVE_HELPERS_QUICK_ARGS
)

cutlass_example_add_executable(
  18_bmg_row_scale_bf16
  18_bmg_row_scale_bf16.cpp
  TEST_COMMAND_OPTIONS
  BMG_RELATIVE_HELPERS_QUICK_ARGS
)

cutlass_example_add_executable(
  18_bmg_row_compact_bf16
  18_bmg_row_compact_bf16.cpp
  TEST_COMMAND_OPTIONS
  BMG_RELATIVE_HELPERS_QUICK_ARGS
)
