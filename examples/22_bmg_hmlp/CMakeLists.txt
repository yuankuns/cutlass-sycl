# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_HMLP_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_HMLP_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  22_bmg_hmlp_fold_timespace_to_depth
  22_bmg_hmlp_fold_timespace_to_depth.cpp
  TEST_COMMAND_OPTIONS
  BMG_HMLP_QUICK_ARGS
)
