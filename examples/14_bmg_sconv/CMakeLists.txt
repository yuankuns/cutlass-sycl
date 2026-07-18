# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(SCONV_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(SCONV_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  14_bmg_sconv
  14_bmg_sconv.cpp
  TEST_COMMAND_OPTIONS
  SCONV_QUICK_ARGS
)

cutlass_example_add_executable(
  14_bmg_update_sconv_cache
  14_bmg_update_sconv_cache.cpp
  TEST_COMMAND_OPTIONS
  SCONV_QUICK_ARGS
)
