# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_RELATIVE_ATTENTION_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_RELATIVE_ATTENTION_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  17_bmg_score_mod_relative_bias
  17_bmg_score_mod_relative_bias.cpp
  TEST_COMMAND_OPTIONS
  BMG_RELATIVE_ATTENTION_QUICK_ARGS
)

cutlass_example_add_executable(
  17_bmg_xpu_flash_attention_window
  17_bmg_xpu_flash_attention_window.cpp
  TEST_COMMAND_OPTIONS
  BMG_RELATIVE_ATTENTION_QUICK_ARGS
)
