# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(ATTN_PROLOGUE_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(ATTN_PROLOGUE_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  15_bmg_attn_prologue_verify
  15_bmg_attn_prologue_verify.cpp
  TEST_COMMAND_OPTIONS
  ATTN_PROLOGUE_QUICK_ARGS
)

cutlass_example_add_executable(
  15_bmg_attn_prologue_extend
  15_bmg_attn_prologue_extend.cpp
  TEST_COMMAND_OPTIONS
  ATTN_PROLOGUE_QUICK_ARGS
)

cutlass_example_add_executable(
  15_bmg_attn_prologue_decode
  15_bmg_attn_prologue_decode.cpp
  TEST_COMMAND_OPTIONS
  ATTN_PROLOGUE_QUICK_ARGS
)

cutlass_example_add_executable(
  15_bmg_attn_prologue_mxfp8_store_tau
  15_bmg_attn_prologue_mxfp8_store_tau.cpp
  TEST_COMMAND_OPTIONS
  ATTN_PROLOGUE_QUICK_ARGS
)
