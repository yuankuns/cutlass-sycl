# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_QUANTIZATION_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_QUANTIZATION_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

cutlass_example_add_executable(
  21_bmg_nvfp4_layout
  21_bmg_nvfp4_layout.cpp
  TEST_COMMAND_OPTIONS
  BMG_QUANTIZATION_QUICK_ARGS
)

cutlass_example_add_executable(
  21_bmg_mxfp4_mapping
  21_bmg_mxfp4_mapping.cpp
  TEST_COMMAND_OPTIONS
  BMG_QUANTIZATION_QUICK_ARGS
)
