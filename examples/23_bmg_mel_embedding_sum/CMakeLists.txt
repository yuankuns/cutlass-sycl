# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(BMG_MEL_EMBEDDING_SUM_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_MEL_EMBEDDING_SUM_QUICK_ARGS "--suite=quick" "--iterations=5" "--verify=1")
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/23_bmg_mel_embedding_sum.cpp")
  cutlass_example_add_executable(
    23_bmg_mel_embedding_sum
    23_bmg_mel_embedding_sum.cpp
    TEST_COMMAND_OPTIONS
    BMG_MEL_EMBEDDING_SUM_QUICK_ARGS
  )
endif()
