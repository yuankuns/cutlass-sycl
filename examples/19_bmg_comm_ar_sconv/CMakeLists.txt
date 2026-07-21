# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_TEST_FOR_CRI)
  set(COMM_AR_SCONV_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(COMM_AR_SCONV_QUICK_ARGS "--suite=quick" "--iterations=3" "--verify=1")
endif()

function(cutlass_example_add_optional_comm_ar_sconv target source)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    cutlass_example_add_executable(
      ${target}
      ${source}
      TEST_COMMAND_OPTIONS
      COMM_AR_SCONV_QUICK_ARGS
    )
  endif()
endfunction()

cutlass_example_add_optional_comm_ar_sconv(19_bmg_all_reduce_variants 19_bmg_all_reduce_variants.cpp)
cutlass_example_add_optional_comm_ar_sconv(19_bmg_ar_fused_decode 19_bmg_ar_fused_decode.cpp)
cutlass_example_add_optional_comm_ar_sconv(19_bmg_ar_scattered_sconv 19_bmg_ar_scattered_sconv.cpp)
cutlass_example_add_optional_comm_ar_sconv(19_bmg_xpu_collective_mapping 19_bmg_xpu_collective_mapping.cpp)
