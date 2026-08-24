# Copyright (C) 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if(NOT "${DPCPP_HOST_COMPILER}" MATCHES "g\\+\\+")
  cutlass_example_add_executable(
    16_bmg_moe_gemm
    16_bmg_moe_gemm.cpp
  )
endif()
