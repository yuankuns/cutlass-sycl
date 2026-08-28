# Copyright (C) 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if(NOT "${DPCPP_HOST_COMPILER}" MATCHES "g\\+\\+")
  cutlass_example_add_executable(
    16_bmg_moe_gemm
    16_bmg_moe_gemm.cpp
  )
  if(DPCPP_SYCL_TARGET MATCHES "bmg")
    # Match the upstream sgl-kernel W4A16 device link: this kernel requests a
    # 256-GRF workgroup and needs IGC's large-GRF allocation mode on BMG.
    target_link_options(
      16_bmg_moe_gemm
      PRIVATE
      -Xs "-options \"-cl-intel-enable-auto-large-GRF-mode\""
    )
  endif()
endif()
