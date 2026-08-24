# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(FMHA_DECODE_TEST_ARGS
  "--seq_len_kv=4097"
  "--num_kv_splits=0"
  "--softmax_scale=1.0"
  "--iterations=2"
  "--warmup=1"
  "--verify=1"
)

cutlass_example_add_executable(
  15_fmha_decode_kvcache
  15_fmha_decode_kvcache.cpp
  TEST_COMMAND_OPTIONS
  FMHA_DECODE_TEST_ARGS
)

# The shared helper uses a relative bin directory. This standalone test is
# registered in its own build subdirectory, so give CTest the top-level path.
set_tests_properties(
  ctest_examples_15_fmha_decode_kvcache
  PROPERTIES
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Relative-attention coverage. The bias barely moves O at realistic magnitudes
# -- O averages thousands of random V rows, so a reweighted softmax stays inside
# the reference tolerance and a kernel that ignored the bias would still pass --
# so this case drops the Q*K term and widens the bias until the softmax is
# decided by the bias alone. rel_extent 128 puts the band's window in the middle
# of the surface (col_origin 3968), exercising the shear rather than just the
# origin-zero case; see the README for the wider sweep.
add_test(
  NAME ctest_examples_15_fmha_decode_kvcache_relative
  COMMAND 15_fmha_decode_kvcache
    --seq_len_kv=4097
    --rel_extent=128
    --softmax_scale=0
    --rel_bias_range=8
    --iterations=2
    --warmup=1
    --verify=1
)

# This example carries the exact Split-K decode implementation used by the
# SGLang XPU path. Keep it private so existing prefill examples retain their
# current application-level FMHA implementation.
target_include_directories(
  15_fmha_decode_kvcache
  BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
