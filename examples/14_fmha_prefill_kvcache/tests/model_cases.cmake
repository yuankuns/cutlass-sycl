function(fmha_prefill_add_case NAME)
  set(options PAGED CAUSAL SINK RELATIVE_BIAS MAIN STRETCH COVERAGE BOUNDARY)
  set(one_value_args
      BATCH
      SEQLEN_Q
      SEQLEN_K
      HEADS_Q
      HEADS_KV
      HEAD_DIM
      HEAD_DIM_V
      PAGE_SIZE
      WINDOW_LEFT
      WINDOW_RIGHT
      REL_EXTENT
      WARMUP
      ITERS
      ATOL
      RTOL)
  cmake_parse_arguments(CASE "${options}" "${one_value_args}" "" ${ARGN})

  foreach(required_arg BATCH SEQLEN_Q SEQLEN_K HEADS_Q HEADS_KV HEAD_DIM)
    if(NOT DEFINED CASE_${required_arg})
      message(FATAL_ERROR "fmha_prefill_add_case(${NAME}) missing ${required_arg}")
    endif()
  endforeach()

  list(FIND FMHA_STANDALONE_HEAD_DIMS "${CASE_HEAD_DIM}" _head_dim_index)
  if(_head_dim_index EQUAL -1)
    message(STATUS "Skipping FMHA prefill test ${NAME}: head_dim=${CASE_HEAD_DIM} was not instantiated")
    return()
  endif()

  if(CASE_PAGED)
    set(_paged 1)
    set(_supported_head_dims ${FMHA_PREFILL_PAGED_HEAD_DIMS})
  else()
    set(_paged 0)
    set(_supported_head_dims ${FMHA_PREFILL_NP_HEAD_DIMS})
  endif()
  list(FIND _supported_head_dims "${CASE_HEAD_DIM}" _mode_head_dim_index)
  if(_mode_head_dim_index EQUAL -1)
    message(STATUS "Skipping FMHA prefill test ${NAME}: head_dim=${CASE_HEAD_DIM} is not supported for paged=${_paged}")
    return()
  endif()

  if(CASE_CAUSAL)
    set(_causal 1)
  else()
    set(_causal 0)
  endif()

  if(CASE_SINK)
    set(_sink 1)
  else()
    set(_sink 0)
  endif()
  if(CASE_RELATIVE_BIAS)
    set(_relative_bias 1)
  else()
    set(_relative_bias 0)
  endif()
  if(DEFINED CASE_REL_EXTENT)
    set(_rel_extent ${CASE_REL_EXTENT})
  else()
    set(_rel_extent 1024)
  endif()

  if(DEFINED CASE_HEAD_DIM_V)
    set(_head_dim_v ${CASE_HEAD_DIM_V})
  else()
    set(_head_dim_v ${CASE_HEAD_DIM})
  endif()

  if(DEFINED CASE_PAGE_SIZE)
    set(_page_size ${CASE_PAGE_SIZE})
  else()
    set(_page_size 64)
  endif()

  if(DEFINED CASE_WINDOW_LEFT)
    set(_window_left ${CASE_WINDOW_LEFT})
  else()
    set(_window_left -1)
  endif()

  if(DEFINED CASE_WINDOW_RIGHT)
    set(_window_right ${CASE_WINDOW_RIGHT})
  else()
    set(_window_right -1)
  endif()

  if(DEFINED CASE_WARMUP)
    set(_warmup ${CASE_WARMUP})
  else()
    set(_warmup ${FMHA_STANDALONE_TEST_WARMUP})
  endif()

  if(DEFINED CASE_ITERS)
    set(_iters ${CASE_ITERS})
  else()
    set(_iters ${FMHA_STANDALONE_TEST_ITERS})
  endif()

  if(DEFINED CASE_ATOL)
    set(_atol ${CASE_ATOL})
  else()
    set(_atol ${FMHA_STANDALONE_TEST_ATOL})
  endif()

  if(DEFINED CASE_RTOL)
    set(_rtol ${CASE_RTOL})
  else()
    set(_rtol ${FMHA_STANDALONE_TEST_RTOL})
  endif()

  set(_test_name "fmha_prefill.${NAME}")
  add_test(
    NAME ${_test_name}
    COMMAND
      $<TARGET_FILE:fmha_prefill_kvcache>
      --batch ${CASE_BATCH}
      --seqlen-q ${CASE_SEQLEN_Q}
      --seqlen-k ${CASE_SEQLEN_K}
      --heads-q ${CASE_HEADS_Q}
      --heads-kv ${CASE_HEADS_KV}
      --head-dim ${CASE_HEAD_DIM}
      --head-dim-v ${_head_dim_v}
      --paged ${_paged}
      --page-size ${_page_size}
      --causal ${_causal}
      --window-left ${_window_left}
      --window-right ${_window_right}
      --sink ${_sink}
      --relative-bias ${_relative_bias}
      --rel-extent ${_rel_extent}
      --warmup ${_warmup}
      --iters ${_iters}
      --verify 1
      --atol ${_atol}
      --rtol ${_rtol})

  set(_labels fmha_prefill)
  if(CASE_MAIN)
    list(APPEND _labels main model)
  endif()
  if(CASE_STRETCH)
    list(APPEND _labels stretch model)
  endif()
  if(CASE_COVERAGE)
    list(APPEND _labels coverage)
  endif()
  if(CASE_BOUNDARY)
    list(APPEND _labels boundary coverage)
  endif()
  if(CASE_PAGED)
    list(APPEND _labels paged)
  else()
    list(APPEND _labels nonpaged)
  endif()
  if(CASE_CAUSAL)
    list(APPEND _labels causal)
  else()
    list(APPEND _labels noncausal)
  endif()
  if(CASE_SINK)
    list(APPEND _labels sink)
  endif()
  if(CASE_RELATIVE_BIAS)
    list(APPEND _labels relative)
  endif()

  set_tests_properties(
    ${_test_name}
    PROPERTIES
      LABELS "${_labels}"
      TIMEOUT ${FMHA_STANDALONE_TEST_TIMEOUT})
endfunction()

# Model-named regression scenarios. Shapes are intentionally scaled down for
# correctness testing while preserving attention traits: head_dim, GQA/MQA
# grouping, paged/non-paged cache, causal/full/local masks, and V head size.

# Main goal
fmha_prefill_add_case(main.gemma4_26b
  MAIN PAGED CAUSAL
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 128 HEADS_Q 16 HEADS_KV 8 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(main.gemma4_31b
  MAIN PAGED CAUSAL
  BATCH 1 SEQLEN_Q 24 SEQLEN_K 128 HEADS_Q 16 HEADS_KV 4 HEAD_DIM 256 PAGE_SIZE 64)

fmha_prefill_add_case(main.qwen3_32b
  MAIN PAGED CAUSAL
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 192 HEADS_Q 16 HEADS_KV 4 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(main.qwen3_30b_a3b
  MAIN PAGED CAUSAL
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 128 HEADS_Q 16 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(main.flux2_dev
  MAIN
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 32 HEADS_Q 8 HEADS_KV 8 HEAD_DIM 128)

# Stretch goal
fmha_prefill_add_case(stretch.qwen35_9b
  STRETCH PAGED CAUSAL
  BATCH 1 SEQLEN_Q 24 SEQLEN_K 128 HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(stretch.qwen35_35b_a3b
  STRETCH PAGED
  BATCH 1 SEQLEN_Q 24 SEQLEN_K 192 HEADS_Q 16 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64
  WINDOW_LEFT 96 WINDOW_RIGHT 0)

fmha_prefill_add_case(stretch.deepseek_ocr2
  STRETCH
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 64 HEADS_Q 8 HEADS_KV 8 HEAD_DIM 96
  WINDOW_LEFT 48 WINDOW_RIGHT 16)

fmha_prefill_add_case(stretch.nemotron3_nano_30b_a3b
  STRETCH PAGED CAUSAL
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 128 HEADS_Q 16 HEADS_KV 4 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(stretch.flux2_klein_4b
  STRETCH
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 32 HEADS_Q 8 HEADS_KV 8 HEAD_DIM 64)

fmha_prefill_add_case(stretch.flux2_klein_9b
  STRETCH
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 64 HEADS_Q 12 HEADS_KV 12 HEAD_DIM 96)

fmha_prefill_add_case(stretch.qwen3_tts
  STRETCH PAGED CAUSAL
  BATCH 1 SEQLEN_Q 24 SEQLEN_K 128 HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(stretch.qwen36_35b_a3b
  STRETCH PAGED CAUSAL
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 256 HEADS_Q 16 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(stretch.z_image_turbo
  STRETCH
  BATCH 1 SEQLEN_Q 32 SEQLEN_K 32 HEADS_Q 8 HEADS_KV 8 HEAD_DIM 64)

# Supplemental kernel-coverage scenarios.
fmha_prefill_add_case(coverage.paged_hd192
  COVERAGE PAGED CAUSAL
  BATCH 1 SEQLEN_Q 16 SEQLEN_K 128 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 192 PAGE_SIZE 64)

fmha_prefill_add_case(coverage.paged_hd512_mqa
  COVERAGE PAGED CAUSAL
  BATCH 1 SEQLEN_Q 8 SEQLEN_K 64 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 512 PAGE_SIZE 64)

fmha_prefill_add_case(coverage.nonpaged_hd72
  COVERAGE CAUSAL
  BATCH 1 SEQLEN_Q 16 SEQLEN_K 64 HEADS_Q 4 HEADS_KV 2 HEAD_DIM 72)

fmha_prefill_add_case(coverage.dv128_from_d64
  COVERAGE PAGED CAUSAL
  BATCH 1 SEQLEN_Q 16 SEQLEN_K 64 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 64 HEAD_DIM_V 128 PAGE_SIZE 64)

fmha_prefill_add_case(coverage.sink_paged_hd64
  COVERAGE PAGED CAUSAL SINK
  BATCH 1 SEQLEN_Q 16 SEQLEN_K 64 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 64 PAGE_SIZE 64)

fmha_prefill_add_case(coverage.local_window_paged
  COVERAGE PAGED
  BATCH 1 SEQLEN_Q 24 SEQLEN_K 128 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 96 PAGE_SIZE 64
  WINDOW_LEFT 64 WINDOW_RIGHT 8)

# Relative-attention coverage on the scoreboard hd128 paged path.
fmha_prefill_add_case(relative.single_q_tile_gqa
  COVERAGE RELATIVE_BIAS PAGED CAUSAL
  BATCH 1 SEQLEN_Q 256 SEQLEN_K 512 HEADS_Q 16 HEADS_KV 8
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 128
  ATOL 5e-3 RTOL 5e-3)

fmha_prefill_add_case(relative.multi_q_tile_gqa
  COVERAGE RELATIVE_BIAS PAGED CAUSAL
  BATCH 1 SEQLEN_Q 512 SEQLEN_K 1024 HEADS_Q 16 HEADS_KV 8
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 128
  ATOL 5e-3 RTOL 5e-3)

fmha_prefill_add_case(relative.qk_tail_mqa
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 1 SEQLEN_Q 300 SEQLEN_K 777 HEADS_Q 4 HEADS_KV 1
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 127
  ATOL 5e-3 RTOL 5e-3)

fmha_prefill_add_case(relative.multi_batch_gqa
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 2 SEQLEN_Q 300 SEQLEN_K 777 HEADS_Q 4 HEADS_KV 2
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 128
  ATOL 5e-3 RTOL 5e-3)

fmha_prefill_add_case(relative.extent_one
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 1 SEQLEN_Q 257 SEQLEN_K 513 HEADS_Q 4 HEADS_KV 1
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 64 REL_EXTENT 1
  ATOL 5e-3 RTOL 5e-3)

fmha_prefill_add_case(relative.extent_larger_than_k
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 1 SEQLEN_Q 256 SEQLEN_K 257 HEADS_Q 4 HEADS_KV 2
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 64 REL_EXTENT 1024
  ATOL 5e-3 RTOL 5e-3)

fmha_prefill_add_case(relative.noncausal_k_tail
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED
  BATCH 1 SEQLEN_Q 256 SEQLEN_K 257 HEADS_Q 4 HEADS_KV 2
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 64 REL_EXTENT 33
  ATOL 5e-3 RTOL 5e-3)

# The bias columns are sheared, so seqlen_k no longer sets the surface geometry.  What
# seqlen_k = 1000 still exercises is the sequence tail: it is not a multiple of the 32-wide
# K tile, so the last block hangs over the end of the sequence.  Two batches and four heads
# also exercise the row/column offsets folded into the surface coordinates.
fmha_prefill_add_case(relative.block2d_k_tail
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 2 SEQLEN_Q 300 SEQLEN_K 1000 HEADS_Q 4 HEADS_KV 2
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 128
  ATOL 5e-3 RTOL 5e-3)

# The three cases below pin rel_bias_can_block_2d.  The column count is rel_extent + 288,
# so rel_extent is what moves the strides now; Inkling's rel_extent % 128 == 0 always lands
# on the fast path, and these are the shapes that do not.
#
# 33 + 288 = 321 columns, an odd width and an odd pitch: no surface can describe it, so the
# whole launch takes the scalar load.
fmha_prefill_add_case(relative.odd_extent_scalar
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 2 SEQLEN_Q 300 SEQLEN_K 512 HEADS_Q 1 HEADS_KV 1
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 33
  ATOL 5e-3 RTOL 5e-3)

# 132 + 288 = 420 columns with one head, so the pitch is 420 elements = 840B: a multiple of
# 4B but not of the 16B a surface pitch needs.  cute's assert does not catch that, so before
# rel_bias_can_block_2d tested for 16B this took the block 2D load and read shifted columns.
fmha_prefill_add_case(relative.unaligned_pitch
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 2 SEQLEN_Q 300 SEQLEN_K 512 HEADS_Q 1 HEADS_KV 1
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 132
  ATOL 5e-3 RTOL 5e-3)

# The mirror image: 130 + 288 = 418 columns is only 2-element aligned, but four heads make
# the pitch 1672 elements = 3344B, a multiple of 16B, so the block 2D load applies.  The
# head column offsets are then 4B- but not 8B-aligned, which is all the x offset needs.
fmha_prefill_add_case(relative.unaligned_head_stride
  COVERAGE BOUNDARY RELATIVE_BIAS PAGED CAUSAL
  BATCH 2 SEQLEN_Q 300 SEQLEN_K 512 HEADS_Q 4 HEADS_KV 2
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 130
  ATOL 5e-3 RTOL 5e-3)

fmha_prefill_add_case(relative.production_4k
  COVERAGE RELATIVE_BIAS PAGED CAUSAL
  BATCH 1 SEQLEN_Q 4096 SEQLEN_K 4096 HEADS_Q 1 HEADS_KV 1
  HEAD_DIM 128 HEAD_DIM_V 128 PAGE_SIZE 128 REL_EXTENT 1024
  WARMUP 1 ITERS 1
  ATOL 5e-3 RTOL 5e-3)

# Tile-boundary scenarios.  These keep head counts small so the CPU reference
# remains practical, but they cover exact tile multiples, non-multiples, +/-1,
# +/-2, half-tile boundaries, page tails, and multi-batch offsets.
fmha_prefill_add_case(boundary.paged_hd128_q_half_minus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q 63 SEQLEN_K 256 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd128_q_half_plus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q 65 SEQLEN_K 256 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd128_q_tile_minus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q 127 SEQLEN_K 256 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd128_q_tile
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q 128 SEQLEN_K 256 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd128_q_tile_plus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q 129 SEQLEN_K 256 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd128_k_tile_minus1
  BOUNDARY PAGED
  BATCH 1 SEQLEN_Q 64 SEQLEN_K 63 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd128_k_tile
  BOUNDARY PAGED
  BATCH 1 SEQLEN_Q 64 SEQLEN_K 64 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd128_k_tile_plus1
  BOUNDARY PAGED
  BATCH 1 SEQLEN_Q 64 SEQLEN_K 65 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.nonpaged_hd128_q_tile_minus1
  BOUNDARY CAUSAL
  BATCH 1 SEQLEN_Q 255 SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128)

fmha_prefill_add_case(boundary.nonpaged_hd128_q_tile
  BOUNDARY CAUSAL
  BATCH 1 SEQLEN_Q 256 SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128)

fmha_prefill_add_case(boundary.nonpaged_hd128_q_tile_plus1
  BOUNDARY CAUSAL
  BATCH 1 SEQLEN_Q 257 SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128)

fmha_prefill_add_case(boundary.nonpaged_hd128_k_half_minus1
  BOUNDARY
  BATCH 1 SEQLEN_Q 128 SEQLEN_K 15 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128)

fmha_prefill_add_case(boundary.nonpaged_hd128_k_half_plus1
  BOUNDARY
  BATCH 1 SEQLEN_Q 128 SEQLEN_K 17 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128)

fmha_prefill_add_case(boundary.paged_hd256_q_tile_plus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q 257 SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 256 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.page128_k_page_minus1
  BOUNDARY PAGED
  BATCH 1 SEQLEN_Q 64 SEQLEN_K 127 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(boundary.page128_k_page_plus1
  BOUNDARY PAGED
  BATCH 1 SEQLEN_Q 64 SEQLEN_K 129 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(boundary.batch3_paged_q_plus1_k_plus1
  BOUNDARY PAGED CAUSAL
  BATCH 3 SEQLEN_Q 129 SEQLEN_K 257 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)
