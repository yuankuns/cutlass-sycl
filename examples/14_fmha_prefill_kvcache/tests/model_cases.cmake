function(fmha_prefill_add_case NAME)
  set(options PAGED CAUSAL SINK MAIN STRETCH COVERAGE BOUNDARY)
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

fmha_prefill_add_case(coverage.nonpaged_hd80
  COVERAGE CAUSAL
  BATCH 1 SEQLEN_Q 16 SEQLEN_K 64 HEADS_Q 4 HEADS_KV 2 HEAD_DIM 80)

fmha_prefill_add_case(coverage.nonpaged_hd192
  COVERAGE CAUSAL
  BATCH 1 SEQLEN_Q 16 SEQLEN_K 64 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 192)

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
