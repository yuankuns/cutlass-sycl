function(fmha_prefill_add_case NAME)
  set(options PAGED CAUSAL SINK MAIN STRETCH COVERAGE BOUNDARY CHUNK APPENDKV PAGE_TABLE_RANDOM)
  set(one_value_args
      BATCH
      SEQLEN_Q
      SEQLEN_K
      SEQLEN_Q_LIST
      PAST_KV
      PAST_KV_LIST
      K_NEW_SEQLENS
      CU_SEQLENS_K_NEW
      CACHE_SEQLENS_OLD
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
  set(_test_command
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

  if(CASE_PAGE_TABLE_RANDOM)
    list(APPEND _test_command --page-table-random 1)
  endif()

  if(DEFINED CASE_SEQLEN_Q_LIST)
    list(APPEND _test_command --seqlen-q-list ${CASE_SEQLEN_Q_LIST})
  endif()
  if(DEFINED CASE_PAST_KV)
    list(APPEND _test_command --past-kv ${CASE_PAST_KV})
  endif()
  if(DEFINED CASE_PAST_KV_LIST)
    list(APPEND _test_command --past-kv-list ${CASE_PAST_KV_LIST})
  endif()
  if(DEFINED CASE_K_NEW_SEQLENS)
    list(APPEND _test_command --k-new-seqlens ${CASE_K_NEW_SEQLENS})
  endif()
  if(DEFINED CASE_CU_SEQLENS_K_NEW)
    list(APPEND _test_command --cu-seqlens-k-new ${CASE_CU_SEQLENS_K_NEW})
  endif()
  if(DEFINED CASE_CACHE_SEQLENS_OLD)
    list(APPEND _test_command --cache-seqlens-old ${CASE_CACHE_SEQLENS_OLD})
  endif()

  add_test(
    NAME ${_test_name}
    COMMAND ${_test_command})

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
  if(CASE_CHUNK)
    list(APPEND _labels chunk coverage)
  endif()
  if(CASE_APPENDKV)
    list(APPEND _labels appendkv coverage)
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
  if(CASE_PAGE_TABLE_RANDOM)
    list(APPEND _labels pagetable_random)
  endif()

  set_tests_properties(
    ${_test_name}
    PROPERTIES
      LABELS "${_labels}"
      TIMEOUT ${FMHA_STANDALONE_TEST_TIMEOUT})
endfunction()

function(fmha_prefill_add_chunk_family NAME)
  set(options PAGED)
  set(one_value_args HEAD_DIM TILE_Q HEADS_Q HEADS_KV PAGE_SIZE)
  cmake_parse_arguments(CHUNK_CASE "${options}" "${one_value_args}" "" ${ARGN})

  foreach(required_arg HEAD_DIM TILE_Q HEADS_Q HEADS_KV)
    if(NOT DEFINED CHUNK_CASE_${required_arg})
      message(FATAL_ERROR "fmha_prefill_add_chunk_family(${NAME}) missing ${required_arg}")
    endif()
  endforeach()

  if(DEFINED CHUNK_CASE_PAGE_SIZE)
    set(_chunk_page_size ${CHUNK_CASE_PAGE_SIZE})
  else()
    set(_chunk_page_size 64)
  endif()

  math(EXPR _tile_minus1 "${CHUNK_CASE_TILE_Q} - 1")
  math(EXPR _tile_plus1 "${CHUNK_CASE_TILE_Q} + 1")
  set(_past_values 0 64 129 ${_tile_minus1} ${CHUNK_CASE_TILE_Q} ${_tile_plus1})
  set(_q_values 1 32 128 ${_tile_minus1})
  list(REMOVE_DUPLICATES _past_values)
  list(REMOVE_DUPLICATES _q_values)

  foreach(_past IN LISTS _past_values)
    foreach(_q IN LISTS _q_values)
      math(EXPR _k "${_past} + ${_q}")
      if(CHUNK_CASE_PAGED)
        fmha_prefill_add_case(chunk.${NAME}_past${_past}_q${_q}
          CHUNK PAGED CAUSAL
          BATCH 1 SEQLEN_Q ${_q} SEQLEN_K ${_k} PAST_KV ${_past}
          HEADS_Q ${CHUNK_CASE_HEADS_Q} HEADS_KV ${CHUNK_CASE_HEADS_KV}
          HEAD_DIM ${CHUNK_CASE_HEAD_DIM} PAGE_SIZE ${_chunk_page_size})
      else()
        fmha_prefill_add_case(chunk.${NAME}_past${_past}_q${_q}
          CHUNK CAUSAL
          BATCH 1 SEQLEN_Q ${_q} SEQLEN_K ${_k} PAST_KV ${_past}
          HEADS_Q ${CHUNK_CASE_HEADS_Q} HEADS_KV ${CHUNK_CASE_HEADS_KV}
          HEAD_DIM ${CHUNK_CASE_HEAD_DIM})
      endif()
    endforeach()
  endforeach()
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

math(EXPR _fmha_prefill_tiled_q_256_plus1 "${FMHA_PREFILL_TILED_Q_256} + 1")

math(EXPR _fmha_prefill_tiled_q_192_minus1 "${FMHA_PREFILL_TILED_Q_192} - 1")
math(EXPR _fmha_prefill_tiled_q_192_plus1 "${FMHA_PREFILL_TILED_Q_192} + 1")

fmha_prefill_add_case(boundary.paged_hd192_q_tile_minus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q ${_fmha_prefill_tiled_q_192_minus1} SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 192 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd192_q_tile
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q ${FMHA_PREFILL_TILED_Q_192} SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 192 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd192_q_tile_plus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q ${_fmha_prefill_tiled_q_192_plus1} SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 192 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.paged_hd256_q_tile_plus1
  BOUNDARY PAGED CAUSAL
  BATCH 1 SEQLEN_Q ${_fmha_prefill_tiled_q_256_plus1} SEQLEN_K 512 HEADS_Q 2 HEADS_KV 1 HEAD_DIM 256 PAGE_SIZE 64)

fmha_prefill_add_case(boundary.page128_k_page_minus1
  BOUNDARY PAGED
  BATCH 1 SEQLEN_Q 64 SEQLEN_K 127 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(boundary.page128_k_page_plus1
  BOUNDARY PAGED
  BATCH 1 SEQLEN_Q 64 SEQLEN_K 129 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(boundary.batch3_paged_q_plus1_k_plus1
  BOUNDARY PAGED CAUSAL
  BATCH 3 SEQLEN_Q 129 SEQLEN_K 257 HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

# Chunk-prefill coverage: Q is only the new chunk while KV contains non-empty
# history plus Q. The list cases exercise heterogeneous per-batch lengths.
fmha_prefill_add_chunk_family(paged_hd64
  PAGED HEAD_DIM 64 TILE_Q 128 HEADS_Q 4 HEADS_KV 1 PAGE_SIZE 64)

fmha_prefill_add_chunk_family(paged_hd128
  PAGED HEAD_DIM 128 TILE_Q ${FMHA_PREFILL_TILED_Q_128} HEADS_Q 4 HEADS_KV 1 PAGE_SIZE 64)

fmha_prefill_add_chunk_family(paged_hd256
  PAGED HEAD_DIM 256 TILE_Q ${FMHA_PREFILL_TILED_Q_256} HEADS_Q 2 HEADS_KV 1 PAGE_SIZE 64)

fmha_prefill_add_chunk_family(nonpaged_hd128
  HEAD_DIM 128 TILE_Q ${FMHA_PREFILL_TILED_Q_NP_128} HEADS_Q 2 HEADS_KV 1)

fmha_prefill_add_case(chunk.hetero_paged_hd128_lists
  CHUNK PAGED CAUSAL
  BATCH 3 SEQLEN_Q 128 SEQLEN_K 257 SEQLEN_Q_LIST 1,32,127 PAST_KV_LIST 0,64,130
  HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(chunk.hetero_nonpaged_hd128_lists
  CHUNK CAUSAL
  BATCH 3 SEQLEN_Q 128 SEQLEN_K 257 SEQLEN_Q_LIST 1,32,255 PAST_KV_LIST 0,64,130
  HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128)

# AppendKV coverage: cache contains the old prefix; k_new/v_new are appended
# inside the FMHA kernel before attention reads the final KV length.
fmha_prefill_add_case(append.paged.hd64_zero_page
  APPENDKV PAGED CAUSAL
  BATCH 2 SEQLEN_Q 32 SEQLEN_K 96 SEQLEN_Q_LIST 1,32
  CACHE_SEQLENS_OLD 0,64 K_NEW_SEQLENS 1,32
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 64 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd64_tile_edge
  APPENDKV PAGED CAUSAL
  BATCH 2 SEQLEN_Q 128 SEQLEN_K 257 SEQLEN_Q_LIST 128,127
  CACHE_SEQLENS_OLD 129,127 K_NEW_SEQLENS 128,127
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 64 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd128_decode_boundary
  APPENDKV PAGED CAUSAL
  BATCH 2 SEQLEN_Q 1 SEQLEN_K 66 SEQLEN_Q_LIST 1,1
  CACHE_SEQLENS_OLD 63,65 K_NEW_SEQLENS 1
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd128_chunk_random_pt
  APPENDKV PAGED CAUSAL PAGE_TABLE_RANDOM
  BATCH 2 SEQLEN_Q 127 SEQLEN_K 254 SEQLEN_Q_LIST 32,127
  CACHE_SEQLENS_OLD 64,127 K_NEW_SEQLENS 32,127
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd128_mixed_general
  APPENDKV PAGED CAUSAL
  BATCH 2 SEQLEN_Q 128 SEQLEN_K 192 SEQLEN_Q_LIST 1,128
  CACHE_SEQLENS_OLD 128,64 K_NEW_SEQLENS 1,128
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd128_new65_page_tail
  APPENDKV PAGED CAUSAL
  BATCH 1 SEQLEN_Q 65 SEQLEN_K 128
  CACHE_SEQLENS_OLD 63 K_NEW_SEQLENS 65
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd96_new65_page_tail
  APPENDKV PAGED CAUSAL
  BATCH 1 SEQLEN_Q 65 SEQLEN_K 128
  CACHE_SEQLENS_OLD 63 K_NEW_SEQLENS 65
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 96 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd128_new129_cross_page
  APPENDKV PAGED CAUSAL
  BATCH 1 SEQLEN_Q 129 SEQLEN_K 256
  CACHE_SEQLENS_OLD 127 K_NEW_SEQLENS 129
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(append.paged.hd128_new511_large_nondiv
  APPENDKV PAGED CAUSAL
  BATCH 1 SEQLEN_Q 128 SEQLEN_K 8192
  CACHE_SEQLENS_OLD 7681 K_NEW_SEQLENS 511
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(append.nopaged.hd128_zero
  APPENDKV CAUSAL
  BATCH 2 SEQLEN_Q 32 SEQLEN_K 96 SEQLEN_Q_LIST 1,32
  CACHE_SEQLENS_OLD 0,64 K_NEW_SEQLENS 1,32
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128)

fmha_prefill_add_case(append.nopaged.hd128_tile_edge
  APPENDKV CAUSAL
  BATCH 2 SEQLEN_Q 128 SEQLEN_K 257 SEQLEN_Q_LIST 128,127
  CACHE_SEQLENS_OLD 129,127 K_NEW_SEQLENS 128,127
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128)

fmha_prefill_add_case(append.nopaged.hd128_boundary
  APPENDKV CAUSAL
  BATCH 2 SEQLEN_Q 1 SEQLEN_K 66 SEQLEN_Q_LIST 1,1
  CACHE_SEQLENS_OLD 63,65 K_NEW_SEQLENS 1
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128)

fmha_prefill_add_case(append.nopaged.hd128_cu_new
  APPENDKV CAUSAL
  BATCH 2 SEQLEN_Q 127 SEQLEN_K 254 SEQLEN_Q_LIST 32,127
  CACHE_SEQLENS_OLD 64,127 CU_SEQLENS_K_NEW 0,32,159
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128)

fmha_prefill_add_case(append.nopaged.hd128_mixed_general
  APPENDKV CAUSAL
  BATCH 2 SEQLEN_Q 128 SEQLEN_K 192 SEQLEN_Q_LIST 1,128
  CACHE_SEQLENS_OLD 128,64 K_NEW_SEQLENS 1,128
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128)

fmha_prefill_add_case(append.nopaged.hd128_new65_threshold
  APPENDKV CAUSAL
  BATCH 1 SEQLEN_Q 65 SEQLEN_K 128
  CACHE_SEQLENS_OLD 63 K_NEW_SEQLENS 65
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128)

fmha_prefill_add_case(append.nopaged.hd96_new65_threshold
  APPENDKV CAUSAL
  BATCH 1 SEQLEN_Q 65 SEQLEN_K 128
  CACHE_SEQLENS_OLD 63 K_NEW_SEQLENS 65
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 96)

fmha_prefill_add_case(append.nopaged.hd128_new129_nondiv
  APPENDKV CAUSAL
  BATCH 1 SEQLEN_Q 129 SEQLEN_K 256
  CACHE_SEQLENS_OLD 127 K_NEW_SEQLENS 129
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128)

# Generalized chunk-prefill accuracy cases, following the coverage dimensions
# used by vLLM/SGLang tests: mixed query/context lengths, decode-like q=1,
# page/block tails, GQA/MQA, local windows, and multi-request batches.
fmha_prefill_add_case(chunk.generalized_paged_batch2_q1024_k1024
  CHUNK PAGED CAUSAL
  BATCH 2 SEQLEN_Q 1024 SEQLEN_K 1024 SEQLEN_Q_LIST 1024,1024 PAST_KV_LIST 0,0
  HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(chunk.generalized_paged_mixed_decode_extend
  CHUNK PAGED CAUSAL
  BATCH 6 SEQLEN_Q 255 SEQLEN_K 1024 SEQLEN_Q_LIST 1,16,33,64,128,255
  PAST_KV_LIST 1023,17,64,0,129,257
  HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(chunk.generalized_paged_page128_tails
  CHUNK PAGED CAUSAL
  BATCH 4 SEQLEN_Q 129 SEQLEN_K 384 SEQLEN_Q_LIST 1,31,127,129
  PAST_KV_LIST 127,128,129,255
  HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 128)

fmha_prefill_add_case(chunk.generalized_paged_random_page_table
  CHUNK PAGED CAUSAL PAGE_TABLE_RANDOM
  BATCH 4 SEQLEN_Q 96 SEQLEN_K 320 SEQLEN_Q_LIST 1,32,64,96
  PAST_KV_LIST 63,64,129,224
  HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(chunk.generalized_paged_gqa_mixed_prefix
  CHUNK PAGED CAUSAL
  BATCH 4 SEQLEN_Q 128 SEQLEN_K 320 SEQLEN_Q_LIST 17,64,96,128
  PAST_KV_LIST 15,96,127,192
  HEADS_Q 8 HEADS_KV 2 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(chunk.generalized_paged_sglang_prefix_lens
  CHUNK PAGED CAUSAL
  BATCH 4 SEQLEN_Q 64 SEQLEN_K 128 SEQLEN_Q_LIST 64,64,64,64
  PAST_KV_LIST 16,32,48,64
  HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64)

fmha_prefill_add_case(chunk.generalized_paged_local_window_offsets
  CHUNK PAGED
  BATCH 3 SEQLEN_Q 10 SEQLEN_K 17 SEQLEN_Q_LIST 4,10,5 PAST_KV_LIST 2,7,4
  HEADS_Q 4 HEADS_KV 1 HEAD_DIM 128 PAGE_SIZE 64
  WINDOW_LEFT 4 WINDOW_RIGHT 0)

fmha_prefill_add_case(chunk.generalized_nonpaged_mixed_prefix
  CHUNK CAUSAL
  BATCH 4 SEQLEN_Q 129 SEQLEN_K 193 SEQLEN_Q_LIST 1,17,64,129
  PAST_KV_LIST 16,32,48,64
  HEADS_Q 2 HEADS_KV 1 HEAD_DIM 128)
