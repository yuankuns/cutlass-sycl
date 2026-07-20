if (CUTLASS_TEST_FOR_CRI)
  set(BMG_MOE_GATE_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
  set(BMG_MOE_GATE_GEMV_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
  set(BMG_MOE_GATE_GEMV_FUSED_QUICK_ARGS "--suite=quick" "--iterations=${CRI_DEFAULT_ITERATIONS}" "--verify=${CRI_DEFAULT_VERIFY}")
else()
  set(BMG_MOE_GATE_QUICK_ARGS "--suite=quick" "--iterations=20" "--verify=1")
  set(BMG_MOE_GATE_GEMV_QUICK_ARGS "--suite=quick" "--iterations=20" "--verify=1")
  set(BMG_MOE_GATE_GEMV_FUSED_QUICK_ARGS "--suite=quick" "--iterations=20" "--verify=1")
endif()

cutlass_example_add_executable(
  16_bmg_moe_gate_topk_renorm
  16_bmg_moe_gate_topk_renorm.cpp
  TEST_COMMAND_OPTIONS
  BMG_MOE_GATE_QUICK_ARGS
)

cutlass_example_add_executable(
  16_bmg_moe_gate_gemv
  16_bmg_moe_gate_gemv.cpp
  TEST_COMMAND_OPTIONS
  BMG_MOE_GATE_GEMV_QUICK_ARGS
)

cutlass_example_add_executable(
  16_bmg_moe_gate_gemv_fused
  16_bmg_moe_gate_gemv_fused.cpp
  TEST_COMMAND_OPTIONS
  BMG_MOE_GATE_GEMV_FUSED_QUICK_ARGS
)
