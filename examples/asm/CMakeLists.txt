set(TEST_GROUPS --groups=128)

cutlass_example_add_executable(
  test_lsc 
  test_lsc.cpp
)

target_link_options(test_lsc PUBLIC)
