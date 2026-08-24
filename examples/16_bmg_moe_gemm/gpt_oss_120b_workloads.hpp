// Copyright (C) 2026 Intel Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gpt_oss_120b {

struct Workload {
  std::string name;
  std::vector<int32_t> rows;
  int n;
  int k;
};

inline const std::array<int32_t, 128> kTp4L0{
    50,32,48,37,60,54,32,267,100,33,72,12,1422,47,70,176,474,21,76,35,47,72,45,37,41,101,27,102,699,48,65,80,
    43,46,62,59,114,12,33,4,56,74,86,56,74,78,199,3,138,36,98,81,42,228,58,205,117,82,88,47,107,58,56,78,
    69,128,51,88,110,55,83,349,51,67,67,30,1028,58,46,55,39,39,52,538,398,112,72,190,196,21,61,33,35,23,
    208,43,39,18,46,137,53,42,57,682,48,124,32,32,151,29,56,82,53,59,98,88,39,151,51,124,39,45,49,80,
    85,40,20,2340};
inline const std::array<int32_t, 128> kTp4L14{
    283,2509,0,1,0,0,210,0,0,0,6,30,6,2,13,0,0,0,0,5,0,1,79,0,0,0,0,70,0,0,3487,0,
    3,0,0,131,32,1229,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,177,0,43,0,0,0,0,17,0,0,0,
    0,237,4,48,0,0,98,0,2,0,0,3,0,123,11,0,208,0,0,9,0,0,0,0,155,139,1,0,0,3,
    0,0,4,0,0,2,156,1,0,0,1617,404,0,13,0,69,0,1,4096,0,0,51,0,0,581,10,0,0,0,
    0,0,0,0};
inline const std::array<int32_t, 128> kTp4L35{
    0,0,0,76,22,32,0,1,0,0,0,3573,0,0,0,1,0,0,0,0,0,0,0,0,2,0,396,0,0,0,1,0,
    0,0,64,0,478,0,0,0,0,0,3,0,49,0,22,0,0,0,0,0,92,0,0,0,0,0,0,0,0,100,2,0,
    0,0,75,1,0,0,0,199,0,0,0,0,0,0,7,0,4,35,0,0,0,0,2,0,298,4,0,0,0,0,0,2736,
    0,0,5,0,12,0,541,12,23,40,0,0,0,0,0,26,0,398,0,0,4080,0,0,0,48,0,16,0,2,0,0,2906};
inline const std::array<int32_t, 128> kTp8L0{
    73,29,29,45,53,43,27,208,122,29,72,16,1461,53,47,173,502,18,60,41,48,54,39,24,42,113,28,104,733,64,104,60,
    45,53,80,57,120,9,32,4,46,61,94,59,72,75,245,5,172,38,75,72,48,223,45,177,126,78,90,43,68,44,65,59,
    64,145,70,84,85,60,68,322,55,65,64,28,1101,55,46,63,35,45,43,542,350,122,75,186,176,16,64,35,17,25,
    186,57,26,18,52,134,52,51,38,651,52,156,49,30,148,19,46,91,52,67,115,86,39,146,54,136,35,49,37,49,
    81,39,25,2418};
inline const std::array<int32_t, 128> kTp8L14{
    329,2489,0,1,0,0,213,0,0,0,4,30,7,0,8,0,0,0,0,10,0,0,97,0,0,0,0,68,0,0,3510,0,
    6,0,0,98,27,1461,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,139,0,49,0,0,0,0,39,0,0,0,
    0,265,3,45,0,0,107,0,5,0,0,1,0,115,12,0,164,0,0,13,0,0,0,0,112,138,3,0,0,5,
    0,1,2,0,0,0,120,0,0,0,1549,336,0,7,0,74,0,4,4094,0,0,52,0,0,557,11,0,0,0,
    0,0,0,0};
inline const std::array<int32_t, 128> kTp8L35{
    0,0,0,107,13,34,0,1,0,0,0,3501,0,0,1,2,0,0,0,1,0,0,0,0,0,0,368,0,0,0,2,0,
    0,0,57,0,650,0,0,0,0,0,2,0,90,0,14,0,0,0,0,0,83,1,0,0,0,0,0,0,0,133,3,0,
    0,0,40,3,0,0,0,179,0,0,0,0,1,0,4,0,4,66,0,0,0,0,2,0,299,4,0,0,0,0,0,2713,
    2,0,11,0,14,0,585,12,29,24,0,0,0,0,0,15,0,377,0,0,4076,0,0,0,33,0,29,0,1,0,0,2798};

template <size_t N>
inline std::vector<int32_t> rows(const std::array<int32_t, N>& values) {
  return {values.begin(), values.end()};
}

inline std::vector<int32_t> decode_rows(std::initializer_list<int> ids) {
  std::vector<int32_t> values(128, 0);
  for (int id : ids) values[id] = 1;
  return values;
}

inline std::vector<Workload> workloads() {
  constexpr int tp4_gemm1_n = 1472, tp4_gemm1_k = 2880, tp4_gemm2_n = 2880, tp4_gemm2_k = 736;
  constexpr int tp8_gemm1_n = 768, tp8_gemm1_k = 2880, tp8_gemm2_n = 2880, tp8_gemm2_k = 384;
  return {
      {"gpt-oss-120b-prefill-l0-gemm1", rows(kTp4L0), tp4_gemm1_n, tp4_gemm1_k},
      {"gpt-oss-120b-prefill-l0-gemm2", rows(kTp4L0), tp4_gemm2_n, tp4_gemm2_k},
      {"gpt-oss-120b-prefill-l14-gemm1", rows(kTp4L14), tp4_gemm1_n, tp4_gemm1_k},
      {"gpt-oss-120b-prefill-l14-gemm2", rows(kTp4L14), tp4_gemm2_n, tp4_gemm2_k},
      {"gpt-oss-120b-prefill-l35-gemm1", rows(kTp4L35), tp4_gemm1_n, tp4_gemm1_k},
      {"gpt-oss-120b-prefill-l35-gemm2", rows(kTp4L35), tp4_gemm2_n, tp4_gemm2_k},
      {"gpt-oss-120b-decode-gemm1", decode_rows({14, 49, 79, 118}), tp4_gemm1_n, tp4_gemm1_k},
      {"gpt-oss-120b-decode-gemm2", decode_rows({14, 49, 79, 118}), tp4_gemm2_n, tp4_gemm2_k},
      {"gpt-oss-120b-tp8-prefill-l0-gemm1", rows(kTp8L0), tp8_gemm1_n, tp8_gemm1_k},
      {"gpt-oss-120b-tp8-prefill-l0-gemm2", rows(kTp8L0), tp8_gemm2_n, tp8_gemm2_k},
      {"gpt-oss-120b-tp8-prefill-l14-gemm1", rows(kTp8L14), tp8_gemm1_n, tp8_gemm1_k},
      {"gpt-oss-120b-tp8-prefill-l14-gemm2", rows(kTp8L14), tp8_gemm2_n, tp8_gemm2_k},
      {"gpt-oss-120b-tp8-prefill-l35-gemm1", rows(kTp8L35), tp8_gemm1_n, tp8_gemm1_k},
      {"gpt-oss-120b-tp8-prefill-l35-gemm2", rows(kTp8L35), tp8_gemm2_n, tp8_gemm2_k},
      {"gpt-oss-120b-tp8-decode-gemm1", decode_rows({22, 36, 73, 115}), tp8_gemm1_n, tp8_gemm1_k},
      {"gpt-oss-120b-tp8-decode-gemm2", decode_rows({22, 36, 73, 115}), tp8_gemm2_n, tp8_gemm2_k},
  };
}

}  // namespace gpt_oss_120b
