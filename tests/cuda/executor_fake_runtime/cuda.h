#pragma once
#include <cstddef>
typedef int CUresult;
constexpr CUresult CUDA_SUCCESS = 0;
inline CUresult cuInit(unsigned) { return CUDA_SUCCESS; }
